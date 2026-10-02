#!/usr/bin/env python3
"""Render a three-light material calibration sheet with Blender 4.2 Cycles.

blender --background --python-exit-code 1 --python render_calibration.py -- \
  --material-json canonical_material.json --profile renderer_profile.json \
  --output-root new_calibration_run --base-scene-blendfile base_scene.blend \
  --sphere-blendfile Sphere.blend

The full counterfactual grid is deliberately unavailable from this entrypoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material-json", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-scene-blendfile", type=Path, required=True)
    parser.add_argument("--sphere-blendfile", type=Path, required=True)
    return parser.parse_args(argv)


def load_inputs(args: argparse.Namespace) -> tuple[dict, dict]:
    material = json.loads(args.material_json.read_text(encoding="utf-8"))
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    for key in ("roughness", "metallic"):
        value = material[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"Canonical {key} must be a finite number in [0, 1]")
    if profile["schema_version"] != 1 or profile["profile_id"] != "natural_material_calibration_v1":
        raise ValueError("Unsupported calibration renderer profile")
    if len(profile["lighting_conditions"]) != 3:
        raise ValueError("Calibration requires exactly three lighting conditions")
    if tuple(profile["blender"]["version"]) != bpy.app.version:
        raise RuntimeError(f"Blender version mismatch: {bpy.app.version}")
    if profile["blender"]["render_engine"] != "CYCLES":
        raise ValueError("Calibration requires Cycles")
    for path in (args.base_scene_blendfile, args.sphere_blendfile):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_root.exists():
        raise FileExistsError(f"Use a new output directory: {args.output_root}")
    return material, profile


def configure_scene(profile: dict, seed: int, condition: dict) -> tuple[object, dict, list[dict]]:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = profile["blender"]["cycles_samples"]
    scene.cycles.seed = seed
    scene.cycles.device = profile["blender"]["cycles_device"]
    expected_visible = profile["blender"]["cuda_visible_devices"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be {expected_visible}")
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "CUDA"
    preferences.get_devices()
    gpu_devices = [device for device in preferences.devices if device.type == "CUDA"]
    if len(gpu_devices) != 1 or profile["blender"]["expected_gpu_name_contains"] not in gpu_devices[0].name:
        raise RuntimeError(f"Expected one visible V100 CUDA device, got {[d.name for d in gpu_devices]}")
    for device in preferences.devices:
        device.use = device.type == "CUDA"
    device_records = [{"name": device.name, "type": device.type, "id": device.id}
                      for device in gpu_devices]
    scene.render.resolution_x, scene.render.resolution_y = profile["blender"]["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.film_transparent = False

    world = scene.world
    world.use_nodes = True
    background = next(node for node in world.node_tree.nodes if node.type == "BACKGROUND")
    background.inputs["Color"].default_value = profile["background"]["world_rgba"]
    background.inputs["Strength"].default_value = condition["world_strength"]
    ground = bpy.data.objects["Ground"]
    ground_mat = bpy.data.materials.new("CalibrationNeutralGround")
    ground_mat.use_nodes = True
    ground_bsdf = ground_mat.node_tree.nodes.get("Principled BSDF")
    ground_bsdf.inputs["Base Color"].default_value = profile["background"]["ground_rgba"]
    ground_bsdf.inputs["Metallic"].default_value = 0.0
    ground_bsdf.inputs["Roughness"].default_value = 1.0
    ground.data.materials.clear()
    ground.data.materials.append(ground_mat)

    light_records = {}
    for name, spec in condition["lights"].items():
        light = bpy.data.objects[name]
        if light.type != "LIGHT":
            raise ValueError(f"Expected Blender light: {name}")
        light.location = [a + b for a, b in zip(light.location, spec["offset_xyz"])]
        light.data.energy *= spec["energy_scale"]
        light.data.color = (1.0, 1.0, 1.0)
        light_records[name] = {
            "location_xyz": list(light.location), "energy": light.data.energy,
            "type": light.data.type, "color": list(light.data.color),
        }
    return scene, light_records, device_records


def append_sphere(path: Path, profile: dict):
    object_name = profile["object"]["asset_name"]
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.append(directory=str(path / "Object") + os.sep, filename=object_name, link=False)
    appended = [obj for obj in bpy.data.objects if obj.name not in before]
    if len(appended) != 1 or appended[0].type != "MESH":
        raise RuntimeError(f"Expected one sphere mesh in {path}")
    obj = appended[0]
    scale = profile["object"]["scale"]
    obj.location = (0.0, 0.0, scale)
    obj.scale = (scale, scale, scale)
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.pass_index = 1
    return obj


def set_material(obj, material: dict, profile: dict) -> None:
    shader = bpy.data.materials.new("ExtractedCanonicalMaterial")
    shader.use_nodes = True
    nodes = shader.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = profile["material"]["base_color_linear_rgba"]
    bsdf.inputs["Roughness"].default_value = material["roughness"]
    bsdf.inputs["Metallic"].default_value = material["metallic"]
    shader.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(shader)


def configure_camera(obj, profile: dict) -> dict:
    camera = bpy.data.objects[profile["camera"]["name"]]
    for constraint in camera.constraints:
        constraint.mute = True
    camera.data.dof.use_dof = False
    direction = Vector(obj.location) - camera.location
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = direction.to_track_quat("-Z", "Y")
    bpy.context.scene.camera = camera
    return {
        "location_xyz": list(camera.location),
        "rotation_quaternion_wxyz": list(camera.rotation_quaternion),
        "lens": camera.data.lens,
    }


def configure_mask(output_dir: Path) -> None:
    scene = bpy.context.scene
    scene.use_nodes = True
    bpy.context.view_layer.use_pass_object_index = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    layers = nodes.new("CompositorNodeRLayers")
    composite = nodes.new("CompositorNodeComposite")
    scene.node_tree.links.new(layers.outputs["Image"], composite.inputs["Image"])
    mask = nodes.new("CompositorNodeIDMask")
    mask.index = 1
    mask.use_antialiasing = True
    scene.node_tree.links.new(layers.outputs["IndexOB"], mask.inputs["ID value"])
    output = nodes.new("CompositorNodeOutputFile")
    output.base_path = str(output_dir)
    output.file_slots[0].path = "__mask_"
    output.format.file_format = "PNG"
    output.format.color_mode = "BW"
    output.format.color_depth = "8"
    scene.node_tree.links.new(mask.outputs["Alpha"], output.inputs[0])
    set_alpha = nodes.new("CompositorNodeSetAlpha")
    scene.node_tree.links.new(layers.outputs["Image"], set_alpha.inputs["Image"])
    scene.node_tree.links.new(mask.outputs["Alpha"], set_alpha.inputs["Alpha"])
    rgba_output = nodes.new("CompositorNodeOutputFile")
    rgba_output.base_path = str(output_dir)
    rgba_output.file_slots[0].path = "__input_rgba_"
    rgba_output.format.file_format = "PNG"
    rgba_output.format.color_mode = "RGBA"
    rgba_output.format.color_depth = "8"
    scene.node_tree.links.new(set_alpha.outputs["Image"], rgba_output.inputs[0])


def render_one(args: argparse.Namespace, material: dict, profile: dict,
               name: str, condition: dict, seed: int, run_hashes: dict) -> dict:
    bpy.ops.wm.open_mainfile(filepath=str(args.base_scene_blendfile.resolve()))
    scene, lights, devices = configure_scene(profile, seed, condition)
    for existing in list(scene.objects):
        if existing.type == "MESH" and existing.name != "Ground":
            bpy.data.objects.remove(existing, do_unlink=True)
    obj = append_sphere(args.sphere_blendfile.resolve(), profile)
    set_material(obj, material, profile)
    camera = configure_camera(obj, profile)
    output_dir = args.output_root / name
    output_dir.mkdir(parents=True)
    configure_mask(output_dir)
    image = output_dir / "image.png"
    scene.render.filepath = str(image)
    bpy.ops.render.render(write_still=True)
    generated_masks = list(output_dir.glob("__mask_*.png"))
    if len(generated_masks) != 1:
        raise RuntimeError(f"Expected exactly one object mask, got {generated_masks}")
    mask = output_dir / "object_mask.png"
    generated_masks[0].rename(mask)
    generated_rgba = list(output_dir.glob("__input_rgba_*.png"))
    if len(generated_rgba) != 1:
        raise RuntimeError(f"Expected exactly one masked RGBA image, got {generated_rgba}")
    input_rgba = output_dir / "input_rgba.png"
    generated_rgba[0].rename(input_rgba)
    if not image.is_file() or not mask.is_file() or not input_rgba.is_file():
        raise RuntimeError("Render, object mask, or RGBA input was not written")
    record = {
        "condition": name, "seed": seed, "roughness": material["roughness"],
        "metallic": material["metallic"],
        "base_color_linear_rgba": profile["material"]["base_color_linear_rgba"],
        "camera": camera, "lights": lights,
        "world_strength": condition["world_strength"],
        "renderer": {
            "blender_version": list(bpy.app.version), "engine": scene.render.engine,
            "cycles_samples": scene.cycles.samples, "cycles_device": scene.cycles.device,
            "cuda_devices": devices,
            "resolution": [scene.render.resolution_x, scene.render.resolution_y],
            "color_management": {
                "display_device": scene.display_settings.display_device,
                "view_transform": scene.view_settings.view_transform,
                "look": scene.view_settings.look,
                "exposure": scene.view_settings.exposure, "gamma": scene.view_settings.gamma,
            },
        },
        "input_sha256": run_hashes,
        "image_sha256": sha256(image), "object_mask_sha256": sha256(mask),
        "input_rgba_sha256": sha256(input_rgba),
        "git_commit": git_commit(),
    }
    write_json(output_dir / "metadata.json", record)
    record["metadata_sha256"] = sha256(output_dir / "metadata.json")
    return record


def main() -> None:
    args = parse_args()
    args.material_json = args.material_json.resolve()
    args.profile = args.profile.resolve()
    args.output_root = args.output_root.resolve()
    material, profile = load_inputs(args)
    run_hashes = {
        "canonical_material_json": sha256(args.material_json),
        "renderer_profile_json": sha256(args.profile),
        "clevr_base_scene_blend": sha256(args.base_scene_blendfile),
        "clevr_sphere_blend": sha256(args.sphere_blendfile),
    }
    args.output_root.mkdir(parents=True)
    records = []
    for name, condition in profile["lighting_conditions"].items():
        records.append(render_one(args, material, profile, name, condition,
                                  profile["blender"]["seed"], run_hashes))
    write_json(args.output_root / "manifest.json", {
        "status": "calibration_complete", "image_count": len(records),
        "input_sha256": run_hashes, "git_commit": git_commit(), "records": records,
    })


if __name__ == "__main__":
    main()
