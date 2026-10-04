"""Render fixed SuperMat A/R/M texture maps across shape, light, and view."""

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, os.environ.get("COLORPEEL_PROJECT_ROOT",
                                  str(Path(__file__).resolve().parents[2])))
from experiments.natural_material_supermat_v1 import render_calibration as base
from experiments.natural_material_threeway_v1.render_grid import append_shape


def set_full_material(obj, texture_dir, repeats):
    shader = bpy.data.materials.new("SuperMatFullMaps")
    shader.use_nodes = True
    nodes = shader.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    coords = nodes.new("ShaderNodeTexCoord")
    scale = nodes.new("ShaderNodeVectorMath")
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = (repeats, repeats, repeats)
    shader.node_tree.links.new(coords.outputs["Generated"], scale.inputs[0])
    for channel, socket in (("albedo", "Base Color"),
                            ("roughness", "Roughness"), ("metallic", "Metallic")):
        path = texture_dir / f"{channel}.png"
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = bpy.data.images.load(str(path.resolve()), check_existing=False)
        texture.image.colorspace_settings.name = "sRGB" if channel == "albedo" else "Non-Color"
        texture.projection = "BOX"
        texture.projection_blend = 0.2
        texture.extension = "REPEAT"
        shader.node_tree.links.new(scale.outputs["Vector"], texture.inputs["Vector"])
        shader.node_tree.links.new(texture.outputs["Color"], bsdf.inputs[socket])
    shader.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(shader)


def render_one(args, manifest, profile, grid, spec, index):
    shape, light, azimuth, height = spec
    view = f"az{azimuth:03d}_h{height:g}"
    out = args.output_root / shape / light / view
    out.mkdir(parents=True, exist_ok=False)
    bpy.ops.wm.open_mainfile(filepath=str(args.base_scene.resolve()))
    scene, lights, devices = base.configure_scene(
        profile, grid["seed"] + index, profile["lighting_conditions"][light])
    for existing in list(scene.objects):
        if existing.type == "MESH" and existing.name != "Ground":
            bpy.data.objects.remove(existing, do_unlink=True)
    obj = append_shape(shape, args.shape_dir, grid["object_scale"])
    set_full_material(obj, args.maps_root / "textures" / args.material_id,
                      grid["texture_repeats"])
    camera = bpy.data.objects[profile["camera"]["name"]]
    offset = camera.location - Vector(obj.location)
    camera.location = (Vector(obj.location) +
                       Matrix.Rotation(math.radians(azimuth), 4, "Z") @ offset)
    camera.location.z += height
    camera_record = base.configure_camera(obj, profile)
    base.configure_mask(out)
    image = out / "image.png"
    scene.render.filepath = str(image)
    bpy.ops.render.render(write_still=True)
    for prefix, name in (("__mask_", "object_mask.png"),
                         ("__input_rgba_", "input_rgba.png")):
        files = list(out.glob(prefix + "*.png"))
        if len(files) != 1:
            raise RuntimeError(f"Expected one {prefix} file in {out}: {files}")
        files[0].rename(out / name)
    record = {
        "index": index, "shape": shape, "light": light, "view": view,
        "azimuth_degrees": azimuth, "camera_height_offset": height,
        "seed": grid["seed"] + index,
        "split": "held_out_shape" if shape == grid["held_out_shape"] else "train",
        "material_id": args.material_id,
        "texture_sha256": {channel: manifest["materials"][args.material_id]["maps"][channel]["texture_sha256"]
                           for channel in ("albedo", "roughness", "metallic")},
        "texture_projection": "Generated coordinates, box projection, repeat",
        "texture_repeats": grid["texture_repeats"],
        "object_scale": grid["object_scale"],
        "camera": camera_record, "lights": lights, "cuda_devices": devices,
        "cycles_samples": scene.cycles.samples,
        "image_sha256": base.sha256(image),
        "mask_sha256": base.sha256(out / "object_mask.png"),
        "input_rgba_sha256": base.sha256(out / "input_rgba.png"),
    }
    base.write_json(out / "metadata.json", record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-id", choices=("mailbox", "metal_spoon", "wood_spoon"), required=True)
    parser.add_argument("--maps-root", type=Path, required=True)
    parser.add_argument("--grid-config", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--base-scene", type=Path, required=True)
    parser.add_argument("--shape-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    for path in (args.maps_root, args.grid_config, args.profile, args.base_scene, args.shape_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu:
        raise ValueError("GPU argument and CUDA_VISIBLE_DEVICES disagree")
    manifest = json.loads((args.maps_root / "manifest.json").read_text())
    grid = json.loads(args.grid_config.read_text())
    profile = json.loads(args.profile.read_text())
    profile["blender"]["cuda_visible_devices"] = args.gpu
    if manifest["status"] != "complete" or grid["schema_version"] != 1:
        raise ValueError("Invalid full-map preparation or grid config")
    for channel in ("albedo", "roughness", "metallic"):
        path = args.maps_root / "textures" / args.material_id / f"{channel}.png"
        expected = manifest["materials"][args.material_id]["maps"][channel]["texture_sha256"]
        if base.sha256(path) != expected:
            raise ValueError(f"Texture changed: {path}")
    specs = list(itertools.product(grid["shapes"], profile["lighting_conditions"],
                                   grid["azimuth_degrees"], grid["camera_height_offsets"]))
    args.output_root.mkdir(parents=True)
    records = [render_one(args, manifest, profile, grid, spec, index)
               for index, spec in enumerate(specs[:args.max_images])]
    base.write_json(args.output_root / "manifest.json", {
        "status": "complete" if len(records) == len(specs) else "smoke",
        "material_id": args.material_id, "expected_images": len(specs),
        "image_count": len(records), "git_commit": base.git_commit(),
        "input_sha256": {"maps_manifest": base.sha256(args.maps_root / "manifest.json"),
                         "grid_config": base.sha256(args.grid_config),
                         "profile": base.sha256(args.profile),
                         "base_scene": base.sha256(args.base_scene),
                         "script": base.sha256(Path(__file__))},
        "blender_version": list(bpy.app.version), "records": records,
    })


if __name__ == "__main__":
    main()
