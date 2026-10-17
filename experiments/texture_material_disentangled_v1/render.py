"""Render the isolated Phase 1 oracle plan using the existing Cycles scene.

Run with Blender --background --python-exit-code 1 --python this_file --
--plan new_plan/plan.json --output new_renders --gpu 3.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys


PROJECT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("protocol.json")
sys.path.insert(0, str(PROJECT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_sha256(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def validate_plan(plan, protocol):
    """Validate immutable inputs before opening a scene or allocating a GPU."""
    if plan.get("status") != "complete" or plan.get("protocol_sha256") != protocol_sha256(protocol):
        raise ValueError("Plan is incomplete or belongs to a different protocol")
    rows = plan["records"]
    expected_eval = len(protocol["dataset"]["heldout_contexts"]) * 4
    if len(rows) != protocol["dataset"]["expected_train_images"] + expected_eval:
        raise ValueError("Plan image count differs from the Phase 1 protocol")
    seen = set()
    checked_textures = {}
    for row in rows:
        identifier = row["id"]
        if (not isinstance(identifier, str) or not identifier or identifier in (".", "..")
                or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in identifier)):
            raise ValueError(f"Unsafe render ID: {identifier!r}")
        if identifier in seen:
            raise ValueError(f"Duplicate render ID: {identifier}")
        seen.add(identifier)
        if row["split"] not in ("train", "eval"):
            raise ValueError(f"Invalid split: {identifier}")
        if row["shape"] not in ("sphere", "cube", "cylinder", "cone"):
            raise ValueError(f"Unsupported shape: {identifier}")
        if row["texture_id"] not in protocol["textures"] or row["material_id"] not in protocol["materials"]:
            raise ValueError(f"Unknown concept: {identifier}")
        if row["color"] not in protocol["dataset"]["colors_linear_rgb"]:
            raise ValueError(f"Unknown color: {identifier}")
        if row["light"] not in protocol["dataset"]["light_phrases"]:
            raise ValueError(f"Unknown light: {identifier}")
        if isinstance(row["seed"], bool) or not isinstance(row["seed"], int) or row["seed"] < 0:
            raise ValueError(f"Invalid render seed: {identifier}")
        texture = Path(row["texture_path"])
        if not texture.is_absolute() or not texture.is_file():
            raise ValueError(f"Texture must be an existing absolute path: {texture}")
        if texture not in checked_textures:
            checked_textures[texture] = sha256(texture)
        if checked_textures[texture] != row["texture_sha256"]:
            raise ValueError(f"Texture hash mismatch: {texture}")
    return rows


def ensure_uv(obj, shape, bpy):
    policy = "asset_uv"
    if not obj.data.uv_layers:
        for selected in bpy.context.selected_objects:
            selected.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        if shape == "cube":
            bpy.ops.uv.cube_project(cube_size=2.0)
            policy = "generated_cube_project"
        elif shape in ("cylinder", "cone"):
            bpy.ops.uv.cylinder_project()
            policy = "generated_cylinder_project"
        else:
            bpy.ops.uv.sphere_project()
            policy = "generated_sphere_project"
        bpy.ops.object.mode_set(mode="OBJECT")
    layer = obj.data.uv_layers.active
    if layer is None or not len(layer.data):
        raise RuntimeError(f"No UV coordinates for {shape}")
    digest = hashlib.sha256()
    for loop in layer.data:
        coordinates = tuple(float(value) for value in loop.uv)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"Nonfinite UV coordinates: {shape}")
        digest.update(struct.pack("<2f", *coordinates))
    return {"policy": policy, "layer": layer.name, "loop_count": len(layer.data)}, digest.hexdigest()


def set_material(obj, row, protocol, bpy):
    material = protocol["materials"][row["material_id"]]
    color = protocol["dataset"]["colors_linear_rgb"][row["color"]]
    shader = bpy.data.materials.new("TextureMaterialOracle")
    shader.use_nodes = True
    nodes = shader.node_tree.nodes
    nodes.clear()
    links = shader.node_tree.links
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    for name, value in (("Roughness", material["roughness"]), ("Metallic", material["metallic"]),
                        ("IOR", 1.5), ("Specular IOR Level", 0.5), ("Coat Weight", 0.0),
                        ("Transmission Weight", 0.0), ("Alpha", 1.0)):
        bsdf.inputs[name].default_value = value
    coords = nodes.new("ShaderNodeTexCoord")
    scale = nodes.new("ShaderNodeVectorMath")
    scale.operation = "MULTIPLY"
    repeats = protocol["texture_extraction"]["uv_repeats"]
    scale.inputs[1].default_value = (repeats, repeats, repeats)
    links.new(coords.outputs["UV"], scale.inputs[0])
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(row["texture_path"], check_existing=False)
    # Stored values are already linear grayscale reflectance, not sRGB RGB.
    texture.image.colorspace_settings.name = "Non-Color"
    texture.projection = "FLAT"
    texture.extension = "REPEAT"
    texture.interpolation = "Linear"
    links.new(scale.outputs["Vector"], texture.inputs["Vector"])
    multiply = nodes.new("ShaderNodeMixRGB")
    multiply.blend_type = "MULTIPLY"
    multiply.inputs[0].default_value = 1.0
    multiply.inputs[2].default_value = (*color, 1.0)
    links.new(texture.outputs["Color"], multiply.inputs[1])
    links.new(multiply.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(shader)
    return {"roughness": material["roughness"], "metallic": material["metallic"],
            "base_color_linear_rgb": color, "texture_projection": "UV", "uv_repeats": repeats,
            "texture_colorspace": "Non-Color", "normal_bump_displacement": False,
            "shader": "Principled BSDF", "ior": 1.5, "specular_ior_level": 0.5,
            "coat_weight": 0.0, "transmission_weight": 0.0}


def render_one(row, output, protocol, profile, bpy, base, append_shape):
    bpy.ops.wm.open_mainfile(filepath=protocol["renderer"]["base_scene"])
    scene, lights, devices = base.configure_scene(
        profile, row["seed"], profile["lighting_conditions"][row["light"]])
    for existing in list(scene.objects):
        if existing.type == "MESH" and existing.name != "Ground":
            bpy.data.objects.remove(existing, do_unlink=True)
    obj = append_shape(row["shape"], Path(protocol["renderer"]["shape_dir"]),
                       protocol["dataset"]["object_scale"])
    uv_mapping, uv_hash = ensure_uv(obj, row["shape"], bpy)
    shader = set_material(obj, row, protocol, bpy)
    camera = base.configure_camera(obj, profile)
    folder = output / row["id"]
    folder.mkdir()
    base.configure_mask(folder)
    image = folder / "image.png"
    scene.render.filepath = str(image)
    bpy.ops.render.render(write_still=True)
    for prefix, name in (("__mask_", "object_mask.png"), ("__input_rgba_", "input_rgba.png")):
        files = list(folder.glob(prefix + "*.png"))
        if len(files) != 1:
            raise RuntimeError(f"Expected one {prefix} output: {files}")
        files[0].rename(folder / name)
    mask, rgba = folder / "object_mask.png", folder / "input_rgba.png"
    record = {**row, **shader, "uv_mapping": uv_mapping, "uv_sha256": uv_hash,
              "image": str(image), "mask": str(mask), "input_rgba": str(rgba),
              "image_sha256": sha256(image), "mask_sha256": sha256(mask),
              "input_rgba_sha256": sha256(rgba), "camera": camera, "lights": lights,
              "world_strength": profile["lighting_conditions"][row["light"]]["world_strength"],
              "cuda_devices": devices, "cycles_samples": scene.cycles.samples,
              "blender_version": list(bpy.app.version), "render_seed": scene.cycles.seed,
              "object_scale": protocol["dataset"]["object_scale"],
              "resolution": [scene.render.resolution_x, scene.render.resolution_y]}
    metadata = folder / "metadata.json"
    base.write_json(metadata, record)
    return {**record, "metadata": str(metadata), "metadata_sha256": sha256(metadata)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu:
        raise ValueError("GPU argument and CUDA_VISIBLE_DEVICES disagree")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    rows = validate_plan(plan, protocol)
    renderer = protocol["renderer"]
    scene = Path(renderer["base_scene"])
    if sha256(scene) != renderer["base_scene_sha256"]:
        raise ValueError("Base scene hash differs from the frozen oracle protocol")
    profile_path = PROJECT / renderer["profile"]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["blender"]["cuda_visible_devices"] = args.gpu
    profile["blender"]["cycles_samples"] = renderer["cycles_samples"]
    profile["blender"]["resolution"] = [renderer["resolution"]] * 2
    if any(row["light"] not in profile["lighting_conditions"] for row in rows):
        raise ValueError("Plan lighting condition is absent from the renderer profile")
    from experiments.natural_material_threeway_v1.render_grid import ASSETS
    shape_hashes = {shape: sha256(Path(renderer["shape_dir"]) / f"{ASSETS[shape]}.blend")
                    for shape in sorted({row["shape"] for row in rows} - {"cone"})}
    import bpy
    from experiments.natural_material_supermat_v1 import render_calibration as base
    from experiments.natural_material_threeway_v1.render_grid import append_shape
    if tuple(bpy.app.version) != (4, 2, 11):
        raise RuntimeError(f"Expected Blender 4.2.11, got {bpy.app.version}")
    evidence = {"protocol_sha256": protocol_sha256(protocol), "plan_sha256": sha256(args.plan),
                "profile_sha256": sha256(profile_path), "base_scene_sha256": sha256(scene),
                "shape_sha256": shape_hashes, "script_sha256": sha256(__file__),
                "git_commit": base.git_commit(), "expected_images": len(rows)}
    output.mkdir(parents=True)
    records = []
    try:
        for row in rows:
            records.append(render_one(row, output, protocol, profile, bpy, base, append_shape))
        base.write_json(output / "manifest.json", {**evidence, "status": "complete",
                        "image_count": len(records), "records": records})
    except Exception as error:
        base.write_json(output / "manifest.json", {**evidence, "status": "failed",
                        "image_count": len(records), "records": records,
                        "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
