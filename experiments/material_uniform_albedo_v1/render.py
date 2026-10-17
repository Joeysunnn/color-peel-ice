"""Render uniform-albedo Material counterfactuals with the existing Cycles scene."""

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from experiments.material_uniform_albedo_v1 import protocol as schema
from experiments.texture_material_disentangled_v1.render import sha256


def validate_plan(plan, protocol):
    """Reject changed or unsafe render inputs before importing Blender."""
    if plan.get("status") != "complete" or plan.get("protocol_sha256") != schema.protocol_sha256(protocol):
        raise ValueError("Plan is incomplete or belongs to a different protocol")
    rows = plan["records"]
    for row in rows:
        identifier = row["id"]
        if (not isinstance(identifier, str) or not identifier or identifier in (".", "..")
                or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                       for character in identifier)):
            raise ValueError(f"Unsafe render ID: {identifier!r}")
    if rows != schema.records(protocol):
        raise ValueError("Plan records differ from the frozen uniform Material protocol")
    return rows


def set_material(obj, row, protocol, bpy):
    material = protocol["materials"][row["material_id"]]
    color = protocol["dataset"]["colors_linear_rgb"][row["color"]]
    rgba = [0.5 * channel for channel in color] + [1.0]
    shader = bpy.data.materials.new("UniformMaterialOracle")
    shader.use_nodes = True
    nodes = shader.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    for name, value in (("Base Color", rgba), ("Roughness", material["roughness"]),
                        ("Metallic", material["metallic"]), ("IOR", 1.5),
                        ("Specular IOR Level", 0.5), ("Coat Weight", 0.0),
                        ("Transmission Weight", 0.0), ("Alpha", 1.0)):
        bsdf.inputs[name].default_value = value
    shader.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(shader)
    return {"roughness": material["roughness"], "metallic": material["metallic"],
            "base_color_linear_rgb": color, "uniform_base_color_linear_rgba": rgba,
            "albedo_mode": "uniform", "albedo_level": 0.5,
            "texture_image_nodes": 0, "normal_bump_displacement": False,
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
    record = {**row, **shader, "image": str(image), "mask": str(mask), "input_rgba": str(rgba),
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
    protocol = schema.read_protocol()
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
                    for shape in sorted({row["shape"] for row in rows})}
    import bpy
    from experiments.natural_material_supermat_v1 import render_calibration as base
    from experiments.natural_material_threeway_v1.render_grid import append_shape
    if tuple(bpy.app.version) != (4, 2, 11):
        raise RuntimeError(f"Expected Blender 4.2.11, got {bpy.app.version}")
    evidence = {"protocol_sha256": schema.protocol_sha256(protocol), "plan_sha256": sha256(args.plan),
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
