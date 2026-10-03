"""Render one fixed R/M material over a deterministic Cycles counterfactual grid."""

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.natural_material_supermat_v1 import render_calibration as base


ASSETS = {"sphere": "Sphere", "cube": "SmoothCube_v2",
          "cylinder": "SmoothCylinder"}


def append_shape(shape, shape_dir, scale):
    if shape == "cone":
        bpy.ops.mesh.primitive_cone_add(vertices=64, radius1=1, radius2=0,
                                        depth=2, location=(0, 0, scale))
        obj = bpy.context.object
        obj.name = "CounterfactualCone"
        obj.scale = (scale, scale, scale)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
    else:
        name = ASSETS[shape]
        path = shape_dir / f"{name}.blend"
        before = set(bpy.data.objects)
        bpy.ops.wm.append(directory=str(path / "Object") + os.sep,
                          filename=name, link=False)
        added = [obj for obj in bpy.data.objects if obj not in before]
        if len(added) != 1 or added[0].type != "MESH":
            raise RuntimeError(f"Expected one mesh from {path}: {added}")
        obj = added[0]
        obj.location = (0, 0, scale)
        obj.scale = (scale, scale, scale)
        obj.rotation_euler = (0, 0, 0)
    obj.pass_index = 1
    return obj


def render_one(args, material, profile, grid, spec, index):
    shape, color, light, view = spec
    out = args.output_root / shape / color / light / view
    out.mkdir(parents=True, exist_ok=False)
    bpy.ops.wm.open_mainfile(filepath=str(args.base_scene.resolve()))
    scene, lights, devices = base.configure_scene(
        profile, grid["seed"] + index, profile["lighting_conditions"][light])
    for existing in list(scene.objects):
        if existing.type == "MESH" and existing.name != "Ground":
            bpy.data.objects.remove(existing, do_unlink=True)
    obj = append_shape(shape, args.shape_dir, grid["object_scale"])
    own_profile = dict(profile)
    own_profile["material"] = {"base_color_linear_rgba": grid["colors_linear_rgba"][color]}
    base.set_material(obj, material, own_profile)
    camera = bpy.data.objects[profile["camera"]["name"]]
    if view == "oblique_45":
        offset = camera.location - Vector(obj.location)
        camera.location = Vector(obj.location) + Matrix.Rotation(math.radians(45), 4, "Z") @ offset
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
        "index": index, "shape": shape, "color": color, "light": light,
        "view": view, "seed": grid["seed"] + index,
        "split": "held_out_shape" if shape == grid["held_out_shape"] else "train",
        "roughness": material["roughness"], "metallic": material["metallic"],
        "base_color_linear_rgba": grid["colors_linear_rgba"][color],
        "object_scale": grid["object_scale"],
        "world_strength": profile["lighting_conditions"][light]["world_strength"],
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
    parser.add_argument("--material-json", type=Path, required=True)
    parser.add_argument("--grid-config", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--base-scene", type=Path, required=True)
    parser.add_argument("--shape-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    for path in (args.material_json, args.grid_config, args.profile,
                 args.base_scene, args.shape_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu:
        raise ValueError("GPU argument and CUDA_VISIBLE_DEVICES disagree")
    material = json.loads(args.material_json.read_text())
    grid = json.loads(args.grid_config.read_text())
    profile = json.loads(args.profile.read_text())
    profile["blender"]["cuda_visible_devices"] = args.gpu
    for key in ("roughness", "metallic"):
        if not 0 <= material[key] <= 1:
            raise ValueError(f"Invalid {key}: {material[key]}")
    specs = list(itertools.product(grid["shapes"], grid["colors_linear_rgba"],
                                   profile["lighting_conditions"], grid["viewpoints"]))
    args.output_root.mkdir(parents=True)
    records = []
    for index, spec in enumerate(specs[:args.max_images]):
        records.append(render_one(args, material, profile, grid, spec, index))
    base.write_json(args.output_root / "manifest.json", {
        "status": "complete" if len(records) == len(specs) else "smoke",
        "material_id": material["material_id"], "expected_images": len(specs),
        "image_count": len(records), "git_commit": base.git_commit(),
        "input_sha256": {"material": base.sha256(args.material_json),
                         "grid_config": base.sha256(args.grid_config),
                         "profile": base.sha256(args.profile),
                         "base_scene": base.sha256(args.base_scene),
                         "shape_assets": {name: base.sha256(args.shape_dir / f"{asset}.blend")
                                          for name, asset in ASSETS.items()},
                         "script": base.sha256(Path(__file__))},
        "blender_version": list(bpy.app.version), "records": records,
    })


if __name__ == "__main__":
    main()
