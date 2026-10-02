#!/usr/bin/env python3
"""Render the mailbox roughness sweep or the two-shape validation with Cycles.

blender --background --python-exit-code 1 --python render_roughness_study.py -- \
  --phase sweep --study-config roughness_study.json \
  --source-material-json canonical_material_v2.json \
  --profile renderer_profile.json --output-root new_study_run \
  --base-scene-blendfile base_scene.blend --sphere-blendfile Sphere.blend

The validation phase additionally requires --calibrated-roughness. Each phase
must have a new output directory. This entrypoint cannot generate the full grid.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_calibration as base


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("sweep", "validation"), required=True)
    parser.add_argument("--study-config", type=Path, required=True)
    parser.add_argument("--source-material-json", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-scene-blendfile", type=Path, required=True)
    parser.add_argument("--sphere-blendfile", type=Path, required=True)
    parser.add_argument("--calibrated-roughness", type=float)
    return parser.parse_args(argv)


def validate(args: argparse.Namespace) -> tuple[dict, dict]:
    paths = (args.study_config, args.source_material_json, args.profile,
             args.base_scene_blendfile, args.sphere_blendfile)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if (args.output_root / args.phase).exists():
        raise FileExistsError(f"Phase output already exists: {args.output_root / args.phase}")
    study = json.loads(args.study_config.read_text(encoding="utf-8"))
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if study["schema_version"] != 1 or study["study_id"] != "mailbox_roughness_space_calibration_v1":
        raise ValueError("Unsupported roughness study config")
    if profile["schema_version"] != 1 or profile["profile_id"] != "natural_material_calibration_v1":
        raise ValueError("Unsupported renderer profile")
    if tuple(profile["blender"]["version"]) != bpy.app.version:
        raise RuntimeError(f"Blender version mismatch: {bpy.app.version}")
    if len(profile["lighting_conditions"]) != 3:
        raise ValueError("Exactly three original lighting conditions are required")
    if study["render_metallic"] != 0.0:
        raise ValueError("This study fixes the dielectric coating metallic input at zero")
    if study["validation_colors_linear_rgba"]["gray"] != profile["material"]["base_color_linear_rgba"]:
        raise ValueError("Sweep gray must match the existing calibration renderer profile")
    if args.phase == "validation" and (args.calibrated_roughness is None or
                                        not 0 <= args.calibrated_roughness <= 1):
        raise ValueError("Validation requires --calibrated-roughness in [0, 1]")
    if args.phase == "sweep" and args.calibrated_roughness is not None:
        raise ValueError("--calibrated-roughness is only for validation")
    return study, profile


def add_cube(profile: dict):
    """Make a cube with the sphere's 2*scale width and center height."""
    scale = profile["object"]["scale"]
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, scale))
    obj = bpy.context.object
    obj.name = "ValidationCube"
    obj.scale = (scale, scale, scale)
    obj.pass_index = 1
    return obj


def render_cell(args: argparse.Namespace, profile: dict, phase: str, cell_id: str,
                shape: str, color_name: str, base_color: list[float], roughness: float,
                light_name: str, condition: dict, hashes: dict) -> dict:
    bpy.ops.wm.open_mainfile(filepath=str(args.base_scene_blendfile))
    seed = profile["blender"]["seed"]
    scene, lights, devices = base.configure_scene(profile, seed, condition)
    for existing in list(scene.objects):
        if existing.type == "MESH" and existing.name != "Ground":
            bpy.data.objects.remove(existing, do_unlink=True)
    if shape == "sphere":
        obj = base.append_sphere(args.sphere_blendfile, profile)
    elif shape == "cube":
        obj = add_cube(profile)
    else:
        raise ValueError(f"Unsupported shape: {shape}")
    material = {"roughness": roughness, "metallic": 0.0}
    cell_profile = deepcopy(profile)
    cell_profile["material"]["base_color_linear_rgba"] = base_color
    base.set_material(obj, material, cell_profile)
    camera = base.configure_camera(obj, profile)
    output_dir = args.output_root / phase / cell_id
    output_dir.mkdir(parents=True)
    base.configure_mask(output_dir)
    image = output_dir / "image.png"
    scene.render.filepath = str(image)
    bpy.ops.render.render(write_still=True)
    outputs = (("__mask_*.png", "object_mask.png"),
               ("__input_rgba_*.png", "input_rgba.png"))
    for pattern, filename in outputs:
        generated = list(output_dir.glob(pattern))
        if len(generated) != 1:
            raise RuntimeError(f"Expected one {filename} for {cell_id}: {generated}")
        generated[0].rename(output_dir / filename)
    mask = output_dir / "object_mask.png"
    rgba = output_dir / "input_rgba.png"
    if not all(path.is_file() for path in (image, mask, rgba)):
        raise RuntimeError(f"Missing render artifacts for {cell_id}")
    record = {
        "phase": phase, "cell_id": cell_id, "shape": shape,
        "color": color_name, "base_color_linear_rgba": base_color,
        "lighting": light_name, "roughness": roughness, "metallic": 0.0,
        "seed": seed, "camera": camera, "lights": lights,
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
                "exposure": scene.view_settings.exposure,
                "gamma": scene.view_settings.gamma,
            },
        },
        "geometry": ({"asset_name": profile["object"]["asset_name"],
                      "scale": profile["object"]["scale"]} if shape == "sphere" else
                     {"primitive": "cube", "size": 2.0,
                      "scale": profile["object"]["scale"],
                      "location": [0.0, 0.0, profile["object"]["scale"]]}),
        "input_sha256": hashes,
        "renderer_script_sha256": base.sha256(Path(__file__).resolve()),
        "calibration_script_sha256": base.sha256(Path(base.__file__).resolve()),
        "image_sha256": base.sha256(image),
        "object_mask_sha256": base.sha256(mask),
        "input_rgba_sha256": base.sha256(rgba),
        "git_commit": base.git_commit(),
    }
    base.write_json(output_dir / "metadata.json", record)
    return record


def main() -> None:
    args = parse_args()
    for name in ("study_config", "source_material_json", "profile", "output_root",
                 "base_scene_blendfile", "sphere_blendfile"):
        setattr(args, name, getattr(args, name).resolve())
    study, profile = validate(args)
    hashes = {
        "study_config": base.sha256(args.study_config),
        "source_material_json": base.sha256(args.source_material_json),
        "renderer_profile_json": base.sha256(args.profile),
        "clevr_base_scene_blend": base.sha256(args.base_scene_blendfile),
        "clevr_sphere_blend": base.sha256(args.sphere_blendfile),
    }
    phase_dir = args.output_root / args.phase
    phase_dir.mkdir(parents=True)
    records = []
    if args.phase == "sweep":
        for roughness in study["sweep_roughness"]:
            if not isinstance(roughness, (int, float)) or not 0 <= roughness <= 1:
                raise ValueError(f"Invalid sweep roughness: {roughness}")
            for light_name, condition in profile["lighting_conditions"].items():
                cell_id = f"r{round(roughness * 100):03d}_{light_name}"
                records.append(render_cell(args, profile, "sweep", cell_id,
                                           "sphere", "gray",
                                           study["validation_colors_linear_rgba"]["gray"],
                                           roughness, light_name, condition, hashes))
    else:
        for shape in study["validation_shapes"]:
            for color_name, color in study["validation_colors_linear_rgba"].items():
                for light_name, condition in profile["lighting_conditions"].items():
                    cell_id = f"{shape}_{color_name}_{light_name}"
                    records.append(render_cell(args, profile, "validation", cell_id,
                                               shape, color_name, color,
                                               args.calibrated_roughness, light_name,
                                               condition, hashes))
    base.write_json(phase_dir / "manifest.json", {
        "phase": args.phase, "status": "render_complete",
        "image_count": len(records),
        "target_supermat_roughness": study["target_supermat_roughness"],
        "calibrated_roughness": args.calibrated_roughness,
        "input_sha256": hashes, "git_commit": base.git_commit(),
        "records": records,
    })


if __name__ == "__main__":
    main()
