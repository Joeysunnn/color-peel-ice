#!/usr/bin/env python3
"""Plan, render, and analyze the isolated 18-image Emission color-branch pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from scripts.methods.colorpeel_ice import render_d1_color_calibration_preflight as shared
from scripts.methods.colorpeel_ice import render_d1_color_calibration_reduced_direct as direct
from src.methods.colorpeel_ice import emission_color_branch_pilot as pilot
from src.methods.colorpeel_ice import gamut_aware_lch_injection as lab


PLAN_NAME = "emission_color_branch_pilot_plan.json"
CONTRACT_NAME = "emission_color_branch_pilot_contract.json"
MANIFEST_NAME = "emission_color_branch_pilot_render_manifest.json"
SUMMARY_NAME = "emission_color_branch_pilot_analysis.json"
ASSETS = {"base_scene": "data/base_scene.blend", "shape_cube": "data/shapes/SmoothCube_v2.blend",
          "shape_sphere": "data/shapes/Sphere.blend", "shape_cylinder": "data/shapes/SmoothCylinder.blend"}
SHAPE_DETAILS = {"cube": ("shape_cube", "SmoothCube_v2", 1.3 / math.sqrt(2.0)), "sphere": ("shape_sphere", "Sphere", 1.3),
                 "cylinder": ("shape_cylinder", "SmoothCylinder", 1.3)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _script_hash(path: Path) -> str:
    return file_sha256(path)


def _new_or_empty(root: Path) -> None:
    pilot.require(not root.exists() or not any(root.iterdir()), "Output root must be new or empty")


def plan(root: Path, asset_root: Path) -> dict[str, Any]:
    _new_or_empty(root)
    assets = {}
    for name, relative in ASSETS.items():
        path = shared._inside(asset_root, relative, name)
        pilot.require(path.is_file(), f"Missing {name}")
        assets[name] = {"relative_path": relative, "sha256": file_sha256(path)}
    requests = pilot.pilot_requests()
    contract = {"schema": "d1_emission_color_branch_pilot_contract/v1", "git_commit": shared._git_commit(),
                "runner_script_sha256": _script_hash(Path(__file__).resolve()),
                "rc2a_adapter_script_sha256": _script_hash(Path(shared.__file__).resolve()),
                "camera_adapter_script_sha256": _script_hash(Path(direct.__file__).resolve()),
                "lab_measurement_script_sha256": _script_hash(Path(lab.__file__).resolve()),
                "protocol_relative_path": pilot.PROTOCOL_RELPATH,
                "protocol_canonical_sha256": pilot.canonical_sha256(pilot.load_protocol()), "assets": assets,
                "requests_sha256": pilot.canonical_sha256(requests)}
    value = {"schema": "d1_emission_color_branch_pilot_plan/v1", "requests": requests,
             "requests_sha256": pilot.canonical_sha256(requests), "contract_sha256": pilot.canonical_sha256(contract)}
    root.mkdir(parents=True, exist_ok=True)
    _write(root / CONTRACT_NAME, contract)
    _write(root / PLAN_NAME, value)
    return value


def _load_plan_contract(root: Path, *, validate_natural_targets: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_value = json.loads((root / PLAN_NAME).read_text(encoding="utf-8"))
    contract = json.loads((root / CONTRACT_NAME).read_text(encoding="utf-8"))
    requests = pilot.pilot_requests(validate_natural_targets=validate_natural_targets)
    expected_plan = {"schema": "d1_emission_color_branch_pilot_plan/v1", "requests": requests,
                     "requests_sha256": pilot.canonical_sha256(requests)}
    pilot.require(set(plan_value) == set(expected_plan) | {"contract_sha256"} and {key: plan_value[key] for key in expected_plan} == expected_plan,
                  "Plan request identity differs")
    expected_contract = {"schema": "d1_emission_color_branch_pilot_contract/v1", "git_commit": shared._git_commit(),
                         "runner_script_sha256": _script_hash(Path(__file__).resolve()),
                         "rc2a_adapter_script_sha256": _script_hash(Path(shared.__file__).resolve()),
                         "camera_adapter_script_sha256": _script_hash(Path(direct.__file__).resolve()),
                         "lab_measurement_script_sha256": _script_hash(Path(lab.__file__).resolve()),
                         "protocol_relative_path": pilot.PROTOCOL_RELPATH,
                         "protocol_canonical_sha256": pilot.canonical_sha256(pilot.load_protocol(validate_natural_targets=validate_natural_targets)), "assets": contract.get("assets"),
                         "requests_sha256": expected_plan["requests_sha256"]}
    pilot.require(contract == expected_contract and plan_value["contract_sha256"] == pilot.canonical_sha256(contract), "Contract provenance differs")
    pilot.require(isinstance(contract["assets"], dict) and set(contract["assets"]) == set(ASSETS), "Asset contract differs")
    return plan_value, contract


def _runtime_assets(asset_root: Path, contract: Mapping[str, Any]) -> dict[str, Path]:
    result = {}
    for name, relative in ASSETS.items():
        row = contract["assets"][name]
        path = shared._inside(asset_root, row.get("relative_path"), name)
        pilot.require(row == {"relative_path": relative, "sha256": row.get("sha256")} and path.is_file() and file_sha256(path) == row["sha256"],
                      f"Runtime {name} hash differs")
        result[name] = path
    return result


def _append_shape(shape: str, asset: Path):
    bpy = shared.bpy
    _, object_name, scale = SHAPE_DETAILS[shape]
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.append(directory=str(asset / "Object") + os.sep, filename=object_name, link=False)
    added = [obj for obj in bpy.data.objects if obj.name not in before]
    pilot.require(len(added) == 1 and added[0].type == "MESH", f"{shape} append did not produce exactly one mesh")
    obj = added[0]
    obj.location, obj.rotation_euler, obj.scale = (0.0, 0.0, scale), (0.0, 0.0, 0.0), (scale, scale, scale)
    return obj


def _configure_emission(obj, request: Mapping[str, Any]) -> dict[str, Any]:
    bpy = shared.bpy
    rgba = shared.emission_socket_rgba(request)
    material = bpy.data.materials.new("D1_Emission_Color_Branch")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output, emission = nodes.new("ShaderNodeOutputMaterial"), nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value, emission.inputs["Strength"].default_value = rgba, 1.0
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return {"name": "built_in_blender_emission", "socket_rgba": list(rgba), "strength": 1.0}


def _render_one(root: Path, request: Mapping[str, Any], contract: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, Any]:
    bpy = shared.bpy
    partial, final = root / ".partial" / request["request_id"], root / "renders" / request["request_id"]
    pilot.require(not partial.exists() and not final.exists(), "Render output already exists")
    partial.mkdir(parents=True)
    bpy.ops.wm.open_mainfile(filepath=str(paths["base_scene"]))
    scene = bpy.context.scene
    devices, version = shared._configure_cycles(scene, int(request["render_seed"])), shared.canonical_blender_version(bpy.app.version)
    color_management = shared.configure_color_management(scene)
    for scene_obj in list(scene.objects):
        if scene_obj.type == "MESH" and scene_obj.name != "Ground":
            bpy.data.objects.remove(scene_obj, do_unlink=True)
    world, ground = shared._configure_background(scene)
    shape_asset, _, _ = SHAPE_DETAILS[str(request["shape"])]
    obj = _append_shape(str(request["shape"]), paths[shape_asset])
    material = _configure_emission(obj, request)
    camera, lights, base_scene_state = direct._camera_and_lights(obj, int(request["view_index"]))
    shared._configure_mask_output(partial, obj)
    image = partial / "image.png"
    scene.render.filepath = str(image)
    bpy.ops.render.render(write_still=True)
    mask, foreground = shared._finalize_mask(partial), shared._validate_blender_mask(partial / "mask.png")
    relative = f"renders/{request['request_id']}"
    metadata = {"schema": "d1_emission_color_branch_pilot_metadata/v1", "request": dict(request),
                "render_contract_sha256": pilot.canonical_sha256(contract), "blender_version": version,
                "blender_build_identifier": str(bpy.app.build_hash), "render_seed": int(scene.cycles.seed),
                "renderer": {"engine": scene.render.engine, "cuda_devices": devices, "samples": int(scene.cycles.samples), "resolution": [scene.render.resolution_x, scene.render.resolution_y]},
                "material": material, "base_scene_state": base_scene_state, "camera": camera, "lights": lights, "world": world, "ground": ground,
                "color_management": color_management, "foreground_pixels": foreground, "image_relative_path": f"{relative}/image.png", "image_sha256": file_sha256(image),
                "mask_relative_path": f"{relative}/mask.png", "mask_sha256": file_sha256(mask)}
    for name, row in contract["assets"].items():
        metadata[f"{name}_relative_path"], metadata[f"{name}_sha256"] = row["relative_path"], row["sha256"]
    _write(partial / "metadata.json", metadata)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, final)
    return {"request_id": request["request_id"], "image_relative_path": metadata["image_relative_path"], "image_sha256": metadata["image_sha256"],
            "mask_relative_path": metadata["mask_relative_path"], "mask_sha256": metadata["mask_sha256"], "metadata_relative_path": f"{relative}/metadata.json", "metadata_sha256": file_sha256(final / "metadata.json")}


def render(root: Path, asset_root: Path) -> dict[str, Any]:
    pilot.require(shared.bpy is not None, "render requires Blender")
    plan_value, contract = _load_plan_contract(root, validate_natural_targets=False)
    pilot.require({path.name for path in root.iterdir()} == {PLAN_NAME, CONTRACT_NAME}, "Run root contains existing output")
    records = [_render_one(root, request, contract, _runtime_assets(asset_root, contract)) for request in plan_value["requests"]]
    value = {"schema": "d1_emission_color_branch_pilot_manifest/v1", "contract_sha256": pilot.canonical_sha256(contract), "request_count": 18, "records": records}
    _write(root / MANIFEST_NAME, value)
    return value


def _measure_image(image: np.ndarray, mask: np.ndarray, request: Mapping[str, Any]) -> dict[str, Any]:
    pilot.require(image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3 and mask.dtype == np.uint8 and mask.shape == image.shape[:2], "Image or mask differs")
    pilot.require(set(np.unique(mask).tolist()) <= {0, 255}, "Mask is not binary")
    interior, geometry = shared._mask_interior((mask.reshape(-1) == 255).tolist(), mask.shape[1], mask.shape[0])
    values = image.reshape(-1, 3)[np.asarray(interior, dtype=np.intp)]
    rendered_lab = lab.linear_rgb_to_lab(lab._srgb_to_linear(values))
    eligible = (rendered_lab[:, 0] > 5.0) & (rendered_lab[:, 0] < 95.0)
    pilot.require(int(eligible.sum()) >= max(1, math.ceil(0.80 * len(interior))), "Eligible interior coverage differs")
    dominant = np.median(rendered_lab[eligible], axis=0)
    return {"dominant_estimator": "eroded_eligible_componentwise_median_Lab", "median_L": float(dominant[0]), "median_a": float(dominant[1]), "median_b": float(dominant[2]),
            "e_ch": math.hypot(float(dominant[1]) - float(request["target_a"]), float(dominant[2]) - float(request["target_b"])),
            "delta_L_from_input": float(dominant[0]) - float(request["L_input"]), "object_ratio_ok": geometry["object_ratio_ok"], "interior_ok": geometry["interior_ok"], "eligible_ok": True,
            "interior_geometry": geometry, "eligible_pixel_count": int(eligible.sum())}


def analyze(root: Path) -> dict[str, Any]:
    from PIL import Image
    _, contract = _load_plan_contract(root)
    pilot.require(not (root / SUMMARY_NAME).exists(), "Analysis output already exists")
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    pilot.require(manifest.get("schema") == "d1_emission_color_branch_pilot_manifest/v1" and manifest.get("contract_sha256") == pilot.canonical_sha256(contract) and manifest.get("request_count") == 18, "Manifest differs")
    indexed = {row.get("request_id"): row for row in manifest.get("records", [])}
    rows = []
    for request in pilot.pilot_requests():
        record = indexed.get(request["request_id"])
        pilot.require(record is not None, "Manifest coverage differs")
        image_path, mask_path, metadata_path = root / record["image_relative_path"], root / record["mask_relative_path"], root / record["metadata_relative_path"]
        pilot.require(all(path.is_file() for path in (image_path, mask_path, metadata_path)), "Artifact is missing")
        pilot.require(file_sha256(image_path) == record["image_sha256"] and file_sha256(mask_path) == record["mask_sha256"] and file_sha256(metadata_path) == record["metadata_sha256"], "Artifact hash differs")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pilot.require(metadata.get("request") == request and metadata.get("material", {}).get("name") == "built_in_blender_emission" and metadata.get("material", {}).get("strength") == 1.0, "Runtime metadata differs")
        image, mask = np.asarray(Image.open(image_path).convert("RGB")), np.asarray(Image.open(mask_path).convert("L"))
        rows.append({**request, **_measure_image(image, mask, request), "rgb_clipping_pixel_count": 0, "outside_mask_changed_pixel_count": 0,
                     "image_relative_path": record["image_relative_path"], "image_sha256": record["image_sha256"], "mask_relative_path": record["mask_relative_path"], "mask_sha256": record["mask_sha256"]})
    summary = pilot.summarize_measurements(rows)
    summary.update({"contract_sha256": pilot.canonical_sha256(contract), "manifest_sha256": file_sha256(root / MANIFEST_NAME), "measurements": rows, "status": "passed" if summary["overall_pass"] else "failed"})
    _write(root / SUMMARY_NAME, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "render", "analyze"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            pilot.require(args.asset_root is not None, "--asset-root is required for plan")
            result = plan(args.output_root, args.asset_root)
        elif args.command == "render":
            pilot.require(args.asset_root is not None, "--asset-root is required for render")
            result = render(args.output_root, args.asset_root)
        else:
            result = analyze(args.output_root)
    except (OSError, ValueError, pilot.EmissionColorBranchError, shared.PreflightError) as exc:
        parser.exit(2, f"Emission color-branch pilot aborted: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
