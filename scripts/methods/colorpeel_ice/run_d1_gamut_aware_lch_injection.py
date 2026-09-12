#!/usr/bin/env python3
"""Plan, render, inject, and analyze the independent D1 gamut-aware LCh pilot."""

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
from src.methods.colorpeel_ice import gamut_aware_lch_injection as injection


PLAN_NAME = "gamut_aware_lch_plan.json"
CONTRACT_NAME = "gamut_aware_lch_contract.json"
NEUTRAL_MANIFEST_NAME = "gamut_aware_lch_neutral_manifest.json"
RESULTS_NAME = "gamut_aware_lch_results.json"
SUMMARY_NAME = "gamut_aware_lch_analysis.json"
ASSETS = {
    "base_scene": "data/base_scene.blend",
    "material_asset": "data/materials/Rubber.blend",
    "shape_cube": "data/shapes/SmoothCube_v2.blend",
    "shape_sphere": "data/shapes/Sphere.blend",
    "shape_cylinder": "data/shapes/SmoothCylinder.blend",
}
SHAPE_DETAILS = {
    "cube": ("shape_cube", "SmoothCube_v2", 1.3 / math.sqrt(2.0)),
    "sphere": ("shape_sphere", "Sphere", 1.3),
    "cylinder": ("shape_cylinder", "SmoothCylinder", 1.3),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _script_hash() -> str:
    return file_sha256(Path(__file__).resolve())


def _protocol_hash() -> str:
    return file_sha256(REPO_ROOT / injection.PROTOCOL_RELPATH)


def _new_or_empty(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise injection.GamutAwareLchError("Output root must be new or empty")


def plan(root: Path, asset_root: Path) -> dict[str, Any]:
    _new_or_empty(root)
    assets = {}
    for name, relative in ASSETS.items():
        path = shared._inside(asset_root, relative, name)
        injection.require(path.is_file(), f"Missing {name}")
        assets[name] = {"relative_path": relative, "sha256": file_sha256(path)}
    neutral, requests = injection.neutral_requests(), injection.pilot_requests()
    contract = {
        "schema": "d1_gamut_aware_lch_contract/v1", "git_commit": shared._git_commit(),
        "adapter_script_sha256": _script_hash(), "protocol_relative_path": injection.PROTOCOL_RELPATH,
        "protocol_sha256": _protocol_hash(), "selection_canonical_sha256": injection.SELECTION_CANONICAL_SHA256,
        "renderer_protocol_canonical_sha256": injection.RENDERER_PROTOCOL_CANONICAL_SHA256,
        "assets": assets, "neutral_requests_sha256": injection.canonical_sha256(neutral),
        "injection_requests_sha256": injection.canonical_sha256(requests),
    }
    value = {
        "schema": "d1_gamut_aware_lch_plan/v1", "neutral_requests": neutral, "injection_requests": requests,
        "neutral_requests_sha256": injection.canonical_sha256(neutral),
        "injection_requests_sha256": injection.canonical_sha256(requests), "contract_sha256": injection.canonical_sha256(contract),
    }
    root.mkdir(parents=True, exist_ok=True)
    _write(root / CONTRACT_NAME, contract)
    _write(root / PLAN_NAME, value)
    return value


def _load_plan_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_value = json.loads((root / PLAN_NAME).read_text(encoding="utf-8"))
    contract = json.loads((root / CONTRACT_NAME).read_text(encoding="utf-8"))
    expected_plan = {
        "schema": "d1_gamut_aware_lch_plan/v1", "neutral_requests": injection.neutral_requests(),
        "injection_requests": injection.pilot_requests(),
        "neutral_requests_sha256": injection.canonical_sha256(injection.neutral_requests()),
        "injection_requests_sha256": injection.canonical_sha256(injection.pilot_requests()),
    }
    injection.require(set(plan_value) == set(expected_plan) | {"contract_sha256"}, "Plan fields differ")
    injection.require({key: plan_value[key] for key in expected_plan} == expected_plan, "Plan request identity differs")
    expected_contract = {
        "schema": "d1_gamut_aware_lch_contract/v1", "git_commit": shared._git_commit(),
        "adapter_script_sha256": _script_hash(), "protocol_relative_path": injection.PROTOCOL_RELPATH,
        "protocol_sha256": _protocol_hash(), "selection_canonical_sha256": injection.SELECTION_CANONICAL_SHA256,
        "renderer_protocol_canonical_sha256": injection.RENDERER_PROTOCOL_CANONICAL_SHA256,
        "assets": contract.get("assets"), "neutral_requests_sha256": expected_plan["neutral_requests_sha256"],
        "injection_requests_sha256": expected_plan["injection_requests_sha256"],
    }
    injection.require(contract == expected_contract, "Contract provenance differs")
    injection.require(plan_value["contract_sha256"] == injection.canonical_sha256(contract), "Plan contract hash differs")
    assets = contract["assets"]
    injection.require(isinstance(assets, dict) and set(assets) == set(ASSETS), "Contract assets differ")
    for name, relative in ASSETS.items():
        row = assets[name]
        injection.require(isinstance(row, dict) and row.get("relative_path") == relative and isinstance(row.get("sha256"), str) and len(row["sha256"]) == 64, f"Contract {name} differs")
    return plan_value, contract


def _runtime_assets(asset_root: Path, contract: Mapping[str, Any]) -> dict[str, Path]:
    paths = {}
    for name, row in contract["assets"].items():
        path = shared._inside(asset_root, row["relative_path"], name)
        injection.require(path.is_file() and file_sha256(path) == row["sha256"], f"Runtime {name} hash differs")
        paths[name] = path
    return paths


def _append_shape(shape: str, asset: Path):
    bpy = shared.bpy
    asset_name, object_name, scale = SHAPE_DETAILS[shape]
    del asset_name
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.append(directory=str(asset / "Object") + os.sep, filename=object_name, link=False)
    added = [obj for obj in bpy.data.objects if obj.name not in before]
    injection.require(len(added) == 1 and added[0].type == "MESH", f"{shape} append did not produce exactly one mesh")
    obj = added[0]
    obj.location = (0.0, 0.0, scale)
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.scale = (scale, scale, scale)
    return obj


def _render_one(root: Path, request: Mapping[str, Any], contract: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, Any]:
    bpy = shared.bpy
    partial = root / ".partial" / request["request_id"]
    final = root / "neutral" / "renders" / request["request_id"]
    injection.require(not partial.exists() and not final.exists(), "Render output already exists")
    partial.mkdir(parents=True)
    bpy.ops.wm.open_mainfile(filepath=str(paths["base_scene"]))
    version = shared.canonical_blender_version(bpy.app.version)
    scene = bpy.context.scene
    devices = shared._configure_cycles(scene, int(request["render_seed"]))
    color_management = shared.configure_color_management(scene)
    for obj in list(scene.objects):
        if obj.type == "MESH" and obj.name != "Ground":
            bpy.data.objects.remove(obj, do_unlink=True)
    world, ground = shared._configure_background(scene)
    shape_asset, _, _ = SHAPE_DETAILS[str(request["shape"])]
    obj = _append_shape(str(request["shape"]), paths[shape_asset])
    linear = injection.lab_to_linear_rgb(np.array([injection.NEUTRAL_LAB]))[0].tolist()
    rubber_request = {"material": "Rubber", "linear_rgb": linear, "socket_rgba": linear + [1.0]}
    material = direct._configure_rubber(obj, rubber_request, paths["material_asset"])
    camera, lights, base = direct._camera_and_lights(obj, int(request["view_index"]))
    shared._configure_mask_output(partial, obj)
    scene.render.filepath = str(partial / "image.png")
    bpy.ops.render.render(write_still=True)
    mask = shared._finalize_mask(partial)
    foreground = shared._validate_blender_mask(mask)
    relative = f"neutral/renders/{request['request_id']}"
    metadata = {
        "schema": "d1_gamut_aware_lch_neutral_metadata/v1", "request": dict(request),
        "render_contract_sha256": injection.canonical_sha256(contract), "blender_version": version,
        "blender_build_identifier": str(bpy.app.build_hash), "render_seed": scene.cycles.seed,
        "renderer": {"engine": scene.render.engine, "cuda_devices": devices, "samples": scene.cycles.samples,
                     "resolution": [scene.render.resolution_x, scene.render.resolution_y],
                     "image": {"format": scene.render.image_settings.file_format, "mode": scene.render.image_settings.color_mode,
                               "bits_per_channel": int(scene.render.image_settings.color_depth)}},
        "material": material, "base_scene_state": base, "object": {"shape": request["shape"], "location": [float(v) for v in obj.location],
                   "scale": [float(v) for v in obj.scale], "rotation_euler": [float(v) for v in obj.rotation_euler]},
        "color_management": color_management, "camera": camera, "lights": lights, "world": world, "ground": ground,
        "foreground_pixels": foreground, "image_relative_path": f"{relative}/image.png", "image_sha256": file_sha256(partial / "image.png"),
        "mask_relative_path": f"{relative}/mask.png", "mask_sha256": file_sha256(mask),
    }
    for name, row in contract["assets"].items():
        metadata[f"{name}_relative_path"] = row["relative_path"]
        metadata[f"{name}_sha256"] = row["sha256"]
    _write(partial / "metadata.json", metadata)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, final)
    return {"request_id": request["request_id"], "shape": request["shape"], "view_index": request["view_index"],
            "image_relative_path": metadata["image_relative_path"], "image_sha256": metadata["image_sha256"],
            "mask_relative_path": metadata["mask_relative_path"], "mask_sha256": metadata["mask_sha256"],
            "metadata_relative_path": f"{relative}/metadata.json", "metadata_sha256": file_sha256(final / "metadata.json")}


def render_neutral(root: Path, asset_root: Path) -> dict[str, Any]:
    injection.require(shared.bpy is not None, "render-neutral requires Blender")
    plan_value, contract = _load_plan_contract(root)
    injection.require({path.name for path in root.iterdir()} == {PLAN_NAME, CONTRACT_NAME}, "Run root contains existing output")
    paths = _runtime_assets(asset_root, contract)
    records = [_render_one(root, request, contract, paths) for request in plan_value["neutral_requests"]]
    value = {"schema": "d1_gamut_aware_lch_neutral_manifest/v1", "contract_sha256": injection.canonical_sha256(contract), "request_count": 9, "records": records}
    _write(root / NEUTRAL_MANIFEST_NAME, value)
    return value


def _neutral_records(root: Path, contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    manifest = json.loads((root / NEUTRAL_MANIFEST_NAME).read_text(encoding="utf-8"))
    injection.require(manifest.get("schema") == "d1_gamut_aware_lch_neutral_manifest/v1" and manifest.get("contract_sha256") == injection.canonical_sha256(contract), "Neutral manifest contract differs")
    records = manifest.get("records")
    injection.require(isinstance(records, list) and len(records) == 9, "Neutral manifest coverage differs")
    expected = {row["request_id"] for row in injection.neutral_requests()}
    result = {row.get("request_id"): row for row in records}
    injection.require(set(result) == expected, "Neutral manifest request identity differs")
    for row in records:
        for kind in ("image", "mask", "metadata"):
            path = root / row[f"{kind}_relative_path"]
            injection.require(path.is_file() and file_sha256(path) == row[f"{kind}_sha256"], f"Neutral {kind} hash differs")
    return result


def inject(root: Path) -> dict[str, Any]:
    from PIL import Image
    plan_value, contract = _load_plan_contract(root)
    injection.require(not (root / RESULTS_NAME).exists() and not (root / "injected").exists() and not (root / "chroma_maps").exists(), "Injection output already exists")
    neutral = _neutral_records(root, contract)
    rows = []
    for request in plan_value["injection_requests"]:
        source_id = f"neutral__{request['shape']}__v{request['view_index']}"
        source = neutral[source_id]
        image_path, mask_path = root / source["image_relative_path"], root / source["mask_relative_path"]
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L"))
        output, evidence, maps = injection.inject_rgb(image, mask, float(request["target_a"]), float(request["target_b"]))
        out_path = root / "injected" / f"{request['request_id']}.png"
        out_path.parent.mkdir(exist_ok=True)
        Image.fromarray(output, "RGB").save(out_path)
        map_path = root / "chroma_maps" / f"{request['request_id']}.npz"
        map_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(map_path, **maps)
        rows.append({**request, **evidence, "neutral_request_id": source_id, "neutral_image_relative_path": source["image_relative_path"],
                     "neutral_image_sha256": source["image_sha256"], "neutral_mask_relative_path": source["mask_relative_path"],
                     "neutral_mask_sha256": source["mask_sha256"], "image_relative_path": str(out_path.relative_to(root)),
                     "image_sha256": file_sha256(out_path), "chroma_map_relative_path": str(map_path.relative_to(root)), "chroma_map_sha256": file_sha256(map_path)})
    value = {"schema": "d1_gamut_aware_lch_results/v1", "contract_sha256": injection.canonical_sha256(contract), "rows": rows}
    _write(root / RESULTS_NAME, value)
    return value


def _measure_pair(source: np.ndarray, output: np.ndarray, mask: np.ndarray, request: Mapping[str, Any]) -> dict[str, Any]:
    interior, geometry = shared._mask_interior((mask.reshape(-1) == 255).tolist(), mask.shape[1], mask.shape[0])
    interior = np.asarray(interior, dtype=np.intp)
    source_lab = injection.linear_rgb_to_lab(injection._srgb_to_linear(source.reshape(-1, 3)[interior]))
    output_lab = injection.linear_rgb_to_lab(injection._srgb_to_linear(output.reshape(-1, 3)[interior]))
    eligible = (source_lab[:, 0] > 5.0) & (source_lab[:, 0] < 95.0)
    injection.require(eligible.any(), "Eroded interior has no eligible pixels")
    selected = output_lab[eligible]
    dominant = np.median(selected, axis=0)
    e_ch = math.hypot(float(dominant[1]) - float(request["target_a"]), float(dominant[2]) - float(request["target_b"]))
    delta_L = output_lab[eligible, 0] - source_lab[eligible, 0]
    target_h = math.radians(float(request["target_h_degrees"]))
    output_h = np.arctan2(selected[:, 2], selected[:, 1])
    hue_error = np.abs(np.angle(np.exp(1j * (output_h - target_h))))
    return {"dominant_estimator": "eroded_eligible_componentwise_median_Lab", "median_L": float(dominant[0]), "dominant_a": float(dominant[1]),
            "dominant_b": float(dominant[2]), "e_ch": e_ch, "delta_L_median": float(np.median(delta_L)),
            "delta_L_abs_p90": injection.nearest_rank_p90(np.abs(delta_L).tolist()), "hue_error_degrees_median": float(np.degrees(np.median(hue_error))),
            "hue_error_degrees_p90": float(np.degrees(injection.nearest_rank_p90(hue_error.tolist()))), "interior_geometry": geometry}


def analyze(root: Path) -> dict[str, Any]:
    from PIL import Image
    _, contract = _load_plan_contract(root)
    injection.require(not (root / SUMMARY_NAME).exists(), "Analysis output already exists")
    rows = json.loads((root / RESULTS_NAME).read_text(encoding="utf-8")).get("rows")
    injection.require(isinstance(rows, list) and len(rows) == 18, "Results coverage differs")
    for row in rows:
        source = np.asarray(Image.open(root / row["neutral_image_relative_path"]).convert("RGB"))
        output = np.asarray(Image.open(root / row["image_relative_path"]).convert("RGB"))
        mask = np.asarray(Image.open(root / row["neutral_mask_relative_path"]).convert("L"))
        injection.require(file_sha256(root / row["image_relative_path"]) == row["image_sha256"], "Injected image hash differs")
        injection.require(file_sha256(root / row["chroma_map_relative_path"]) == row["chroma_map_sha256"], "Chroma map hash differs")
        outside = mask == 0
        delta = np.abs(output.astype(np.int16) - source.astype(np.int16))
        row["outside_mask_changed_pixel_count"] = int(np.any(delta[outside] != 0, axis=1).sum())
        row["outside_mask_max_abs_rgb_change"] = int(delta[outside].max()) if outside.any() else 0
        row.update(_measure_pair(source, output, mask, row))
    summary = injection.summarize_measurements(rows)
    summary.update({"contract_sha256": injection.canonical_sha256(contract), "results_sha256": file_sha256(root / RESULTS_NAME), "measurements": rows,
                    "status": "passed" if summary["overall_pass"] else "failed"})
    _write(root / SUMMARY_NAME, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "render-neutral", "inject", "analyze"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--asset-root", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            injection.require(args.asset_root is not None, "--asset-root is required for plan")
            result = plan(args.output_root, args.asset_root)
        elif args.command == "render-neutral":
            injection.require(args.asset_root is not None, "--asset-root is required for render-neutral")
            result = render_neutral(args.output_root, args.asset_root)
        elif args.command == "inject":
            result = inject(args.output_root)
        else:
            result = analyze(args.output_root)
    except (OSError, ValueError, injection.GamutAwareLchError, shared.PreflightError) as exc:
        parser.exit(2, f"gamut-aware LCh injection aborted: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
