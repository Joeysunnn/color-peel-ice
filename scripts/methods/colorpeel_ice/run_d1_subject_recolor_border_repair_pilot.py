#!/usr/bin/env python3
"""Run the independent, approved bottom-border repair subject recoloring pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image

from scripts.methods.colorpeel_ice import render_d1_color_calibration_preflight as shared
from src.methods.colorpeel_ice import natural_subject_recolor_border_repair as repair_core
from src.methods.colorpeel_ice import natural_subject_recolor_pilot as recolor_core


PROTOCOL = REPO_ROOT / "experiments/natural_image_subject_color_pilot/configs/d1_subject_recolor_border_repair_protocol_v1.json"
PLAN_NAME = "subject_recolor_border_repair_plan.json"
REPAIR_NAME = "subject_recolor_border_repair_manifest.json"
RESULTS_NAME = "subject_recolor_border_repair_results.json"
ANALYSIS_NAME = "subject_recolor_border_repair_analysis.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    recolor_core.require(isinstance(value, dict), f"{path} must contain an object")
    return value


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def source_paths(source_root: Path, protocol: dict[str, Any]) -> tuple[Path, dict[str, Path]]:
    source_manifest = source_root / "manifests" / "pilot_mask_manifest.json"
    recolor_core.require(source_manifest.is_file() and sha256(source_manifest) == protocol["source"]["mask_manifest_sha256"], "Source manifest differs")
    source = next((row for row in read_json(source_manifest).get("samples", []) if row.get("stable_id") == protocol["source"]["stable_id"]), None)
    recolor_core.require(isinstance(source, dict) and source.get("status") == "PASS", "Source subject is not PASS")
    outputs = source.get("outputs", {})
    paths = {name: source_root / outputs.get(name, "") for name in ("raw_image", "raw_mask")}
    recolor_core.require(all(path.is_file() for path in paths.values()), "Source artifact is missing")
    recolor_core.require(sha256(paths["raw_image"]) == protocol["source"]["raw_image_sha256"], "Source image differs")
    recolor_core.require(sha256(paths["raw_mask"]) == protocol["source"]["raw_mask_sha256"], "Source mask differs")
    return source_manifest, paths


def generated_requests(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    generated = [row for row in protocol["auxiliary_palette"] if row["name"] in protocol["held_out_guard"]["generated_palette_names"]]
    excluded = [row["name"] for row in protocol["auxiliary_palette"] if row["name"] in protocol["held_out_guard"]["excluded_palette_names"]]
    recolor_core.require(len(generated) == protocol["expected_image_count"] and excluded == protocol["held_out_guard"]["excluded_palette_names"], "Palette identity differs")
    original, threshold = protocol["source"]["original_hue_degrees"], protocol["held_out_guard"]["minimum_hue_separation_degrees"]
    recolor_core.require(all(float(recolor_core.hue_distance_degrees(np.array([row["hue_degrees"]]), original)[0]) >= threshold for row in generated), "Held-out hue guard differs")
    return generated


def plan(run_root: Path, source_root: Path) -> dict[str, Any]:
    protocol = read_json(PROTOCOL)
    recolor_core.require(protocol.get("schema") == "natural_subject_recolor_border_repair_protocol/v1", "Protocol differs")
    source_root = source_root.resolve()
    source_manifest, paths = source_paths(source_root, protocol)
    value = {
        "schema": "natural_subject_recolor_border_repair_plan/v1",
        "git_commit": git_commit(),
        "adapter_script_sha256": sha256(Path(__file__).resolve()),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "protocol_sha256": sha256(PROTOCOL),
        "source_root": str(source_root),
        "source_manifest_sha256": sha256(source_manifest),
        "source": {"paths": {name: str(path) for name, path in paths.items()}, "sha256": {name: sha256(path) for name, path in paths.items()}},
        "mask_repair": protocol["mask_repair"],
        "requests": generated_requests(protocol),
    }
    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / PLAN_NAME, value)
    return value


def repair_mask(run_root: Path) -> dict[str, Any]:
    plan_value = read_json(run_root / PLAN_NAME)
    paths = {name: Path(path) for name, path in plan_value["source"]["paths"].items()}
    recolor_core.require(all(sha256(paths[name]) == plan_value["source"]["sha256"][name] for name in paths), "Source artifact hash differs")
    raw_mask = np.asarray(Image.open(paths["raw_mask"]).convert("L"), dtype=np.uint8)
    repaired, added = repair_core.repair_bottom_border(raw_mask, plan_value["mask_repair"])
    alpha, alpha_derivation = repair_core.derive_repaired_alpha(repaired)
    mask_path = run_root / "masks" / "repaired_mask.png"
    alpha_path = run_root / "masks" / "repaired_alpha_u16.png"
    recolor_core.require(not mask_path.exists() and not alpha_path.exists(), "Repair output already exists")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((repaired.astype(np.uint8) * 255), mode="L").save(mask_path)
    Image.fromarray(alpha, mode="I;16").save(alpha_path)
    value = {
        "schema": "natural_subject_recolor_border_repair_manifest/v1",
        "plan_sha256": sha256(run_root / PLAN_NAME),
        "source_raw_mask_sha256": sha256(paths["raw_mask"]),
        "repaired_mask_relative_path": str(mask_path.relative_to(run_root)),
        "repaired_mask_sha256": sha256(mask_path),
        "repaired_alpha_relative_path": str(alpha_path.relative_to(run_root)),
        "repaired_alpha_sha256": sha256(alpha_path),
        "added_pixel_count": int(added.sum()),
        "added_bbox_xyxy_inclusive": plan_value["mask_repair"]["rectangle_xyxy_inclusive"],
        "alpha_derivation": alpha_derivation,
    }
    write_json(run_root / REPAIR_NAME, value)
    return value


def load_repair(run_root: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    repair = read_json(run_root / REPAIR_NAME)
    mask_path = run_root / repair["repaired_mask_relative_path"]
    alpha_path = run_root / repair["repaired_alpha_relative_path"]
    recolor_core.require(sha256(mask_path) == repair["repaired_mask_sha256"] and sha256(alpha_path) == repair["repaired_alpha_sha256"], "Repaired mask artifact differs")
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    alpha = np.asarray(Image.open(alpha_path), dtype=np.uint16)
    recolor_core.require(mask.shape == alpha.shape == (512, 512) and set(np.unique(mask).tolist()) <= {0, 255} and np.all(alpha[mask == 0] == 0), "Repaired mask or alpha differs")
    return repair, mask, alpha


def recolor(run_root: Path) -> dict[str, Any]:
    plan_value = read_json(run_root / PLAN_NAME)
    _, mask, alpha = load_repair(run_root)
    image_path = Path(plan_value["source"]["paths"]["raw_image"])
    recolor_core.require(sha256(image_path) == plan_value["source"]["sha256"]["raw_image"], "Source image hash differs")
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    records = []
    for request in plan_value["requests"]:
        output, metrics, maps = recolor_core.recolor(image, mask, alpha, float(request["hue_degrees"]))
        image_path = run_root / "images" / f"{request['name']}.png"
        map_path = run_root / "maps" / f"{request['name']}.npz"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        map_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(output).save(image_path)
        np.savez_compressed(map_path, **maps)
        records.append({"name": request["name"], "hue_degrees": request["hue_degrees"], "image_relative_path": str(image_path.relative_to(run_root)), "image_sha256": sha256(image_path), "map_relative_path": str(map_path.relative_to(run_root)), "map_sha256": sha256(map_path), "metrics": metrics})
    value = {"schema": "natural_subject_recolor_border_repair_results/v1", "plan_sha256": sha256(run_root / PLAN_NAME), "repair_manifest_sha256": sha256(run_root / REPAIR_NAME), "record_count": len(records), "records": records}
    write_json(run_root / RESULTS_NAME, value)
    return value


def analyze(run_root: Path) -> dict[str, Any]:
    plan_value = read_json(run_root / PLAN_NAME)
    results = read_json(run_root / RESULTS_NAME)
    repair, mask, _ = load_repair(run_root)
    recolor_core.require(
        results.get("plan_sha256") == sha256(run_root / PLAN_NAME)
        and results.get("repair_manifest_sha256") == sha256(run_root / REPAIR_NAME),
        "Results provenance differs",
    )
    recolor_core.require(results.get("record_count") == 5 and len(results.get("records", [])) == 5, "Output coverage differs")
    source = np.asarray(Image.open(plan_value["source"]["paths"]["raw_image"]).convert("RGB"), dtype=np.uint8)
    original_mask = np.asarray(Image.open(plan_value["source"]["paths"]["raw_mask"]).convert("L"), dtype=np.uint8)
    added = (mask == 255) & (original_mask == 0)
    interior, geometry = shared._mask_interior((mask.reshape(-1) == 255).tolist(), mask.shape[1], mask.shape[0])
    indices = np.asarray(interior, dtype=np.intp)
    source_lab = recolor_core.gamut.linear_rgb_to_lab(recolor_core.gamut._srgb_to_linear(source.reshape(-1, 3)[indices]))
    rows = results["records"]
    for row in rows:
        output_path = run_root / row["image_relative_path"]
        map_path = run_root / row["map_relative_path"]
        recolor_core.require(sha256(output_path) == row["image_sha256"] and sha256(map_path) == row["map_sha256"], "Output artifact differs")
        output = np.asarray(Image.open(output_path).convert("RGB"), dtype=np.uint8)
        delta = np.abs(output.astype(np.int16) - source.astype(np.int16))
        output_lab = recolor_core.gamut.linear_rgb_to_lab(recolor_core.gamut._srgb_to_linear(output.reshape(-1, 3)[indices]))
        eligible = (source_lab[:, 0] > 5.0) & (source_lab[:, 0] < 95.0)
        selected = output_lab[eligible]
        hue = np.degrees(np.arctan2(selected[:, 2], selected[:, 1])) % 360.0
        delta_l = selected[:, 0] - source_lab[eligible, 0]
        row["repaired_boundary_qc"] = {"added_mask_pixel_count": int(added.sum()), "added_mask_unchanged_output_count": int(np.all(output == source, axis=2)[added].sum()), "outside_repaired_mask_changed_pixel_count": int(np.any(delta[mask == 0] != 0, axis=1).sum()), "outside_repaired_mask_max_abs_rgb_change": int(delta[mask == 0].max())}
        row["eroded_interior_measurement"] = {
            "estimator": "eroded_eligible_Lab",
            "geometry": geometry,
            "eligible_pixel_count": int(selected.shape[0]),
            "median_a": float(np.median(selected[:, 1])),
            "median_b": float(np.median(selected[:, 2])),
            "median_C": float(np.median(np.hypot(selected[:, 1], selected[:, 2]))),
            "delta_L_median": float(np.median(delta_l)),
            "delta_L_abs_p90": float(np.percentile(np.abs(delta_l), 90)),
            "hue_error_median_degrees": float(np.median(recolor_core.hue_distance_degrees(hue, float(row["hue_degrees"])))),
            "hue_error_p90_degrees": float(np.percentile(recolor_core.hue_distance_degrees(hue, float(row["hue_degrees"])), 90)),
        }
    outside_ok = all(row["repaired_boundary_qc"]["outside_repaired_mask_changed_pixel_count"] == 0 for row in rows)
    repaired_ok = all(row["repaired_boundary_qc"]["added_mask_unchanged_output_count"] == 0 for row in rows)
    value = {"schema": "natural_subject_recolor_border_repair_analysis/v1", "record_count": 5, "repair_manifest_sha256": sha256(run_root / REPAIR_NAME), "outside_repaired_mask_ok": outside_ok, "added_bottom_region_recolored": repaired_ok, "automatic_safety_pass": outside_ok and repaired_ok and all(row["metrics"]["rgb_clipping_pixel_count"] == 0 for row in rows), "manual_review_required": ["identity_preservation", "shading_preservation", "bottom_boundary_halo_review"], "records": rows}
    write_json(run_root / ANALYSIS_NAME, value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "repair-mask", "recolor", "analyze"))
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        if args.source_root is None:
            parser.error("--source-root is required for plan")
        value = plan(args.run_root, args.source_root)
    elif args.command == "repair-mask":
        value = repair_mask(args.run_root)
    elif args.command == "recolor":
        value = recolor(args.run_root)
    else:
        value = analyze(args.run_root)
    print(json.dumps({"status": "ok", "keys": sorted(value)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
