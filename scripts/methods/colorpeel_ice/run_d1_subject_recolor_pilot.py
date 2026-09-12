#!/usr/bin/env python3
"""Plan, recolor, and analyze the five-image natural subject-only pilot."""

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
from src.methods.colorpeel_ice import natural_subject_recolor_pilot as core


PROTOCOL = REPO_ROOT / "experiments/natural_image_subject_color_pilot/configs/d1_subject_recolor_pilot_protocol_v1.json"
PLAN_NAME, MANIFEST_NAME, ANALYSIS_NAME = "subject_recolor_plan.json", "subject_recolor_manifest.json", "subject_recolor_analysis.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    core.require(isinstance(value, dict), f"{path} must contain an object")
    return value


def plan(run_root: Path, source_root: Path) -> dict[str, Any]:
    protocol, source_root = read_json(PROTOCOL), source_root.resolve()
    core.require(protocol.get("schema") == "natural_subject_recolor_pilot_protocol/v1", "Protocol differs")
    source_manifest = source_root / "manifests" / "pilot_mask_manifest.json"
    core.require(source_manifest.is_file() and sha256(source_manifest) == protocol["source"]["mask_manifest_sha256"], "Source manifest differs")
    source = next((row for row in read_json(source_manifest).get("samples", []) if row.get("stable_id") == protocol["source"]["stable_id"]), None)
    core.require(isinstance(source, dict) and source.get("status") == "PASS", "Source subject is not PASS")
    outputs = source.get("outputs", {})
    paths = {name: source_root / outputs.get(name, "") for name in ("raw_image", "raw_mask", "alpha_canonical")}
    core.require(all(path.is_file() for path in paths.values()), "Source artifact is missing")
    core.require(sha256(paths["raw_image"]) == protocol["source"]["raw_image_sha256"] and sha256(paths["raw_mask"]) == protocol["source"]["raw_mask_sha256"], "Source raw artifact differs")
    palette = protocol["auxiliary_palette"]
    generated = [row for row in palette if row["name"] in protocol["held_out_guard"]["generated_palette_names"]]
    core.require(len(generated) == protocol["expected_image_count"] and [row["name"] for row in palette if row["name"] in protocol["held_out_guard"]["excluded_palette_names"]] == protocol["held_out_guard"]["excluded_palette_names"], "Palette identity differs")
    threshold, original = protocol["held_out_guard"]["minimum_hue_separation_degrees"], protocol["source"]["original_hue_degrees"]
    core.require(all(float(core.hue_distance_degrees(np.array([row["hue_degrees"]]), original)[0]) >= threshold for row in generated), "Held-out hue guard differs")
    value = {
        "schema": "natural_subject_recolor_pilot_plan/v1",
        "git_commit": git_commit(),
        "adapter_script_sha256": sha256(Path(__file__).resolve()),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "protocol_sha256": sha256(PROTOCOL),
        "source_root": str(source_root),
        "source_manifest_sha256": sha256(source_manifest),
        "source": {"stable_id": source["stable_id"], "paths": {name: str(path) for name, path in paths.items()}, "sha256": {name: sha256(path) for name, path in paths.items()}},
        "requests": generated,
    }
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / PLAN_NAME).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def recolor(run_root: Path) -> dict[str, Any]:
    value = read_json(run_root / PLAN_NAME)
    paths = {name: Path(path) for name, path in value["source"]["paths"].items()}
    core.require(all(sha256(paths[name]) == value["source"]["sha256"][name] for name in paths), "Source artifact hash differs")
    image = np.asarray(Image.open(paths["raw_image"]).convert("RGB"), dtype=np.uint8)
    mask = np.asarray(Image.open(paths["raw_mask"]).convert("L"), dtype=np.uint8)
    alpha = np.asarray(Image.open(paths["alpha_canonical"]), dtype=np.uint16)
    records = []
    for request in value["requests"]:
        output, metrics, maps = core.recolor(image, mask, alpha, float(request["hue_degrees"]))
        image_path, map_path = run_root / "images" / f"{request['name']}.png", run_root / "maps" / f"{request['name']}.npz"
        image_path.parent.mkdir(parents=True, exist_ok=True); map_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(output).save(image_path); np.savez_compressed(map_path, **maps)
        records.append({"name": request["name"], "hue_degrees": request["hue_degrees"], "image_relative_path": str(image_path.relative_to(run_root)), "image_sha256": sha256(image_path), "map_relative_path": str(map_path.relative_to(run_root)), "map_sha256": sha256(map_path), "metrics": metrics})
    manifest = {"schema": "natural_subject_recolor_pilot_manifest/v1", "plan_sha256": sha256(run_root / PLAN_NAME), "record_count": len(records), "records": records}
    (run_root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def analyze(run_root: Path) -> dict[str, Any]:
    manifest = read_json(run_root / MANIFEST_NAME)
    core.require(manifest.get("record_count") == 5 and len(manifest.get("records", [])) == 5, "Output coverage differs")
    for row in manifest["records"]:
        core.require(sha256(run_root / row["image_relative_path"]) == row["image_sha256"] and sha256(run_root / row["map_relative_path"]) == row["map_sha256"], "Output artifact differs")
    plan_value = read_json(run_root / PLAN_NAME)
    source = np.asarray(Image.open(plan_value["source"]["paths"]["raw_image"]).convert("RGB"), dtype=np.uint8)
    mask = np.asarray(Image.open(plan_value["source"]["paths"]["raw_mask"]).convert("L"), dtype=np.uint8)
    interior, geometry = shared._mask_interior((mask.reshape(-1) == 255).tolist(), mask.shape[1], mask.shape[0])
    interior = np.asarray(interior, dtype=np.intp)
    source_lab = core.gamut.linear_rgb_to_lab(core.gamut._srgb_to_linear(source.reshape(-1, 3)[interior]))
    rows = manifest["records"]
    for row in rows:
        output = np.asarray(Image.open(run_root / row["image_relative_path"]).convert("RGB"), dtype=np.uint8)
        output_lab = core.gamut.linear_rgb_to_lab(core.gamut._srgb_to_linear(output.reshape(-1, 3)[interior]))
        selected = output_lab[(source_lab[:, 0] > 5.0) & (source_lab[:, 0] < 95.0)]
        core.require(selected.size > 0, "Eroded interior has no eligible pixels")
        hue = np.degrees(np.arctan2(selected[:, 2], selected[:, 1])) % 360.0
        delta_l = selected[:, 0] - source_lab[(source_lab[:, 0] > 5.0) & (source_lab[:, 0] < 95.0), 0]
        row["eroded_interior_measurement"] = {
            "estimator": "eroded_eligible_Lab",
            "geometry": geometry,
            "eligible_pixel_count": int(selected.shape[0]),
            "median_a": float(np.median(selected[:, 1])),
            "median_b": float(np.median(selected[:, 2])),
            "median_C": float(np.median(np.hypot(selected[:, 1], selected[:, 2]))),
            "delta_L_median": float(np.median(delta_l)),
            "delta_L_abs_p90": float(np.percentile(np.abs(delta_l), 90)),
            "hue_error_median_degrees": float(np.median(core.hue_distance_degrees(hue, float(row["hue_degrees"])))),
            "hue_error_p90_degrees": float(np.percentile(core.hue_distance_degrees(hue, float(row["hue_degrees"])), 90)),
        }
    outside_mask_ok = all(row["metrics"]["outside_mask_changed_pixel_count"] == 0 for row in rows)
    no_rgb_clipping = all(row["metrics"]["rgb_clipping_pixel_count"] == 0 for row in rows)
    alpha_outside_ok = all(row["metrics"]["alpha_outside_nonzero_pixel_count"] == 0 for row in rows)
    value = {"schema": "natural_subject_recolor_pilot_analysis/v1", "record_count": 5, "outside_mask_ok": outside_mask_ok, "no_rgb_clipping": no_rgb_clipping, "alpha_outside_ok": alpha_outside_ok, "automatic_safety_pass": outside_mask_ok and no_rgb_clipping and alpha_outside_ok, "manual_review_required": ["identity_preservation", "shading_preservation", "halo_artifact_review"], "records": rows}
    (run_root / ANALYSIS_NAME).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "recolor", "analyze")); parser.add_argument("--run-root", type=Path, required=True); parser.add_argument("--source-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        if args.source_root is None: parser.error("--source-root is required for plan")
        result = plan(args.run_root, args.source_root)
    elif args.command == "recolor": result = recolor(args.run_root)
    else: result = analyze(args.run_root)
    print(json.dumps({"status": "ok", "keys": sorted(result)}, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
