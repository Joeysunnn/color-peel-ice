#!/usr/bin/env python3
"""Build deterministic background/pose counterfactuals from repaired D1 mailbox assets."""

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

from src.methods.colorpeel_ice import natural_subject_counterfactual_v2 as core
from src.methods.colorpeel_ice import natural_subject_recolor_pilot as recolor_core


PROTOCOL = REPO_ROOT / "experiments/natural_image_subject_color_pilot/configs/d1_subject_counterfactual_v2_protocol_v1.json"
PLAN_NAME = "subject_counterfactual_v2_plan.json"
RESULTS_NAME = "subject_counterfactual_v2_results.json"
ANALYSIS_NAME = "subject_counterfactual_v2_analysis.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    core.require(isinstance(value, dict), f"{path} must contain an object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _source_records(source_root: Path, protocol: dict[str, Any]) -> tuple[dict[str, Path], np.ndarray, np.ndarray, np.ndarray, Path]:
    source = protocol["source_repaired_subject"]
    analysis_path = source_root / source["analysis_relative_path"]
    results_path = source_root / source["results_relative_path"]
    repair_path = source_root / source["repair_manifest_relative_path"]
    source_plan_path = source_root / "subject_recolor_border_repair_plan.json"
    core.require(
        analysis_path.is_file() and results_path.is_file() and repair_path.is_file()
        and sha256(analysis_path) == source["analysis_sha256"]
        and sha256(results_path) == source["results_sha256"],
        "Repaired source provenance differs",
    )
    analysis, results, repair, source_plan = read_json(analysis_path), read_json(results_path), read_json(repair_path), read_json(source_plan_path)
    core.require(
        analysis.get("automatic_safety_pass") is True and results.get("record_count") == 5
        and results.get("repair_manifest_sha256") == sha256(repair_path)
        and repair.get("plan_sha256") == sha256(source_plan_path),
        "Repaired source safety differs",
    )
    mask_path, alpha_path = source_root / repair["repaired_mask_relative_path"], source_root / repair["repaired_alpha_relative_path"]
    core.require(
        sha256(mask_path) == source["repaired_mask_sha256"],
        "Repaired mask provenance differs",
    )
    records = {row["name"]: row for row in results["records"]}
    expected = protocol["auxiliary_colors"]
    core.require(set(records) == set(expected), "Auxiliary color coverage differs")
    images = {}
    for name, color in expected.items():
        record = records[name]
        image_path = source_root / record["image_relative_path"]
        core.require(record["hue_degrees"] == color["hue_degrees"] and sha256(image_path) == color["image_sha256"], f"Source image differs: {name}")
        images[name] = image_path
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    alpha = np.asarray(Image.open(alpha_path), dtype=np.uint16)
    source_image_path = Path(source_plan["source"]["paths"]["raw_image"])
    source_image = np.asarray(Image.open(source_image_path).convert("RGB"), dtype=np.uint8)
    core.require(
        source_image_path.is_file() and sha256(source_image_path) == source["raw_image_sha256"]
        and source_image.shape[:2] == mask.shape == alpha.shape == (512, 512) and np.all(alpha[mask == 0] == 0),
        "Raw image or repaired mask differs",
    )
    return images, mask, alpha, source_image, source_image_path


def _requests(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    source_hue, threshold = protocol["held_out_guard"]["source_hue_degrees"], protocol["held_out_guard"]["minimum_hue_separation_degrees"]
    requests = []
    for name, color in protocol["auxiliary_colors"].items():
        distance = float(recolor_core.hue_distance_degrees(np.array([color["hue_degrees"]]), source_hue)[0])
        core.require(distance >= threshold, f"Held-out hue guard differs: {name}")
        for variant in protocol["variants"]:
            requests.append({"record_id": f"{name}__{variant['id']}", "color_name": name, "hue_degrees": color["hue_degrees"], "hue_distance_degrees": distance, "variant": variant})
    core.require(len(requests) == protocol["expected_record_count"], "Counterfactual coverage differs")
    return requests


def plan(run_root: Path, source_root: Path, protocol_path: Path = PROTOCOL) -> dict[str, Any]:
    protocol_path, source_root = protocol_path.resolve(), source_root.resolve()
    protocol = read_json(protocol_path)
    core.require(protocol.get("schema") == "natural_subject_counterfactual_v2_protocol/v1", "Protocol differs")
    core.require(protocol["source_repaired_subject"]["stable_id"] == "D1GT:81/130.png", "Subject differs")
    images, _, _, _, source_image_path = _source_records(source_root, protocol)
    core.require(not run_root.exists(), "Output root must be new")
    value = {
        "schema": "natural_subject_counterfactual_v2_plan/v1",
        "git_commit": git_commit(),
        "adapter_script_sha256": sha256(Path(__file__).resolve()),
        "core_module_sha256": sha256(REPO_ROOT / "src/methods/colorpeel_ice/natural_subject_counterfactual_v2.py"),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "protocol_sha256": sha256(protocol_path),
        "source_root": str(source_root),
        "source_raw_image": str(source_image_path),
        "source_raw_image_sha256": sha256(source_image_path),
        "source_image_sha256": {name: sha256(path) for name, path in images.items()},
        "requests": _requests(protocol),
    }
    run_root.mkdir(parents=True)
    write_json(run_root / PLAN_NAME, value)
    return value


def generate(run_root: Path) -> dict[str, Any]:
    plan_value = read_json(run_root / PLAN_NAME)
    protocol = read_json(PROTOCOL) if sha256(PROTOCOL) == plan_value["protocol_sha256"] else None
    core.require(protocol is not None, "Protocol hash differs")
    images, mask, alpha, source_image, source_image_path = _source_records(Path(plan_value["source_root"]), protocol)
    core.require(str(source_image_path) == plan_value["source_raw_image"], "Raw source image path differs")
    core.require(sha256(Path(plan_value["source_raw_image"])) == plan_value["source_raw_image_sha256"], "Raw source image provenance differs")
    records = []
    for request in plan_value["requests"]:
        name, record_id = request["color_name"], request["record_id"]
        recolored = np.asarray(Image.open(images[name]).convert("RGB"), dtype=np.uint8)
        output, transformed_mask, metrics = core.apply_variant(recolored, source_image, mask, alpha, request["variant"])
        image_path, mask_path = run_root / "images" / f"{record_id}.png", run_root / "masks" / f"{record_id}.png"
        image_path.parent.mkdir(exist_ok=True); mask_path.parent.mkdir(exist_ok=True)
        core.require(not image_path.exists() and not mask_path.exists(), f"Output already exists: {record_id}")
        Image.fromarray(output).save(image_path)
        Image.fromarray(transformed_mask, mode="L").save(mask_path)
        records.append({**request, "image_relative_path": image_path.relative_to(run_root).as_posix(), "image_sha256": sha256(image_path), "mask_relative_path": mask_path.relative_to(run_root).as_posix(), "mask_sha256": sha256(mask_path), "metrics": metrics})
    value = {"schema": "natural_subject_counterfactual_v2_results/v1", "plan_sha256": sha256(run_root / PLAN_NAME), "record_count": len(records), "records": records}
    write_json(run_root / RESULTS_NAME, value)
    return value


def analyze(run_root: Path) -> dict[str, Any]:
    plan_value, results = read_json(run_root / PLAN_NAME), read_json(run_root / RESULTS_NAME)
    core.require(results.get("plan_sha256") == sha256(run_root / PLAN_NAME), "Results provenance differs")
    rows = results.get("records", [])
    core.require(len(rows) == results.get("record_count") == len(plan_value["requests"]), "Output coverage differs")
    checks = []
    for row in rows:
        image_path, mask_path = run_root / row["image_relative_path"], run_root / row["mask_relative_path"]
        core.require(sha256(image_path) == row["image_sha256"] and sha256(mask_path) == row["mask_sha256"], "Output hash differs")
        image, mask = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8), np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        background = np.asarray(row["variant"]["background_rgb"], dtype=np.uint8)
        checks.append({"record_id": row["record_id"], "mask_binary": set(np.unique(mask).tolist()) <= {0, 255}, "mask_nonempty": bool(np.any(mask == 255)), "background_exact_outside_mask": bool(np.all(image[mask == 0] == background))})
    safe = all(all(check.values()) for check in checks)
    value = {"schema": "natural_subject_counterfactual_v2_analysis/v1", "record_count": len(checks), "automatic_safety_pass": safe, "manual_review_required": ["mailbox identity preservation", "background seam and halo review", "scale/translation coverage"], "records": checks}
    write_json(run_root / ANALYSIS_NAME, value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "generate", "analyze"))
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args(argv)
    if args.command == "plan":
        if args.source_root is None:
            parser.error("--source-root is required for plan")
        value = plan(args.run_root, args.source_root, args.protocol)
    elif args.command == "generate":
        value = generate(args.run_root)
    else:
        value = analyze(args.run_root)
    print(json.dumps({"status": "ok", "keys": sorted(value)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
