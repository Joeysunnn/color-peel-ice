#!/usr/bin/env python3
"""Plan, inject, and analyze the D1 gamut-only LCh pilot using frozen neutral bases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from scripts.methods.colorpeel_ice import render_d1_color_calibration_preflight as shared
from src.methods.colorpeel_ice import gamut_only_lch_injection as injection
from src.methods.colorpeel_ice import gamut_aware_lch_injection as source_batch


PLAN_NAME = "gamut_only_lch_plan.json"
CONTRACT_NAME = "gamut_only_lch_contract.json"
RESULTS_NAME = "gamut_only_lch_results.json"
SUMMARY_NAME = "gamut_only_lch_analysis.json"
SOURCE_MANIFEST_NAME = "gamut_aware_lch_neutral_manifest.json"
SOURCE_CONTRACT_NAME = "gamut_aware_lch_contract.json"


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


def _source_records(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_path, contract_path = root / SOURCE_MANIFEST_NAME, root / SOURCE_CONTRACT_NAME
    injection.require(manifest_path.is_file() and contract_path.is_file(), "Frozen neutral source is incomplete")
    injection.require(file_sha256(manifest_path) == injection.SOURCE_NEUTRAL_MANIFEST_SHA256, "Frozen neutral manifest hash differs")
    injection.require(file_sha256(contract_path) == injection.SOURCE_NEUTRAL_CONTRACT_SHA256, "Frozen neutral contract hash differs")
    manifest, contract = json.loads(manifest_path.read_text(encoding="utf-8")), json.loads(contract_path.read_text(encoding="utf-8"))
    injection.require(manifest == {"schema": "d1_gamut_aware_lch_neutral_manifest/v1", "contract_sha256": source_batch.canonical_sha256(contract), "request_count": 9, "records": manifest.get("records")}, "Frozen neutral manifest structure differs")
    records = manifest["records"]
    injection.require(isinstance(records, list) and len(records) == 9, "Frozen neutral record coverage differs")
    by_id = {row.get("request_id"): row for row in records}
    injection.require(set(by_id) == {row["request_id"] for row in injection.neutral_requests()}, "Frozen neutral request identity differs")
    for row in records:
        for kind in ("image", "mask", "metadata"):
            path = root / row[f"{kind}_relative_path"]
            injection.require(path.is_file() and file_sha256(path) == row[f"{kind}_sha256"], f"Frozen neutral {kind} hash differs")
    return by_id, manifest


def plan(root: Path, neutral_root: Path) -> dict[str, Any]:
    injection.require(not root.exists() or not any(root.iterdir()), "Output root must be new or empty")
    _, source_manifest = _source_records(neutral_root)
    requests = injection.pilot_requests()
    contract = {
        "schema": "d1_gamut_only_lch_contract/v1", "git_commit": shared._git_commit(), "adapter_script_sha256": _script_hash(),
        "protocol_relative_path": injection.PROTOCOL_RELPATH, "protocol_sha256": _protocol_hash(),
        "selection_canonical_sha256": injection.SELECTION_CANONICAL_SHA256,
        "renderer_protocol_canonical_sha256": injection.RENDERER_PROTOCOL_CANONICAL_SHA256,
        "source_neutral_manifest_sha256": injection.SOURCE_NEUTRAL_MANIFEST_SHA256,
        "source_neutral_contract_sha256": injection.SOURCE_NEUTRAL_CONTRACT_SHA256,
        "source_neutral_records_sha256": injection.canonical_sha256(source_manifest["records"]),
        "injection_requests_sha256": injection.canonical_sha256(requests),
    }
    value = {"schema": "d1_gamut_only_lch_plan/v1", "injection_requests": requests,
             "injection_requests_sha256": injection.canonical_sha256(requests), "contract_sha256": injection.canonical_sha256(contract)}
    root.mkdir(parents=True, exist_ok=True)
    _write(root / CONTRACT_NAME, contract)
    _write(root / PLAN_NAME, value)
    return value


def _load_plan_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_value = json.loads((root / PLAN_NAME).read_text(encoding="utf-8"))
    contract = json.loads((root / CONTRACT_NAME).read_text(encoding="utf-8"))
    requests = injection.pilot_requests()
    expected_plan = {"schema": "d1_gamut_only_lch_plan/v1", "injection_requests": requests,
                     "injection_requests_sha256": injection.canonical_sha256(requests)}
    injection.require(set(plan_value) == set(expected_plan) | {"contract_sha256"} and {key: plan_value[key] for key in expected_plan} == expected_plan, "Plan request identity differs")
    expected_contract = {"schema": "d1_gamut_only_lch_contract/v1", "git_commit": shared._git_commit(), "adapter_script_sha256": _script_hash(),
                         "protocol_relative_path": injection.PROTOCOL_RELPATH, "protocol_sha256": _protocol_hash(),
                         "selection_canonical_sha256": injection.SELECTION_CANONICAL_SHA256,
                         "renderer_protocol_canonical_sha256": injection.RENDERER_PROTOCOL_CANONICAL_SHA256,
                         "source_neutral_manifest_sha256": injection.SOURCE_NEUTRAL_MANIFEST_SHA256,
                         "source_neutral_contract_sha256": injection.SOURCE_NEUTRAL_CONTRACT_SHA256,
                         "source_neutral_records_sha256": contract.get("source_neutral_records_sha256"),
                         "injection_requests_sha256": injection.canonical_sha256(requests)}
    injection.require(contract == expected_contract and plan_value["contract_sha256"] == injection.canonical_sha256(contract), "Contract provenance differs")
    return plan_value, contract


def inject(root: Path, neutral_root: Path) -> dict[str, Any]:
    from PIL import Image
    plan_value, contract = _load_plan_contract(root)
    injection.require(not (root / RESULTS_NAME).exists() and not (root / "injected").exists() and not (root / "chroma_maps").exists(), "Injection output already exists")
    neutral, manifest = _source_records(neutral_root)
    injection.require(injection.canonical_sha256(manifest["records"]) == contract["source_neutral_records_sha256"], "Frozen neutral records differ from contract")
    rows = []
    for request in plan_value["injection_requests"]:
        source_id = f"neutral__{request['shape']}__v{request['view_index']}"
        source = neutral[source_id]
        image_path, mask_path = neutral_root / source["image_relative_path"], neutral_root / source["mask_relative_path"]
        image, mask = np.asarray(Image.open(image_path).convert("RGB")), np.asarray(Image.open(mask_path).convert("L"))
        output, evidence, maps = injection.inject_rgb(image, mask, float(request["target_a"]), float(request["target_b"]))
        output_path, map_path = root / "injected" / f"{request['request_id']}.png", root / "chroma_maps" / f"{request['request_id']}.npz"
        output_path.parent.mkdir(exist_ok=True)
        map_path.parent.mkdir(exist_ok=True)
        Image.fromarray(output, "RGB").save(output_path)
        np.savez_compressed(map_path, **maps)
        rows.append({**request, **evidence, "source_neutral_request_id": source_id,
                     "source_neutral_image_relative_path": source["image_relative_path"], "source_neutral_image_sha256": source["image_sha256"],
                     "source_neutral_mask_relative_path": source["mask_relative_path"], "source_neutral_mask_sha256": source["mask_sha256"],
                     "image_relative_path": str(output_path.relative_to(root)), "image_sha256": file_sha256(output_path),
                     "chroma_map_relative_path": str(map_path.relative_to(root)), "chroma_map_sha256": file_sha256(map_path)})
    value = {"schema": "d1_gamut_only_lch_results/v1", "contract_sha256": injection.canonical_sha256(contract), "rows": rows}
    _write(root / RESULTS_NAME, value)
    return value


def _measure_pair(source: np.ndarray, output: np.ndarray, mask: np.ndarray, request: Mapping[str, Any]) -> dict[str, Any]:
    interior, geometry = shared._mask_interior((mask.reshape(-1) == 255).tolist(), mask.shape[1], mask.shape[0])
    indices = np.asarray(interior, dtype=np.intp)
    source_lab = source_batch.linear_rgb_to_lab(source_batch._srgb_to_linear(source.reshape(-1, 3)[indices]))
    output_lab = source_batch.linear_rgb_to_lab(source_batch._srgb_to_linear(output.reshape(-1, 3)[indices]))
    eligible = (source_lab[:, 0] > 5.0) & (source_lab[:, 0] < 95.0)
    injection.require(eligible.any(), "Eroded interior has no eligible pixels")
    selected = output_lab[eligible]
    dominant = np.median(selected, axis=0)
    delta_L = output_lab[eligible, 0] - source_lab[eligible, 0]
    chroma = np.hypot(selected[:, 1], selected[:, 2])
    hue_eligible = chroma > 0.5
    target_h = math.radians(float(request["target_h_degrees"]))
    hue_error = np.abs(np.angle(np.exp(1j * (np.arctan2(selected[hue_eligible, 2], selected[hue_eligible, 1]) - target_h))))
    return {"dominant_estimator": "eroded_eligible_componentwise_median_Lab", "median_L": float(dominant[0]), "dominant_a": float(dominant[1]), "dominant_b": float(dominant[2]),
            "e_ch": math.hypot(float(dominant[1]) - float(request["target_a"]), float(dominant[2]) - float(request["target_b"])),
            "delta_L_median": float(np.median(delta_L)), "delta_L_abs_p90": injection.nearest_rank_p90(np.abs(delta_L).tolist()),
            "hue_error_measurement_min_C": 0.5, "hue_eligible_pixel_count": int(hue_eligible.sum()),
            "hue_error_degrees_median": float(np.degrees(np.median(hue_error))) if hue_error.size else None,
            "hue_error_degrees_p90": float(np.degrees(injection.nearest_rank_p90(hue_error.tolist()))) if hue_error.size else None,
            "interior_geometry": geometry}


def analyze(root: Path, neutral_root: Path) -> dict[str, Any]:
    from PIL import Image
    _, contract = _load_plan_contract(root)
    injection.require(not (root / SUMMARY_NAME).exists(), "Analysis output already exists")
    neutral, manifest = _source_records(neutral_root)
    injection.require(injection.canonical_sha256(manifest["records"]) == contract["source_neutral_records_sha256"], "Frozen neutral records differ from contract")
    rows = json.loads((root / RESULTS_NAME).read_text(encoding="utf-8")).get("rows")
    injection.require(isinstance(rows, list) and len(rows) == 18, "Results coverage differs")
    for row in rows:
        source = neutral[row["source_neutral_request_id"]]
        source_image = np.asarray(Image.open(neutral_root / source["image_relative_path"]).convert("RGB"))
        mask = np.asarray(Image.open(neutral_root / source["mask_relative_path"]).convert("L"))
        output = np.asarray(Image.open(root / row["image_relative_path"]).convert("RGB"))
        injection.require(file_sha256(root / row["image_relative_path"]) == row["image_sha256"] and file_sha256(root / row["chroma_map_relative_path"]) == row["chroma_map_sha256"], "Injected artifact hash differs")
        delta = np.abs(output.astype(np.int16) - source_image.astype(np.int16))
        outside = mask == 0
        row["outside_mask_changed_pixel_count"] = int(np.any(delta[outside] != 0, axis=1).sum())
        row["outside_mask_max_abs_rgb_change"] = int(delta[outside].max()) if outside.any() else 0
        row.update(_measure_pair(source_image, output, mask, row))
    summary = injection.summarize_measurements(rows)
    summary.update({"contract_sha256": injection.canonical_sha256(contract), "results_sha256": file_sha256(root / RESULTS_NAME), "measurements": rows, "status": "passed" if summary["overall_pass"] else "failed"})
    _write(root / SUMMARY_NAME, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "inject", "analyze"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--neutral-root", required=True, type=Path)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    try:
        result = plan(args.output_root, args.neutral_root) if args.command == "plan" else inject(args.output_root, args.neutral_root) if args.command == "inject" else analyze(args.output_root, args.neutral_root)
    except (OSError, ValueError, injection.GamutOnlyLchError, shared.PreflightError) as exc:
        parser.exit(2, f"gamut-only LCh injection aborted: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
