#!/usr/bin/env python3
"""Stage the nine verified Emission images for one `<C*>` ColorPeel short run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.colorpeel_ice import emission_color_branch_pilot as source_pilot

PROTOCOL_RELPATH = "experiments/natural_image_subject_color_pilot/configs/d1_emission_color_transfer_orange_protocol_v1.json"
ANALYSIS_NAME = "emission_color_branch_pilot_analysis.json"
MANIFEST_NAME = "emission_color_branch_pilot_render_manifest.json"


class StagingError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StagingError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StagingError(f"Cannot read {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path} must contain an object")
    return value


def protocol() -> dict[str, Any]:
    value = read_json(REPO_ROOT / PROTOCOL_RELPATH)
    require(value.get("schema") == "emission_color_transfer_protocol/v1", "Protocol schema differs")
    require(value.get("target", {}).get("stable_id") == "D1GT:81/130.png", "Target differs")
    training = value.get("training", {})
    require(training.get("modifier_token") == "<C*>" and training.get("initializer_token") == "orange", "Token identity differs")
    require(training.get("subjects") == ["cube", "sphere", "cylinder"] and training.get("view_indices") == [0, 8, 16] and training.get("image_count") == 9, "Training matrix differs")
    require(value.get("approval_state", {}).get("short_transfer_training_approved") is True, "Training is not approved")
    return value


def expected_requests(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    target, training = value["target"], value["training"]
    rows = {}
    for shape in training["subjects"]:
        for view in training["view_indices"]:
            request_id = f"emission__D1GT_81_130__{shape}__v{view:02d}"
            rows[request_id] = {"shape": shape, "view_index": view,
                                "prompt": training["prompt_template"].format(subject=shape)}
    require(len(rows) == 9, "Expected request identity differs")
    return rows


def verified_source(source_root: Path, value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    analysis_path, manifest_path = source_root / ANALYSIS_NAME, source_root / MANIFEST_NAME
    source = value["source_pilot"]
    require(analysis_path.is_file() and sha256(analysis_path) == source["analysis_sha256"], "Source analysis hash differs")
    analysis = read_json(analysis_path)
    require(analysis.get("status") == source["required_status"] and analysis.get("overall_pass") is True and analysis.get("request_count") == 18, "Source analysis differs")
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == "d1_emission_color_branch_pilot_manifest/v1" and manifest.get("request_count") == 18, "Source manifest differs")
    records = {row.get("request_id"): row for row in manifest.get("records", []) if isinstance(row, dict)}
    expected = expected_requests(value)
    require(set(expected) <= set(records), "Source target coverage differs")
    selected = {}
    for request_id, request in expected.items():
        row = records[request_id]
        image = source_root / row.get("image_relative_path", "")
        metadata = source_root / row.get("metadata_relative_path", "")
        require(image.is_file() and sha256(image) == row.get("image_sha256") and metadata.is_file() and sha256(metadata) == row.get("metadata_sha256"), "Source artifact hash differs")
        runtime = read_json(metadata)
        payload = runtime.get("request", {})
        require(payload.get("stable_id") == value["target"]["stable_id"] and payload.get("shape") == request["shape"] and payload.get("view_index") == request["view_index"], "Source request differs")
        require(payload.get("material") == "Emission" and payload.get("emission_strength") == 1.0, "Source material differs")
        selected[request_id] = {**request, "image": image, "image_sha256": row["image_sha256"]}
    return selected


def stage(source_root: Path, output_root: Path) -> dict[str, Any]:
    value = protocol()
    selected = verified_source(source_root.resolve(), value)
    require(not output_root.exists() or not any(output_root.iterdir()), "Output root must be new or empty")
    output_root.mkdir(parents=True, exist_ok=True)
    concepts, records = [], []
    for request_id, row in selected.items():
        directory = output_root / request_id
        directory.mkdir()
        destination = directory / "img.png"
        try:
            os.symlink(row["image"], destination)
            method = "symlink"
        except OSError:
            shutil.copy2(row["image"], destination)
            method = "copy_fallback"
        require(sha256(destination) == row["image_sha256"], "Staged image hash differs")
        concepts.append({"instance_prompt": [row["prompt"]], "instance_data_dir": str(directory)})
        records.append({"request_id": request_id, "shape": row["shape"], "view_index": row["view_index"], "prompt": row["prompt"], "source_image_sha256": row["image_sha256"], "staging_method": method})
    concepts_path, manifest_path = output_root / "concepts.json", output_root / "staging_manifest.json"
    concepts_path.write_text(json.dumps(concepts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps({"schema": "d1_emission_color_transfer_staging_manifest/v1", "source_analysis_sha256": value["source_pilot"]["analysis_sha256"], "protocol_sha256": sha256(REPO_ROOT / PROTOCOL_RELPATH), "record_count": 9, "records": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "staged", "concepts": str(concepts_path), "manifest": str(manifest_path), "record_count": 9}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(stage(args.source_root, args.output_root), sort_keys=True))
    except (OSError, StagingError) as exc:
        parser.exit(2, f"Emission transfer staging aborted: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
