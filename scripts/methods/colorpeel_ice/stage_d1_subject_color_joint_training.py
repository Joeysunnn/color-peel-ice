#!/usr/bin/env python3
"""Stage a balanced, unpaired `<S*>`/`<C*>` shared-K/V joint-training pilot."""

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

from scripts.methods.colorpeel_ice import stage_d1_emission_color_transfer as color_stage
from scripts.methods.colorpeel_ice import stage_d1_subject_recolor_training as subject_stage


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} must contain an object")
    return value


def protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    require(value.get("schema") == "natural_subject_color_joint_training_protocol/v1", "Protocol schema differs")
    training, schedule = value.get("training", {}), value.get("schedule", {})
    require(training == {
        "modifier_tokens": ["<S*>", "<C*>"],
        "initializer_tokens": ["mailbox", "orange"],
        "shared_kv": True,
        "cos_weight": 0.0,
        "joint_two_object_binding": False,
        "original_subject_color_record": "forbidden",
        "paired_subject_color_rows": "forbidden",
    }, "Joint-token contract differs")
    require(schedule == {"subject_rows": 45, "color_rows": 45, "ordering": "strict_alternation_subject_then_color"}, "Schedule differs")
    for name in ("subject", "color"):
        branch = value.get("branches", {}).get(name, {})
        relative, expected = branch.get("source_protocol_relative_path"), branch.get("source_protocol_sha256")
        require(isinstance(relative, str) and isinstance(expected, str) and len(expected) == 64, f"{name} source contract differs")
    require(value.get("approval_state", {}).get("shared_kv_joint_pilot_approved") is True, "Joint pilot is not approved")
    return value


def source_protocol(branch: dict[str, Any]) -> Path:
    path = REPO_ROOT / branch["source_protocol_relative_path"]
    require(path.is_file() and sha256(path) == branch["source_protocol_sha256"], f"Source protocol differs: {path}")
    return path


def link_or_copy(source: Path, destination: Path, expected_sha256: str) -> str:
    try:
        os.symlink(source, destination)
        method = "symlink"
    except OSError:
        shutil.copy2(source, destination)
        method = "copy_fallback"
    require(sha256(destination) == expected_sha256, f"Staged hash differs: {destination}")
    return method


def build_schedule(subject_rows: list[dict[str, Any]], color_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    require(len(subject_rows) == 5 and len(color_rows) == 9, "Physical branch coverage differs")
    rows = []
    for index in range(45):
        subject, color = subject_rows[index % len(subject_rows)], color_rows[index % len(color_rows)]
        require("<S*>" in subject["instance_prompt"][0] and "<C*>" not in subject["instance_prompt"][0], "Subject row is paired")
        require("<C*>" in color["instance_prompt"][0] and "<S*>" not in color["instance_prompt"][0], "Color row is paired")
        rows.extend((subject, color))
    require(len(rows) == 90 and all(("<S*>" in row["instance_prompt"][0]) != ("<C*>" in row["instance_prompt"][0]) for row in rows), "Joint rows differ")
    return rows


def stage(subject_root: Path, color_root: Path, output_root: Path, protocol_path: Path) -> dict[str, Any]:
    value = protocol(protocol_path)
    require(not output_root.exists() or not any(output_root.iterdir()), "Output root must be new or empty")
    require(subject_root.resolve() == Path(value["branches"]["subject"]["source_root"]).resolve(), "Subject source root differs")
    require(color_root.resolve() == Path(value["branches"]["color"]["source_root"]).resolve(), "Color source root differs")
    subject_value = subject_stage.protocol(source_protocol(value["branches"]["subject"]))
    subject_images, subject_mask = subject_stage.verified_source(subject_root.resolve(), subject_value)
    color_value = color_stage.protocol(source_protocol(value["branches"]["color"]))
    color_images = color_stage.verified_source(color_root.resolve(), color_value)
    output_root.mkdir(parents=True, exist_ok=True)
    physical_records, subject_rows, color_rows = [], [], []
    for name in subject_value["training_data"]["image_names"]:
        directory = output_root / "subject" / name
        image_dir, mask_dir = directory / "images", directory / "masks"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir()
        image_hash = subject_value["training_data"]["expected_image_sha256"][name]
        image_method = link_or_copy(subject_images[name], image_dir / "image.png", image_hash)
        mask_method = link_or_copy(subject_mask, mask_dir / "image.png", subject_value["source_pilot"]["repaired_mask_sha256"])
        subject_rows.append({"instance_prompt": [subject_value["training_data"]["prompt_by_image"][name]], "instance_data_dir": str(image_dir), "instance_mask_dir": str(mask_dir)})
        physical_records.append({"branch": "subject", "name": name, "image_sha256": image_hash, "mask_sha256": subject_value["source_pilot"]["repaired_mask_sha256"], "image_staging_method": image_method, "mask_staging_method": mask_method})
    for request_id, row in color_images.items():
        directory = output_root / "color" / request_id
        directory.mkdir(parents=True)
        method = link_or_copy(row["image"], directory / "img.png", row["image_sha256"])
        color_rows.append({"instance_prompt": [row["prompt"]], "instance_data_dir": str(directory)})
        physical_records.append({"branch": "color", "name": request_id, "image_sha256": row["image_sha256"], "staging_method": method})
    concepts = build_schedule(subject_rows, color_rows)
    concepts_path, manifest_path = output_root / "concepts.json", output_root / "staging_manifest.json"
    concepts_path.write_text(json.dumps(concepts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "natural_subject_color_joint_staging_manifest/v1",
        "protocol_sha256": sha256(protocol_path),
        "physical_record_count": len(physical_records),
        "training_row_count": len(concepts),
        "subject_training_rows": 45,
        "color_training_rows": 45,
        "records": physical_records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "staged", "concepts": str(concepts_path), "manifest": str(manifest_path), "record_count": len(concepts)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--color-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(stage(args.subject_root, args.color_root, args.output_root, args.protocol), sort_keys=True))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Joint staging aborted: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
