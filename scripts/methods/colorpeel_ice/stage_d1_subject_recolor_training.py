#!/usr/bin/env python3
"""Stage the five repaired auxiliary-color images for one `<S*>` short run."""

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

from src.methods.colorpeel_ice.natural_subject_recolor_pilot import require


PROTOCOL_RELPATH = "experiments/natural_image_subject_color_pilot/configs/d1_subject_recolor_gorilla_training_protocol_v1.json"
ANALYSIS_NAME = "subject_recolor_border_repair_analysis.json"
RESULTS_NAME = "subject_recolor_border_repair_results.json"
REPAIR_NAME = "subject_recolor_border_repair_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} must contain an object")
    return value


def protocol(path: Path | None = None) -> dict[str, Any]:
    value = read_json(path or REPO_ROOT / PROTOCOL_RELPATH)
    schema = value.get("schema")
    require(schema in {"natural_subject_recolor_training_protocol/v1", "natural_subject_recolor_training_protocol/v2", "natural_subject_recolor_training_protocol/v3", "natural_subject_recolor_training_protocol/v4", "natural_subject_recolor_training_protocol/v5", "natural_subject_recolor_training_protocol/v6", "natural_subject_recolor_training_protocol/v7", "natural_subject_recolor_training_protocol/v8"}, "Protocol differs")
    subject, data, approval = value.get("subject", {}), value.get("training_data", {}), value.get("approval_state", {})
    require(subject.get("modifier_token") == "<S*>", "Subject token contract differs")
    image_names = ("red", "yellow", "green", "cyan", "blue")
    if schema.endswith("/v1"):
        require(subject.get("initializer_token") == "gorilla", "Subject token contract differs")
        require(subject.get("prompt") == "a photo of <S*>", "Subject token contract differs")
        prompt_template = "a photo of <S*> in {color} color"
        require(approval.get("subject_only_short_training_approved") is True, "Training approval differs")
    elif schema.endswith("/v2"):
        require(subject.get("initializer_token") == "gorilla", "Subject token contract differs")
        prompt_template = "a photo of <S*> statue in {color} color"
        require(subject.get("training_prompt_template") == prompt_template, "Statue prompt contract differs")
        require(approval.get("statue_prompt_underfit_followup_approved") is True, "Training approval differs")
    elif schema.endswith("/v3"):
        require(subject.get("initializer_token") == "statue", "Subject token contract differs")
        prompt_template = "a photo of <S*> gorilla statue in {color} color"
        require(subject.get("training_prompt_template") == prompt_template, "Statue category prompt contract differs")
        require(approval.get("initializer_kv_ablation_approved") is True, "Training approval differs")
    elif schema.endswith("/v4"):
        require(subject.get("initializer_token") == "statue", "Subject token contract differs")
        prompt_template = "a photo of <S*> gorilla statue in {color} color"
        require(subject.get("training_prompt_template") == prompt_template, "Statue category prompt contract differs")
        require(approval.get("step_dose_response_approved") is True, "Training approval differs")
        require(value.get("followup_training", {}).get("authorized_steps") == [750, 1000], "Step-dose contract differs")
    elif schema.endswith("/v5"):
        require(subject.get("modifier_token") == "<S*>" and subject.get("initializer_tokens") == ["statue", "gorilla", "sculpture"], "Initializer screen contract differs")
        prompt_template = "a photo of <S*> statue in {color} color"
        require(subject.get("training_prompt_template") == prompt_template, "No-gorilla prompt contract differs")
        require(approval.get("initializer_screen_approved") is True, "Training approval differs")
        require(value.get("screen", {}).get("authorized_steps") == 500 and value["screen"].get("kv_learning_rate") == 1.0e-5, "Initializer screen parameters differ")
    elif schema.endswith("/v6"):
        require(subject.get("modifier_token") == "<S*>" and subject.get("initializer_tokens") == ["statue", "gorilla", "sculpture"], "Initializer screen contract differs")
        prompt_template = "a photo of <S*> statue in {color} color"
        require(subject.get("training_prompt_template") == prompt_template, "No-gorilla prompt contract differs")
        require(approval.get("initializer_step_screen_approved") is True, "Training approval differs")
        require(value.get("screen", {}).get("authorized_steps") == [500, 750] and value["screen"].get("kv_learning_rate") == 1.0e-5, "Initializer step-screen parameters differ")
    else:
        require(subject.get("initializer_token") == "mailbox", "Mailbox initializer contract differs")
        prompt_template = subject.get("training_prompt_template")
        require(prompt_template in {"a photo of <S*> mailbox in {color} color", "a photo of <S*> in {color} color"}, "Mailbox prompt contract differs")
        image_names = ("red", "green", "cyan", "blue", "magenta")
        require(approval.get("mailbox_prompt_ablation_approved") is True, "Training approval differs")
        expected_steps = [750] if schema.endswith("/v7") else [1000]
        require(value.get("training", {}).get("authorized_steps") == expected_steps and value["training"].get("kv_learning_rate") == 1.0e-5, "Mailbox full-K/V parameters differ")
    expected_prompts = {name: prompt_template.format(color=name) for name in image_names}
    require(data.get("image_names") == list(expected_prompts) and data.get("prompt_by_image") == expected_prompts and data.get("original_subject_color_record") == "forbidden" and data.get("color_modifier_token") == "forbidden", "Training data identity differs")
    require(value.get("caa") == {"enabled": False, "cos_weight": 0.0, "reason": "one learned modifier token cannot form a learned-token attention pair"}, "CAA contract differs")
    require(data.get("use_repaired_binary_instance_mask") is True, "Training mask contract differs")
    require(approval.get("mixed_shared_checkpoint_training_approved") is False, "Mixed training must remain forbidden")
    return value


def link_or_copy(source: Path, destination: Path, expected_sha256: str) -> str:
    try:
        os.symlink(source, destination)
        method = "symlink"
    except OSError:
        shutil.copy2(source, destination)
        method = "copy_fallback"
    require(sha256(destination) == expected_sha256, f"Staged hash differs: {destination}")
    return method


def verified_source(source_root: Path, value: dict[str, Any]) -> tuple[dict[str, Path], Path]:
    source = value["source_pilot"]
    analysis_path, results_path, repair_path = source_root / ANALYSIS_NAME, source_root / RESULTS_NAME, source_root / REPAIR_NAME
    require(analysis_path.is_file() and sha256(analysis_path) == source["analysis_sha256"], "Source analysis differs")
    require(results_path.is_file() and sha256(results_path) == source["results_sha256"], "Source results differ")
    analysis, results, repair = read_json(analysis_path), read_json(results_path), read_json(repair_path)
    require(analysis.get("automatic_safety_pass") is source["required_automatic_safety_pass"] and analysis.get("added_bottom_region_recolored") is True and analysis.get("record_count") == 5, "Source analysis status differs")
    require(repair.get("repaired_mask_sha256") == source["repaired_mask_sha256"], "Repaired mask manifest differs")
    mask_path = source_root / repair.get("repaired_mask_relative_path", "")
    require(mask_path.is_file() and sha256(mask_path) == source["repaired_mask_sha256"], "Repaired mask differs")
    records = {row.get("name"): row for row in results.get("records", []) if isinstance(row, dict)}
    data = value["training_data"]
    require(set(records) == set(data["image_names"]), "Source image coverage differs")
    images = {}
    for name in data["image_names"]:
        record = records[name]
        image = source_root / record.get("image_relative_path", "")
        expected = data["expected_image_sha256"][name]
        require(image.is_file() and record.get("image_sha256") == expected and sha256(image) == expected, f"Source image differs: {name}")
        images[name] = image
    return images, mask_path


def stage(source_root: Path, output_root: Path, protocol_path: Path | None = None) -> dict[str, Any]:
    effective_protocol = protocol_path or REPO_ROOT / PROTOCOL_RELPATH
    value = protocol(effective_protocol)
    images, source_mask = verified_source(source_root.resolve(), value)
    require(not output_root.exists() or not any(output_root.iterdir()), "Output root must be new or empty")
    output_root.mkdir(parents=True)
    concepts, records = [], []
    for name in value["training_data"]["image_names"]:
        image_dir, mask_dir = output_root / name / "images", output_root / name / "masks"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir()
        image_destination, mask_destination = image_dir / "image.png", mask_dir / "image.png"
        expected_image = value["training_data"]["expected_image_sha256"][name]
        image_method = link_or_copy(images[name], image_destination, expected_image)
        mask_method = link_or_copy(source_mask, mask_destination, value["source_pilot"]["repaired_mask_sha256"])
        prompt = value["training_data"]["prompt_by_image"][name]
        concepts.append({"instance_prompt": [prompt], "instance_data_dir": str(image_dir), "instance_mask_dir": str(mask_dir)})
        records.append({"name": name, "prompt": prompt, "image_sha256": expected_image, "mask_sha256": value["source_pilot"]["repaired_mask_sha256"], "image_staging_method": image_method, "mask_staging_method": mask_method})
    concepts_path, manifest_path = output_root / "concepts.json", output_root / "staging_manifest.json"
    concepts_path.write_text(json.dumps(concepts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps({"schema": "natural_subject_recolor_training_staging_manifest/v1", "protocol_sha256": sha256(effective_protocol), "source_analysis_sha256": value["source_pilot"]["analysis_sha256"], "record_count": len(records), "records": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "staged", "concepts": str(concepts_path), "manifest": str(manifest_path), "record_count": len(records)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(stage(args.source_root, args.output_root, args.protocol), sort_keys=True))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Subject recolor staging aborted: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
