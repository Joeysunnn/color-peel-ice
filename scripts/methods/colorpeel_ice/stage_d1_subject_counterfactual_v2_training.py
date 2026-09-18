#!/usr/bin/env python3
"""Stage the reviewed 25-pair D1 mailbox counterfactual-v2 subject dataset."""

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

import numpy as np
from PIL import Image

from src.methods.colorpeel_ice.natural_subject_counterfactual_v2 import require


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


def protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    require(value.get("schema") == "natural_subject_counterfactual_v2_training_protocol/v1", "Protocol differs")
    require(value.get("subject") == {
        "stable_id": "D1GT:81/130.png",
        "modifier_token": "<S*>",
        "initializer_token": "mailbox",
        "training_prompt_template": "a photo of <S*> mailbox in {color} color",
    }, "Subject contract differs")
    require(value.get("training") == {
        "seed": 42,
        "max_train_steps": 1000,
        "embedding_learning_rate": 1.0e-5,
        "kv_learning_rate": 1.0e-5,
        "full_kv": True,
        "hflip": False,
        "from_scratch": True,
    }, "Training contract differs")
    require(value.get("caa") == {"enabled": False, "cos_weight": 0.0, "reason": "one learned modifier token cannot form a learned-token attention pair"}, "CAA contract differs")
    require(value.get("approval_state") == {"subject_counterfactual_v2_training_approved": True, "joint_training_approved": False}, "Approval differs")
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


def verified_source(source_root: Path, value: dict[str, Any]) -> list[dict[str, Any]]:
    source = value["source_counterfactual"]
    paths = {name: source_root / name for name in ("subject_counterfactual_v2_plan.json", "subject_counterfactual_v2_results.json", "subject_counterfactual_v2_analysis.json")}
    expected = {"subject_counterfactual_v2_plan.json": source["plan_sha256"], "subject_counterfactual_v2_results.json": source["results_sha256"], "subject_counterfactual_v2_analysis.json": source["analysis_sha256"]}
    require(all(path.is_file() and sha256(path) == expected[name] for name, path in paths.items()), "Counterfactual source provenance differs")
    plan, results, analysis = (read_json(paths[name]) for name in paths)
    source_protocol = REPO_ROOT / source["source_protocol_relative_path"]
    require(source_protocol.is_file() and sha256(source_protocol) == source["source_protocol_sha256"] and plan.get("protocol_sha256") == source["source_protocol_sha256"], "Counterfactual protocol differs")
    require(analysis.get("automatic_safety_pass") is True and results.get("plan_sha256") == sha256(paths["subject_counterfactual_v2_plan.json"]), "Counterfactual QC differs")
    records = {row.get("record_id"): row for row in results.get("records", []) if isinstance(row, dict)}
    colors, variants = value["training_data"]["color_names"], value["training_data"]["variant_ids"]
    expected_ids = {f"{color}__{variant}" for color in colors for variant in variants}
    require(set(records) == expected_ids and len(records) == value["training_data"]["record_count"], "Counterfactual record coverage differs")
    verified = []
    for record_id in sorted(expected_ids):
        row = records[record_id]
        color, variant = record_id.split("__", 1)
        require(row.get("color_name") == color and row.get("variant", {}).get("id") == variant and row.get("hue_distance_degrees", 0.0) >= value["training_data"]["minimum_hue_separation_degrees"], f"Held-out lineage differs: {record_id}")
        image, mask = source_root / row["image_relative_path"], source_root / row["mask_relative_path"]
        require(image.is_file() and mask.is_file() and sha256(image) == row["image_sha256"] and sha256(mask) == row["mask_sha256"], f"Counterfactual artifact differs: {record_id}")
        mask_array = np.asarray(Image.open(mask).convert("L"), dtype=np.uint8)
        require(mask_array.shape == (512, 512) and set(np.unique(mask_array).tolist()) <= {0, 255} and np.any(mask_array == 255), f"Counterfactual mask differs: {record_id}")
        verified.append({"record_id": record_id, "color_name": color, "variant_id": variant, "image": image, "mask": mask, "image_sha256": row["image_sha256"], "mask_sha256": row["mask_sha256"], "hue_degrees": row["hue_degrees"], "hue_distance_degrees": row["hue_distance_degrees"]})
    return verified


def stage(source_root: Path, output_root: Path, protocol_path: Path) -> dict[str, Any]:
    value = protocol(protocol_path)
    require(source_root.resolve() == Path(value["source_counterfactual"]["source_root"]).resolve(), "Source root differs")
    records = verified_source(source_root.resolve(), value)
    require(not output_root.exists() or not any(output_root.iterdir()), "Output root must be new or empty")
    output_root.mkdir(parents=True, exist_ok=True)
    concepts, staged = [], []
    template = value["subject"]["training_prompt_template"]
    for row in records:
        image_dir, mask_dir = output_root / row["record_id"] / "images", output_root / row["record_id"] / "masks"
        image_dir.mkdir(parents=True); mask_dir.mkdir()
        image_method = link_or_copy(row["image"], image_dir / "image.png", row["image_sha256"])
        mask_method = link_or_copy(row["mask"], mask_dir / "image.png", row["mask_sha256"])
        prompt = template.format(color=row["color_name"])
        concepts.append({"instance_prompt": [prompt], "instance_data_dir": str(image_dir), "instance_mask_dir": str(mask_dir)})
        staged.append({**{key: row[key] for key in ("record_id", "color_name", "variant_id", "hue_degrees", "hue_distance_degrees", "image_sha256", "mask_sha256")}, "prompt": prompt, "image_staging_method": image_method, "mask_staging_method": mask_method})
    concepts_path, manifest_path = output_root / "concepts.json", output_root / "staging_manifest.json"
    concepts_path.write_text(json.dumps(concepts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps({"schema": "natural_subject_counterfactual_v2_training_staging_manifest/v1", "protocol_sha256": sha256(protocol_path), "source_results_sha256": value["source_counterfactual"]["results_sha256"], "record_count": len(staged), "records": staged}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "staged", "concepts": str(concepts_path), "manifest": str(manifest_path), "record_count": len(staged)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(stage(args.source_root, args.output_root, args.protocol), sort_keys=True))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Counterfactual-v2 staging aborted: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
