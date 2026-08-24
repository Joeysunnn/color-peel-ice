"""Merge nine successful multiview generation runs into one audited campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from src.methods.colorpeel_ice.multiview_evaluation_protocol import (
    build_manifest,
    read_evaluation_protocol,
    validate_campaign_manifest,
)


FOLDS = ("A", "B", "C")
TRAINING_SEEDS = (42, 43, 44)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_contact_sheets(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir()
    paths = []
    for fold_id in FOLDS:
        for training_seed in TRAINING_SEEDS:
            checkpoint_rows = [
                row
                for row in rows
                if row["fold_id"] == fold_id and row["training_seed"] == training_seed
            ]
            canvas = Image.new("RGB", (12 * 128, 15 * 128), "white")
            draw = ImageDraw.Draw(canvas)
            for index, row in enumerate(checkpoint_rows):
                x = (index % 12) * 128
                y = (index // 12) * 128
                with Image.open(row["image_path"]) as source:
                    thumb = source.convert("RGB")
                    thumb.thumbnail((120, 104))
                canvas.paste(thumb, (x + 4, y + 20))
                split = "H" if row["held_out"] else "S"
                draw.text(
                    (x + 4, y + 4),
                    f"{split} {row['subject_label'][0]}-{row['color_label'][0]} {row['generation_seed']}",
                    fill="red" if row["held_out"] else "black",
                )
            path = contact_dir / f"fold_{fold_id.lower()}_train{training_seed}_all180.jpg"
            canvas.save(path, quality=90)
            paths.append(str(path.resolve()))
    return paths


def expected_variant(fold_id: str, training_seed: int) -> str:
    return f"multiview_v2_heldout_generate_fold_{fold_id.lower()}_train{training_seed}"


def discover_runs(generation_root: Path) -> dict[tuple[str, int], tuple[Path, dict[str, Any]]]:
    expected = {
        expected_variant(fold_id, seed): (fold_id, seed)
        for fold_id in FOLDS
        for seed in TRAINING_SEEDS
    }
    found: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    for manifest_path in sorted(generation_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        variant = manifest.get("run", {}).get("variant")
        if variant not in expected or manifest.get("status") != "succeeded":
            continue
        if manifest.get("returncode") != 0 or manifest.get("stage") != "generate_multiview":
            raise ValueError(f"invalid successful generation manifest: {manifest_path}")
        key = expected[variant]
        if key in found:
            raise ValueError(f"multiple successful generation runs found for {key}")
        if int(manifest.get("run", {}).get("seed")) != key[1]:
            raise ValueError(f"launcher seed does not match generation variant: {manifest_path}")
        found[key] = (manifest_path.parent.resolve(), manifest)
    expected_keys = {(fold, seed) for fold in FOLDS for seed in TRAINING_SEEDS}
    if set(found) != expected_keys:
        raise ValueError(
            f"expected nine successful generation runs, found {sorted(found)}"
        )
    return found


def validate_one_run(
    run_dir: Path,
    launcher_manifest: dict[str, Any],
    protocol: dict[str, Any],
    fold_id: str,
    training_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inference_dir = run_dir / "inference"
    manifest_path = inference_dir / "generation_manifest.jsonl"
    status_path = inference_dir / "generation_status.jsonl"
    rows = read_jsonl(manifest_path)
    statuses = read_jsonl(status_path)
    expected_ids = {
        row["id"]
        for row in build_manifest(
            protocol, fold_id=fold_id, training_seed=training_seed
        )
    }
    ids = [row.get("id") for row in rows]
    if len(rows) != 180 or len(ids) != len(set(ids)) or set(ids) != expected_ids:
        raise ValueError(f"generation manifest is not the locked 180-row slice: {run_dir}")
    status_index = {row.get("id"): row for row in statuses}
    if len(statuses) != 180 or len(status_index) != 180 or set(status_index) != expected_ids:
        raise ValueError(f"generation status is not the locked 180-row slice: {run_dir}")

    merged_rows = []
    merged_statuses = []
    for row in rows:
        status = status_index[row["id"]]
        image_path = (inference_dir / row["image_path"]).resolve()
        if status.get("status") != "ok" or not image_path.is_file():
            raise ValueError(f"generation item is not successful: {row['id']}")
        if status.get("image_sha256") != sha256_file(image_path):
            raise ValueError(f"generation image hash mismatch: {row['id']}")
        if status.get("model_fingerprint_sha256") != row.get("model_fingerprint_sha256"):
            raise ValueError(f"generation model fingerprint mismatch: {row['id']}")
        if status.get("protocol_fingerprint_sha256") != row.get("protocol_fingerprint_sha256"):
            raise ValueError(f"generation protocol fingerprint mismatch: {row['id']}")
        with Image.open(image_path) as image:
            image.load()
            if image.size != (512, 512) or image.mode != "RGB":
                raise ValueError(f"generation image is not 512x512 RGB: {row['id']}")
        merged_rows.append(
            {
                **row,
                "source_generation_run": str(run_dir),
                "source_generation_git_commit": launcher_manifest["git"]["commit"],
                "source_image_path": row["image_path"],
                "image_path": str(image_path),
            }
        )
        merged_statuses.append(
            {
                **status,
                "source_generation_run": str(run_dir),
                "image_path": str(image_path),
            }
        )
    return merged_rows, merged_statuses


def bundle(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = read_evaluation_protocol(args.evaluation_protocol)
    runs = discover_runs(args.generation_root.resolve())
    rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    run_inventory = []
    for key in sorted(runs):
        run_dir, launcher_manifest = runs[key]
        run_rows, run_statuses = validate_one_run(
            run_dir, launcher_manifest, protocol, key[0], key[1]
        )
        rows.extend(run_rows)
        statuses.extend(run_statuses)
        run_inventory.append(
            {
                "fold_id": key[0],
                "training_seed": key[1],
                "run_dir": str(run_dir),
                "generation_git_commit": launcher_manifest["git"]["commit"],
                "model_fingerprint_sha256": run_rows[0]["model_fingerprint_sha256"],
                "images": 180,
            }
        )
    validate_campaign_manifest(rows, protocol)
    if len({status["image_sha256"] for status in statuses}) != 1620:
        raise ValueError("campaign contains duplicate generated image hashes")

    write_jsonl(rows, output_dir / "campaign_generation_manifest.jsonl")
    write_jsonl(statuses, output_dir / "campaign_generation_status.jsonl")
    write_jsonl(run_inventory, output_dir / "generation_run_inventory.jsonl")
    write_csv([], output_dir / "generation_failures.csv")

    review_rows = [
        {
            "review_order": index,
            "generation_id": row["id"],
            "fold_id": row["fold_id"],
            "training_seed": row["training_seed"],
            "generation_seed": row["generation_seed"],
            "combination_status": row["combination_status"],
            "requested_shape": row["expected_shape"],
            "requested_color": row["expected_color"],
            "image_path": row["image_path"],
            "observed_shape": "",
            "observed_color": "",
            "shape_correct": "",
            "color_correct": "",
            "joint_correct": "",
            "all_black": "",
            "confidence": "",
            "comment": "",
        }
        for index, row in enumerate(random.Random(42).sample(rows, len(rows)), 1)
    ]
    write_csv(review_rows, output_dir / "human_review.csv")
    contact_sheets = write_contact_sheets(rows, output_dir)
    qwen_config = {
        "schema_version": 1,
        "status": "ready_after_human_review",
        "stage": "predict_qwen",
        "run": {
            "study": "clevr_subject_color_3x3",
            "variant": "multiview_v2_heldout_qwen_campaign",
            "seed": 42,
        },
        "environment": {"CUDA_VISIBLE_DEVICES": "3"},
        "data_manifest": str(
            (output_dir / "campaign_generation_manifest.jsonl").resolve()
        ),
        "args": {
            "manifest": str(
                (output_dir / "campaign_generation_manifest.jsonl").resolve()
            ),
            "image-dir": "/",
            "evaluation-protocol": str(args.evaluation_protocol.resolve()),
            "device": "cuda:0",
            "protocol": "multiview-heldout",
        },
        "protocol": {
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "torch_dtype": "float16",
            "do_sample": False,
            "max_new_tokens": 128,
            "human_review_is_primary": True,
        },
    }
    (output_dir / "qwen_config.json").write_text(
        json.dumps(qwen_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "status": "ready_for_human_review",
        "evaluation_protocol_id": protocol["protocol_id"],
        "evaluation_protocol_sha256": sha256_file(args.evaluation_protocol),
        "generation_runs": 9,
        "images": 1620,
        "seen_images": 1080,
        "held_out_images": 540,
        "human_review_random_seed": 42,
        "contact_sheets": contact_sheets,
        "qwen_config": str((output_dir / "qwen_config.json").resolve()),
    }
    (output_dir / "campaign_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return provenance


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(bundle(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
