"""Merge nine successful 360-image material runs for human review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from src.methods.colorpeel_ice.material_evaluation_protocol import (
    build_manifest, read_protocol, validate_campaign,
)

FOLDS = ("A", "B", "C")
TRAINING_SEEDS = (42, 43, 44)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def expected_variant(fold: str, seed: int) -> str:
    return f"material_heldout_generate_fold_{fold.lower()}_train{seed}"


def discover_runs(root: Path) -> dict[tuple[str, int], tuple[Path, dict[str, Any]]]:
    wanted = {expected_variant(fold, seed): (fold, seed) for fold in FOLDS for seed in TRAINING_SEEDS}
    found = {}
    for path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8")); variant = manifest.get("run", {}).get("variant")
        if variant not in wanted or manifest.get("status") != "succeeded":
            continue
        if manifest.get("returncode") != 0 or manifest.get("stage") != "generate_material_multiview":
            raise ValueError(f"invalid material generation manifest: {path}")
        key = wanted[variant]
        if key in found:
            raise ValueError(f"duplicate material generation run: {key}")
        found[key] = (path.parent.resolve(), manifest)
    expected = {(fold, seed) for fold in FOLDS for seed in TRAINING_SEEDS}
    if set(found) != expected:
        raise ValueError(f"expected nine material generation runs, found {sorted(found)}")
    return found


def validate_run(run_dir: Path, launcher: dict[str, Any], protocol: dict[str, Any],
                 fold: str, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inference = run_dir / "inference"; rows = read_jsonl(inference / "generation_manifest.jsonl")
    statuses = read_jsonl(inference / "generation_status.jsonl")
    expected_ids = {row["id"] for row in build_manifest(protocol, fold_id=fold, training_seed=seed)}
    ids = [row.get("id") for row in rows]; status_index = {row.get("id"): row for row in statuses}
    if len(rows) != 360 or len(set(ids)) != 360 or set(ids) != expected_ids:
        raise ValueError(f"generation manifest is not a locked 360-row slice: {run_dir}")
    if len(statuses) != 360 or len(status_index) != 360 or set(status_index) != expected_ids:
        raise ValueError(f"generation status is not a locked 360-row slice: {run_dir}")
    merged, merged_status = [], []
    for row in rows:
        status = status_index[row["id"]]; path = (inference / row["image_path"]).resolve()
        if status.get("status") != "ok" or not path.is_file() or status.get("image_sha256") != sha256_file(path):
            raise ValueError(f"generation item failed or changed: {row['id']}")
        with Image.open(path) as image:
            image.load()
            if image.size != (512, 512) or image.mode != "RGB":
                raise ValueError(f"generation image is not 512x512 RGB: {row['id']}")
        merged.append({**row, "source_generation_run": str(run_dir),
                       "source_generation_git_commit": launcher["git"]["commit"],
                       "source_image_path": row["image_path"], "image_path": str(path)})
        merged_status.append({**status, "source_generation_run": str(run_dir), "image_path": str(path)})
    return merged, merged_status


def contact_sheets(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    directory = output_dir / "contact_sheets"; directory.mkdir()
    paths = []
    for fold in FOLDS:
        for seed in TRAINING_SEEDS:
            subset = [row for row in rows if row["fold_id"] == fold and row["training_seed"] == seed]
            canvas = Image.new("RGB", (20 * 96, 18 * 96), "white"); draw = ImageDraw.Draw(canvas)
            for index, row in enumerate(subset):
                x, y = (index % 20) * 96, (index // 20) * 96
                with Image.open(row["image_path"]) as source:
                    tile = source.convert("RGB"); tile.thumbnail((90, 76))
                canvas.paste(tile, (x + 3, y + 18)); split = "H" if row["held_out"] else "S"
                draw.text((x + 2, y + 2), f"{split}-{row['material_label'][0]}-{row['generation_seed']}",
                          fill="red" if row["held_out"] else "black")
            path = directory / f"fold_{fold.lower()}_train{seed}_all360.jpg"
            canvas.save(path, quality=90); paths.append(str(path.resolve()))
    return paths


def bundle(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True); protocol = read_protocol(args.evaluation_protocol)
    rows, statuses, inventory = [], [], []
    for key, (run_dir, launcher) in sorted(discover_runs(args.generation_root.resolve()).items()):
        run_rows, run_statuses = validate_run(run_dir, launcher, protocol, key[0], key[1])
        rows.extend(run_rows); statuses.extend(run_statuses)
        inventory.append({"fold_id": key[0], "training_seed": key[1], "run_dir": str(run_dir),
                          "images": 360, "model_fingerprint_sha256": run_rows[0]["model_fingerprint_sha256"]})
    validate_campaign(rows, protocol)
    if len({status["image_sha256"] for status in statuses}) != 3240:
        raise ValueError("material campaign contains duplicate image hashes")
    write_jsonl(rows, output_dir / "campaign_generation_manifest.jsonl")
    write_jsonl(statuses, output_dir / "campaign_generation_status.jsonl")
    write_jsonl(inventory, output_dir / "generation_run_inventory.jsonl")
    write_csv([], output_dir / "generation_failures.csv")
    review = [{"review_order": index, "generation_id": row["id"], "fold_id": row["fold_id"],
               "training_seed": row["training_seed"], "generation_seed": row["generation_seed"],
               "combination_status": row["combination_status"], "requested_shape": row["expected_shape"],
               "requested_color": row["expected_color"], "requested_material": row["expected_material"],
               "image_path": row["image_path"], "observed_shape": "", "observed_color": "",
               "observed_material": "", "shape_correct": "", "color_correct": "", "material_correct": "",
               "joint_correct": "", "black_cap": "", "all_black": "", "confidence": "", "comment": ""}
              for index, row in enumerate(random.Random(42).sample(rows, len(rows)), 1)]
    write_csv(review, output_dir / "human_review.csv"); sheets = contact_sheets(rows, output_dir)
    gate = {"status": "pending_human_review", "qwen_authorized": False,
            "required_checks": ["shape_stability", "color_stability", "material_stability",
                                "red_rubber_sphere_no_persistent_metal_black_cap"]}
    (output_dir / "human_gate_decision.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    qwen = {"schema_version": 1, "status": "ready_after_human_review", "stage": "predict_qwen_material",
            "run": {"study": "clevr_subject_color_material_3x3x2", "variant": "material_heldout_qwen_campaign", "seed": 42},
            "environment": {"CUDA_VISIBLE_DEVICES": "3"}, "args": {
                "manifest": str((output_dir / "campaign_generation_manifest.jsonl").resolve()), "image-dir": "/",
                "evaluation-protocol": str(args.evaluation_protocol.resolve()), "device": "cuda:0"},
            "protocol": {"model": "Qwen/Qwen3-VL-8B-Instruct", "torch_dtype": "float16",
                         "do_sample": False, "max_new_tokens": 128, "human_review_is_primary": True}}
    (output_dir / "qwen_config.json").write_text(json.dumps(qwen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {"status": "ready_for_human_review", "images": 3240, "seen_images": 2160,
              "held_out_images": 1080, "generation_runs": 9, "contact_sheets": sheets,
              "qwen_config": str((output_dir / "qwen_config.json").resolve())}
    (output_dir / "campaign_provenance.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                                          encoding="utf-8")
    return result


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
