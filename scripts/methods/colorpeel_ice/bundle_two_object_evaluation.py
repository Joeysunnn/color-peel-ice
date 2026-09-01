"""Validate 720 two-object generations and build the human-review packet."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.colorpeel_ice.two_object_evaluation_protocol import (  # noqa: E402
    build_manifest,
    file_sha256,
    read_protocol,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(generation_root: Path, protocol_path: Path) -> list[dict]:
    protocol = read_protocol(protocol_path)
    expected = build_manifest(protocol)
    rows = read_jsonl(generation_root / "generation_manifest.jsonl")
    statuses = read_jsonl(generation_root / "generation_status.jsonl")
    if len(rows) != len(statuses) or len(rows) != protocol["expected_image_count"]:
        raise ValueError("generation manifest/status must contain exactly 720 rows")
    expected_by_id = {row["id"]: row for row in expected}
    rows_by_id = {row.get("id"): row for row in rows}
    status_by_id = {row.get("id"): row for row in statuses}
    if len(rows_by_id) != len(rows) or len(status_by_id) != len(statuses):
        raise ValueError("generation ledgers contain duplicate ids")
    if set(rows_by_id) != set(expected_by_id) or set(status_by_id) != set(expected_by_id):
        raise ValueError("generation ids differ from the locked protocol")
    hashes_by_scene: dict[str, set[str]] = {}
    for item_id, expected_row in expected_by_id.items():
        row, status = rows_by_id[item_id], status_by_id[item_id]
        for field in (
            "pair_group", "pair_index", "orientation", "scene_id", "generation_seed",
            "prompt", "left", "right", "num_inference_steps", "guidance_scale", "image_path",
        ):
            if row.get(field) != expected_row[field]:
                raise ValueError(f"generation row changed {field}: {item_id}")
        if not row.get("safety_checker_disabled") or not row.get("safety_risk_acknowledged"):
            raise ValueError(f"safety policy not recorded: {item_id}")
        if status.get("status") != "ok" or status.get("failure_reason") is not None:
            raise ValueError(f"generation failed: {item_id}")
        image_path = generation_root / row["image_path"]
        with Image.open(image_path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (512, 512):
                raise ValueError(f"invalid generated RGB: {item_id}")
        image_hash = file_sha256(image_path)
        if status.get("image_sha256") != image_hash:
            raise ValueError(f"generated image hash changed: {item_id}")
        for field in ("model_fingerprint_sha256", "protocol_fingerprint_sha256"):
            if status.get(field) != row.get(field):
                raise ValueError(f"generation provenance changed {field}: {item_id}")
        hashes_by_scene.setdefault(row["scene_id"], set()).add(image_hash)
    if len(hashes_by_scene) != 36 or set(map(len, hashes_by_scene.values())) != {20}:
        raise ValueError("every scene must have 20 unique generated RGB hashes")
    return rows


def _tile(image_path: Path) -> Image.Image:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    image.thumbnail((192, 192), Image.Resampling.LANCZOS)
    return image


def build_sheet(rows: list[dict], generation_root: Path, output: Path) -> None:
    selected = (42, 46, 50, 54, 58)
    by_key = {(row["scene_id"], row["generation_seed"]): row for row in rows}
    scenes = sorted({row["scene_id"] for row in rows})
    sheet = Image.new("RGB", (1000, len(scenes) * 224), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, scene_id in enumerate(scenes):
        for column, seed in enumerate(selected):
            row = by_key[(scene_id, seed)]
            tile = _tile(generation_root / row["image_path"])
            sheet.paste(tile, (column * 200 + (200 - tile.width) // 2, row_index * 224 + 24))
            draw.text((column * 200 + 4, row_index * 224 + 4), f"{scene_id} s{seed}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def build_pair_sheets(rows: list[dict], generation_root: Path, output_dir: Path) -> list[str]:
    selected = (42, 46, 50, 54, 58)
    by_key = {(row["pair_group"], row["pair_index"], row["orientation"], row["generation_seed"]): row
              for row in rows}
    paths = []
    for pair_index in range(9):
        sheet = Image.new("RGB", (1000, 4 * 224), "white")
        draw = ImageDraw.Draw(sheet)
        for row_index, (group, orientation) in enumerate(
            (("seen", "forward"), ("seen", "swapped"), ("unseen", "forward"), ("unseen", "swapped"))
        ):
            for column, seed in enumerate(selected):
                row = by_key[(group, pair_index, orientation, seed)]
                tile = _tile(generation_root / row["image_path"])
                sheet.paste(tile, (column * 200 + (200 - tile.width) // 2, row_index * 224 + 24))
                draw.text((column * 200 + 4, row_index * 224 + 4), f"{group} {orientation} s{seed}", fill="black")
        path = output_dir / f"pair_{pair_index:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path)
        paths.append(str(path))
    return paths


def write_review(rows: list[dict], generation_root: Path, output: Path) -> None:
    rows = list(rows)
    random.Random(42).shuffle(rows)
    fields = [
        "review_index", "generation_id", "pair_group", "pair_index", "orientation", "seed", "image",
        "expected_left", "expected_right", "two_objects_present", "left_bundle_correct", "right_bundle_correct",
        "attribute_swap", "missing_object", "extra_object", "severe_occlusion", "all_black", "confidence", "comment",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow({
                "review_index": index,
                "generation_id": row["id"],
                "pair_group": row["pair_group"],
                "pair_index": row["pair_index"],
                "orientation": row["orientation"],
                "seed": row["generation_seed"],
                "image": str((generation_root / row["image_path"]).resolve()),
                "expected_left": row["left"]["state_id"],
                "expected_right": row["right"]["state_id"],
            })


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("output-dir must be new or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = validate(args.generation_root.resolve(), args.evaluation_protocol)
    review = args.output_dir / "two_object_generation_human_review.csv"
    main_sheet = args.output_dir / "two_object_generation_36scene_contact_sheet.png"
    write_review(rows, args.generation_root.resolve(), review)
    build_sheet(rows, args.generation_root.resolve(), main_sheet)
    pair_sheets = build_pair_sheets(rows, args.generation_root.resolve(), args.output_dir / "pair_sheets")
    result = {
        "status": "validated_pending_human_review",
        "generation_rows": 720,
        "seen_rows": 360,
        "unseen_rows": 360,
        "scene_count": 36,
        "training_or_generation_changed": False,
        "human_review_csv": str(review),
        "contact_sheet": str(main_sheet),
        "pair_sheets": pair_sheets,
    }
    (args.output_dir / "bundle_status.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
