"""Build a randomized, blinded human-review packet for cyan diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


EXPECTED_ROWS = 540
REVIEW_FIELDS = (
    "review_id",
    "noun",
    "seed",
    "target_color",
    "image_path",
    "color_fidelity_rating",
    "prompt_alignment_rating",
    "visual_quality_rating",
    "invalid_or_artifact",
    "reviewer_id",
    "notes",
)


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
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} diagnostic rows, found {len(rows)}")
    ids = set()
    for row in rows:
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id or item_id in ids:
            raise ValueError("diagnostic row ids must be present and unique")
        ids.add(item_id)
        if row.get("category") != "cyan_diagnostic":
            raise ValueError(f"unexpected category for {item_id}")


def validate_image(path: Path) -> None:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (512, 512):
            raise ValueError(f"expected RGB 512x512 image: {path}")


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_packet(
    rows: list[dict[str, Any]],
    image_dir: Path,
    blinded_image_dir: Path,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_rows(rows)
    rng = random.Random(random_seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    review_rows = []
    key_rows = []
    blinded_image_dir.mkdir(parents=True, exist_ok=True)
    for review_index, item in enumerate(shuffled, 1):
        review_id = f"review-{review_index:04d}"
        source = image_dir / Path(item["image_path"])
        validate_image(source)
        blinded = blinded_image_dir / f"{review_id}.png"
        if blinded.exists():
            raise FileExistsError(f"refusing to overwrite blinded image: {blinded}")
        shutil.copy2(source, blinded)
        review = {
            "review_id": review_id,
            "noun": item["noun"],
            "seed": item["seed"],
            "target_color": "cyan",
            "image_path": str(blinded),
            "color_fidelity_rating": "",
            "prompt_alignment_rating": "",
            "visual_quality_rating": "",
            "invalid_or_artifact": "",
            "reviewer_id": "",
            "notes": "",
        }
        key = {
            "review_id": review_id,
            "id": item["id"],
            "pair_id": item["pair_id"],
            "comparison_id": item["comparison_id"],
            "model_variant": item["model_variant"],
            "template_family": item["template_family"],
            "color_candidate": item["color_candidate"],
            "condition": item["condition"],
            "prompt": item["prompt"],
            "source_path": str(source),
        }
        review_rows.append(review)
        key_rows.append(key)
    return review_rows, key_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--blinded-image-dir", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--key-csv", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=20260822)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    review_rows, key_rows = build_packet(
        read_jsonl(args.manifest),
        args.image_dir,
        args.blinded_image_dir,
        args.random_seed,
    )
    write_csv(args.review_csv, REVIEW_FIELDS, review_rows)
    key_fields = (
        "review_id",
        "id",
        "pair_id",
        "comparison_id",
        "model_variant",
        "template_family",
        "color_candidate",
        "condition",
        "prompt",
        "source_path",
    )
    write_csv(args.key_csv, key_fields, key_rows)


if __name__ == "__main__":
    main()
