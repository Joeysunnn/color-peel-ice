#!/usr/bin/env python3
"""Validate and bundle two 18-image single-object checkpoint smokes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.colorpeel_ice.material_evaluation_protocol import (  # noqa: E402
    build_full_grid_manifest,
    read_protocol,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_generation(root: Path, protocol: dict[str, Any], expected_kind: str) -> list[dict[str, Any]]:
    expected = {row["id"]: row for row in build_full_grid_manifest(protocol, generation_seeds=[42])}
    rows = _read_jsonl(root / "generation_manifest.jsonl")
    statuses = _read_jsonl(root / "generation_status.jsonl")
    indexed = {row.get("id"): row for row in rows}
    status_index = {row.get("id"): row for row in statuses}
    if len(rows) != 18 or len(indexed) != 18 or set(indexed) != set(expected):
        raise ValueError("generation manifest must contain the exact 18-cell seed-42 smoke")
    if len(statuses) != 18 or len(status_index) != 18 or set(status_index) != set(expected):
        raise ValueError("generation status must contain the exact 18-cell seed-42 smoke")
    provenance = json.loads((root / "generation_provenance.json").read_text(encoding="utf-8"))
    if provenance.get("checkpoint_kind") != expected_kind:
        raise ValueError(f"expected checkpoint kind {expected_kind}")
    output = []
    for item_id, locked in expected.items():
        row, status = indexed[item_id], status_index[item_id]
        for field, value in locked.items():
            if field != "image_path" and row.get(field) != value:
                raise ValueError(f"generation row changed {field}: {item_id}")
        path = (root / row["image_path"]).resolve()
        if status.get("status") != "ok" or status.get("image_sha256") != _sha256(path):
            raise ValueError(f"generation image failed or changed: {item_id}")
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (512, 512):
                raise ValueError(f"generation image must be 512x512 RGB: {item_id}")
        output.append({**row, "resolved_image_path": str(path), "checkpoint_kind": expected_kind})
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _contact_sheet(by_key: dict[tuple[str, str, str, str], dict[str, Any]], output: Path) -> None:
    columns = (
        ("material-baseline", "metal"), ("joint-binding", "metal"),
        ("material-baseline", "rubber"), ("joint-binding", "rubber"),
    )
    rows = [(shape, color) for shape in ("cube", "sphere", "cylinder") for color in ("red", "cyan", "gray")]
    tile, header = 224, 28
    canvas = Image.new("RGB", (tile * len(columns), (tile + header) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (shape, color) in enumerate(rows):
        for column_index, (kind, material) in enumerate(columns):
            item = by_key[(kind, shape, color, material)]
            with Image.open(item["resolved_image_path"]) as source:
                image = source.convert("RGB")
                image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
            x, y = column_index * tile, row_index * (tile + header)
            canvas.paste(image, (x + (tile - image.width) // 2, y + header))
            draw.text((x + 4, y + 4), f"{shape} {color} | {kind} {material}", fill="black")
    canvas.save(output)


def bundle(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    protocol = read_protocol(args.evaluation_protocol)
    baseline = _validate_generation(args.baseline_root.resolve(), protocol, "material-baseline")
    joint = _validate_generation(args.joint_root.resolve(), protocol, "joint-binding")
    combined = baseline + joint
    by_key = {
        (row["checkpoint_kind"], row["expected_shape"], row["expected_color"], row["expected_material"]): row
        for row in combined
    }
    if len(by_key) != 36:
        raise ValueError("comparison must contain 18 semantic cells from each checkpoint")
    review = []
    for order, row in enumerate(random.Random(42).sample(combined, len(combined)), start=1):
        review.append({
            "review_order": order,
            "checkpoint_kind": row["checkpoint_kind"],
            "generation_id": row["id"],
            "requested_shape": row["expected_shape"],
            "requested_color": row["expected_color"],
            "requested_material": row["expected_material"],
            "image_path": row["resolved_image_path"],
            "observed_shape": "", "observed_color": "", "observed_material": "",
            "shape_correct": "", "color_correct": "", "material_correct": "",
            "joint_correct": "", "confidence": "", "comment": "",
        })
    _write_csv(output / "human_review.csv", review)
    _contact_sheet(by_key, output / "single_object_checkpoint_comparison.png")
    result = {
        "schema_version": 1,
        "status": "pending_human_review",
        "images_per_checkpoint": 18,
        "total_images": 36,
        "generation_seed": 42,
        "full_bundle_only": True,
        "full_360_generation_authorized": False,
        "contact_sheet": str((output / "single_object_checkpoint_comparison.png").resolve()),
        "review_csv": str((output / "human_review.csv").resolve()),
    }
    (output / "comparison_status.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--joint-root", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(bundle(parser.parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
