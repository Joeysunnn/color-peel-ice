"""Validate and prepare the 360-image full-grid material human gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from PIL import Image, ImageDraw

from src.methods.colorpeel_ice.material_evaluation_protocol import build_full_grid_manifest, read_protocol


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def bundle(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True); protocol = read_protocol(args.evaluation_protocol)
    rows, statuses = read_jsonl(args.manifest), read_jsonl(args.status)
    expected = {row["id"]: row for row in build_full_grid_manifest(protocol)}
    indexed = {row.get("id"): row for row in rows}; status_index = {row.get("id"): row for row in statuses}
    if len(rows) != 360 or len(indexed) != 360 or set(indexed) != set(expected):
        raise ValueError("baseline generation manifest must contain exact 360-row full grid")
    if len(statuses) != 360 or len(status_index) != 360 or set(status_index) != set(expected):
        raise ValueError("baseline generation status must contain exact 360-row full grid")
    resolved = []
    for item_id, locked in expected.items():
        row, status = indexed[item_id], status_index[item_id]
        for field, value in locked.items():
            if field != "image_path" and row.get(field) != value:
                raise ValueError(f"baseline row changed {field}: {item_id}")
        path = (args.image_root / row["image_path"]).resolve()
        if status.get("status") != "ok" or not path.is_file() or status.get("image_sha256") != sha256_file(path):
            raise ValueError(f"baseline image failed or changed: {item_id}")
        with Image.open(path) as image:
            image.load()
            if image.size != (512, 512) or image.mode != "RGB":
                raise ValueError(f"baseline image is not 512x512 RGB: {item_id}")
        resolved.append({**row, "image_path": str(path)})
    review = [{"review_order": index, "generation_id": row["id"], "generation_seed": row["generation_seed"],
               "requested_shape": row["expected_shape"], "requested_color": row["expected_color"],
               "requested_material": row["expected_material"], "image_path": row["image_path"],
               "observed_shape": "", "observed_color": "", "observed_material": "", "shape_correct": "",
               "color_correct": "", "material_correct": "", "joint_correct": "", "black_cap": "",
               "all_black": "", "confidence": "", "comment": ""}
              for index, row in enumerate(random.Random(42).sample(resolved, len(resolved)), 1)]
    write_csv(review, output / "human_review.csv")
    red_sphere = [row for row in review if row["requested_shape"] == "sphere" and row["requested_color"] == "red"]
    write_csv(red_sphere, output / "red_sphere_material_regression.csv")
    pair_dir = output / "material_pair_sheets"; pair_dir.mkdir()
    sheets = []
    for shape in ("cube", "sphere", "cylinder"):
        for color in ("red", "cyan", "gray"):
            subset = [row for row in resolved if row["expected_shape"] == shape and row["expected_color"] == color]
            by_key = {(row["expected_material"], row["generation_seed"]): row for row in subset}
            canvas = Image.new("RGB", (400, 20 * 224), "white"); draw = ImageDraw.Draw(canvas)
            for line, seed in enumerate(range(42, 62)):
                for column, material in enumerate(("metal", "rubber")):
                    row = by_key[(material, seed)]
                    with Image.open(row["image_path"]) as source:
                        tile = source.convert("RGB"); tile.thumbnail((192, 192), Image.Resampling.LANCZOS)
                    canvas.paste(tile, (column * 200 + (200 - tile.width) // 2, line * 224 + 24))
                    draw.text((column * 200 + 4, line * 224 + 4), f"{material} seed {seed}", fill="black")
            path = pair_dir / f"{shape}_{color}.png"; canvas.save(path); sheets.append(str(path.resolve()))
    gate = {"status": "pending_human_review", "heldout_training_authorized": False,
            "required_checks": ["metal_rubber_distinguishable", "material_swap_preserves_shape_color",
                                "red_rubber_sphere_no_persistent_metal_black_cap"]}
    (output / "human_gate_decision.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    result = {"status": "ready_for_human_review", "images": 360, "review_rows": 360,
              "red_sphere_rows": 40, "material_pair_sheets": sheets}
    (output / "baseline_provenance.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                                      encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(bundle(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
