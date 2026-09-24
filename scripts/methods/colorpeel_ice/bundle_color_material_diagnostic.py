"""Validate and contact-sheet the fixed 36-image color/material comparison."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw


CONDITIONS = ("base", "color_only", "material_only", "color_material")
OBJECTS = ("cube", "sphere", "mug")
SEEDS = (42, 43, 44)
CELL = 250
LABEL = 245
ROW = 275


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bundle(run: Path, protocol: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("status") != "succeeded" or provenance.get("protocol_sha256") != sha256(protocol):
        raise ValueError("run did not finish under the locked protocol")
    manifest = read_jsonl(run / "generation_manifest.jsonl")
    statuses = read_jsonl(run / "generation_status.jsonl")
    by_id = {row["id"]: row for row in statuses}
    if len(manifest) != 36 or len(statuses) != 36 or len(by_id) != 36 or {row["id"] for row in manifest} != set(by_id):
        raise ValueError("comparison must have 36 unique matching rows")
    for row in manifest:
        result = by_id[row["id"]]
        if any(result.get(key) != value for key, value in row.items()):
            raise ValueError(f"generation row differs: {row['id']}")
        path = run / row["image_path"]
        if result["status"] not in {"ok", "safety_filtered"} or sha256(path) != result["image_sha256"]:
            raise ValueError(f"image hash or status differs: {row['id']}")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (512, 512):
                raise ValueError(f"image mode or size differs: {row['id']}")
    output.mkdir(parents=True, exist_ok=True)
    for object_name in OBJECTS:
        sheet = Image.new("RGB", (LABEL + 3 * CELL, 4 * ROW), "white")
        draw = ImageDraw.Draw(sheet)
        for row_index, condition in enumerate(CONDITIONS):
            samples = [row for row in manifest if row["object"] == object_name and row["condition"] == condition]
            if [row["seed"] for row in samples] != list(SEEDS):
                raise ValueError(f"comparison grid differs: {object_name} / {condition}")
            y = row_index * ROW
            draw.text((8, y + 8), condition, fill="black")
            for line_index, line in enumerate(textwrap.wrap(samples[0]["prompt"], width=30)):
                draw.text((8, y + 40 + line_index * 16), line, fill="black")
            for column, sample in enumerate(samples):
                result = by_id[sample["id"]]
                x = LABEL + column * CELL
                with Image.open(run / sample["image_path"]) as image:
                    sheet.paste(image.resize((CELL - 8, CELL - 8), Image.Resampling.LANCZOS), (x, y))
                draw.text((x + 4, y + CELL), f"seed {sample['seed']}  {result['status']}", fill="black")
        sheet.save(output / f"{object_name}.png")
    summary = {
        "run": str(run.resolve()), "protocol_sha256": sha256(protocol),
        "manifest_sha256": sha256(run / "generation_manifest.jsonl"),
        "generation_status_sha256": sha256(run / "generation_status.jsonl"),
        "status_counts": dict(Counter(row["status"] for row in statuses)),
        "filtered_ids": [row["id"] for row in manifest if by_id[row["id"]]["status"] == "safety_filtered"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(bundle(args.run, args.protocol, args.output), indent=2))


if __name__ == "__main__":
    main()
