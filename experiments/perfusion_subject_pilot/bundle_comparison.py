"""Validate and arrange a completed Perfusion B0/P1 comparison for visual review."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def bundle(run: Path, protocol: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("status") != "succeeded" or provenance.get("protocol_sha256") != sha256(protocol):
        raise ValueError("comparison did not finish under the selected protocol")
    expected = jsonl(run / "generation_manifest.jsonl")
    observed = jsonl(run / "generation_status.jsonl")
    by_id = {row["id"]: row for row in observed}
    if len(expected) != 36 or len(observed) != 36 or len(by_id) != 36 or set(by_id) != {row["id"] for row in expected}:
        raise ValueError("expected 36 unique completed samples")
    by_key = {}
    for row in expected:
        status = by_id[row["id"]]
        key = (row["group"], row["condition"], row["seed"], row["arm"])
        if (key in by_key or any(status.get(field) != value for field, value in row.items())
                or status["status"] not in {"ok", "safety_filtered"}):
            raise ValueError(f"sample row differs: {row['id']}")
        image_path = run / row["image_path"]
        if sha256(image_path) != status["image_sha256"]:
            raise ValueError(f"image hash differs: {row['id']}")
        with Image.open(image_path) as image:
            if image.mode != "RGB" or image.size != (512, 512):
                raise ValueError(f"image dimensions differ: {row['id']}")
        by_key[key] = row
    output.mkdir(parents=True, exist_ok=True)
    cell, label, row_height = 384, 135, 412
    for group in ("plain", "red", "blue"):
        sheet = Image.new("RGB", (label + 2 * cell, 6 * row_height + 35), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((label + 6, 8), "B0  token-local K/V", fill="black")
        draw.text((label + cell + 6, 8), "P1  mailbox Key + rank-1 Value", fill="black")
        for seed_index, seed in enumerate((42, 43, 44)):
            for condition_index, condition in enumerate(("subject_only", "subject_material")):
                y = 35 + (2 * seed_index + condition_index) * row_height
                draw.text((8, y + 8), f"seed {seed}", fill="black")
                draw.text((8, y + 28), "S only" if condition_index == 0 else "S + M", fill="black")
                for arm_index, arm in enumerate(("B0", "P1")):
                    row = by_key[(group, condition, seed, arm)]
                    with Image.open(run / row["image_path"]) as image:
                        sheet.paste(image.resize((cell, cell), Image.Resampling.LANCZOS),
                                    (label + arm_index * cell, y))
                    draw.text((label + arm_index * cell + 6, y + cell + 4),
                              by_id[row["id"]]["status"], fill="black")
        sheet.save(output / f"{group}.png")
    summary = {
        "run": str(run.resolve()),
        "cohort": provenance["cohort"],
        "protocol_sha256": sha256(protocol),
        "generation_manifest_sha256": sha256(run / "generation_manifest.jsonl"),
        "generation_status_sha256": sha256(run / "generation_status.jsonl"),
        "status_counts": dict(Counter(row["status"] for row in observed)),
        "filtered_ids": [row["id"] for row in observed if row["status"] == "safety_filtered"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(bundle(args.run, args.protocol, args.output), indent=2))
