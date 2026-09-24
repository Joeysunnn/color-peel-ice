"""Pair the fixed material pilot outputs from the corrected and original ground-reflection data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw


GROUPS = ("seen_reconstruction", "color_invariance", "unseen_object_transfer", "lighting_robustness")
CELL = 202
LABEL_WIDTH = 260
ROW_HEIGHT = 226


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_run(root: Path) -> tuple[list[dict], dict[str, dict], dict]:
    generation = root / "generation"
    manifest = read_jsonl(generation / "generation_manifest.jsonl")
    statuses = read_jsonl(generation / "generation_status.jsonl")
    provenance = json.loads((generation / "provenance.json").read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in statuses}
    if len(manifest) != 60 or len(statuses) != 60 or len(by_id) != 60:
        raise ValueError(f"Expected 60 unique generation rows in {root}")
    if {row["id"] for row in manifest} != set(by_id):
        raise ValueError(f"Manifest and statuses differ in {root}")
    for row in statuses:
        image = generation / row["image_path"]
        if row["status"] not in {"ok", "safety_filtered"} or sha256(image) != row["image_sha256"]:
            raise ValueError(f"Invalid generation status or image hash: {row['id']}")
        with Image.open(image) as opened:
            if opened.mode != "RGB" or opened.size != (512, 512):
                raise ValueError(f"Invalid image mode or size: {row['id']}")
    return manifest, by_id, provenance


def draw_sheet(group: str, manifest: list[dict], corrected: dict[str, dict], reflection: dict[str, dict],
               corrected_root: Path, reflection_root: Path, output: Path) -> None:
    prompts = list(dict.fromkeys(row["prompt"] for row in manifest if row["group"] == group))
    sheet = Image.new("RGB", (LABEL_WIDTH + CELL * 5, ROW_HEIGHT * 2 * len(prompts)), "white")
    draw = ImageDraw.Draw(sheet)
    for prompt_index, prompt in enumerate(prompts):
        samples = [row for row in manifest if row["group"] == group and row["prompt"] == prompt]
        if len(samples) != 5 or [row["seed"] for row in samples] != [42, 43, 44, 45, 46]:
            raise ValueError(f"Unexpected seed grid: {group} / {prompt}")
        for variant_index, (label, root, records) in enumerate((
            ("corrected", corrected_root, corrected),
            ("ground reflection", reflection_root, reflection),
        )):
            y = (prompt_index * 2 + variant_index) * ROW_HEIGHT
            for line_index, line in enumerate(textwrap.wrap(prompt, width=32)):
                draw.text((8, y + 8 + line_index * 14), line, fill="black")
            draw.text((8, y + 74), label, fill="black")
            for seed_index, sample in enumerate(samples):
                record = records[sample["id"]]
                x = LABEL_WIDTH + seed_index * CELL
                with Image.open(root / "generation" / record["image_path"]) as opened:
                    sheet.paste(opened.resize((CELL - 8, CELL - 8), Image.Resampling.LANCZOS), (x, y))
                draw.text((x + 4, y + CELL - 3), f"seed {sample['seed']}  {record['status']}", fill="black")
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrected-run", type=Path, required=True)
    parser.add_argument("--reflection-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Comparison output directory must be new or empty")
    corrected_manifest, corrected, corrected_provenance = load_run(args.corrected_run)
    reflection_manifest, reflection, reflection_provenance = load_run(args.reflection_run)
    if corrected_manifest != reflection_manifest or corrected_provenance["protocol_sha256"] != reflection_provenance["protocol_sha256"]:
        raise ValueError("Comparison runs do not have the same prompt and seed protocol")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for group in GROUPS:
        draw_sheet(group, corrected_manifest, corrected, reflection, args.corrected_run,
                   args.reflection_run, args.output_dir / f"{group}.png")
    summary = {
        "corrected_run": str(args.corrected_run.resolve()),
        "reflection_run": str(args.reflection_run.resolve()),
        "generation_manifest_sha256": corrected_provenance["generation_manifest_sha256"],
        "evaluation_protocol_sha256": corrected_provenance["protocol_sha256"],
        "corrected_status_counts": dict(Counter(row["status"] for row in corrected.values())),
        "reflection_status_counts": dict(Counter(row["status"] for row in reflection.values())),
        "corrected_filtered_ids": [row["id"] for row in corrected_manifest if corrected[row["id"]]["status"] == "safety_filtered"],
        "reflection_filtered_ids": [row["id"] for row in corrected_manifest if reflection[row["id"]]["status"] == "safety_filtered"],
    }
    (args.output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
