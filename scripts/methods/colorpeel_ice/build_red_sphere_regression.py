"""Build the 180-row legacy red-sphere material-artifact regression ledger."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.methods.colorpeel_ice.multiview_evaluation_protocol import (
    read_evaluation_protocol,
    validate_campaign_manifest,
)

OBSERVATION = (
    "User review: Fold C red sphere persistently shows a black upper cap; similar "
    "view-dependent dark metal reflections occur less severely in other folds."
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = read_evaluation_protocol(args.evaluation_protocol)
    rows = [row for row in validate_campaign_manifest(read_jsonl(args.manifest), protocol)
            if row["expected_shape"] == "sphere" and row["expected_color"] == "red"]
    if len(rows) != 180:
        raise ValueError(f"expected 180 legacy red-sphere images, found {len(rows)}")
    output = [{"generation_id": row["id"], "fold_id": row["fold_id"],
               "training_seed": row["training_seed"], "generation_seed": row["generation_seed"],
               "combination_status": row["combination_status"], "image_path": row["image_path"],
               "black_cap": "", "shape_correct": "", "color_correct": "", "view_dependent": "",
               "confidence": "", "comment": "", "prior_human_observation": OBSERVATION}
              for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)


if __name__ == "__main__":
    main()
