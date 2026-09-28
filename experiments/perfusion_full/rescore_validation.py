"""Correct validation aggregation from stored per-image CLIP scores without rerendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rescore(directory: Path) -> None:
    selection_path = directory / "selection.json"
    previous = json.loads(selection_path.read_text())
    if (previous.get("schema") != "perfusion_full_validation/v1"
            or not previous.get("complete_validation")
            or len(previous.get("results", [])) != 16):
        raise ValueError("expected complete original validation")
    backup = directory / "selection_initial_mean_of_harmonics.json"
    if backup.exists():
        raise FileExistsError(backup)
    rows = [json.loads(line) for line in (directory / "validation.jsonl").read_text().splitlines()]
    if len(rows) != 16 * 8:
        raise ValueError("expected 128 per-image validation rows")
    summaries = []
    for step in range(25, 401, 25):
        group = [row for row in rows if row["step"] == step]
        if ([row["prompt_index"] for row in group] != list(range(8))
                or any(row["seed"] != 42 + row["prompt_index"] for row in group)):
            raise ValueError("validation prompt/seed rows differ")
        visible = [row for row in group if not row["safety_filtered"]]
        if visible:
            mean_i = sum(row["clip_i"] for row in visible) / len(visible)
            mean_t = sum(row["clip_t"] for row in visible) / len(visible)
            harmonic = 2 * mean_i * mean_t / (mean_i + mean_t)
        else:
            mean_i = mean_t = harmonic = None
        summaries.append({"step": step, "valid_count": len(visible),
                          "clip_i": mean_i, "clip_t": mean_t,
                          "harmonic": harmonic})
    eligible = [row for row in summaries if row["valid_count"] == 8]
    selected = max(eligible, key=lambda row: row["harmonic"]) if eligible else None
    corrected = {**previous, "results": summaries,
                 "selected_step": selected["step"] if selected else None,
                 "score_aggregation": "harmonic_of_mean_clip_i_and_mean_clip_t",
                 "rescore_source": backup.name}
    backup.write_bytes(selection_path.read_bytes())
    selection_path.write_text(json.dumps(corrected, indent=2) + "\n")
    print(json.dumps({"directory": str(directory), "selected_step": corrected["selected_step"],
                      "selected_harmonic": selected["harmonic"] if selected else None}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    rescore(parser.parse_args().directory)
