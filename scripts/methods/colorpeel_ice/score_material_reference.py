"""Score Qwen on accepted renderer references without selecting a pass threshold."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from scripts.methods.colorpeel_ice.predict_qwen_material_reference import validate_items


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(manifest: Iterable[dict[str, Any]], predictions: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    items = {row["id"]: row for row in validate_items(list(manifest))}
    predicted = {row["id"]: row for row in predictions}
    if len(items) != 360 or len(predicted) != 360 or set(items) != set(predicted):
        raise ValueError("reference scoring requires 360 matching unique rows")
    scored = []
    for item_id, item in sorted(items.items()):
        prediction = predicted[item_id]
        ok = prediction.get("status") == "ok"
        row = {**item, "prediction_status": "ok" if ok else "failure",
               "failure_reason": None if ok else prediction.get("failure_reason")}
        for axis in ("shape", "color", "material"):
            value = prediction.get(f"predicted_{axis}") if ok else None
            row[f"predicted_{axis}"] = value
            row[f"{axis}_correct"] = ok and value == item[f"expected_{axis}"]
        row["joint_correct"] = all(row[f"{axis}_correct"] for axis in ("shape", "color", "material"))
        scored.append(row)
    valid = sum(row["prediction_status"] == "ok" for row in scored)
    metrics: dict[str, Any] = {"expected": 360, "prediction_successes": valid,
                               "prediction_failures": 360 - valid, "coverage": valid / 360}
    for axis in ("shape", "color", "material", "joint"):
        correct = sum(row[f"{axis}_correct"] for row in scored)
        metrics[f"{axis}_correct"] = correct
        metrics[f"{axis}_accuracy_all_expected"] = correct / 360
        metrics[f"{axis}_accuracy_valid_only"] = correct / valid if valid else None
    confusion = Counter((row["expected_material"], row["predicted_material"]) for row in scored)
    metrics["material_confusion"] = {f"{expected}->{observed}": count
                                     for (expected, observed), count in sorted(confusion.items(), key=str)}
    return metrics, scored


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, scored = score(read_jsonl(args.manifest), read_jsonl(args.predictions))
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
    with (args.output_dir / "scored_rows.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    confusion = Counter((row["expected_material"], row["predicted_material"]) for row in scored)
    write_csv([{"expected_material": key[0], "predicted_material": key[1], "count": value}
               for key, value in sorted(confusion.items(), key=str)], args.output_dir / "material_confusion.csv")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        groups[(row["expected_shape"], row["expected_color"], row["expected_material"])].append(row)
    write_csv([{"shape": key[0], "color": key[1], "material": key[2], "expected": len(rows),
                "material_correct": sum(row["material_correct"] for row in rows),
                "material_accuracy": sum(row["material_correct"] for row in rows) / len(rows)}
               for key, rows in sorted(groups.items())], args.output_dir / "by_shape_color_material.csv")
    write_csv([row for row in scored if row["prediction_status"] != "ok"], args.output_dir / "failures.csv")


if __name__ == "__main__":
    main()
