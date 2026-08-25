"""Score three-axis seen/held-out predictions without a composite score."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.methods.colorpeel_ice.material_evaluation_protocol import read_protocol, validate_campaign

SHAPES = ("cube", "sphere", "cylinder", "other")
COLORS = ("red", "cyan", "gray", "other")
MATERIALS = ("metal", "rubber", "other")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index_unique(rows: Iterable[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id or item_id in result:
            raise ValueError(f"{source}: missing or duplicate id")
        result[item_id] = row
    return result


def empty_counts() -> dict[str, int]:
    return {"expected": 0, "prediction_successes": 0, "prediction_failures": 0,
            "shape_correct": 0, "color_correct": 0, "material_correct": 0, "joint_correct": 0}


def finish(counts: dict[str, int]) -> dict[str, Any]:
    result = dict(counts); expected, valid = counts["expected"], counts["prediction_successes"]
    result["coverage"] = valid / expected if expected else None
    for axis in ("shape", "color", "material", "joint"):
        correct = counts[f"{axis}_correct"]
        result[f"{axis}_accuracy_all_expected"] = correct / expected if expected else None
        result[f"{axis}_accuracy_valid_only"] = correct / valid if valid else None
    return result


def prediction_ok(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("status") == "ok" and row.get("predicted_shape") in SHAPES
                and row.get("predicted_color") in COLORS and row.get("predicted_material") in MATERIALS)


def score(manifest_rows: Iterable[dict[str, Any]], prediction_rows: Iterable[dict[str, Any]],
          protocol: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    manifest = index_unique(validate_campaign(manifest_rows, protocol), "manifest")
    predictions = index_unique(prediction_rows, "predictions")
    if set(predictions) - set(manifest):
        raise ValueError("predictions contain unknown ids")
    overall, by_split, by_checkpoint, by_cell = empty_counts(), defaultdict(empty_counts), defaultdict(empty_counts), defaultdict(empty_counts)
    scored, failures = [], []
    for item_id, item in manifest.items():
        prediction = predictions.get(item_id); ok = prediction_ok(prediction)
        values = {axis: prediction.get(f"predicted_{axis}") if ok else None
                  for axis in ("shape", "color", "material")}
        correct = {axis: ok and values[axis] == item[f"expected_{axis}"]
                   for axis in ("shape", "color", "material")}
        correct["joint"] = all(correct.values())
        split = item["combination_status"]
        keys = (overall, by_split[split], by_checkpoint[(item["fold_id"], item["training_seed"], split)],
                by_cell[(item["fold_id"], item["training_seed"], item["subject_label"],
                         item["color_label"], item["material_label"], split)])
        for counts in keys:
            counts["expected"] += 1; counts["prediction_successes" if ok else "prediction_failures"] += 1
            for axis in correct:
                counts[f"{axis}_correct"] += int(correct[axis])
        row = {**item, "prediction_status": "ok" if ok else "failure",
               "prediction_failure_reason": None if ok else
               (prediction.get("failure_reason") if prediction else "prediction_missing"),
               **{f"predicted_{axis}": values[axis] for axis in values},
               **{f"{axis}_correct": correct[axis] for axis in correct}}
        scored.append(row)
        if not ok:
            failures.append(row)

    checkpoint_rows = [{"fold_id": key[0], "training_seed": key[1], "combination_status": key[2],
                        **finish(value)} for key, value in sorted(by_checkpoint.items())]
    cell_rows = [{"fold_id": key[0], "training_seed": key[1], "shape": key[2], "color": key[3],
                  "material": key[4], "combination_status": key[5], **finish(value)}
                 for key, value in sorted(by_cell.items())]

    def intervention(group_field: str, changed_axis: str, expected_size: int, expected_groups: int) -> list[dict[str, Any]]:
        groups = defaultdict(list)
        for row in scored:
            groups[row[group_field]].append(row)
        if len(groups) != expected_groups:
            raise ValueError(f"expected {expected_groups} {changed_axis} intervention groups, got {len(groups)}")
        output = []
        fixed_axes = [axis for axis in ("shape", "color", "material") if axis != changed_axis]
        for group_id, rows in sorted(groups.items()):
            if len(rows) != expected_size:
                raise ValueError(f"intervention group {group_id} has wrong size")
            split_pattern = f"{sum(row['held_out'] is False for row in rows)}seen+{sum(row['held_out'] is True for row in rows)}heldout"
            available = [row for row in rows if row["prediction_status"] == "ok"]
            record = {"group_id": group_id, "fold_id": rows[0]["fold_id"],
                      "training_seed": rows[0]["training_seed"], "generation_seed": rows[0]["generation_seed"],
                      "changed_axis": changed_axis, "split_pattern": split_pattern, "expected_rows": expected_size,
                      "valid_rows": len(available), "coverage": len(available) / expected_size,
                      "all_predictions_available": len(available) == expected_size,
                      "all_joint_correct": len(available) == expected_size and all(row["joint_correct"] for row in rows)}
            for axis in fixed_axes:
                record[f"fixed_{axis}_value"] = rows[0][f"expected_{axis}"]
                record[f"fixed_{axis}_correct_all_expected"] = sum(row[f"{axis}_correct"] for row in rows) / expected_size
                record[f"fixed_{axis}_consistent"] = (
                    len(available) == expected_size and len({row[f"predicted_{axis}"] for row in rows}) == 1
                )
            output.append(record)
        return output

    material_rows = intervention("fixed_shape_color_group", "material", 2, 1620)
    if {row["split_pattern"] for row in material_rows} != {"2seen+0heldout", "1seen+1heldout"}:
        raise ValueError("material intervention strata changed")
    shape_material_rows = intervention("fixed_shape_material_group", "color", 3, 1080)
    color_material_rows = intervention("fixed_color_material_group", "shape", 3, 1080)
    if any(row["split_pattern"] != "2seen+1heldout" for row in shape_material_rows + color_material_rows):
        raise ValueError("shape/color intervention strata changed")
    metrics = {"protocol_id": protocol["protocol_id"], "manifest_items": len(manifest),
               "prediction_rows": len(predictions), "overall": finish(overall),
               "by_split": {key: finish(value) for key, value in sorted(by_split.items())},
               "intervention_groups": {"material_replacement": 1620, "color_replacement": 1080,
                                       "shape_replacement": 1080}}
    return metrics, scored, {"checkpoint_split": checkpoint_rows, "cell": cell_rows,
                             "material_intervention": material_rows, "color_intervention": shape_material_rows,
                             "shape_intervention": color_material_rows, "failures": failures}


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv); protocol = read_protocol(args.evaluation_protocol)
    metrics, scored, tables = score(read_jsonl(args.manifest), read_jsonl(args.predictions), protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(scored, args.output_dir / "scored_rows.jsonl")
    for name, rows in tables.items():
        write_csv(rows, args.output_dir / f"{name}.csv")


if __name__ == "__main__":
    main()
