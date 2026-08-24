"""Score complete-bundle multiview seen/held-out predictions without one score."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.methods.colorpeel_ice.multiview_evaluation_protocol import (
    read_evaluation_protocol,
    validate_campaign_manifest,
)


EXPECTED_ITEMS = 1620
PROTOCOL_ID = "clevr_subject_color_3x3_multiview_heldout_bundle_v1"
SHAPES = ("cube", "sphere", "cylinder")
COLORS = ("red", "cyan", "gray")


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
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def index_unique(rows: Iterable[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{source}: missing id")
        if item_id in result:
            raise ValueError(f"{source}: duplicate id {item_id!r}")
        result[item_id] = row
    return result


def _empty_counts() -> dict[str, int]:
    return {
        "expected": 0,
        "prediction_successes": 0,
        "prediction_failures": 0,
        "shape_correct": 0,
        "color_correct": 0,
        "joint_correct": 0,
    }


def _finish_counts(counts: dict[str, int]) -> dict[str, Any]:
    expected = counts["expected"]
    valid = counts["prediction_successes"]
    result: dict[str, Any] = dict(counts)
    result["coverage"] = valid / expected if expected else None
    for name in ("shape", "color", "joint"):
        correct = counts[f"{name}_correct"]
        result[f"{name}_accuracy_all_expected"] = correct / expected if expected else None
        result[f"{name}_accuracy_valid_only"] = correct / valid if valid else None
    return result


def _prediction_ok(row: dict[str, Any] | None) -> bool:
    return bool(
        row is not None
        and row.get("status") == "ok"
        and row.get("predicted_shape") in (*SHAPES, "other")
        and row.get("predicted_color") in (*COLORS, "other")
    )


def score(
    manifest_rows: Iterable[dict[str, Any]],
    prediction_rows: Iterable[dict[str, Any]],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    manifest = index_unique(
        validate_campaign_manifest(manifest_rows, protocol), "manifest"
    )
    predictions = index_unique(prediction_rows, "predictions")
    unknown = sorted(set(predictions) - set(manifest))
    if unknown:
        raise ValueError(f"unknown prediction ids: {unknown[:5]}")

    overall = _empty_counts()
    by_split = defaultdict(_empty_counts)
    by_checkpoint_split = defaultdict(_empty_counts)
    by_cell = defaultdict(_empty_counts)
    scored_rows = []
    failures = []

    for item_id, item in manifest.items():
        prediction = predictions.get(item_id)
        ok = _prediction_ok(prediction)
        predicted_shape = prediction.get("predicted_shape") if ok else None
        predicted_color = prediction.get("predicted_color") if ok else None
        shape_correct = ok and predicted_shape == item["expected_shape"]
        color_correct = ok and predicted_color == item["expected_color"]
        joint_correct = shape_correct and color_correct
        split = item["combination_status"]
        checkpoint_key = (item["fold_id"], item["training_seed"], split)
        cell_key = (
            item["fold_id"],
            item["training_seed"],
            item["subject_label"],
            item["color_label"],
            split,
        )
        for counts in (
            overall,
            by_split[split],
            by_checkpoint_split[checkpoint_key],
            by_cell[cell_key],
        ):
            counts["expected"] += 1
            counts["prediction_successes" if ok else "prediction_failures"] += 1
            counts["shape_correct"] += int(shape_correct)
            counts["color_correct"] += int(color_correct)
            counts["joint_correct"] += int(joint_correct)
        merged = {
            **item,
            "prediction_status": "ok" if ok else "failure",
            "prediction_failure_reason": None if ok else (
                prediction.get("failure_reason") if prediction else "prediction_missing"
            ),
            "predicted_shape": predicted_shape,
            "predicted_color": predicted_color,
            "shape_correct": shape_correct,
            "color_correct": color_correct,
            "joint_correct": joint_correct,
        }
        scored_rows.append(merged)
        if not ok:
            failures.append(merged)

    checkpoint_rows = []
    for (fold_id, training_seed, split), counts in sorted(by_checkpoint_split.items()):
        checkpoint_rows.append(
            {
                "fold_id": fold_id,
                "training_seed": training_seed,
                "combination_status": split,
                **_finish_counts(counts),
            }
        )
    cell_rows = []
    for (fold_id, training_seed, subject, color, split), counts in sorted(by_cell.items()):
        cell_rows.append(
            {
                "fold_id": fold_id,
                "training_seed": training_seed,
                "subject": subject,
                "color": color,
                "combination_status": split,
                **_finish_counts(counts),
            }
        )

    def intervention_rows(group_field: str, fixed_axis: str) -> list[dict[str, Any]]:
        groups = defaultdict(list)
        for row in scored_rows:
            groups[row[group_field]].append(row)
        result = []
        for group_id, rows in sorted(groups.items()):
            if len(rows) != 3:
                raise ValueError(f"intervention group {group_id!r} does not contain 3 rows")
            splits = [row["combination_status"] for row in rows]
            if splits.count("seen") != 2 or splits.count("held_out") != 1:
                raise ValueError(f"intervention group {group_id!r} is not 2 seen + 1 held-out")
            available = all(row["prediction_status"] == "ok" for row in rows)
            held_out = next(row for row in rows if row["combination_status"] == "held_out")
            seen = [row for row in rows if row["combination_status"] == "seen"]
            axis_key = "predicted_shape" if fixed_axis == "subject" else "predicted_color"
            target_key = "expected_shape" if fixed_axis == "subject" else "expected_color"
            seen_available = all(row["prediction_status"] == "ok" for row in seen)
            held_out_available = held_out["prediction_status"] == "ok"
            seen_consensus = seen_available and seen[0][axis_key] == seen[1][axis_key]
            result.append(
                {
                    "group_id": group_id,
                    "fold_id": rows[0]["fold_id"],
                    "training_seed": rows[0]["training_seed"],
                    "generation_seed": rows[0]["generation_seed"],
                    "fixed_axis": fixed_axis,
                    "fixed_value": rows[0][target_key],
                    "all_three_predictions_available": available,
                    "seen_pair_predictions_available": seen_available,
                    "heldout_prediction_available": held_out_available,
                    "seen_pair_consistent": seen_consensus,
                    "heldout_axis_correct": (
                        held_out_available and held_out[axis_key] == held_out[target_key]
                    ),
                    "heldout_matches_seen_consensus": (
                        seen_consensus
                        and held_out_available
                        and held_out[axis_key] == seen[0][axis_key]
                    ),
                    "all_three_target_axis_correct": (
                        available and all(row[axis_key] == row[target_key] for row in rows)
                    ),
                    "all_three_joint_correct": (
                        available and all(row["joint_correct"] for row in rows)
                    ),
                }
            )
        if len(result) != 540:
            raise ValueError(f"expected 540 intervention groups, got {len(result)}")
        return result

    fixed_subject_rows = intervention_rows("fixed_subject_group", "subject")
    fixed_color_rows = intervention_rows("fixed_color_group", "color")

    macro = {}
    for split in ("seen", "held_out"):
        subset = [row for row in checkpoint_rows if row["combination_status"] == split]
        macro[split] = {}
        for metric in (
            "shape_accuracy_all_expected",
            "color_accuracy_all_expected",
            "joint_accuracy_all_expected",
        ):
            values = [row[metric] for row in subset]
            macro[split][metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values),
            }

    metrics = {
        "protocol_id": PROTOCOL_ID,
        "manifest_items": len(manifest),
        "prediction_rows": len(predictions),
        "overall": _finish_counts(overall),
        "by_split": {
            split: _finish_counts(counts) for split, counts in sorted(by_split.items())
        },
        "checkpoint_macro": macro,
        "intervention_groups": {
            "fixed_subject_color_replacement": len(fixed_subject_rows),
            "fixed_color_subject_replacement": len(fixed_color_rows),
        },
    }
    tables = {
        "checkpoint_split": checkpoint_rows,
        "cell": cell_rows,
        "fixed_subject": fixed_subject_rows,
        "fixed_color": fixed_color_rows,
        "failures": failures,
    }
    return metrics, scored_rows, tables


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    protocol = read_evaluation_protocol(args.evaluation_protocol)
    metrics, scored, tables = score(
        read_jsonl(args.manifest), read_jsonl(args.predictions), protocol
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_jsonl(scored, args.output_dir / "scored_rows.jsonl")
    write_csv(tables["checkpoint_split"], args.output_dir / "metrics_by_checkpoint_split.csv")
    write_csv(tables["cell"], args.output_dir / "metrics_by_cell.csv")
    write_csv(tables["fixed_subject"], args.output_dir / "shape_under_color_intervention.csv")
    write_csv(tables["fixed_color"], args.output_dir / "color_under_subject_intervention.csv")
    write_csv(tables["failures"], args.output_dir / "qwen_failures.csv")


if __name__ == "__main__":
    main()
