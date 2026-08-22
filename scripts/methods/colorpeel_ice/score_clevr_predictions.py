"""Score external VLM JSONL predictions for the CLEVR evaluation manifest.

The scorer deliberately does not import or run a vision-language model. Each
prediction line must contain ``id``, ``predicted_shape`` and ``predicted_color``.
Multiple prediction files are accepted and merged by manifest item id.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SHAPES = ("cube", "sphere", "cylinder")
COLORS = ("red", "cyan", "gray")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _index_unique(rows: Iterable[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{source}: every row must have a non-empty string id")
        if item_id in indexed:
            raise ValueError(f"{source}: duplicate id {item_id!r}")
        indexed[item_id] = row
    return indexed


def load_predictions(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        for item_id, row in _index_unique(read_jsonl(path), str(path)).items():
            if item_id in merged:
                raise ValueError(f"duplicate prediction id across files: {item_id!r}")
            merged[item_id] = row
    return merged


def _normalize(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value or None


def _bucket(value: str | None, known: tuple[str, ...]) -> str:
    if value is None:
        return "missing"
    return value if value in known else "other"


def _empty_table(rows: tuple[str, ...], columns: tuple[str, ...]) -> dict[str, Any]:
    all_columns = (*columns, "other", "missing")
    return {
        "rows": list(rows),
        "columns": list(all_columns),
        "counts": {row: {column: 0 for column in all_columns} for row in rows},
    }


def score(
    manifest_rows: Iterable[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _index_unique(manifest_rows, "manifest")
    unknown_prediction_ids = sorted(set(predictions) - set(manifest))
    if unknown_prediction_ids:
        raise ValueError(
            "predictions contain ids not present in manifest: "
            + ", ".join(unknown_prediction_ids[:10])
        )

    grid_total = 0
    shape_correct = 0
    color_correct = 0
    joint_correct = 0
    predicted_count = 0
    category_counts: Counter[str] = Counter()
    subject_color = _empty_table(SHAPES, COLORS)
    color_shape = _empty_table(COLORS, SHAPES)
    merged_rows: list[dict[str, Any]] = []

    for item_id, item in manifest.items():
        prediction = predictions.get(item_id)
        predicted_shape = _normalize(
            prediction.get("predicted_shape") if prediction is not None else None
        )
        predicted_color = _normalize(
            prediction.get("predicted_color") if prediction is not None else None
        )
        if prediction is not None:
            predicted_count += 1
        category = item.get("category")
        category_counts[str(category)] += 1

        shape_match = predicted_shape == item.get("subject_label")
        color_match = predicted_color == item.get("color_label")
        if category == "grid":
            grid_total += 1
            shape_correct += int(shape_match)
            color_correct += int(color_match)
            joint_correct += int(shape_match and color_match)
        elif category == "subject_only":
            row = item.get("subject_label")
            if row in SHAPES:
                column = _bucket(predicted_color, COLORS)
                subject_color["counts"][row][column] += 1
        elif category == "color_only":
            row = item.get("color_label")
            if row in COLORS:
                column = _bucket(predicted_shape, SHAPES)
                color_shape["counts"][row][column] += 1

        merged_rows.append(
            {
                **item,
                "predicted_shape": predicted_shape,
                "predicted_color": predicted_color,
                "shape_match": shape_match if category == "grid" else None,
                "color_match": color_match if category == "grid" else None,
                "joint_match": (shape_match and color_match) if category == "grid" else None,
            }
        )

    def accuracy(correct: int) -> float | None:
        return correct / grid_total if grid_total else None

    metrics = {
        "manifest_items": len(manifest),
        "predicted_items": predicted_count,
        "missing_prediction_items": len(manifest) - predicted_count,
        "category_counts": dict(sorted(category_counts.items())),
        "grid": {
            "total": grid_total,
            "shape_correct": shape_correct,
            "color_correct": color_correct,
            "joint_correct": joint_correct,
            "shape_accuracy": accuracy(shape_correct),
            "color_accuracy": accuracy(color_correct),
            "joint_accuracy": accuracy(joint_correct),
        },
        "axis_contingency": {
            "subject_token_by_predicted_color": subject_color,
            "color_token_by_predicted_shape": color_shape,
        },
    }
    return metrics, merged_rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--merged-output",
        type=Path,
        help="Defaults to OUTPUT with a .merged.jsonl suffix.",
    )
    args = parser.parse_args(argv)
    if args.merged_output is None:
        args.merged_output = args.output.with_suffix(".merged.jsonl")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_rows = read_jsonl(args.manifest)
    predictions = load_predictions(args.predictions)
    metrics, merged_rows = score(manifest_rows, predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(merged_rows, args.merged_output)


if __name__ == "__main__":
    main()
