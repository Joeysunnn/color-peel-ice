"""Run the locked material Qwen prompt on 360 accepted renderer references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from scripts.methods.colorpeel_ice.predict_qwen_material import (
    MODEL_ID, QwenPredictor, append_jsonl, completed_ids, parse_prediction, read_jsonl,
)
from src.methods.colorpeel_ice.prepare_material_calibration import sha256_file


def validate_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids = [row.get("id") for row in rows]
    factors = {(row.get("expected_shape"), row.get("expected_color"), row.get("expected_material"),
                row.get("view_index")) for row in rows}
    expected = {(shape, color, material, view) for shape in ("cube", "sphere", "cylinder")
                for color in ("red", "cyan", "gray") for material in ("metal", "rubber")
                for view in range(20)}
    if len(rows) != 360 or len(set(ids)) != 360 or factors != expected:
        raise ValueError("reference manifest must contain the locked 360-image grid")
    return rows


def status_base(item: dict[str, Any]) -> dict[str, Any]:
    return {"id": item["id"], "image_path": item["image_path"], "model": MODEL_ID,
            "local_files_only": True, "torch_dtype": "float16", "do_sample": False,
            "max_new_tokens": 128, "status": None, "failure_reason": None,
            "predicted_shape": None, "predicted_color": None, "predicted_material": None,
            "raw_response": None, "expected_shape": item["expected_shape"],
            "expected_color": item["expected_color"], "expected_material": item["expected_material"],
            "view_index": item["view_index"], "source": item["source"]}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    items = validate_items(read_jsonl(args.manifest))
    done = completed_ids(args.output, items, args.resume)
    pending = [row for row in items if row["id"] not in done]
    if not pending:
        return 0
    try:
        predictor = QwenPredictor(args.device)
    except Exception as exc:
        for item in pending:
            row = status_base(item)
            row.update(status="failure", failure_reason=f"model_load_error:{type(exc).__name__}:{exc}")
            append_jsonl(row, args.output)
        return 1
    for item in pending:
        row = status_base(item)
        path = Path(item["image_path"])
        if not path.is_file():
            row.update(status="failure", failure_reason="image_missing")
        elif sha256_file(path) != item["image_sha256"]:
            row.update(status="failure", failure_reason="image_hash_mismatch")
        else:
            try:
                with Image.open(path) as handle:
                    handle.load()
                    if handle.size != (512, 512) or handle.mode != "RGB":
                        raise ValueError("reference image is not 512x512 RGB")
                    image = handle.copy()
                raw = predictor(image)
                prediction = parse_prediction(raw)
                row.update(status="ok", raw_response=raw,
                           predicted_shape=prediction["shape"], predicted_color=prediction["color"],
                           predicted_material=prediction["material"])
            except Exception as exc:
                row.update(status="failure", failure_reason=f"prediction_error:{type(exc).__name__}:{exc}")
        append_jsonl(row, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
