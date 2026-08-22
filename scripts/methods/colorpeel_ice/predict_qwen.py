"""Predict CLEVR shape/color labels with a frozen Qwen3-VL model.

This stage is independent of ICE. It processes exactly the 300 non-transfer
manifest items and writes one success or failure JSON object per image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MAX_NEW_TOKENS = 128
EXPECTED_ITEMS = 300
LOCAL_FILES_ONLY = True
SHAPES = ("cube", "sphere", "cylinder", "other")
COLORS = ("red", "cyan", "gray", "other")
CLASSIFICATION_PROMPT = """Inspect the main foreground object in this image.
Return exactly one JSON object with exactly these two keys:
{"shape":"cube|sphere|cylinder|other","color":"red|cyan|gray|other"}
Choose exactly one listed value for each key. Do not add prose or Markdown."""


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


def non_transfer_items(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed_categories = {"grid", "subject_only", "color_only"}
    items = [row for row in rows if row.get("category") in allowed_categories]
    if len(items) != EXPECTED_ITEMS:
        raise ValueError(f"expected {EXPECTED_ITEMS} non-transfer items, found {len(items)}")
    ids = [item.get("id") for item in items]
    if any(not isinstance(item_id, str) or not item_id for item_id in ids):
        raise ValueError("every non-transfer item must have a non-empty string id")
    if len(ids) != len(set(ids)):
        raise ValueError("non-transfer manifest contains duplicate ids")
    return items


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_prediction(text: str) -> dict[str, str]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response does not contain a JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict) or set(payload) != {"shape", "color"}:
        raise ValueError("response JSON must contain exactly shape and color")
    shape = payload["shape"]
    color = payload["color"]
    if shape not in SHAPES:
        raise ValueError(f"invalid shape label: {shape!r}")
    if color not in COLORS:
        raise ValueError(f"invalid color label: {color!r}")
    return {"shape": shape, "color": color}


class QwenPredictor:
    def __init__(self, device: str) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = device
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, local_files_only=LOCAL_FILES_ONLY
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=LOCAL_FILES_ONLY,
        ).to(device)
        self.model.eval()

    def __call__(self, image: Image.Image) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": CLASSIFICATION_PROMPT},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(
                dtype=self.torch.float16
            )
        with self.torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


def _status_base(item: dict[str, Any], image_path: Path) -> dict[str, Any]:
    return {
        "id": item["id"],
        "category": item["category"],
        "image_path": str(image_path),
        "model": MODEL_ID,
        "local_files_only": LOCAL_FILES_ONLY,
        "torch_dtype": "float16",
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "status": None,
        "failure_reason": None,
        "predicted_shape": None,
        "predicted_color": None,
        "raw_response": None,
    }


def run_predictions(
    items: Iterable[dict[str, Any]], image_dir: Path, predictor: Any
) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for item in items:
        image_path = image_dir / Path(str(item.get("image_path", "")))
        status = _status_base(item, image_path)
        if not image_path.is_file():
            status.update(status="failure", failure_reason="image_missing")
            statuses.append(status)
            continue
        try:
            with Image.open(image_path) as handle:
                image = handle.convert("RGB")
                image.load()
            response = predictor(image)
            status["raw_response"] = response
            prediction = parse_prediction(response)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            status.update(
                status="failure",
                failure_reason=f"image_or_response_error:{type(error).__name__}:{error}",
            )
            statuses.append(status)
            continue
        except Exception as error:  # Preserve per-image model failures in JSONL.
            status.update(
                status="failure",
                failure_reason=f"model_inference_error:{type(error).__name__}:{error}",
            )
            statuses.append(status)
            continue
        status.update(
            status="ok",
            predicted_shape=prediction["shape"],
            predicted_color=prediction["color"],
        )
        statuses.append(status)
    return statuses


def model_load_failures(items: Iterable[dict[str, Any]], image_dir: Path, error: Exception) -> list[dict[str, Any]]:
    statuses = []
    reason = f"model_load_error:{type(error).__name__}:{error}"
    for item in items:
        image_path = image_dir / Path(str(item.get("image_path", "")))
        status = _status_base(item, image_path)
        status.update(status="failure", failure_reason=reason)
        statuses.append(status)
    return statuses


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    items = non_transfer_items(read_jsonl(args.manifest))
    try:
        predictor = QwenPredictor(args.device)
    except Exception as error:
        write_jsonl(model_load_failures(items, args.image_dir, error), args.output)
        return 1
    write_jsonl(run_predictions(items, args.image_dir, predictor), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
