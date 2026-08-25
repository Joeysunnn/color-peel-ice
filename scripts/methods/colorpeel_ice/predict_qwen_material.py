"""Predict locked shape/color/material JSON for the 3240-image campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from src.methods.colorpeel_ice.material_evaluation_protocol import read_protocol, validate_campaign

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
SHAPES = ("cube", "sphere", "cylinder", "other")
COLORS = ("red", "cyan", "gray", "other")
MATERIALS = ("metal", "rubber", "other")
PROMPT = """Inspect the main foreground object in this image.
Return exactly one JSON object with exactly these three keys:
{"shape":"cube|sphere|cylinder|other","color":"red|cyan|gray|other","material":"metal|rubber|other"}
Choose exactly one listed value for each key. Do not add prose or Markdown."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"); handle.flush()


def parse_prediction(text: str) -> dict[str, str]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response does not contain JSON")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict) or set(value) != {"shape", "color", "material"}:
        raise ValueError("response JSON must contain exactly shape, color, material")
    if value["shape"] not in SHAPES or value["color"] not in COLORS or value["material"] not in MATERIALS:
        raise ValueError("response contains an invalid label")
    return value


class QwenPredictor:
    def __init__(self, device: str):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device, self.torch = device, torch
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=True
        ).to(device)
        self.model.eval()

    def __call__(self, image: Image.Image) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                     {"type": "text", "text": PROMPT}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.torch.float16)
        with self.torch.no_grad():
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=128)
        trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]


def status_base(item: dict[str, Any], path: Path) -> dict[str, Any]:
    return {"id": item["id"], "image_path": str(path), "model": MODEL_ID, "local_files_only": True,
            "torch_dtype": "float16", "do_sample": False, "max_new_tokens": 128, "status": None,
            "failure_reason": None, "predicted_shape": None, "predicted_color": None,
            "predicted_material": None, "raw_response": None, "fold_id": item["fold_id"],
            "training_seed": item["training_seed"], "generation_seed": item["generation_seed"],
            "combination_status": item["combination_status"]}


def run_predictions(items: Iterable[dict[str, Any]], image_dir: Path, predictor: Any) -> list[dict[str, Any]]:
    output = []
    for item in items:
        path = image_dir / Path(item["image_path"]); status = status_base(item, path)
        if not path.is_file():
            status.update(status="failure", failure_reason="image_missing"); output.append(status); continue
        try:
            with Image.open(path) as handle:
                image = handle.convert("RGB"); image.load()
            response = predictor(image); prediction = parse_prediction(response)
            status.update(status="ok", raw_response=response,
                          predicted_shape=prediction["shape"], predicted_color=prediction["color"],
                          predicted_material=prediction["material"])
        except Exception as exc:
            status.update(status="failure", failure_reason=f"prediction_error:{type(exc).__name__}:{exc}")
        output.append(status)
    return output


def completed_ids(output: Path, items: list[dict[str, Any]], resume: bool) -> set[str]:
    if not output.exists():
        return set()
    if not resume:
        raise FileExistsError(f"prediction output already exists; use --resume: {output}")
    rows = read_jsonl(output); valid_ids = {row["id"] for row in items}; ids = [row.get("id") for row in rows]
    if len(ids) != len(set(ids)) or any(item_id not in valid_ids for item_id in ids):
        raise ValueError("prediction output contains duplicate or unknown ids")
    return set(ids)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv); protocol = read_protocol(args.evaluation_protocol)
    items = validate_campaign(read_jsonl(args.manifest), protocol)
    done = completed_ids(args.output, items, args.resume); pending = [row for row in items if row["id"] not in done]
    if not pending:
        return 0
    try:
        predictor = QwenPredictor(args.device)
    except Exception as exc:
        for item in pending:
            status = status_base(item, args.image_dir / Path(item["image_path"]))
            status.update(status="failure", failure_reason=f"model_load_error:{type(exc).__name__}:{exc}")
            append_jsonl(status, args.output)
        return 1
    for item in pending:
        append_jsonl(run_predictions([item], args.image_dir, predictor)[0], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
