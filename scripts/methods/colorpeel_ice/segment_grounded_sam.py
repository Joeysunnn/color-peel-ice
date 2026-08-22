"""Create transfer-object masks with Grounding DINO and Segment Anything.

This stage is self-contained and does not import ICE. It consumes the locked
generation manifest, processes exactly its 600 transfer items, mirrors image
paths below ``--mask-dir``, and writes one explicit status JSON object per item.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM_MODEL = "facebook/sam-vit-base"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
MIN_MASK_RATIO = 0.005
MAX_MASK_RATIO = 0.90
EXPECTED_TRANSFER_ITEMS = 600
LOCAL_FILES_ONLY = True
SEGMENTATION_QUERIES = (
    "bowl.",
    "bowling ball.",
    "plate.",
    "vase.",
    "pants.",
    "teddy bear.",
    "snooker ball.",
    "parrot.",
    "sofa.",
    "rose.",
)


class NoDetectionError(RuntimeError):
    pass


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


def transfer_items(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    items = [row for row in rows if row.get("category") == "transfer"]
    if len(items) != EXPECTED_TRANSFER_ITEMS:
        raise ValueError(
            f"expected {EXPECTED_TRANSFER_ITEMS} transfer items, found {len(items)}"
        )
    ids = [item.get("id") for item in items]
    if any(not isinstance(item_id, str) or not item_id for item_id in ids):
        raise ValueError("every transfer item must have a non-empty string id")
    if len(ids) != len(set(ids)):
        raise ValueError("transfer manifest contains duplicate ids")
    return items


def query_for_item(item: dict[str, Any]) -> str:
    index = item.get("transfer_template_index")
    if not isinstance(index, int) or not 0 <= index < len(SEGMENTATION_QUERIES):
        raise ValueError(f"invalid transfer_template_index for {item.get('id')!r}: {index!r}")
    return SEGMENTATION_QUERIES[index]


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class GroundedSamSegmenter:
    def __init__(self, device: str, dtype: str) -> None:
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            SamModel,
            SamProcessor,
        )

        self.torch = torch
        self.device = device
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        self.torch_dtype = torch_dtype
        self.grounding_processor = AutoProcessor.from_pretrained(
            GROUNDING_DINO_MODEL, local_files_only=LOCAL_FILES_ONLY
        )
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_DINO_MODEL,
            torch_dtype=torch_dtype,
            local_files_only=LOCAL_FILES_ONLY,
        ).to(device)
        self.sam_processor = SamProcessor.from_pretrained(
            SAM_MODEL, local_files_only=LOCAL_FILES_ONLY
        )
        self.sam_model = SamModel.from_pretrained(
            SAM_MODEL,
            torch_dtype=torch_dtype,
            local_files_only=LOCAL_FILES_ONLY,
        ).to(device)
        self.grounding_model.eval()
        self.sam_model.eval()

    def __call__(self, image: Image.Image, query: str) -> np.ndarray:
        torch = self.torch
        inputs = self.grounding_processor(
            images=image, text=query, return_tensors="pt"
        ).to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.torch_dtype)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        detection = self.grounding_processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = detection["boxes"].detach().cpu()
        if boxes.shape[0] == 0:
            raise NoDetectionError("Grounding DINO returned no boxes")

        sam_inputs = self.sam_processor(
            images=image,
            input_boxes=[boxes.tolist()],
            return_tensors="pt",
        ).to(self.device)
        sam_inputs["pixel_values"] = sam_inputs["pixel_values"].to(
            dtype=self.torch_dtype
        )
        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs, multimask_output=True)
        masks = self.sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.detach().cpu(),
            sam_inputs["original_sizes"].detach().cpu(),
            sam_inputs["reshaped_input_sizes"].detach().cpu(),
        )[0]
        scores = sam_outputs.iou_scores.detach().cpu()[0]
        best_indices = scores.argmax(dim=-1)
        best_masks = masks[torch.arange(masks.shape[0]), best_indices]
        return best_masks.to(torch.bool).any(dim=0).numpy()


def _status_base(item: dict[str, Any], image_path: Path, mask_path: Path, query: str) -> dict[str, Any]:
    return {
        "id": item["id"],
        "category": item["category"],
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "query": query,
        "grounding_model": GROUNDING_DINO_MODEL,
        "sam_model": SAM_MODEL,
        "local_files_only": LOCAL_FILES_ONLY,
        "box_threshold": BOX_THRESHOLD,
        "text_threshold": TEXT_THRESHOLD,
        "min_mask_ratio": MIN_MASK_RATIO,
        "max_mask_ratio": MAX_MASK_RATIO,
        "status": None,
        "failure_reason": None,
        "mask_pixels": None,
        "mask_ratio": None,
    }


def run_segmentation(
    items: Iterable[dict[str, Any]],
    image_dir: Path,
    mask_dir: Path,
    segmenter: Any,
) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for item in items:
        relative_path = Path(str(item.get("image_path", "")))
        image_path = image_dir / relative_path
        mask_path = mask_dir / relative_path
        query = query_for_item(item)
        status = _status_base(item, image_path, mask_path, query)
        if not image_path.is_file():
            status.update(status="failure", failure_reason="image_missing")
            statuses.append(status)
            continue
        try:
            with Image.open(image_path) as handle:
                image = handle.convert("RGB")
                image.load()
            mask = np.asarray(segmenter(image, query), dtype=bool)
        except NoDetectionError:
            status.update(status="failure", failure_reason="no_detection")
            statuses.append(status)
            continue
        except (OSError, ValueError) as error:
            status.update(
                status="failure",
                failure_reason=f"image_or_segmentation_error:{type(error).__name__}:{error}",
            )
            statuses.append(status)
            continue
        except Exception as error:  # Keep every model failure auditable per image.
            status.update(
                status="failure",
                failure_reason=f"model_inference_error:{type(error).__name__}:{error}",
            )
            statuses.append(status)
            continue
        if mask.shape != (image.height, image.width):
            status.update(status="failure", failure_reason="mask_size_mismatch")
            statuses.append(status)
            continue
        mask_pixels = int(mask.sum())
        mask_ratio = mask_pixels / mask.size
        status.update(mask_pixels=mask_pixels, mask_ratio=mask_ratio)
        if not MIN_MASK_RATIO <= mask_ratio <= MAX_MASK_RATIO:
            status.update(status="failure", failure_reason="mask_ratio_out_of_range")
            statuses.append(status)
            continue
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
        status["status"] = "ok"
        statuses.append(status)
    return statuses


def model_load_failures(items: Iterable[dict[str, Any]], image_dir: Path, mask_dir: Path, error: Exception) -> list[dict[str, Any]]:
    statuses = []
    reason = f"model_load_error:{type(error).__name__}:{error}"
    for item in items:
        relative_path = Path(str(item.get("image_path", "")))
        status = _status_base(
            item,
            image_dir / relative_path,
            mask_dir / relative_path,
            query_for_item(item),
        )
        status.update(status="failure", failure_reason=reason)
        statuses.append(status)
    return statuses


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    items = transfer_items(read_jsonl(args.manifest))
    try:
        segmenter = GroundedSamSegmenter(args.device, args.dtype)
    except Exception as error:
        write_jsonl(model_load_failures(items, args.image_dir, args.mask_dir, error), args.output)
        return 1
    write_jsonl(run_segmentation(items, args.image_dir, args.mask_dir, segmenter), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
