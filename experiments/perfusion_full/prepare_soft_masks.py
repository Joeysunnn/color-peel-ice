"""Prepare CLIPSeg soft foreground masks for a full-Perfusion concept source.

The frozen source concepts and images remain read-only. Each CLIPSeg probability
map is bilinearly resized to the source image, divided by its own maximum, and
stored as an 8-bit grayscale PNG without thresholding. The source's binary mask
is used only for diagnostic comparison; it never changes the generated mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


MODEL_ID = "CIDAS/clipseg-rd64-refined"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_path(raw: str, concepts_path: Path) -> Path:
    path = Path(raw)
    return (path if path.is_absolute() else concepts_path.parent / path).resolve()


def source_rows(concepts_path: Path) -> list[tuple[dict, list[tuple[Path, Path | None]]]]:
    concepts = json.loads(concepts_path.read_text(encoding="utf-8"))
    if not isinstance(concepts, list) or not concepts:
        raise ValueError("concepts JSON must be a nonempty list")
    rows = []
    for index, concept in enumerate(concepts):
        if not isinstance(concept, dict) or "instance_data_dir" not in concept:
            raise ValueError(f"concept row {index} lacks instance_data_dir")
        if "instance_mask_dirs" in concept:
            raise ValueError("joint two-object mask sources are outside this experiment")
        image_dir = _source_path(concept["instance_data_dir"], concepts_path)
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        images = sorted(path for path in image_dir.iterdir() if path.is_file())
        if not images or any(path.suffix.lower() not in IMAGE_SUFFIXES for path in images):
            raise ValueError(f"concept row {index} must contain only supported images: {image_dir}")
        stems = [path.stem for path in images]
        if len(stems) != len(set(stems)):
            raise ValueError(f"concept row {index} contains duplicate image stems")
        old_mask_dir = concept.get("instance_mask_dir")
        masks = None
        if old_mask_dir is not None:
            mask_dir = _source_path(old_mask_dir, concepts_path)
            if not mask_dir.is_dir():
                raise FileNotFoundError(mask_dir)
            mask_files = [path for path in mask_dir.iterdir() if path.is_file()]
            masks = {path.stem: path for path in mask_files}
            if (set(masks) != set(stems) or len(masks) != len(stems)
                    or len(mask_files) != len(stems)):
                raise ValueError(f"concept row {index} binary masks do not match source images")
        rows.append((concept, [(image, masks[image.stem] if masks is not None else None) for image in images]))
    return rows


def normalized_soft_mask(logits: torch.Tensor, size: tuple[int, int]) -> tuple[np.ndarray, dict]:
    """Sigmoid, resize to (width, height), and divide by the prediction maximum."""
    if logits.ndim != 2:
        raise ValueError("CLIPSeg logits must be one [height, width] map")
    if not torch.isfinite(logits).all():
        raise ValueError("CLIPSeg produced nonfinite logits")
    probabilities = logits.float().sigmoid()[None, None]
    resized = F.interpolate(probabilities, size=(size[1], size[0]), mode="bilinear", align_corners=False)[0, 0]
    maximum = float(resized.max())
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("CLIPSeg produced an empty or invalid probability map")
    normalized = (resized / maximum).clamp(0, 1).cpu().numpy().astype(np.float32)
    return normalized, {
        "raw_probability_min": float(resized.min()),
        "raw_probability_max": maximum,
        "normalized_min": float(normalized.min()),
        "normalized_max": float(normalized.max()),
        "normalized_mean": float(normalized.mean()),
        "normalized_sum": float(normalized.sum()),
    }


def binary_mask_audit(soft: np.ndarray, mask_path: Path | None) -> dict | None:
    if mask_path is None:
        return None
    with Image.open(mask_path) as image:
        binary_u8 = np.asarray(image.convert("L"), dtype=np.uint8)
    if binary_u8.shape != soft.shape:
        raise ValueError(f"source binary mask and image size differ: {mask_path}")
    if not set(np.unique(binary_u8)).issubset({0, 255}):
        raise ValueError(f"source audit mask is not binary: {mask_path}")
    binary = binary_u8.astype(np.float32) / 255
    overlap = float(np.minimum(soft, binary).sum())
    union = float(np.maximum(soft, binary).sum())
    return {
        "binary_mask_sha256": sha256(mask_path),
        "binary_foreground_fraction": float(binary.mean()),
        "soft_iou_with_binary": overlap / union if union > 0 else None,
        "soft_mean_inside_binary": float(soft[binary > 0].mean()) if binary.any() else None,
        "soft_mean_outside_binary": float(soft[binary == 0].mean()) if (binary == 0).any() else None,
    }


def prepare(concepts_path: Path, query: str, output_dir: Path,
            predict: Callable[[Image.Image, str], torch.Tensor], model_provenance: dict) -> dict:
    """Write derived concepts, one soft-mask directory per source concept row, and a ledger."""
    concepts_path = concepts_path.resolve()
    output_dir = output_dir.resolve()
    if not query.strip():
        raise ValueError("CLIPSeg query cannot be empty")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    rows = source_rows(concepts_path)
    for concept, sources in rows:
        image_dir = _source_path(concept["instance_data_dir"], concepts_path)
        if output_dir == image_dir or output_dir.is_relative_to(image_dir):
            raise ValueError("output directory must be outside source image directories")
        if concept.get("instance_mask_dir"):
            old_mask_dir = _source_path(concept["instance_mask_dir"], concepts_path)
            if output_dir == old_mask_dir or output_dir.is_relative_to(old_mask_dir):
                raise ValueError("output directory must be outside source binary-mask directories")
    output_dir.mkdir(parents=True, exist_ok=True)
    derived = []
    ledger_path = output_dir / "soft_mask_manifest.jsonl"
    with ledger_path.open("w", encoding="utf-8") as ledger:
        for row_index, (concept, sources) in enumerate(rows):
            new_mask_dir = output_dir / "masks" / f"row_{row_index:04d}"
            new_mask_dir.mkdir(parents=True)
            derived_concept = dict(concept)
            derived_concept["instance_mask_dir"] = str(new_mask_dir)
            derived.append(derived_concept)
            for image_path, binary_path in sources:
                with Image.open(image_path) as source_image:
                    rgb = source_image.convert("RGB")
                    logits = predict(rgb, query)
                    soft, stats = normalized_soft_mask(logits, rgb.size)
                mask_path = new_mask_dir / f"{image_path.stem}.png"
                quantized = np.rint(soft * 255).astype(np.uint8)
                Image.fromarray(quantized, mode="L").save(mask_path)
                audit = binary_mask_audit(soft, binary_path)
                entry = {
                    "row_index": row_index,
                    "source_image": str(image_path),
                    "source_image_sha256": sha256(image_path),
                    "source_binary_mask": str(binary_path) if binary_path is not None else None,
                    "soft_mask": str(mask_path),
                    "soft_mask_sha256": sha256(mask_path),
                    "image_size": list(rgb.size),
                    "query": query,
                    **stats,
                    "binary_audit": audit,
                }
                ledger.write(json.dumps(entry, sort_keys=True) + "\n")
                ledger.flush()
    derived_path = output_dir / "derived_concepts.json"
    derived_path.write_text(json.dumps(derived, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    provenance = {
        "schema": "perfusion_full_clipseg_soft_masks/v1",
        "source_concepts": str(concepts_path),
        "source_concepts_sha256": sha256(concepts_path),
        "derived_concepts": str(derived_path),
        "derived_concepts_sha256": sha256(derived_path),
        "soft_mask_manifest_sha256": sha256(ledger_path),
        "concept_rows": len(rows),
        "images": sum(len(sources) for _, sources in rows),
        "query": query,
        "normalization": "sigmoid logits, bilinear resize, divide by per-mask maximum; 8-bit PNG, no threshold",
        "binary_mask_role": "audit only; never affects CLIPSeg prediction",
        **model_provenance,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concepts", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    processor = CLIPSegProcessor.from_pretrained(MODEL_ID)
    model = CLIPSegForImageSegmentation.from_pretrained(MODEL_ID).to(args.device)
    model.eval().requires_grad_(False)

    def predict(image: Image.Image, query: str) -> torch.Tensor:
        batch = processor(text=[query], images=[image], padding=True, return_tensors="pt")
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.inference_mode():
            logits = model(**batch).logits
        return logits[0].detach().cpu()

    provenance = prepare(args.concepts, args.query, args.output_dir, predict, {
        "model": MODEL_ID,
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "model_config_sha256": hashlib.sha256(model.config.to_json_string().encode()).hexdigest(),
        "device": args.device,
    })
    print(f"Prepared {provenance['images']} CLIPSeg soft masks in {args.output_dir}")


if __name__ == "__main__":
    main()
