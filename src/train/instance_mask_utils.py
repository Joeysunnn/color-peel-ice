"""Optional instance-mask helpers for the controlled two-object training stage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def pair_instance_images_and_masks(image_dir: Path, mask_dir: Path | None):
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
    if mask_dir is None:
        return [(path, None) for path in image_paths]
    if not mask_dir.is_dir():
        raise ValueError(f"instance_mask_dir does not exist: {mask_dir}")
    masks_by_stem = {path.stem: path for path in mask_dir.iterdir() if path.is_file()}
    image_stems = {path.stem for path in image_paths}
    if set(masks_by_stem) != image_stems:
        raise ValueError(f"instance_mask_dir must contain exactly one matching file per image: {mask_dir}")
    return [(path, masks_by_stem[path.stem]) for path in image_paths]


def pair_joint_instance_images_and_masks(
    image_dir: Path, left_mask_dir: Path, right_mask_dir: Path
):
    """Pair every full RGB image with its left and right instance masks."""
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
    if not left_mask_dir.is_dir() or not right_mask_dir.is_dir():
        raise ValueError("joint binding requires both left and right mask directories")
    masks = {
        "left": {path.stem: path for path in left_mask_dir.iterdir() if path.is_file()},
        "right": {path.stem: path for path in right_mask_dir.iterdir() if path.is_file()},
    }
    image_stems = {path.stem for path in image_paths}
    for side in ("left", "right"):
        if set(masks[side]) != image_stems:
            raise ValueError(f"{side} mask directory must contain exactly one matching file per image")
    return [(path, masks["left"][path.stem], masks["right"][path.stem]) for path in image_paths]


def load_latent_instance_mask(mask_path: Path, image_size: int, mask_size: int) -> np.ndarray:
    with Image.open(mask_path) as source:
        source = source.convert("L")
        if source.size != (image_size, image_size):
            raise ValueError(f"Instance mask must be {image_size}x{image_size}: {mask_path}")
        values = {value for value, count in enumerate(source.histogram()) if count}
        if not values <= {0, 255}:
            raise ValueError(f"Instance mask must be strictly binary: {mask_path}")
        source = source.resize((mask_size, mask_size), resample=Image.NEAREST)
        result = np.asarray(source, dtype=np.float32) / 255.0
    if result.sum() <= 0:
        raise ValueError(f"Instance mask has no latent foreground pixels: {mask_path}")
    return result
