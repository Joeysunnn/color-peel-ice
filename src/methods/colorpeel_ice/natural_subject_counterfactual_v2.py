"""Deterministic foreground/background counterfactuals for the D1 mailbox."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


class CounterfactualError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CounterfactualError(message)


def _coefficients(shape: tuple[int, int], scale: float, translate_x: int, translate_y: int, mirror: bool) -> tuple[float, float, float, float, float, float]:
    height, width = shape
    require(0.0 < scale <= 1.0, "Scale must be in (0, 1]")
    sign = -1.0 if mirror else 1.0
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
    return (
        sign / scale,
        0.0,
        center_x - sign * (center_x + translate_x) / scale,
        0.0,
        1.0 / scale,
        center_y - (center_y + translate_y) / scale,
    )


def _transform(array: np.ndarray, coefficients: tuple[float, float, float, float, float, float], resample: Image.Resampling) -> np.ndarray:
    image = Image.fromarray(array)
    return np.asarray(image.transform((array.shape[1], array.shape[0]), Image.Transform.AFFINE, coefficients, resample=resample, fillcolor=0))


def _transform_premultiplied(array: np.ndarray, coefficients: tuple[float, float, float, float, float, float]) -> np.ndarray:
    channels = []
    for channel in range(3):
        image = Image.fromarray(array[..., channel].astype(np.float32), mode="F")
        channels.append(np.asarray(image.transform((array.shape[1], array.shape[0]), Image.Transform.AFFINE, coefficients, resample=Image.Resampling.BICUBIC, fillcolor=0.0)))
    return np.stack(channels, axis=2)


def apply_variant(
    recolored_image: np.ndarray,
    source_image: np.ndarray,
    binary_mask: np.ndarray,
    alpha_u16: np.ndarray,
    variant: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Place one recolored subject onto a fixed background with a shared affine map."""
    recolored_image, source_image = np.asarray(recolored_image), np.asarray(source_image)
    binary_mask, alpha_u16 = np.asarray(binary_mask), np.asarray(alpha_u16)
    require(
        recolored_image.dtype == np.uint8
        and recolored_image.shape == source_image.shape
        and recolored_image.ndim == 3
        and recolored_image.shape[2] == 3,
        "Images must be matching uint8 RGB arrays",
    )
    require(binary_mask.shape == recolored_image.shape[:2] and set(np.unique(binary_mask).tolist()) <= {0, 255}, "Binary mask differs")
    require(alpha_u16.shape == binary_mask.shape and np.issubdtype(alpha_u16.dtype, np.integer), "Soft alpha differs")
    require(np.all(alpha_u16[binary_mask == 0] == 0), "Alpha extends outside the binary mask")
    background = np.asarray(variant["background_rgb"], dtype=np.uint8)
    require(background.shape == (3,), "Background must be RGB")
    scale, translate_x, translate_y, mirror = float(variant["scale"]), int(variant["translate_x"]), int(variant["translate_y"]), bool(variant["mirror"])

    alpha = alpha_u16.astype(np.float64) / 65535.0
    foreground = source_image.copy()
    inside = alpha > 0.0
    foreground[inside] = np.clip(
        np.floor((recolored_image[inside].astype(np.float64) - (1.0 - alpha[inside, None]) * source_image[inside]) / alpha[inside, None] + 0.5),
        0,
        255,
    ).astype(np.uint8)
    if mirror:
        foreground, binary_mask, alpha = np.fliplr(foreground), np.fliplr(binary_mask), np.fliplr(alpha)
    coefficients = _coefficients(binary_mask.shape, scale, translate_x, translate_y, False)
    premultiplied = foreground.astype(np.float64) * alpha[..., None]
    premultiplied[binary_mask == 0] = 0.0
    transformed_mask = _transform(binary_mask, coefficients, Image.Resampling.NEAREST).astype(np.uint8)
    transformed_alpha = _transform(alpha.astype(np.float32), coefficients, Image.Resampling.BILINEAR).astype(np.float64)
    transformed_alpha = np.clip(transformed_alpha, 0.0, 1.0)
    transformed_alpha[transformed_mask == 0] = 0.0
    transformed_premultiplied = np.clip(_transform_premultiplied(premultiplied, coefficients), 0.0, 255.0)
    transformed_premultiplied[transformed_mask == 0] = 0.0
    output = np.floor(
        transformed_premultiplied + (1.0 - transformed_alpha[..., None]) * background.astype(np.float64)
        + 0.5
    ).astype(np.uint8)
    require(np.any(transformed_mask == 255), "Variant removed the complete subject")
    require(np.all(output[transformed_mask == 0] == background), "Background changed outside transformed mask")
    ys, xs = np.nonzero(transformed_mask == 255)
    metrics = {
        "background_rgb": background.tolist(),
        "mask_pixel_count": int(transformed_mask.astype(bool).sum()),
        "bbox_xyxy_inclusive": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "outside_mask_background_pixel_count": int((transformed_mask == 0).sum()),
        "mirror": mirror,
        "scale": scale,
        "translate_x": translate_x,
        "translate_y": translate_y,
    }
    return output, transformed_mask, metrics
