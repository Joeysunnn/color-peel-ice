"""Numerics for the single-subject auxiliary-hue recoloring pilot."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.methods.colorpeel_ice import gamut_aware_lch_injection as gamut


GAMUT_MARGIN = 0.995
GAMUT_ITERATIONS = 32


class SubjectRecolorError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SubjectRecolorError(message)


def hue_distance_degrees(first: np.ndarray, second: float) -> np.ndarray:
    return np.abs((np.asarray(first) - second + 180.0) % 360.0 - 180.0)


def recolor(image: np.ndarray, binary_mask: np.ndarray, alpha_u16: np.ndarray, hue_degrees: float) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    image, binary_mask, alpha_u16 = np.asarray(image), np.asarray(binary_mask), np.asarray(alpha_u16)
    require(image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3, "Image must be uint8 RGB")
    require(binary_mask.shape == image.shape[:2] and set(np.unique(binary_mask).tolist()) <= {0, 255}, "Binary mask differs")
    require(alpha_u16.shape == image.shape[:2] and np.issubdtype(alpha_u16.dtype, np.integer), "Soft alpha differs")
    inside = binary_mask == 255
    require(inside.any() and ((alpha_u16[~inside] == 0).all()), "Mask or alpha contract differs")
    source_lab = gamut.linear_rgb_to_lab(gamut._srgb_to_linear(image[inside]))
    source_c = np.hypot(source_lab[:, 1], source_lab[:, 2])
    radians = math.radians(hue_degrees)
    max_c = gamut.max_srgb_chroma(source_lab[:, 0], radians, iterations=GAMUT_ITERATIONS)
    final_c = np.minimum(source_c, max_c * GAMUT_MARGIN)
    recolored_lab = np.column_stack((source_lab[:, 0], final_c * math.cos(radians), final_c * math.sin(radians)))
    linear = gamut.lab_to_linear_rgb(recolored_lab)
    require(np.isfinite(linear).all() and ((0.0 <= linear) & (linear <= 1.0)).all(), "Gamut solver produced invalid RGB")
    hard = image.copy()
    hard[inside] = gamut._linear_to_srgb_uint8(linear)
    alpha = alpha_u16.astype(np.float64) / 65535.0
    composite = image.copy()
    blended = alpha[..., None] * hard.astype(np.float64) + (1.0 - alpha[..., None]) * image.astype(np.float64)
    composite[inside] = np.floor(blended[inside] + 0.5).astype(np.uint8)
    require(np.array_equal(composite[~inside], image[~inside]), "Outside-mask RGB changed")
    measured = gamut.linear_rgb_to_lab(gamut._srgb_to_linear(hard[inside]))
    measured_hue = np.degrees(np.arctan2(measured[:, 2], measured[:, 1])) % 360.0
    gamut_limited = final_c + 1e-10 < source_c
    reduction = np.zeros_like(source_c)
    nonzero = source_c > 1e-12
    reduction[nonzero] = 1.0 - final_c[nonzero] / source_c[nonzero]
    maps = {"source_C": np.zeros(binary_mask.shape, dtype=np.float32), "C_max": np.zeros(binary_mask.shape, dtype=np.float32), "C_final": np.zeros(binary_mask.shape, dtype=np.float32), "chroma_reduction_fraction": np.zeros(binary_mask.shape, dtype=np.float32), "gamut_limited": np.zeros(binary_mask.shape, dtype=np.uint8)}
    for name, values in (("source_C", source_c), ("C_max", max_c), ("C_final", final_c), ("chroma_reduction_fraction", reduction)):
        maps[name][inside] = values.astype(np.float32)
    maps["gamut_limited"][inside] = gamut_limited.astype(np.uint8)
    metrics = {"object_pixel_count": int(inside.sum()), "outside_mask_changed_pixel_count": int(np.any(composite != image, axis=2)[~inside].sum()), "rgb_clipping_pixel_count": 0,
               "delta_L_median": float(np.median(np.abs(measured[:, 0] - source_lab[:, 0]))), "delta_L_p95": float(np.percentile(np.abs(measured[:, 0] - source_lab[:, 0]), 95)),
               "hue_error_median_degrees": float(np.median(hue_distance_degrees(measured_hue, hue_degrees))), "hue_error_p95_degrees": float(np.percentile(hue_distance_degrees(measured_hue, hue_degrees), 95)),
               "gamut_reduced_pixel_count": int(gamut_limited.sum()), "gamut_reduced_pixel_ratio": float(gamut_limited.mean()), "chroma_reduction_median": float(np.median(reduction)), "chroma_reduction_p95": float(np.percentile(reduction, 95)),
               "soft_edge_pixel_count": int((inside & (alpha_u16 < 65535)).sum()), "alpha_outside_nonzero_pixel_count": int((alpha_u16[~inside] != 0).sum()), "halo_check": "outside_binary_mask_unchanged"}
    return composite, metrics, maps
