"""Approved, source-specific bottom-border mask repair for the D1 subject pilot."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.methods.colorpeel_ice import natural_image_masks as masks
from src.methods.colorpeel_ice.natural_subject_recolor_pilot import require


def contiguous_runs(values: np.ndarray) -> list[list[int]]:
    indices = np.flatnonzero(values)
    if indices.size == 0:
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return [[int(group[0]), int(group[-1])] for group in groups]


def repair_bottom_border(raw_mask_u8: np.ndarray, repair: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw_mask_u8 = np.asarray(raw_mask_u8)
    require(raw_mask_u8.shape == (512, 512) and set(np.unique(raw_mask_u8).tolist()) <= {0, 255}, "Raw mask differs")
    expected_runs = repair["pre_repair_mask_runs"]
    for row_text, expected in expected_runs.items():
        row = int(row_text)
        require(contiguous_runs(raw_mask_u8[row] == 255) == expected, f"Raw mask row {row} differs")
    x0, y0, x1, y1 = repair["rectangle_xyxy_inclusive"]
    require(0 <= x0 <= x1 < 512 and 0 <= y0 <= y1 < 512 and y1 == 511, "Repair rectangle differs")
    repaired = raw_mask_u8 == 255
    before = repaired.copy()
    repaired[y0 : y1 + 1, x0 : x1 + 1] = True
    added = repaired & ~before
    require(int(added.sum()) == repair["expected_added_pixel_count"], "Repair added-pixel count differs")
    require(np.all(repaired[before]), "Repair removed foreground")
    return repaired, added


def derive_repaired_alpha(repaired: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    repaired = np.asarray(repaired, dtype=bool)
    bbox = masks.half_open_bbox(repaired)
    derived = masks.derive_alpha(repaired, bbox, masks.exact_edt(repaired))
    alpha = derived["alpha_u16"]
    require(alpha.dtype == np.uint16 and np.all(alpha[~repaired] == 0), "Repaired alpha differs")
    return alpha, {key: value for key, value in derived.items() if key not in {"alpha", "alpha_u16"}}
