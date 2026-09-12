"""Fixed-hue Lab injection that reduces chroma only for sRGB gamut legality."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.methods.colorpeel_ice import gamut_aware_lch_injection as color


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_RELPATH = "experiments/natural_image_subject_color_pilot/configs/d1_gamut_only_lch_injection_protocol_v1.json"
SELECTION_CANONICAL_SHA256 = color.SELECTION_CANONICAL_SHA256
RENDERER_PROTOCOL_CANONICAL_SHA256 = color.RENDERER_PROTOCOL_CANONICAL_SHA256
SOURCE_NEUTRAL_MANIFEST_SHA256 = "04d843102f0b5a8ade7c2f74f8af32a6dc406448d49522a0b12a23454a06fa4b"
SOURCE_NEUTRAL_CONTRACT_SHA256 = "cc703f95dbcd9d10345283fb16f21903132129b757dd6bc0dd079d1240649cc8"
SHAPES, VIEWS = color.SHAPES, color.VIEWS
GAMUT_MARGIN = color.GAMUT_MARGIN


class GamutOnlyLchError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GamutOnlyLchError(message)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def target_lch(target_a: float, target_b: float) -> tuple[float, float]:
    chroma = math.hypot(target_a, target_b)
    require(chroma > 0 and math.isfinite(chroma), "Target chroma must be positive and finite")
    return chroma, math.atan2(target_b, target_a)


def neutral_requests() -> list[dict[str, Any]]:
    return color.neutral_requests()


def pilot_requests() -> list[dict[str, Any]]:
    targets = (("D1GT:19/31.png", 9.43264458499643, 17.10049193148857),
               ("D1GT:81/130.png", 63.21805661727065, 54.85769274644318))
    rows = []
    for stable_id, target_a, target_b in targets:
        target_C, target_h = target_lch(target_a, target_b)
        for shape in SHAPES:
            for view in VIEWS:
                safe = stable_id.replace(":", "_").replace("/", "_").replace(".png", "")
                rows.append({"request_id": f"gamut_only__{safe}__{shape}__v{view}", "stable_id": stable_id,
                             "target_a": target_a, "target_b": target_b, "target_C": target_C,
                             "target_h_degrees": math.degrees(target_h), "shape": shape, "view_index": view,
                             "render_seed": 42})
    require(len(rows) == 18 and len({row["request_id"] for row in rows}) == 18, "Pilot request identity differs")
    return rows


def inject_rgb(image: np.ndarray, mask: np.ndarray, target_a: float, target_b: float) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    image, mask = np.asarray(image), np.asarray(mask)
    require(image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3, "Image must be uint8 RGB")
    require(mask.shape == image.shape[:2] and mask.dtype == np.uint8, "Mask dimensions or dtype differ")
    require(set(np.unique(mask).tolist()) <= {0, 255}, "Mask must be binary 0/255")
    inside = mask == 255
    require(inside.any(), "Mask has no object pixels")
    source_lab = color.linear_rgb_to_lab(color._srgb_to_linear(image[inside]))
    target_C, target_h = target_lch(target_a, target_b)
    desired_C = np.full(source_lab.shape[0], target_C)
    max_C = color.max_srgb_chroma(source_lab[:, 0], target_h)
    final_C = np.minimum(desired_C, max_C * GAMUT_MARGIN)
    output_lab = np.column_stack((source_lab[:, 0], final_C * math.cos(target_h), final_C * math.sin(target_h)))
    linear = color.lab_to_linear_rgb(output_lab)
    require(np.isfinite(linear).all() and ((0 <= linear) & (linear <= 1)).all(), "Gamut solver produced invalid RGB")
    output = image.copy()
    output[inside] = color._linear_to_srgb_uint8(linear)
    require(np.array_equal(output[~inside], image[~inside]), "Background bytes changed")
    gamut_limited = final_C + 1e-10 < desired_C
    reduction = 1.0 - final_C / target_C
    maps = {"C_desired": np.zeros(mask.shape, dtype=np.float32), "C_max": np.zeros(mask.shape, dtype=np.float32),
            "C_final": np.zeros(mask.shape, dtype=np.float32), "gamut_reduction_fraction": np.zeros(mask.shape, dtype=np.float32),
            "gamut_limited": np.zeros(mask.shape, dtype=np.uint8)}
    for key, values in (("C_desired", desired_C), ("C_max", max_C), ("C_final", final_C), ("gamut_reduction_fraction", reduction)):
        maps[key][inside] = values.astype(np.float32)
    maps["gamut_limited"][inside] = gamut_limited.astype(np.uint8)
    evidence = {"status": "injected", "object_pixel_count": int(inside.sum()), "target_C": target_C,
                "target_h_degrees": math.degrees(target_h), "rgb_clipping_pixel_count": 0,
                "gamut_limited_pixel_count": int(gamut_limited.sum()), "gamut_limited_pixel_ratio": float(gamut_limited.mean()),
                "gamut_reduction_median": float(np.median(reduction)), "gamut_reduction_p95": float(np.percentile(reduction, 95))}
    return output, evidence, maps


def nearest_rank_p90(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    require(ordered, "P90 needs a non-empty sequence")
    return ordered[math.ceil(0.9 * len(ordered) - 1)]


def summarize_measurements(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {row["request_id"] for row in pilot_requests()}
    require(len(rows) == 18 and {row["request_id"] for row in rows} == expected, "Measurements do not cover pilot")
    by_target: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_target.setdefault(str(row["stable_id"]), []).append(row)
    targets = []
    for stable_id, target_rows in sorted(by_target.items()):
        require(len(target_rows) == 9, "Target shape/view coverage differs")
        errors = [float(row["e_ch"]) for row in target_rows]
        ratios = [float(row["gamut_limited_pixel_ratio"]) for row in target_rows]
        medians = [float(row["gamut_reduction_median"]) for row in target_rows]
        p95s = [float(row["gamut_reduction_p95"]) for row in target_rows]
        gates = {"color": float(np.median(errors)) <= 3.0 and nearest_rank_p90(errors) <= 5.0,
                 "severe_gamut": float(np.median(ratios)) <= 0.5 and float(np.median(medians)) <= 0.25 and max(p95s) <= 0.5}
        targets.append({"stable_id": stable_id, "render_count": 9, "median_e_ch": float(np.median(errors)), "p90_e_ch": nearest_rank_p90(errors),
                        "median_gamut_limited_pixel_ratio": float(np.median(ratios)), "median_gamut_reduction": float(np.median(medians)),
                        "max_p95_gamut_reduction": max(p95s), "passes": all(gates.values()), "gates": gates})
    overall_p90 = nearest_rank_p90([float(row["e_ch"]) for row in rows])
    outside_ok = all(int(row["outside_mask_changed_pixel_count"]) == 0 for row in rows)
    clipping_ok = all(int(row["rgb_clipping_pixel_count"]) == 0 for row in rows)
    return {"schema": "d1_gamut_only_lch_injection_analysis/v1", "request_count": 18, "targets": targets,
            "overall_render_p90_e_ch": overall_p90, "outside_mask_ok": outside_ok, "rgb_clipping_ok": clipping_ok,
            "overall_pass": all(row["passes"] for row in targets) and overall_p90 <= 5.0 and outside_ok and clipping_ok}
