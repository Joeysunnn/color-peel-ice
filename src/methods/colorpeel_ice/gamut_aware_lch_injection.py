"""Deterministic fixed-hue, gamut-aware LCh injection for a new C pilot."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_RELPATH = "experiments/natural_image_subject_color_pilot/configs/d1_gamut_aware_lch_injection_protocol_v1.json"
SELECTION_CANONICAL_SHA256 = "33d9b2d5fb404786d63ac23b161c7578a08442ecd24691ca4c221721e88ee0a6"
RENDERER_PROTOCOL_CANONICAL_SHA256 = "476d01c5a383ead7d1ceec5c276bb26d809cf2fabf51cd81e0c49695cc4aa48e"
SHAPES = ("cube", "sphere", "cylinder")
VIEWS = (0, 8, 16)
NEUTRAL_LAB = (50.0, 0.0, 0.0)
GAMUT_MARGIN = 0.995
GAMUT_ITERATIONS = 32


class GamutAwareLchError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GamutAwareLchError(message)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64) / 255.0
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb_uint8(values: np.ndarray) -> np.ndarray:
    require(np.isfinite(values).all() and ((0 <= values) & (values <= 1)).all(), "Linear RGB is out of gamut")
    srgb = np.where(values <= 0.0031308, 12.92 * values, 1.055 * values ** (1 / 2.4) - 0.055)
    return np.floor(srgb * 255.0 + 0.5).astype(np.uint8)


def linear_rgb_to_lab(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 2 and values.shape[1] == 3, "RGB must be Nx3")
    xyz = values @ np.array([[0.4124564, 0.2126729, 0.0193339], [0.3575761, 0.7151522, 0.1191920], [0.1804375, 0.0721750, 0.9503041]])
    xyz /= np.array([0.95047, 1.0, 1.08883])
    epsilon, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > epsilon, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    return np.column_stack((116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])))


def lab_to_linear_rgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    require(lab.ndim == 2 and lab.shape[1] == 3 and np.isfinite(lab).all(), "Lab must be finite Nx3")
    fy = (lab[:, 0] + 16) / 116
    fx = fy + lab[:, 1] / 500
    fz = fy - lab[:, 2] / 200
    epsilon, kappa = 216 / 24389, 24389 / 27
    f = np.column_stack((fx, fy, fz))
    cube = f ** 3
    xyz = np.where(cube > epsilon, cube, (116 * f - 16) / kappa)
    xyz *= np.array([0.95047, 1.0, 1.08883])
    return xyz @ np.array([[3.2404542, -0.9692660, 0.0556434], [-1.5371385, 1.8760108, -0.2040259], [-0.4985314, 0.0415560, 1.0572252]])


def target_lch(target_a: float, target_b: float) -> tuple[float, float]:
    chroma = math.hypot(target_a, target_b)
    require(chroma > 0 and math.isfinite(chroma), "Target chroma must be positive and finite")
    return chroma, math.atan2(target_b, target_a)


def smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def shading_attenuation(lightness: np.ndarray) -> np.ndarray:
    lightness = np.asarray(lightness, dtype=np.float64)
    return smoothstep((lightness - 20.0) / 20.0) * (1.0 - smoothstep((lightness - 70.0) / 20.0))


def max_srgb_chroma(lightness: np.ndarray, hue_radians: float, *, iterations: int = GAMUT_ITERATIONS) -> np.ndarray:
    """Return the largest in-gamut chroma at every fixed Lab L and hue."""
    lightness = np.asarray(lightness, dtype=np.float64).reshape(-1)
    require(np.isfinite(lightness).all() and ((0.0 <= lightness) & (lightness <= 100.0)).all(), "Lightness must be finite in [0,100]")
    require(iterations == GAMUT_ITERATIONS, "Gamut iteration count differs from protocol")
    lo, hi = np.zeros_like(lightness), np.full_like(lightness, 200.0)
    neutral = lab_to_linear_rgb(np.column_stack((lightness, lo, lo)))
    require(np.isfinite(neutral).all() and ((0 <= neutral) & (neutral <= 1)).all(), "Neutral Lab source is not in gamut")
    for _ in range(iterations):
        middle = (lo + hi) / 2.0
        candidate = lab_to_linear_rgb(np.column_stack((lightness, middle * math.cos(hue_radians), middle * math.sin(hue_radians))))
        valid = np.isfinite(candidate).all(axis=1) & (candidate >= 0).all(axis=1) & (candidate <= 1).all(axis=1)
        lo = np.where(valid, middle, lo)
        hi = np.where(valid, hi, middle)
    return lo


def neutral_requests() -> list[dict[str, Any]]:
    return [{"request_id": f"neutral__{shape}__v{view}", "shape": shape, "view_index": view,
             "render_seed": 42, "neutral_lab": list(NEUTRAL_LAB)} for shape in SHAPES for view in VIEWS]


def pilot_requests() -> list[dict[str, Any]]:
    targets = (("D1GT:19/31.png", 9.43264458499643, 17.10049193148857),
               ("D1GT:81/130.png", 63.21805661727065, 54.85769274644318))
    rows = []
    for stable_id, target_a, target_b in targets:
        target_C, target_h = target_lch(target_a, target_b)
        for shape in SHAPES:
            for view in VIEWS:
                safe = stable_id.replace(":", "_").replace("/", "_").replace(".png", "")
                rows.append({"request_id": f"inject__{safe}__{shape}__v{view}", "stable_id": stable_id,
                             "target_a": target_a, "target_b": target_b, "target_C": target_C,
                             "target_h_degrees": math.degrees(target_h), "shape": shape, "view_index": view,
                             "render_seed": 42})
    require(len(rows) == 18 and len({row["request_id"] for row in rows}) == 18, "Pilot request identity differs")
    return rows


def inject_rgb(image: np.ndarray, mask: np.ndarray, target_a: float, target_b: float) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Preserve object-pixel L while deterministically attenuating chroma at fixed hue."""
    image, mask = np.asarray(image), np.asarray(mask)
    require(image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3, "Image must be uint8 RGB")
    require(mask.shape == image.shape[:2] and mask.dtype == np.uint8, "Mask dimensions or dtype differ")
    require(set(np.unique(mask).tolist()) <= {0, 255}, "Mask must be binary 0/255")
    inside = mask == 255
    require(inside.any(), "Mask has no object pixels")
    source_lab = linear_rgb_to_lab(_srgb_to_linear(image[inside]))
    target_C, target_h = target_lch(target_a, target_b)
    attenuation = shading_attenuation(source_lab[:, 0])
    base_C = target_C * attenuation
    max_C = max_srgb_chroma(source_lab[:, 0], target_h)
    final_C = np.minimum(base_C, max_C * GAMUT_MARGIN)
    output_lab = np.column_stack((source_lab[:, 0], final_C * math.cos(target_h), final_C * math.sin(target_h)))
    linear = lab_to_linear_rgb(output_lab)
    require(np.isfinite(linear).all() and ((0 <= linear) & (linear <= 1)).all(), "Gamut solver produced invalid RGB")
    output = image.copy()
    output[inside] = _linear_to_srgb_uint8(linear)
    require(np.array_equal(output[~inside], image[~inside]), "Background bytes changed")
    gamut_limited = final_C + 1e-10 < base_C
    total_reduction = 1.0 - final_C / target_C
    maps = {"r": np.zeros(mask.shape, dtype=np.float32), "C_base": np.zeros(mask.shape, dtype=np.float32),
            "C_max": np.zeros(mask.shape, dtype=np.float32), "C_final": np.zeros(mask.shape, dtype=np.float32),
            "reduction_fraction": np.zeros(mask.shape, dtype=np.float32), "gamut_limited": np.zeros(mask.shape, dtype=np.uint8)}
    for key, values in (("r", attenuation), ("C_base", base_C), ("C_max", max_C), ("C_final", final_C), ("reduction_fraction", total_reduction)):
        maps[key][inside] = values.astype(np.float32)
    maps["gamut_limited"][inside] = gamut_limited.astype(np.uint8)
    evidence = {"status": "injected", "object_pixel_count": int(inside.sum()), "target_C": target_C,
                "target_h_degrees": math.degrees(target_h), "rgb_clipping_pixel_count": 0,
                "gamut_limited_pixel_count": int(gamut_limited.sum()),
                "gamut_limited_pixel_ratio": float(gamut_limited.mean()),
                "total_reduction_median": float(np.median(total_reduction)),
                "total_reduction_p95": float(np.percentile(total_reduction, 95)),
                "gamut_reduction_median": float(np.median(np.maximum(base_C - final_C, 0.0) / target_C)),
                "gamut_reduction_p95": float(np.percentile(np.maximum(base_C - final_C, 0.0) / target_C, 95))}
    return output, evidence, maps


def nearest_rank_p90(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    require(ordered, "P90 needs a non-empty sequence")
    return ordered[math.ceil(0.9 * len(ordered)) - 1]


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
        medians = [float(row["total_reduction_median"]) for row in target_rows]
        p95s = [float(row["total_reduction_p95"]) for row in target_rows]
        gates = {"color": float(np.median(errors)) <= 3.0 and nearest_rank_p90(errors) <= 5.0,
                 "severe_gamut": float(np.median(ratios)) <= 0.5 and float(np.median(medians)) <= 0.25 and max(p95s) <= 0.5}
        targets.append({"stable_id": stable_id, "render_count": 9, "median_e_ch": float(np.median(errors)),
                        "p90_e_ch": nearest_rank_p90(errors), "median_gamut_limited_pixel_ratio": float(np.median(ratios)),
                        "median_total_reduction": float(np.median(medians)), "max_p95_total_reduction": max(p95s),
                        "passes": all(gates.values()), "gates": gates})
    overall_p90 = nearest_rank_p90([float(row["e_ch"]) for row in rows])
    outside_ok = all(int(row["outside_mask_changed_pixel_count"]) == 0 for row in rows)
    clipping_ok = all(int(row["rgb_clipping_pixel_count"]) == 0 for row in rows)
    return {"schema": "d1_gamut_aware_lch_injection_analysis/v1", "request_count": 18, "targets": targets,
            "overall_render_p90_e_ch": overall_p90, "outside_mask_ok": outside_ok, "rgb_clipping_ok": clipping_ok,
            "overall_pass": all(row["passes"] for row in targets) and overall_p90 <= 5.0 and outside_ok and clipping_ok}
