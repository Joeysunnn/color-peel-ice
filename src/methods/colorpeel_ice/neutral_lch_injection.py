"""Independent, strict-gamut neutral-render Lab chromaticity injection."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.methods.colorpeel_ice import natural_image_target_selection as selection
from src.methods.colorpeel_ice.natural_image_masks import canonical_sha256

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_RELPATH = "experiments/natural_image_subject_color_pilot/configs/d1_neutral_lch_injection_protocol_v1.json"
VIEWS = (0, 8, 16)
NEUTRAL_LAB = (50.0, 0.0, 0.0)


class NeutralLchInjectionError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise NeutralLchInjectionError(message)


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64) / 255.0
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
    cube = np.column_stack((fx, fy, fz)) ** 3
    xyz = np.where(cube > epsilon, cube, (116 * np.column_stack((fx, fy, fz)) - 16) / kappa)
    xyz *= np.array([0.95047, 1.0, 1.08883])
    return xyz @ np.array([[3.2404542, -0.9692660, 0.0556434], [-1.5371385, 1.8760108, -0.2040259], [-0.4985314, 0.0415560, 1.0572252]])


def pilot_requests() -> list[dict[str, Any]]:
    rows = []
    for record in selection.pipeline_development_records():
        target = record["target"]
        for view in VIEWS:
            rows.append({"stable_id": record["source"]["stable_id"], "target_a": target["a"], "target_b": target["b"],
                         "shape": "sphere", "view_index": view, "render_seed": 42,
                         "request_id": f"inject__{record['source']['stable_id'].replace(':', '_').replace('/', '_')}__sphere__v{view}"})
    require(len(rows) == 18 and len({row["request_id"] for row in rows}) == 18, "Pilot request identity differs")
    return rows


def neutral_requests() -> list[dict[str, Any]]:
    return [{"request_id": f"neutral__sphere__v{view}", "shape": "sphere", "view_index": view,
             "render_seed": 42, "neutral_lab": list(NEUTRAL_LAB)} for view in VIEWS]


def future_full_requests() -> list[dict[str, Any]]:
    """Define, but never execute, the immutable 90-request full expansion."""
    rows = []
    for record in selection.pipeline_development_records():
        target = record["target"]
        for shape in ("cube", "sphere", "cylinder"):
            for view in (0, 4, 8, 12, 16):
                rows.append({"stable_id": record["source"]["stable_id"], "target_a": target["a"], "target_b": target["b"],
                             "shape": shape, "view_index": view, "render_seed": 42,
                             "request_id": f"inject__{record['source']['stable_id'].replace(':', '_').replace('/', '_')}__{shape}__v{view}"})
    require(len(rows) == 90 and len({row["request_id"] for row in rows}) == 90, "Future full request identity differs")
    return rows


def inject_rgb(image: np.ndarray, mask: np.ndarray, target_a: float, target_b: float) -> tuple[np.ndarray | None, dict[str, Any]]:
    image, mask = np.asarray(image), np.asarray(mask)
    require(image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3, "Image must be uint8 RGB")
    require(mask.shape == image.shape[:2] and mask.dtype == np.uint8, "Mask dimensions or dtype differ")
    values = set(np.unique(mask).tolist())
    require(values <= {0, 255}, "Mask must be binary 0/255")
    inside = mask == 255
    require(inside.any(), "Mask has no object pixels")
    original = image[inside]
    lab = linear_rgb_to_lab(_srgb_to_linear(original))
    replaced = lab.copy()
    replaced[:, 1] = target_a
    replaced[:, 2] = target_b
    linear = lab_to_linear_rgb(replaced)
    invalid = (~np.isfinite(linear)).any(axis=1) | (linear < 0).any(axis=1) | (linear > 1).any(axis=1)
    evidence = {"object_pixel_count": int(inside.sum()), "gamut_rejected_pixel_count": int(invalid.sum()),
                "status": "gamut_rejected" if invalid.any() else "injected"}
    if invalid.any():
        return None, evidence
    output = image.copy()
    output[inside] = _linear_to_srgb_uint8(linear)
    require(np.array_equal(output[~inside], image[~inside]), "Background bytes changed")
    return output, evidence


def summarize_measurements(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(rows) == 18 and {row["request_id"] for row in rows} == {row["request_id"] for row in pilot_requests()}, "Measurements do not cover pilot")
    rejected = [row for row in rows if row["status"] != "injected"]
    if rejected:
        return {"status": "gamut_rejected", "request_count": 18, "rejected_count": len(rejected), "overall_pass": False}
    by_target: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_target.setdefault(str(row["stable_id"]), []).append(row)
    summaries = []
    for stable_id, target_rows in by_target.items():
        require(len(target_rows) == 3, "Target view coverage differs")
        errors = sorted(float(row["e_ch"]) for row in target_rows)
        summaries.append({"stable_id": stable_id, "median_e_ch": errors[1], "p90_e_ch": errors[2],
                          "passes": errors[1] <= 3 and errors[2] <= 5})
    overall_p90 = sorted(float(row["e_ch"]) for row in rows)[16]
    return {"status": "measured", "request_count": 18, "rejected_count": 0, "targets": summaries,
            "overall_render_p90_e_ch": overall_p90, "overall_pass": all(row["passes"] for row in summaries) and overall_p90 <= 5}
