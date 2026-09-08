"""Production mask derivation helpers for the natural-image pilot batch."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


IMAGE_SIZE = (512, 512)
MASK_VALUES = {0, 255}
ROUNDING_SEMANTICS = "round_half_up_for_nonnegative; ceil/floor are math.ceil/math.floor"
COLOR_THRESHOLD_SEMANTICS = "r=0 returns raw; r>0 keeps pixels where EDT(raw_foreground) > r"
ALPHA_THRESHOLD_SEMANTICS = (
    "pixel-centered signed EDT: foreground DT(raw)-0.5, outside -(DT(~raw)-0.5); "
    "w_out=0; raw outside is clamped exactly to zero"
)


class NaturalImageMaskError(ValueError):
    """Raised when a pilot mask input or derivation violates the contract."""


@dataclass(frozen=True)
class ComponentStats:
    component_id: int
    area: int


def require(condition: bool, message: str) -> None:
    if not condition:
        raise NaturalImageMaskError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def round_half_up(value: float) -> int:
    require(value >= 0, "round_half_up only accepts nonnegative values")
    return int(math.floor(value + 0.5))


def clip_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def parse_stable_id(stable_id: str) -> tuple[str, str]:
    prefix, sep, rest = stable_id.partition(":")
    require(prefix == "D1GT" and sep == ":", f"Unsupported stable ID: {stable_id}")
    sample_id, slash, mask_name = rest.partition("/")
    require(slash == "/" and sample_id and mask_name.endswith(".png"), f"Bad stable ID: {stable_id}")
    require(mask_name != "combined_mask.png", f"combined_mask.png is forbidden: {stable_id}")
    return sample_id, mask_name


def safe_slug(stable_id: str, rank: int) -> str:
    sample_id, mask_name = parse_stable_id(stable_id)
    return f"{rank:03d}_D1GT_{sample_id}_{Path(mask_name).stem}"


def load_rgb(path: Path) -> tuple[Image.Image, dict[str, Any]]:
    with Image.open(path) as source:
        source.load()
        mode = source.mode
        size = source.size
        image_format = source.format
        image = source.copy()
    require(mode == "RGB", f"RGB image must decode as RGB: {path}")
    require(size == IMAGE_SIZE, f"RGB image must be 512x512: {path}")
    require(image_format == "JPEG", f"RGB image must decode as JPEG: {path}")
    return image, {"mode": mode, "size": list(size), "format": image_format}


def load_binary_mask(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(path) as source:
        source.load()
        mode = source.mode
        size = source.size
        source_format = source.format
        array = np.asarray(source)
    require(mode == "L", f"Mask must decode as L: {path}")
    require(size == IMAGE_SIZE, f"Mask must be 512x512: {path}")
    values = {int(value) for value in np.unique(array)}
    require(values == MASK_VALUES, f"Mask must contain exactly {{0,255}} values: {path}")
    require(source_format == "PNG", f"Mask must decode as PNG: {path}")
    return array == 255, {"mode": mode, "size": list(size), "values": sorted(values), "format": source_format}


def half_open_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    require(len(xs) > 0, "Cannot compute a bbox for an empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def compute_tight_crop(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int] = IMAGE_SIZE,
    w_out: int = 0,
) -> dict[str, Any]:
    x0, y0, x1, y1 = bbox
    bbox_w, bbox_h = x1 - x0, y1 - y0
    requested = clip_int(max(math.ceil(0.05 * max(bbox_w, bbox_h)), w_out + 2), 8, 32)
    width, height = image_size
    crop_x0 = max(0, x0 - requested)
    crop_y0 = max(0, y0 - requested)
    crop_x1 = min(width, x1 + requested)
    crop_y1 = min(height, y1 + requested)
    applied = {
        "left": x0 - crop_x0,
        "top": y0 - crop_y0,
        "right": crop_x1 - x1,
        "bottom": crop_y1 - y1,
    }
    return {
        "bbox_half_open": [x0, y0, x1, y1],
        "bbox_size": [bbox_w, bbox_h],
        "requested_margin_px": requested,
        "applied_margin_px": applied,
        "crop_origin_xy": [crop_x0, crop_y0],
        "crop_box_half_open": [crop_x0, crop_y0, crop_x1, crop_y1],
        "boundary_clipped": any(value < requested for value in applied.values()),
    }


def _edt_1d(values: np.ndarray) -> np.ndarray:
    n = len(values)
    d = np.empty(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.int64)
    z = np.empty(n + 1, dtype=np.float64)
    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf
    for q in range(1, n):
        p = v[k]
        numerator = (values[q] + q * q) - (values[p] + p * p)
        denominator = 2 * q - 2 * p
        s = numerator / denominator
        while s <= z[k]:
            k -= 1
            p = v[k]
            numerator = (values[q] + q * q) - (values[p] + p * p)
            denominator = 2 * q - 2 * p
            s = numerator / denominator
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        p = v[k]
        d[q] = (q - p) * (q - p) + values[p]
    return d


def exact_edt(binary: np.ndarray) -> np.ndarray:
    """Exact Euclidean distance transform for True pixels to the nearest False pixel."""
    require(binary.ndim == 2, "EDT input must be a 2D mask")
    if not np.any(binary):
        return np.zeros(binary.shape, dtype=np.float64)
    if np.all(binary):
        return np.full(binary.shape, np.inf, dtype=np.float64)
    inf = float(binary.shape[0] * binary.shape[0] + binary.shape[1] * binary.shape[1] + 1)
    grid = np.where(binary, inf, 0.0).astype(np.float64)
    tmp = np.empty_like(grid)
    for x in range(grid.shape[1]):
        tmp[:, x] = _edt_1d(grid[:, x])
    squared = np.empty_like(grid)
    for y in range(grid.shape[0]):
        squared[y, :] = _edt_1d(tmp[y, :])
    return np.sqrt(squared)


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, list[ComponentStats]]:
    labels = np.zeros(mask.shape, dtype=np.int32)
    stats: list[ComponentStats] = []
    component_id = 0
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if labels[y, x] != 0:
            continue
        component_id += 1
        labels[y, x] = component_id
        queue: deque[tuple[int, int]] = deque([(int(y), int(x))])
        area = 0
        while queue:
            cy, cx = queue.popleft()
            area += 1
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if labels[ny, nx] == 0 and mask[ny, nx]:
                        labels[ny, nx] = component_id
                        queue.append((ny, nx))
        stats.append(ComponentStats(component_id=component_id, area=area))
    return labels, stats


def significant_area_threshold(raw_area: int, height: int, width: int) -> int:
    return max(max(4, math.ceil(1e-4 * height * width)), math.ceil(0.01 * raw_area))


def _component_retention(raw_labels: np.ndarray, component: ComponentStats, candidate: np.ndarray) -> float:
    kept = int(np.logical_and(raw_labels == component.component_id, candidate).sum())
    return kept / component.area


def _retained_subcomponent_count(
    raw_labels: np.ndarray,
    component: ComponentStats,
    candidate: np.ndarray,
) -> int:
    retained = np.logical_and(raw_labels == component.component_id, candidate)
    _, retained_components = connected_components(retained)
    return len(retained_components)


def evaluate_color_candidate(
    raw: np.ndarray,
    candidate: np.ndarray,
    raw_labels: np.ndarray,
    raw_components: list[ComponentStats],
) -> dict[str, Any]:
    height, width = raw.shape
    raw_area = int(raw.sum())
    candidate_area = int(candidate.sum())
    threshold = significant_area_threshold(raw_area, height, width)
    significant = [component for component in raw_components if component.area >= threshold]
    _, candidate_components = connected_components(candidate)
    component_retentions = [
        {
            "component_id": component.component_id,
            "raw_area": component.area,
            "retention": _component_retention(raw_labels, component, candidate),
            "retained_subcomponent_count": _retained_subcomponent_count(raw_labels, component, candidate),
        }
        for component in significant
    ]
    gates = {
        "area_min": max(512, math.ceil(0.001 * height * width)),
        "significant_component_area_min": threshold,
        "area_ok": candidate_area >= max(512, math.ceil(0.001 * height * width)),
        "total_retention_min": 0.85,
        "total_retention": candidate_area / raw_area,
        "total_retention_ok": candidate_area / raw_area >= 0.85,
        "significant_component_retention_min": 0.70,
        "significant_component_retention_ok": all(row["retention"] >= 0.70 for row in component_retentions),
        "significant_component_disappearance_ok": all(row["retention"] > 0 for row in component_retentions),
        "significant_component_single_subcomponent_ok": all(
            row["retained_subcomponent_count"] == 1 for row in component_retentions
        ),
        "component_count_raw": len(raw_components),
        "component_count_candidate": len(candidate_components),
        "component_count_ok": len(candidate_components) <= len(raw_components),
        "significant_component_count_raw": len(significant),
    }
    passed = all(
        gates[name]
        for name in (
            "area_ok",
            "total_retention_ok",
            "significant_component_retention_ok",
            "significant_component_disappearance_ok",
            "significant_component_single_subcomponent_ok",
            "component_count_ok",
        )
    )
    return {
        "area": candidate_area,
        "retention": candidate_area / raw_area,
        "component_count": len(candidate_components),
        "significant_component_retentions": component_retentions,
        "gates": gates,
        "passed": passed,
    }


def derive_color_mask(raw: np.ndarray, bbox: tuple[int, int, int, int], d_in: np.ndarray) -> dict[str, Any]:
    x0, y0, x1, y1 = bbox
    s = min(x1 - x0, y1 - y0)
    q50 = float(np.quantile(d_in[raw], 0.5))
    q50_cap = math.floor(q50 / 3.0)
    r_hi = min(round_half_up(0.010 * s), 8, q50_cap)
    r_lo = min(round_half_up(0.005 * s), 6, q50_cap)
    candidates = sorted(set([r_hi, r_lo, 1, 0]), reverse=True)
    raw_labels, raw_components = connected_components(raw)
    evaluations = []
    selected_mask = raw
    selected_radius = 0
    for radius in candidates:
        candidate = raw.copy() if radius == 0 else np.logical_and(raw, d_in > radius)
        evaluation = evaluate_color_candidate(raw, candidate, raw_labels, raw_components)
        evaluation["radius"] = radius
        evaluations.append(evaluation)
        if evaluation["passed"]:
            selected_radius = radius
            selected_mask = candidate
            break
    status = "PASS" if selected_radius > 0 else "REVIEW"
    flags = [] if selected_radius > 0 else ["REVIEW: erosion_fallback"]
    return {
        "mask": selected_mask,
        "selected_radius_px": selected_radius,
        "status": status,
        "flags": flags,
        "s_px": s,
        "q50_d_in": q50,
        "q50_cap_floor": q50_cap,
        "r_hi": r_hi,
        "r_lo": r_lo,
        "candidates_desc": candidates,
        "candidate_evaluations": evaluations,
        "rounding_semantics": ROUNDING_SEMANTICS,
        "threshold_semantics": COLOR_THRESHOLD_SEMANTICS,
    }


def derive_alpha(raw: np.ndarray, bbox: tuple[int, int, int, int], d_in: np.ndarray) -> dict[str, Any]:
    x0, y0, x1, y1 = bbox
    s = min(x1 - x0, y1 - y0)
    q50 = float(np.quantile(d_in[raw], 0.5))
    w_cap = math.floor(q50 / 3.0)
    requested_w_in = max(1, round_half_up(0.005 * s))
    capped_w_in = min(requested_w_in, 4, w_cap)
    min_width_override = capped_w_in < 1
    w_in = 1 if min_width_override else capped_w_in
    w_out = 0
    d_out = exact_edt(~raw)
    delta = np.where(raw, d_in - 0.5, -(d_out - 0.5))
    u = np.clip((delta + w_out) / (w_in + w_out), 0.0, 1.0)
    alpha = 3.0 * u * u - 2.0 * u * u * u
    alpha = np.where(raw, alpha, 0.0)
    alpha_u16 = np.rint(np.clip(alpha, 0.0, 1.0) * 65535.0).astype(np.uint16)
    alpha_u16[~raw] = 0
    return {
        "alpha": alpha,
        "alpha_u16": alpha_u16,
        "w_in_px": int(w_in),
        "w_out_px": w_out,
        "requested_w_in_px": int(requested_w_in),
        "capped_w_in_px": int(capped_w_in),
        "min_width_override": bool(min_width_override),
        "status": "REVIEW" if min_width_override else "PASS",
        "flags": ["REVIEW: alpha_min_width_override"] if min_width_override else [],
        "q50_d_in": q50,
        "q50_cap_floor": w_cap,
        "threshold_semantics": ALPHA_THRESHOLD_SEMANTICS,
        "dtype": "uint16",
        "range": [0, 65535],
    }


def merge_sample_status(
    color_status: str,
    color_flags: Iterable[str],
    alpha_status: str,
    alpha_flags: Iterable[str],
    alpha_outside_nonzero: int = 0,
) -> tuple[str, list[str]]:
    flags = list(color_flags) + list(alpha_flags)
    if alpha_outside_nonzero:
        return "FAIL", flags + ["FAIL: alpha_outside_nonzero"]
    if "FAIL" in {color_status, alpha_status}:
        return "FAIL", flags
    if "REVIEW" in {color_status, alpha_status}:
        return "REVIEW", flags
    return "PASS", flags


def bool_to_l_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def save_alpha_u16(path: Path, alpha_u16: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(alpha_u16, mode="I;16").save(path)


def crop_image(image: Image.Image, crop_box: Iterable[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(tuple(crop_box)).save(path)


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * (rgb ** (1 / 2.4)) - 0.055)


def composite_linear_srgb(rgb_image: Image.Image, alpha: np.ndarray, background: tuple[int, int, int]) -> Image.Image:
    fg = np.asarray(rgb_image, dtype=np.float64) / 255.0
    bg = np.array(background, dtype=np.float64).reshape(1, 1, 3) / 255.0
    fg_linear = _srgb_to_linear(fg)
    bg_linear = _srgb_to_linear(bg)
    a = alpha.reshape(alpha.shape[0], alpha.shape[1], 1)
    out = fg_linear * a + bg_linear * (1.0 - a)
    return Image.fromarray(np.rint(_linear_to_srgb(out) * 255).astype(np.uint8), mode="RGB")


def checkerboard(size: tuple[int, int], tile: int = 24) -> Image.Image:
    width, height = size
    yy, xx = np.indices((height, width))
    cells = ((xx // tile) + (yy // tile)) % 2
    arr = np.where(cells[..., None] == 0, 235, 160).astype(np.uint8)
    return Image.fromarray(np.repeat(arr, 3, axis=2), mode="RGB")


def composite_checker(rgb_image: Image.Image, alpha: np.ndarray) -> Image.Image:
    bg = np.asarray(checkerboard(rgb_image.size), dtype=np.float64) / 255.0
    fg = np.asarray(rgb_image, dtype=np.float64) / 255.0
    out = _srgb_to_linear(fg) * alpha[..., None] + _srgb_to_linear(bg) * (1.0 - alpha[..., None])
    return Image.fromarray(np.rint(_linear_to_srgb(out) * 255).astype(np.uint8), mode="RGB")


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    center = padded[1:-1, 1:-1]
    eroded = center.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            eroded &= padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
    return np.logical_and(mask, ~eroded)


def overlay_mask(rgb_image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> Image.Image:
    arr = np.asarray(rgb_image.convert("RGB"), dtype=np.float64)
    blend = np.array(color, dtype=np.float64).reshape(1, 1, 3)
    arr[mask] = arr[mask] * (1.0 - alpha) + blend * alpha
    return Image.fromarray(np.rint(arr).astype(np.uint8), mode="RGB")


def alpha_heatmap(alpha: np.ndarray, contour_mask: np.ndarray) -> Image.Image:
    a = np.clip(alpha, 0.0, 1.0)
    red = np.clip(2.0 * a - 0.35, 0.0, 1.0)
    green = np.clip(1.4 * a, 0.0, 1.0)
    blue = np.clip(1.0 - 1.3 * a, 0.0, 1.0)
    arr = np.stack([red, green, blue], axis=2)
    arr[contour_mask] = np.array([1.0, 1.0, 1.0])
    return Image.fromarray(np.rint(arr * 255).astype(np.uint8), mode="RGB")


def make_sample_qc(
    rgb_image: Image.Image,
    raw: np.ndarray,
    color_mask: np.ndarray,
    alpha: np.ndarray,
    crop_box: list[int],
    label: str,
) -> Image.Image:
    deleted = np.logical_and(raw, ~color_mask)
    panels = [
        ("raw overlay", overlay_mask(rgb_image, raw, (0, 190, 255), 0.38)),
        ("EDT deleted band", overlay_mask(rgb_image, deleted, (255, 40, 40), 0.72)),
        ("alpha + contours", alpha_heatmap(alpha, mask_boundary(raw))),
        ("black composite", composite_linear_srgb(rgb_image, alpha, (0, 0, 0))),
        ("white composite", composite_linear_srgb(rgb_image, alpha, (255, 255, 255))),
        ("checker composite", composite_checker(rgb_image, alpha)),
        ("risk crop", overlay_mask(rgb_image, deleted, (255, 40, 40), 0.72).crop(tuple(crop_box))),
    ]
    thumb_w, thumb_h = 160, 160
    label_h = 34
    sheet = Image.new("RGB", (len(panels) * thumb_w, thumb_h + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    for col, (name, panel) in enumerate(panels):
        tile = panel.convert("RGB")
        tile.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = col * thumb_w + (thumb_w - tile.width) // 2
        y = label_h + (thumb_h - tile.height) // 2
        sheet.paste(tile, (x, y))
        draw.text((col * thumb_w + 4, 4), f"{label} | {name}", fill="black")
        draw.text((col * thumb_w + 4, 18), "production EDT", fill="black")
    return sheet


def copy_verified(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = file_sha256(source)
    destination_hash = file_sha256(destination)
    require(source_hash == destination_hash, f"Copied bytes changed: {source} -> {destination}")
    stat = source.stat()
    return {
        "source_sha256": source_hash,
        "output_sha256": destination_hash,
        "source_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "bytes_identical": True,
    }
