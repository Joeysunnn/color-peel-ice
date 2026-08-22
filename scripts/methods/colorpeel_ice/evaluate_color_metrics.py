"""Evaluate ColorPeel color fidelity from generated images and external masks.

This program does not load a segmentation model. Masks must already exist below
``--mask-dir`` at the same relative paths used by ``image_path`` in the generation
manifest. Missing and empty masks are emitted as failures rather than skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


FRACTIONS = (0.10, 0.50, 1.00)
METRICS = ("delta_e", "delta_e_ch", "srgb_angular_deg", "hue_angular_deg")
DEFAULT_CATEGORIES = ("transfer",)


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


def load_target_colors(path: Path) -> dict[str, tuple[int, int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    colors = payload.get("colors")
    if not isinstance(colors, list):
        raise ValueError(f"{path}: expected a top-level colors list")
    result: dict[str, tuple[int, int, int]] = {}
    for item in colors:
        name = item.get("name") if isinstance(item, dict) else None
        rgb = item.get("rgb") if isinstance(item, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}: each color needs a non-empty name")
        if (
            not isinstance(rgb, list)
            or len(rgb) != 3
            or any(not isinstance(channel, int) or not 0 <= channel <= 255 for channel in rgb)
        ):
            raise ValueError(f"{path}: invalid 8-bit RGB for {name!r}: {rgb!r}")
        if name in result:
            raise ValueError(f"{path}: duplicate color name {name!r}")
        result[name] = tuple(rgb)
    return result


def derive_source_colors(
    dataset_root: Path, experiment_manifest_path: Path
) -> tuple[dict[str, tuple[float, float, float]], dict[str, Any]]:
    """Derive one robust RGB reference per color from source-image GT masks."""

    manifest = json.loads(experiment_manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{experiment_manifest_path}: expected non-empty samples list")
    pixels_by_color: dict[str, list[np.ndarray]] = {}
    sample_ids_by_color: dict[str, list[str]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("source sample must be a JSON object")
        sample_id = sample.get("id")
        shape = sample.get("shape")
        color = sample.get("color")
        if not all(isinstance(value, str) and value for value in (sample_id, shape, color)):
            raise ValueError(f"invalid source sample metadata: {sample!r}")
        image_path = dataset_root / sample_id / "img.jpg"
        mask_path = dataset_root / sample_id / f"mask_{shape}_0.png"
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f"source image/mask missing for {sample_id}: {image_path}, {mask_path}"
            )
        with Image.open(image_path) as image_handle:
            image = np.asarray(image_handle.convert("RGB"), dtype=np.uint8)
        with Image.open(mask_path) as mask_handle:
            mask = np.asarray(mask_handle.convert("L")) > 0
        if image.shape[:2] != mask.shape:
            raise ValueError(f"source mask size mismatch for {sample_id}")
        if not mask.any():
            raise ValueError(f"source mask empty for {sample_id}")
        pixels_by_color.setdefault(color, []).append(image[mask])
        sample_ids_by_color.setdefault(color, []).append(sample_id)

    references = {
        color: tuple(float(value) for value in np.median(np.concatenate(parts), axis=0))
        for color, parts in pixels_by_color.items()
    }
    audit = {
        "method": "per_channel_median_of_all_gt_mask_pixels",
        "dataset_root": str(dataset_root.resolve()),
        "experiment_manifest": str(experiment_manifest_path.resolve()),
        "colors": [
            {
                "name": color,
                "source_rgb": list(references[color]),
                "sample_ids": sample_ids_by_color[color],
                "pixel_count": int(sum(part.shape[0] for part in pixels_by_color[color])),
            }
            for color in sorted(references)
        ],
    }
    return references, audit


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert nonlinear 0..1 sRGB to CIE Lab using the D65 white point."""

    rgb = np.asarray(rgb, dtype=np.float64)
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = linear @ matrix.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    epsilon = 216 / 24389
    kappa = 24389 / 27
    f_xyz = np.where(
        xyz > epsilon,
        np.cbrt(xyz),
        (kappa * xyz + 16) / 116,
    )
    return np.stack(
        (
            116 * f_xyz[..., 1] - 16,
            500 * (f_xyz[..., 0] - f_xyz[..., 1]),
            200 * (f_xyz[..., 1] - f_xyz[..., 2]),
        ),
        axis=-1,
    )


def _hue_degrees(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return HSV hue in degrees and whether hue is defined (saturation > 0)."""

    rgb = np.asarray(rgb, dtype=np.float64)
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    delta = maximum - minimum
    defined = delta > np.finfo(np.float64).eps
    safe_delta = np.where(defined, delta, 1.0)
    hue = np.zeros_like(maximum)
    red_max = (maximum == rgb[..., 0]) & defined
    green_max = (maximum == rgb[..., 1]) & defined & ~red_max
    blue_max = defined & ~red_max & ~green_max
    hue[red_max] = 60 * (
        ((rgb[..., 1] - rgb[..., 2]) / safe_delta)[red_max] % 6
    )
    hue[green_max] = 60 * (
        ((rgb[..., 2] - rgb[..., 0]) / safe_delta + 2)[green_max]
    )
    hue[blue_max] = 60 * (
        ((rgb[..., 0] - rgb[..., 1]) / safe_delta + 4)[blue_max]
    )
    return hue, defined


def pixel_errors(pixels_u8: np.ndarray, target_rgb: tuple[int, int, int]) -> dict[str, np.ndarray]:
    pixels = np.asarray(pixels_u8, dtype=np.float64) / 255.0
    target = np.asarray(target_rgb, dtype=np.float64)[None, :] / 255.0
    pixels_lab = srgb_to_lab(pixels)
    target_lab = srgb_to_lab(target)[0]
    lab_difference = pixels_lab - target_lab

    denominator = np.linalg.norm(pixels, axis=-1) * np.linalg.norm(target[0])
    cosine = np.divide(
        pixels @ target[0],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    srgb_angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    pixel_hue, pixel_hue_defined = _hue_degrees(pixels)
    target_hue, target_hue_defined = _hue_degrees(target)
    hue_difference = np.abs(pixel_hue - target_hue[0])
    hue_angle = np.minimum(hue_difference, 360.0 - hue_difference)
    hue_valid = pixel_hue_defined & bool(target_hue_defined[0])
    hue_angle = np.where(hue_valid, hue_angle, np.nan)

    return {
        "delta_e": np.linalg.norm(lab_difference, axis=-1),
        "delta_e_ch": np.linalg.norm(lab_difference[..., 1:3], axis=-1),
        "srgb_angular_deg": srgb_angle,
        "hue_angular_deg": hue_angle,
    }


def closest_fraction_mean(values: np.ndarray, fraction: float) -> float | None:
    finite = np.sort(np.asarray(values, dtype=np.float64)[np.isfinite(values)])
    if finite.size == 0:
        return None
    count = max(1, math.ceil(finite.size * fraction))
    return float(finite[:count].mean())


def _base_row(item: dict[str, Any], image_path: Path, mask_path: Path) -> dict[str, Any]:
    row = {
        "id": item.get("id"),
        "category": item.get("category"),
        "prompt": item.get("prompt"),
        "seed": item.get("seed"),
        "color_label": item.get("color_label"),
        "color_token": item.get("color_token"),
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "status": None,
        "failure_reason": None,
        "mask_pixels": None,
        "hue_valid_pixels": None,
        "hue_status": None,
        "target_r": None,
        "target_g": None,
        "target_b": None,
        "nominal_target_r": None,
        "nominal_target_g": None,
        "nominal_target_b": None,
        "source_target_r": None,
        "source_target_g": None,
        "source_target_b": None,
        "source_reference_status": "not_requested",
        "source_hue_valid_pixels": None,
        "source_hue_status": None,
    }
    for metric in METRICS:
        for fraction in FRACTIONS:
            row[f"{metric}_{int(fraction * 100)}pct"] = None
            row[f"nominal_{metric}_{int(fraction * 100)}pct"] = None
            row[f"source_{metric}_{int(fraction * 100)}pct"] = None
    return row


def evaluate_item(
    item: dict[str, Any],
    image_root: Path,
    mask_root: Path,
    target_colors: dict[str, tuple[int, int, int]],
    source_colors: dict[str, tuple[float, float, float]] | None = None,
) -> dict[str, Any]:
    relative_image_path = Path(str(item.get("image_path", "")))
    image_path = image_root / relative_image_path
    mask_path = mask_root / relative_image_path
    row = _base_row(item, image_path, mask_path)
    target_rgb = target_colors.get(str(item.get("color_label")))
    if target_rgb is None:
        row.update(status="failure", failure_reason="target_color_missing")
        return row
    row["target_r"], row["target_g"], row["target_b"] = target_rgb
    row["nominal_target_r"], row["nominal_target_g"], row["nominal_target_b"] = target_rgb
    source_rgb = None
    if source_colors is not None:
        source_rgb = source_colors.get(str(item.get("color_label")))
        if source_rgb is None:
            row.update(status="failure", failure_reason="source_reference_missing")
            return row
        row["source_reference_status"] = "available"
        row["source_target_r"], row["source_target_g"], row["source_target_b"] = source_rgb
    if not image_path.is_file():
        row.update(status="failure", failure_reason="image_missing")
        return row
    if not mask_path.is_file():
        row.update(status="failure", failure_reason="mask_missing")
        return row

    try:
        with Image.open(image_path) as image_handle:
            image = np.asarray(image_handle.convert("RGB"), dtype=np.uint8)
        with Image.open(mask_path) as mask_handle:
            mask = np.asarray(mask_handle.convert("L")) > 0
    except (OSError, ValueError) as error:
        row.update(status="failure", failure_reason=f"image_or_mask_decode_error:{error}")
        return row
    if image.shape[:2] != mask.shape:
        row.update(status="failure", failure_reason="mask_size_mismatch")
        return row
    mask_pixels = int(mask.sum())
    row["mask_pixels"] = mask_pixels
    if mask_pixels == 0:
        row.update(status="failure", failure_reason="mask_empty")
        return row

    errors = pixel_errors(image[mask], target_rgb)
    row["hue_valid_pixels"] = int(np.isfinite(errors["hue_angular_deg"]).sum())
    if len(set(target_rgb)) == 1:
        row["hue_status"] = "undefined_achromatic_target"
    elif row["hue_valid_pixels"] == 0:
        row["hue_status"] = "undefined_no_chromatic_mask_pixels"
    else:
        row["hue_status"] = "ok"
    for metric, values in errors.items():
        for fraction in FRACTIONS:
            value = closest_fraction_mean(values, fraction)
            row[f"{metric}_{int(fraction * 100)}pct"] = value
            row[f"nominal_{metric}_{int(fraction * 100)}pct"] = value
    if source_rgb is not None:
        source_errors = pixel_errors(image[mask], source_rgb)
        row["source_hue_valid_pixels"] = int(
            np.isfinite(source_errors["hue_angular_deg"]).sum()
        )
        if len(set(source_rgb)) == 1:
            row["source_hue_status"] = "undefined_achromatic_target"
        elif row["source_hue_valid_pixels"] == 0:
            row["source_hue_status"] = "undefined_no_chromatic_mask_pixels"
        else:
            row["source_hue_status"] = "ok"
        for metric, values in source_errors.items():
            for fraction in FRACTIONS:
                row[f"source_{metric}_{int(fraction * 100)}pct"] = closest_fraction_mean(
                    values, fraction
                )
    row["status"] = "ok"
    return row


def evaluate_manifest(
    manifest_rows: Iterable[dict[str, Any]],
    image_root: Path,
    mask_root: Path,
    target_colors: dict[str, tuple[int, int, int]],
    categories: set[str],
    source_colors: dict[str, tuple[float, float, float]] | None = None,
) -> list[dict[str, Any]]:
    return [
        evaluate_item(item, image_root, mask_root, target_colors, source_colors)
        for item in manifest_rows
        if item.get("category") in categories
    ]


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no manifest rows matched --categories")
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument(
        "--target-colors-json",
        type=Path,
        default=Path(
            "experiments/clevr_subject_color_3x3/manifests/clevr_3x3_manifest.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument(
        "--source-dataset-root",
        type=Path,
        help="Optional CLEVR dataset root used to derive masked source-color medians.",
    )
    parser.add_argument(
        "--source-reference-output",
        type=Path,
        help="Write the source-reference derivation audit JSON.",
    )
    args = parser.parse_args(argv)
    if args.source_reference_output is not None and args.source_dataset_root is None:
        parser.error("--source-reference-output requires --source-dataset-root")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source_colors = None
    if args.source_dataset_root is not None:
        source_colors, source_audit = derive_source_colors(
            args.source_dataset_root, args.target_colors_json
        )
        if args.source_reference_output is not None:
            args.source_reference_output.parent.mkdir(parents=True, exist_ok=True)
            args.source_reference_output.write_text(
                json.dumps(source_audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    rows = evaluate_manifest(
        read_jsonl(args.manifest),
        args.image_dir,
        args.mask_dir,
        load_target_colors(args.target_colors_json),
        set(args.categories),
        source_colors,
    )
    write_csv(rows, args.output)


if __name__ == "__main__":
    main()
