#!/usr/bin/env python3
"""Derive the locked D1 natural-image pilot masks and QC bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.colorpeel_ice.natural_image_masks import (  # noqa: E402
    ALPHA_THRESHOLD_SEMANTICS,
    COLOR_THRESHOLD_SEMANTICS,
    IMAGE_SIZE,
    ROUNDING_SEMANTICS,
    NaturalImageMaskError,
    bool_to_l_image,
    canonical_sha256,
    compute_tight_crop,
    copy_verified,
    crop_image,
    derive_alpha,
    derive_color_mask,
    exact_edt,
    file_sha256,
    half_open_bbox,
    load_binary_mask,
    load_rgb,
    make_sample_qc,
    merge_sample_status,
    parse_stable_id,
    require,
    safe_slug,
    save_alpha_u16,
    write_json,
)


LOCKED_SELECTION = [
    {"group": "pilot", "rank": 1, "stable_id": "D1GT:3/207.png"},
    {"group": "pilot", "rank": 2, "stable_id": "D1GT:7/116.png"},
    {"group": "pilot", "rank": 3, "stable_id": "D1GT:86/139.png"},
    {"group": "pilot", "rank": 4, "stable_id": "D1GT:76/120.png"},
    {"group": "pilot", "rank": 5, "stable_id": "D1GT:0/158.png"},
    {"group": "pilot", "rank": 6, "stable_id": "D1GT:19/31.png"},
    {"group": "pilot", "rank": 7, "stable_id": "D1GT:46/97.png"},
    {"group": "pilot", "rank": 8, "stable_id": "D1GT:6/42.png"},
    {"group": "pilot", "rank": 9, "stable_id": "D1GT:77/53.png"},
    {"group": "pilot", "rank": 10, "stable_id": "D1GT:81/130.png"},
    {"group": "pilot", "rank": 11, "stable_id": "D1GT:74/104.png"},
    {"group": "pilot", "rank": 12, "stable_id": "D1GT:53/196.png"},
    {"group": "reserve", "rank": 13, "stable_id": "D1GT:88/184.png"},
    {"group": "reserve", "rank": 14, "stable_id": "D1GT:43/156.png"},
    {"group": "reserve", "rank": 15, "stable_id": "D1GT:10/91.png"},
    {"group": "reserve", "rank": 16, "stable_id": "D1GT:71/152.png"},
    {"group": "reserve", "rank": 17, "stable_id": "D1GT:31/98.png"},
    {"group": "reserve", "rank": 18, "stable_id": "D1GT:1/8.png"},
]

EXTERNAL_SELECTION_KEYS = {
    "schema_version",
    "selection_provenance",
    "selection_description",
    "samples",
}
EXTERNAL_SELECTION_ROW_KEYS = {
    "rank",
    "group",
    "sample_id",
    "mask_name",
    "stable_id",
    "selection_reason",
}

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
CANONICAL_ALPHA_MODES = {"I", "I;16", "I;16L", "I;16B"}


def input_identity(image_root: Path, mask_root: Path, image_layout: str) -> dict[str, Any]:
    return {
        "image_root": str(image_root.resolve()),
        "mask_root": str(mask_root.resolve()),
        "image_layout": image_layout,
        "image_layout_semantics": {
            "ice": "<image_root>/<sample_id>/img.jpg",
            "flat": "<image_root>/<sample_id>.jpg",
        },
        "mask_layout_semantics": "<mask_root>/<sample_id>/<mask_name>",
        "note": "locked delegation list is authoritative; mask lineage remains unknown in repo",
    }


def reject_repo_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.resolve()
    repo = REPO_ROOT.resolve()
    require(resolved != repo and repo not in resolved.parents, f"--output-dir must not be inside repo: {resolved}")
    if resolved.exists():
        require(resolved.is_dir(), f"--output-dir exists and is not a directory: {resolved}")
        require(
            not any(resolved.iterdir()),
            f"non-empty output directory will not be overwritten: {resolved}",
        )
    return resolved


def resolve_image_path(image_root: Path, sample_id: str, image_layout: str) -> Path:
    if image_layout == "ice":
        return image_root / sample_id / "img.jpg"
    if image_layout == "flat":
        return image_root / f"{sample_id}.jpg"
    raise NaturalImageMaskError(f"Unsupported image layout: {image_layout}")


def root_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def validate_selection(selection: list[dict[str, Any]]) -> None:
    require(len(selection) == 18, "Expected exactly 18 locked samples")
    stable_ids = [row["stable_id"] for row in selection]
    require(len(set(stable_ids)) == 18, "Stable IDs must be unique")
    sample_ids = [parse_stable_id(stable_id)[0] for stable_id in stable_ids]
    require(len(set(sample_ids)) == 18, "Sample IDs must be unique")
    require([row["rank"] for row in selection] == list(range(1, 19)), "Ranks must stay in locked order")
    require([row["group"] for row in selection[:12]] == ["pilot"] * 12, "Ranks 1-12 must be pilot")
    require([row["group"] for row in selection[12:]] == ["reserve"] * 6, "Ranks 13-18 must be reserve")


def validate_external_selection(selection: Any) -> list[dict[str, Any]]:
    """Fail closed for the separately reviewed three-replacement selection."""
    require(isinstance(selection, list), "selection JSON samples must be a list")
    require(len(selection) == 6, "External selection must contain exactly 6 samples")
    for row in selection:
        require(isinstance(row, dict), "Each external selection row must be an object")
        require(set(row) == EXTERNAL_SELECTION_ROW_KEYS, "External selection row has an invalid schema")
        require(type(row["rank"]) is int, "External selection rank must be an integer")
        require(isinstance(row["group"], str), "External selection group must be a string")
        require(isinstance(row["sample_id"], str) and row["sample_id"], "External selection sample_id must be nonempty")
        require(isinstance(row["mask_name"], str) and row["mask_name"], "External selection mask_name must be nonempty")
        require(isinstance(row["stable_id"], str), "External selection stable_id must be a string")
        require(isinstance(row["selection_reason"], str) and row["selection_reason"],
                "External selection selection_reason must be nonempty")
        sample_id, mask_name = parse_stable_id(row["stable_id"])
        require(sample_id == row["sample_id"] and mask_name == row["mask_name"],
                f"stable_id must match sample_id and mask_name: {row['stable_id']}")

    stable_ids = [row["stable_id"] for row in selection]
    sample_ids = [row["sample_id"] for row in selection]
    require(len(set(stable_ids)) == len(stable_ids), "External stable IDs must be unique")
    require(len(set(sample_ids)) == len(sample_ids), "External sample IDs must be unique")
    require([row["rank"] for row in selection] == list(range(1, 7)),
            "External ranks must be continuous from 1 through 6")
    require([row["group"] for row in selection] == ["replacement"] * 3 + ["reserve"] * 3,
            "External selection must be 3 replacement rows followed by 3 reserve rows")
    locked_sample_ids = {parse_stable_id(row["stable_id"])[0] for row in LOCKED_SELECTION}
    require(not locked_sample_ids.intersection(sample_ids),
            "External selection sample IDs must not overlap the locked 18")
    return selection


def _repo_relative_or_none(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return None


def load_selection_json(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a reviewed external selection without retaining a machine-local path."""
    resolved = path.resolve()
    require(resolved.is_file(), f"Selection JSON does not exist: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NaturalImageMaskError(f"Could not parse selection JSON: {resolved}: {exc}") from exc
    require(isinstance(value, dict) and set(value) == EXTERNAL_SELECTION_KEYS,
            "Selection JSON has an invalid schema")
    require(value["schema_version"] == 1, "Unsupported selection JSON schema_version")
    require(isinstance(value["selection_provenance"], str) and value["selection_provenance"],
            "selection_provenance must be nonempty")
    require(isinstance(value["selection_description"], str) and value["selection_description"],
            "selection_description must be nonempty")
    selection = validate_external_selection(value["samples"])
    return selection, {
        "selection_source_repo_relative_path": _repo_relative_or_none(resolved),
        "selection_file_sha256": file_sha256(resolved),
        "selection_provenance": value["selection_provenance"],
        "selection_description": value["selection_description"],
        "selected_count": len(selection),
        "group_counts": {
            "replacement": sum(row["group"] == "replacement" for row in selection),
            "reserve": sum(row["group"] == "reserve" for row in selection),
        },
        "stable_id_order_sha256": canonical_sha256([row["stable_id"] for row in selection]),
        "human_selection_reasons": [
            {"stable_id": row["stable_id"], "selection_reason": row["selection_reason"]}
            for row in selection
        ],
    }


def _relative_to_output(path: Path, output_dir: Path) -> str:
    return path.resolve().relative_to(output_dir.resolve()).as_posix()


def _image_metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        array = np.asarray(image)
        metadata: dict[str, Any] = {
            "mode": image.mode,
            "size": list(image.size),
            "format": image.format,
            "dtype": str(array.dtype),
        }
        if array.size:
            metadata["range"] = [int(array.min()), int(array.max())]
        if image.mode == "L":
            metadata["values"] = sorted(int(value) for value in np.unique(array))
        return metadata


def read_png_ihdr(path: Path) -> dict[str, int]:
    with path.open("rb") as handle:
        signature = handle.read(8)
        require(signature == PNG_SIGNATURE, f"PNG signature mismatch: {path}")
        length = int.from_bytes(handle.read(4), "big")
        chunk_type = handle.read(4)
        require(length == 13 and chunk_type == b"IHDR", f"PNG IHDR chunk missing or invalid: {path}")
        payload = handle.read(length)
        require(len(payload) == 13, f"PNG IHDR payload is truncated: {path}")
    return {
        "width": int.from_bytes(payload[0:4], "big"),
        "height": int.from_bytes(payload[4:8], "big"),
        "bit_depth": payload[8],
        "color_type": payload[9],
    }


def validate_canonical_alpha_png(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    ihdr = read_png_ihdr(path)
    require(ihdr["bit_depth"] == 16, f"canonical alpha PNG must have bit depth 16: {path}")
    require(ihdr["color_type"] == 0, f"canonical alpha PNG must be grayscale color type 0: {path}")
    metadata = _image_metadata(path)
    require(metadata["mode"] in CANONICAL_ALPHA_MODES, f"canonical alpha decoded mode is not 16-bit grayscale: {path}")
    with Image.open(path) as image:
        image.load()
        array = np.asarray(image).copy()
    require(array.ndim == 2, f"canonical alpha must decode to a 2D array: {path}")
    decoded_is_allowed = array.dtype == np.uint16 or np.issubdtype(array.dtype, np.signedinteger)
    require(decoded_is_allowed, f"canonical alpha decoded dtype is not portable 16-bit-compatible integer: {path}")
    require(array.size == 0 or (int(array.min()) >= 0 and int(array.max()) <= 65535),
            f"canonical alpha decoded values must stay within [0,65535]: {path}")
    metadata.update({
        "storage_dtype": "uint16",
        "storage_bit_depth": ihdr["bit_depth"],
        "storage_color_type": ihdr["color_type"],
    })
    return metadata, array


def _canonical_alpha_metadata(path: Path) -> dict[str, Any]:
    metadata, _ = validate_canonical_alpha_png(path)
    return metadata


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bool_to_l_image(mask).save(path)


def _save_alpha_preview(path: Path, alpha_u16: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = np.rint(alpha_u16.astype(np.float64) / 65535.0 * 255.0).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(path)


def derive_one(
    row: dict[str, Any],
    image_root: Path,
    mask_root: Path,
    image_layout: str,
    output_dir: Path,
) -> dict[str, Any]:
    stable_id = row["stable_id"]
    sample_id, mask_name = parse_stable_id(stable_id)
    source_image_path = resolve_image_path(image_root, sample_id, image_layout)
    source_mask_path = mask_root / sample_id / mask_name
    source_image_relative = root_relative(source_image_path, image_root)
    source_mask_relative = root_relative(source_mask_path, mask_root)
    require(source_image_path.is_file(), f"Missing source RGB: {source_image_path}")
    require(source_mask_path.is_file(), f"Missing source mask: {source_mask_path}")

    rgb_image, image_decode = load_rgb(source_image_path)
    raw, mask_decode = load_binary_mask(source_mask_path)
    bbox = half_open_bbox(raw)
    crop = compute_tight_crop(bbox, IMAGE_SIZE, w_out=0)
    d_in = exact_edt(raw)
    color = derive_color_mask(raw, bbox, d_in)
    alpha = derive_alpha(raw, bbox, d_in)

    sample_dir = output_dir / "samples" / safe_slug(stable_id, row["rank"])
    raw_image_out = sample_dir / "raw" / "img.jpg"
    raw_mask_out = sample_dir / "raw" / "mask.png"
    color_mask_out = sample_dir / "masks" / "color_mask.png"
    alpha_out = sample_dir / "masks" / "alpha_u16.png"
    alpha_preview_out = sample_dir / "masks" / "alpha_preview_u8.png"
    crop_dir = sample_dir / "crops"
    qc_out = sample_dir / "qc" / "sample_qc.png"

    image_copy = copy_verified(source_image_path, raw_image_out)
    mask_copy = copy_verified(source_mask_path, raw_mask_out)
    _write_mask(color_mask_out, color["mask"])
    save_alpha_u16(alpha_out, alpha["alpha_u16"])
    _save_alpha_preview(alpha_preview_out, alpha["alpha_u16"])

    crop_box = crop["crop_box_half_open"]
    crop_image(rgb_image, crop_box, crop_dir / "img_crop.jpg")
    crop_image(bool_to_l_image(raw), crop_box, crop_dir / "raw_mask_crop.png")
    crop_image(bool_to_l_image(color["mask"]), crop_box, crop_dir / "color_mask_crop.png")
    crop_image(Image.fromarray(alpha["alpha_u16"], mode="I;16"), crop_box, crop_dir / "alpha_u16_crop.png")

    qc_out.parent.mkdir(parents=True, exist_ok=True)
    qc_image = make_sample_qc(
        rgb_image,
        raw,
        color["mask"],
        alpha["alpha"],
        crop_box,
        f"{row['rank']:02d} {stable_id}",
    )
    qc_image.save(qc_out)

    raw_area = int(raw.sum())
    color_area = int(color["mask"].sum())
    alpha_outside_nonzero = int(np.count_nonzero(alpha["alpha_u16"][~raw]))
    require(alpha_outside_nonzero == 0, f"Alpha outside raw mask is nonzero: {stable_id}")
    require(mask_copy["source_sha256"] == mask_copy["output_sha256"], f"Raw mask copy hash changed: {stable_id}")
    require(mask_name != "combined_mask.png", f"Forbidden combined mask used: {stable_id}")

    status, flags = merge_sample_status(
        color["status"],
        color["flags"],
        alpha["status"],
        alpha["flags"],
        alpha_outside_nonzero,
    )

    outputs = {
        "raw_image": _relative_to_output(raw_image_out, output_dir),
        "raw_mask": _relative_to_output(raw_mask_out, output_dir),
        "color_mask": _relative_to_output(color_mask_out, output_dir),
        "alpha_canonical": _relative_to_output(alpha_out, output_dir),
        "alpha_preview": _relative_to_output(alpha_preview_out, output_dir),
        "crops": {
            "raw_image": _relative_to_output(crop_dir / "img_crop.jpg", output_dir),
            "raw_mask": _relative_to_output(crop_dir / "raw_mask_crop.png", output_dir),
            "color_mask": _relative_to_output(crop_dir / "color_mask_crop.png", output_dir),
            "alpha_canonical": _relative_to_output(crop_dir / "alpha_u16_crop.png", output_dir),
        },
        "sample_qc": _relative_to_output(qc_out, output_dir),
    }
    record = {
        "rank": row["rank"],
        "group": row["group"],
        "stable_id": stable_id,
        "sample_id": sample_id,
        "mask_name": mask_name,
        "status": status,
        "flags": flags,
        "pilot_eligible": True,
        "human_review": {
            "extraction": "PASS",
            "color": "PASS",
            "compositing": "PASS",
            "overall": "PASS",
        },
        "source": {
            "image": {
                "root_relative_path": source_image_relative,
                "sha256": image_copy["source_sha256"],
                "size_bytes": image_copy["source_size_bytes"],
                "mtime_ns": source_image_path.stat().st_mtime_ns,
                "decode": image_decode,
            },
            "mask": {
                "root_relative_path": source_mask_relative,
                "sha256": mask_copy["source_sha256"],
                "size_bytes": mask_copy["source_size_bytes"],
                "mtime_ns": mask_copy["source_mtime_ns"],
                "decode": mask_decode,
                "foreground_polarity": "255_is_foreground",
                "pairing_rule": (
                    f"stable_id {stable_id}; image_layout={image_layout}; "
                    f"image_root_relative={source_image_relative}; mask_root_relative={source_mask_relative}"
                ),
                "mask_lineage": "user_provided_unknown_in_repo",
            },
        },
        "raw_copy": {
            "image_bytes_identical": image_copy["bytes_identical"],
            "mask_bytes_identical": mask_copy["bytes_identical"],
            "mask_output_sha256": mask_copy["output_sha256"],
        },
        "geometry": {
            "raw_area_px": raw_area,
            "raw_area_ratio": raw_area / (IMAGE_SIZE[0] * IMAGE_SIZE[1]),
            "color_area_px": color_area,
            "color_retention": color_area / raw_area,
            "crop": crop,
        },
        "color_mask_derivation": {
            key: value
            for key, value in color.items()
            if key != "mask"
        },
        "alpha_derivation": {
            key: value
            for key, value in alpha.items()
            if key not in {"alpha", "alpha_u16"}
        },
        "outputs": outputs,
        "output_metadata": {
            "raw_image": _image_metadata(raw_image_out),
            "raw_mask": _image_metadata(raw_mask_out),
            "color_mask": _image_metadata(color_mask_out),
            "alpha_canonical": _canonical_alpha_metadata(alpha_out),
            "alpha_preview": {**_image_metadata(alpha_preview_out), "dtype": "uint8", "range": [0, 255]},
            "crops": {
                "raw_image": _image_metadata(crop_dir / "img_crop.jpg"),
                "raw_mask": _image_metadata(crop_dir / "raw_mask_crop.png"),
                "color_mask": _image_metadata(crop_dir / "color_mask_crop.png"),
                "alpha_canonical": _canonical_alpha_metadata(crop_dir / "alpha_u16_crop.png"),
            },
            "sample_qc": _image_metadata(qc_out),
        },
        "qc": {
            "alpha_outside_nonzero": alpha_outside_nonzero,
            "uses_linear_rgb_premultiplied_alpha": True,
            "label": "production EDT",
        },
        "reconstruction_mask": None,
    }
    if "selection_reason" in row:
        record["selection_reason"] = row["selection_reason"]
    return record


def write_contact_sheet(records: list[dict[str, Any]], output_dir: Path) -> str:
    sample_qcs = []
    for record in records:
        with Image.open(output_dir / record["outputs"]["sample_qc"]) as image:
            image.load()
            sample_qcs.append(image.copy())
    width = max(image.width for image in sample_qcs)
    height = sum(image.height for image in sample_qcs)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for image in sample_qcs:
        sheet.paste(image, (0, y))
        y += image.height
    path = output_dir / "qc" / "production_edt_contact_sheet.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return _relative_to_output(path, output_dir)


def verify_outputs(manifest: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    errors = []
    image_root = Path(manifest["inputs"]["image_root"])
    mask_root = Path(manifest["inputs"]["mask_root"])

    def add_error(stable_id: str, artifact: str, message: str) -> None:
        errors.append(f"{stable_id} {artifact}: {message}")

    def checked_sha(path: Path, stable_id: str, artifact: str) -> str | None:
        try:
            return file_sha256(path)
        except OSError as exc:
            add_error(stable_id, artifact, f"cannot hash {path}: {exc}")
            return None

    def decode(path: Path, stable_id: str, artifact: str) -> tuple[dict[str, Any], np.ndarray] | None:
        try:
            metadata = _image_metadata(path)
            with Image.open(path) as image:
                image.load()
                array = np.asarray(image).copy()
            return metadata, array
        except (OSError, ValueError) as exc:
            add_error(stable_id, artifact, f"cannot decode {path}: {exc}")
            return None

    for record in manifest["samples"]:
        stable_id = record["stable_id"]
        source_image = image_root / record["source"]["image"]["root_relative_path"]
        source_mask = mask_root / record["source"]["mask"]["root_relative_path"]
        raw_image = output_dir / record["outputs"]["raw_image"]
        raw_mask = output_dir / record["outputs"]["raw_mask"]
        source_image_hash = checked_sha(source_image, stable_id, "source_image")
        source_mask_hash = checked_sha(source_mask, stable_id, "source_mask")
        raw_image_hash = checked_sha(raw_image, stable_id, "raw_image")
        raw_mask_hash = checked_sha(raw_mask, stable_id, "raw_mask")
        if source_image_hash and raw_image_hash and source_image_hash != raw_image_hash:
            add_error(stable_id, "raw_image", "hash does not match source image")
        if source_mask_hash and raw_mask_hash and source_mask_hash != raw_mask_hash:
            add_error(stable_id, "raw_mask", "hash does not match source mask")

        decoded = {
            "raw_image": decode(raw_image, stable_id, "raw_image"),
            "raw_mask": decode(raw_mask, stable_id, "raw_mask"),
            "color_mask": decode(output_dir / record["outputs"]["color_mask"], stable_id, "color_mask"),
            "alpha_canonical": decode(output_dir / record["outputs"]["alpha_canonical"], stable_id, "alpha_canonical"),
            "alpha_preview": decode(output_dir / record["outputs"]["alpha_preview"], stable_id, "alpha_preview"),
            "sample_qc": decode(output_dir / record["outputs"]["sample_qc"], stable_id, "sample_qc"),
        }
        for crop_name, crop_path in record["outputs"].get("crops", {}).items():
            decoded[f"crop.{crop_name}"] = decode(output_dir / crop_path, stable_id, f"crop.{crop_name}")

        full_size = list(IMAGE_SIZE)
        for artifact in ("raw_image", "raw_mask", "color_mask", "alpha_canonical", "alpha_preview"):
            item = decoded.get(artifact)
            if item and item[0]["size"] != full_size:
                add_error(stable_id, artifact, f"expected size {full_size}, observed {item[0]['size']}")
        for artifact in ("raw_mask", "color_mask", "alpha_preview"):
            item = decoded.get(artifact)
            if item and item[0]["mode"] != "L":
                add_error(stable_id, artifact, f"expected mode L, observed {item[0]['mode']}")
        for artifact in ("raw_mask", "color_mask"):
            item = decoded.get(artifact)
            if item and set(int(value) for value in np.unique(item[1])) != {0, 255}:
                add_error(stable_id, artifact, "expected strict {0,255} values")

        raw_item = decoded.get("raw_mask")
        alpha_item = decoded.get("alpha_canonical")
        preview_item = decoded.get("alpha_preview")
        alpha_arr = None
        if alpha_item:
            try:
                _, alpha_arr = validate_canonical_alpha_png(output_dir / record["outputs"]["alpha_canonical"])
            except (NaturalImageMaskError, OSError, ValueError) as exc:
                add_error(stable_id, "alpha_canonical", str(exc))
            if raw_item:
                raw_arr = raw_item[1] == 255
                if alpha_arr is not None and int(np.count_nonzero(alpha_arr[~raw_arr])) != 0:
                    add_error(stable_id, "alpha_canonical", "raw exterior contains nonzero alpha")
            if preview_item and alpha_arr is not None:
                expected_preview = np.rint(alpha_arr.astype(np.float64) / 65535.0 * 255.0).astype(np.uint8)
                if not np.array_equal(preview_item[1], expected_preview):
                    add_error(stable_id, "alpha_preview", "does not match uint8 mapping from canonical alpha")

        try:
            crop_box = record["geometry"]["crop"]["crop_box_half_open"]
            x0, y0, x1, y1 = [int(value) for value in crop_box]
            crop_size = [x1 - x0, y1 - y0]
        except (KeyError, TypeError, ValueError) as exc:
            add_error(stable_id, "geometry.crop", f"bad crop_box_half_open: {exc}")
            continue
        crop_sources = {
            "raw_mask": decoded.get("raw_mask"),
            "color_mask": decoded.get("color_mask"),
            "alpha_canonical": decoded.get("alpha_canonical"),
        }
        if decoded.get("crop.alpha_canonical"):
            try:
                validate_canonical_alpha_png(output_dir / record["outputs"]["crops"]["alpha_canonical"])
            except (NaturalImageMaskError, OSError, ValueError) as exc:
                add_error(stable_id, "crop.alpha_canonical", str(exc))
        for crop_name, full_item in crop_sources.items():
            crop_item = decoded.get(f"crop.{crop_name}")
            if crop_item and crop_item[0]["size"] != crop_size:
                add_error(stable_id, f"crop.{crop_name}", f"expected size {crop_size}, observed {crop_item[0]['size']}")
            if crop_item and full_item and not np.array_equal(crop_item[1], full_item[1][y0:y1, x0:x1]):
                add_error(stable_id, f"crop.{crop_name}", "pixels do not match full-size crop slice")
        raw_image_crop = decoded.get("crop.raw_image")
        if raw_image_crop and raw_image_crop[0]["size"] != crop_size:
            add_error(stable_id, "crop.raw_image", f"expected size {crop_size}, observed {raw_image_crop[0]['size']}")
        sample_qc = decoded.get("sample_qc")
        if sample_qc and (sample_qc[0]["size"][0] <= 0 or sample_qc[0]["size"][1] <= 0):
            add_error(stable_id, "sample_qc", f"expected nonzero size, observed {sample_qc[0]['size']}")
    return {
        "checked_samples": len(manifest["samples"]),
        "errors": errors,
        "passed": not errors,
    }


def build_manifest(
    image_root: Path,
    mask_root: Path,
    image_layout: str,
    output_dir: Path,
    selection: list[dict[str, Any]] | None = None,
    selection_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if selection is None:
        selection = LOCKED_SELECTION
        validate_selection(selection)
        manifest_selection = {
            "locked_count": 18,
            "pilot_count": 12,
            "reserve_count": 6,
            "stable_id_order_sha256": canonical_sha256([row["stable_id"] for row in selection]),
        }
    else:
        validate_external_selection(selection)
        require(selection_metadata is not None, "External selection metadata is required")
        manifest_selection = selection_metadata
    output_dir = reject_repo_output_dir(output_dir)
    require(image_root.is_dir(), f"Image root does not exist: {image_root}")
    require(mask_root.is_dir(), f"Mask root does not exist: {mask_root}")
    records = [derive_one(row, image_root, mask_root, image_layout, output_dir) for row in selection]
    contact_sheet = write_contact_sheet(records, output_dir)
    summary = {
        "PASS": sum(1 for row in records if row["status"] == "PASS"),
        "REVIEW": sum(1 for row in records if row["status"] == "REVIEW"),
        "FAIL": sum(1 for row in records if row["status"] == "FAIL"),
    }
    manifest = {
        "schema_version": 1,
        "study": "natural_image_subject_color_pilot",
        "created_by": "prepare_natural_image_pilot.py",
        "production_contract": {
            "raw_mask_source_bytes_immutable": True,
            "combined_mask_forbidden": True,
            "mask_lineage": "user_provided_unknown_in_repo",
            "coordinate_space": "512x512 img.jpg only",
            "color_mask_uses_euclidean_edt": True,
            "soft_alpha_uses_euclidean_signed_edt": True,
            "gaussian_blur_used": False,
            "target_color_extraction_run": False,
            "recoloring_run": False,
            "training_or_model_evaluation_run": False,
            "rounding_semantics": ROUNDING_SEMANTICS,
            "color_threshold_semantics": COLOR_THRESHOLD_SEMANTICS,
            "alpha_threshold_semantics": ALPHA_THRESHOLD_SEMANTICS,
        },
        "inputs": input_identity(image_root, mask_root, image_layout),
        "selection": manifest_selection,
        "summary": summary,
        "contact_sheet": contact_sheet,
        "samples": records,
    }
    verification = verify_outputs(manifest, output_dir)
    manifest["verification"] = verification
    require(verification["passed"], f"Output verification failed: {verification['errors']}")
    return manifest


def write_readme(output_dir: Path, manifest: dict[str, Any]) -> None:
    is_default_selection = "locked_count" in manifest["selection"]
    selection_scope = "12 pilot samples and 6 reserve samples" if is_default_selection else "3 replacement samples and 3 reserve samples"
    introduction = (
        "This directory contains the D1 pilot/reserve mask derivation batch. The locked\n"
        "input list is embedded in `src/methods/colorpeel_ice/prepare_natural_image_pilot.py`."
        if is_default_selection
        else "This directory contains the D1 mask derivation batch. Selection provenance and\n"
             "individual human selection reasons are recorded in the manifest."
    )
    readme = f"""# Natural Image Subject-Color Pilot

{introduction}

Scope:

- {selection_scope}.
- Raw numbered masks are copied byte-for-byte from the provided mask root.
- Color masks and soft alpha masks are derived from exact Euclidean distance transforms.
- `combined_mask.png`, target color extraction, recoloring, training, and model evaluation are not used.

Machine-readable manifest: `manifests/pilot_mask_manifest.json`

Production QC contact sheet: `{manifest["contact_sheet"]}`

Status summary: PASS={manifest["summary"]["PASS"]}, REVIEW={manifest["summary"]["REVIEW"]}, FAIL={manifest["summary"]["FAIL"]}.
Downstream batches should consume only records with `status == "PASS"`.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def write_report(output_dir: Path, manifest: dict[str, Any]) -> None:
    is_default_selection = "locked_count" in manifest["selection"]
    selection_scope = "fixed 12 pilot plus 6 reserve samples" if is_default_selection else "fixed 3 replacement plus 3 reserve samples"
    review_or_fail = [
        row["stable_id"]
        for row in manifest["samples"]
        if row["status"] in {"REVIEW", "FAIL"}
    ]
    lines = [
        "# D1 Pilot Mask Derivation Batch",
        "",
        "Evidence labels: observed for generated files and checks in the generated run/output directory.",
        "",
        "## Scope",
        "",
        f"- observed: Derived raw mask byte copies, color masks, canonical uint16 soft-alpha masks, tight crops, manifest, and QC sheets for the {selection_scope}.",
        "- observed: Did not use `combined_mask.png`; did not run target color extraction, recoloring, training, model evaluation, segmentation, or GPU jobs.",
        "- observed: Mask lineage is recorded only as `user_provided_unknown_in_repo`.",
        "",
        "## Implementation",
        "",
        "- observed: Color masks use exact Euclidean EDT with descending candidate radii and component/retention gates.",
        "- observed: Soft alpha uses pixel-centered signed EDT, `w_out=0`, and clamps raw-mask exterior pixels exactly to zero.",
        "- observed: QC composites use linear-RGB premultiplied alpha over black, white, and checkerboard backgrounds.",
        "",
        "## Results",
        "",
        f"- observed: PASS={manifest['summary']['PASS']}, REVIEW={manifest['summary']['REVIEW']}, FAIL={manifest['summary']['FAIL']}.",
        f"- observed: REVIEW/FAIL stable IDs: {', '.join(review_or_fail) if review_or_fail else 'none'}.",
        f"- observed: Manifest verification passed: {manifest['verification']['passed']}.",
        f"- observed: Production QC contact sheet: `{manifest['contact_sheet']}`.",
        "",
        "## Next Batch Input",
        "",
        "- Use the generated `manifests/pilot_mask_manifest.json` under the explicit run output directory.",
        "- Consume only `samples[]` records with `status == \"PASS\"`.",
        "- Use `outputs.raw_image`, `outputs.raw_mask`, `outputs.color_mask`, `outputs.alpha_canonical`, and `geometry.crop` in the 512x512 `img.jpg` coordinate system.",
        "- Treat `reconstruction_mask` as null for every record.",
    ]
    path = output_dir / "reports" / "01_mask_derivation_batch.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--image-layout", choices=("ice", "flat"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    output_dir = reject_repo_output_dir(args.output_dir)
    selection = None
    selection_metadata = None
    if args.selection_json is not None:
        selection, selection_metadata = load_selection_json(args.selection_json)
    manifest = build_manifest(
        args.image_root.resolve(),
        args.mask_root.resolve(),
        args.image_layout,
        output_dir,
        selection,
        selection_metadata,
    )
    manifest_path = output_dir / "manifests" / "pilot_mask_manifest.json"
    write_json(manifest_path, manifest)
    write_readme(output_dir, manifest)
    write_report(output_dir, manifest)
    print(json.dumps({
        "status": "completed",
        "manifest": str(manifest_path),
        "contact_sheet": str(output_dir / manifest["contact_sheet"]),
        "summary": manifest["summary"],
        "review_or_fail": [
            row["stable_id"] for row in manifest["samples"] if row["status"] in {"REVIEW", "FAIL"}
        ],
    }, indent=2, sort_keys=True))
    return manifest


if __name__ == "__main__":
    try:
        main()
    except NaturalImageMaskError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
