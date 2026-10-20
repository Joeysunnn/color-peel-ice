"""Read-only parameter provenance and linear premultiplied PBR compositing."""

import itertools
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from src.methods.colorpeel_ice.gamut_aware_lch_injection import _srgb_to_linear, _linear_to_srgb_uint8

COLORS = {"pink": 0.0, "green": 120.0, "cyan": 180.0, "blue": 240.0, "purple": 300.0}
METALLIC = {"nonmetallic": 0.0, "metallic": 1.0}
ROUGHNESS = {"low": 0.15, "medium": 0.5, "high": 0.85}


def _check_locks(locks):
    if not locks or any(not Path(path).is_absolute() or sha256(Path(path)) != value
                        for path, value in locks.items()):
        raise ValueError("preview or immutable source hashes differ")


def verify_preview(preview):
    """Validate the existing thirty unrendered proposals without changing their files."""
    preview = Path(preview).resolve()
    status = read_json(preview / "status.json")
    if (status.get("status") != "parameter_preview_verified_renderer_missing"
            or status.get("parameter_configurations") != 30 or status.get("accepted_parameter_edits") != 30
            or status.get("rejected_parameter_edits") != 0 or status.get("rendered_rgb") != 0
            or status.get("training_eligible_images") != 0 or status.get("training_started") is not False):
        raise ValueError("expected a complete accepted parameter preview with no rendered or trained images")
    protocol = read_json(preview / "protocol.json")
    if (protocol.get("schema") != "subject_albedo_pbr_c/v1" or protocol.get("colors") != COLORS
            or protocol.get("metallic") != METALLIC or protocol.get("roughness") != ROUGHNESS
            or protocol.get("training_enabled") is not False):
        raise ValueError("preview A/M/R factor definitions differ")
    source_locks = read_json(preview / "source_locks.json")
    output_hashes = read_json(preview / "output_hashes.json")
    _check_locks(source_locks)
    _check_locks(output_hashes)
    if any(not Path(path).resolve().is_relative_to(preview) for path in output_hashes):
        raise ValueError("preview output receipt points outside its own run")
    required = [preview / name for name in ("protocol.json", "parameter_manifest.jsonl", "source_locks.json",
                "source_receipt.json", "albedo_qc.json", "renderer_readiness.json")]
    required += [preview / "source" / name for name in
                 ("original.jpg", "object_mask.png", "input.png", "albedo.png", "roughness.png", "metallic.png")]
    if any(str(path) not in output_hashes for path in required):
        raise ValueError("preview output receipt omits required metadata or source artifacts")
    locks = dict(source_locks)
    for path, digest in output_hashes.items():
        if path in locks and locks[path] != digest:
            raise ValueError("preview and upstream source receipts conflict")
        locks[path] = digest
    for name in ("status.json", "output_hashes.json"):
        locks[str(preview / name)] = sha256(preview / name)
    rows = [json.loads(line) for line in (preview / "parameter_manifest.jsonl").read_text(encoding="utf8").splitlines()]
    expected = set(itertools.product(COLORS, METALLIC, ROUGHNESS))
    keys = [(row["color"], row["metallic_state"], row["roughness_state"]) for row in rows]
    if len(rows) != 30 or len(set(keys)) != 30 or set(keys) != expected:
        raise ValueError("preview must contain the complete thirty-cell A/M/R factorial")
    with Image.open(preview / "source/object_mask.png") as image:
        mask = np.asarray(image)
        if image.mode != "L" or image.size != (512, 512) or set(np.unique(mask)) != {0, 255}:
            raise ValueError("preview original object mask differs")
    inside = mask == 255
    seen = {channel: {} for channel in ("albedo", "metallic", "roughness")}
    for row in rows:
        color, metallic, roughness = row["color"], row["metallic_state"], row["roughness_state"]
        if (row.get("id") != f"{color}__{metallic}__{roughness}" or row.get("hue_degrees") != COLORS[color]
                or row.get("metallic") != METALLIC[metallic] or row.get("roughness") != ROUGHNESS[roughness]
                or row.get("qc") != "accepted_parameter_edit" or row.get("render_status") != "not_rendered"
                or row.get("rgb_image") is not None or row.get("training_eligible") is not False):
            raise ValueError("preview row parameters, acceptance or render status differ")
        for channel, factor in (("albedo", color), ("metallic", metallic), ("roughness", roughness)):
            record = row["maps"][channel]
            path = Path(record["path"])
            if (not path.is_absolute() or not path.resolve().is_relative_to(preview)
                    or output_hashes.get(str(path)) != record["sha256"]):
                raise ValueError("parameter map is not bound to the immutable preview receipt")
            if factor in seen[channel]:
                if record != seen[channel][factor]:
                    raise ValueError("one parameter map changes with an unrelated A/M/R factor")
                continue
            seen[channel][factor] = record
            if channel == "albedo":
                with Image.open(path) as image:
                    if image.mode != "RGB" or image.size != (512, 512):
                        raise ValueError("Albedo must be a full-size RGB map")
            else:
                value = (METALLIC if channel == "metallic" else ROUGHNESS)[factor]
                pixels = np.load(path, allow_pickle=False)
                with Image.open(preview / "source" / f"{channel}.png") as image:
                    original = np.asarray(image.convert("L"), dtype=np.float32) / 255
                if (pixels.dtype != np.float32 or pixels.shape != mask.shape or not np.isfinite(pixels).all()
                        or np.any(pixels < 0) or np.any(pixels > 1)
                        or not np.all(pixels[inside] == np.float32(value))
                        or not np.array_equal(pixels[~inside], original[~inside])
                        or record.get("value") != value or record.get("qc") != "accepted_parameter_edit"
                        or output_hashes.get(record.get("preview")) != record.get("preview_sha256")):
                    raise ValueError("authoritative M/R map intervention or its preview receipt differs")
    if any(len({record["sha256"] for record in factors.values()}) != len(factors)
           for factors in seen.values()):
        raise ValueError("distinct factor states must select distinct Albedo/M/R maps")
    _check_locks(locks)
    return rows, locks


def composite(premult_rgba_linear_float32, original_uint8RGB, mask_uint8, background=None):
    """Composite fixed-exposure linear foreground only where the original mask permits."""
    rgba, original, mask = map(np.asarray, (premult_rgba_linear_float32, original_uint8RGB, mask_uint8))
    if (rgba.dtype != np.float32 or rgba.ndim != 3 or rgba.shape[2] != 4
            or original.dtype != np.uint8 or original.shape != (*rgba.shape[:2], 3)
            or mask.dtype != np.uint8 or mask.shape != rgba.shape[:2]):
        raise ValueError("expected matching float32 RGBA, uint8 RGB and uint8 object mask")
    if not set(np.unique(mask)) <= {0, 255} or not np.any(mask == 255):
        raise ValueError("object mask must be nonempty binary 0/255")
    if not np.isfinite(rgba).all() or np.any(rgba[:, :, 3] < 0) or np.any(rgba[:, :, 3] > 1):
        raise ValueError("RGBA must be finite with alpha in [0,1]")
    alpha = rgba[:, :, 3]
    if np.any(np.abs(rgba[:, :, :3][alpha == 0]) > 1e-6):
        raise ValueError("premultiplied RGB must be zero wherever alpha is zero")
    inside = mask == 255
    plate = original if background is None else np.asarray(background)
    if plate.dtype != np.uint8 or plate.shape != original.shape:
        raise ValueError("fixed background plate must match original uint8 RGB dimensions")
    if not np.array_equal(plate[~inside], original[~inside]):
        raise ValueError("fixed background plate must preserve every outside-mask original RGB pixel")
    padded = np.pad(inside, 1, constant_values=False)
    core = np.logical_and.reduce([padded[y:y + mask.shape[0], x:x + mask.shape[1]]
                                  for y, x in itertools.product(range(3), repeat=2)])
    if not core.any() or not np.any(alpha[inside] > 0):
        raise ValueError("render must cover a nonempty eroded foreground core")
    coverage = float(np.mean(alpha[core] > 0.99))
    if coverage < 0.995:
        raise ValueError(f"render core coverage {coverage:.6f} is below 99.5%; source foreground cannot fill missing geometry")
    linear = rgba[:, :, :3].astype(np.float64) + _srgb_to_linear(plate) * (1 - alpha[:, :, None])
    foreground = linear[inside]
    clipped = (foreground < 0) | (foreground > 1)
    result = original.copy()
    result[inside] = _linear_to_srgb_uint8(np.clip(foreground, 0, 1))
    qc = {"foreground_pixels": int(inside.sum()), "core_pixels": int(core.sum()),
          "core_coverage_fraction": coverage, "core_alpha_threshold": 0.99,
          "minimum_core_coverage_fraction": 0.995,
          "clipped_pixel_fraction": float(np.mean(np.any(clipped, axis=1))),
          "clipped_channel_fraction": float(clipped.mean()),
          "alpha_min_inside": float(alpha[inside].min()), "alpha_max_inside": float(alpha[inside].max()),
          "alpha_mean_inside": float(alpha[inside].mean()),
          "uncovered_inside_pixels": int(np.count_nonzero(alpha[inside] == 0)),
          "outside_mask_changed_pixels": int(np.count_nonzero(np.any(result[~inside] != original[~inside], axis=1))),
          "background_source": "original_rgb" if background is None else "fixed_background_plate",
          "compositing": "linear premultiplied foreground + linear "
                         + ("original background" if background is None else "fixed background plate")
                         + " * (1-alpha); write only inside original mask",
          "display": "fixed [0,1] clipping then sRGB; no per-image exposure or normalization"}
    return result, qc


def read_exr(path):
    """Load Blender Combined RGBA EXR as premultiplied scene-linear float32 RGBA."""
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    pixels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if (not isinstance(pixels, np.ndarray) or pixels.dtype != np.float32
            or pixels.ndim != 3 or pixels.shape[2] != 4 or not np.isfinite(pixels).all()):
        raise ValueError("Combined EXR must contain four finite scene-linear float32 channels")
    return pixels[:, :, [2, 1, 0, 3]].copy()
