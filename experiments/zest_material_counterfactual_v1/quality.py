"""Estimated-intrinsic QC, never physical PBR ground truth or RGB color leakage.

Thresholds are flat JSON: max_albedo_deltaE76_p95, max_albedo_delta_ab_p95,
min_albedo_edge_precision, min_albedo_edge_recall,
min_albedo_high_frequency_correlation, max_aligned_depth_error,
max_background_mae255 and max_protected_mae255. All default to None, meaning
uncalibrated. min_valid_pixels defaults to 1000. Optional target_metallic and
target_roughness require min_material_direction_delta; their values describe
the reference bank, not guaranteed generated PBR values. Manual review requires
require_manual_review=True (default) and manual_review_passed=True explicitly.
"""

import json

import numpy as np
from PIL import Image, ImageFilter

from src.methods.colorpeel_ice.gamut_aware_lch_injection import (
    _srgb_to_linear, linear_rgb_to_lab,
)


LIMITS = {
    "max_albedo_deltaE76_p95": ("albedo_deltaE76_p95", "max"),
    "max_albedo_delta_ab_p95": ("albedo_delta_ab_p95", "max"),
    "min_albedo_edge_precision": ("albedo_edge_precision", "min"),
    "min_albedo_edge_recall": ("albedo_edge_recall", "min"),
    "min_albedo_high_frequency_correlation": ("albedo_high_frequency_correlation", "min"),
    "max_aligned_depth_error": ("aligned_depth_error", "max"),
    "max_background_mae255": ("background_mae255", "max"),
    "max_protected_mae255": ("protected_mae255", "max"),
}


def _fail(message):
    raise ValueError(message)


def _array(value, name, shape, kind):
    array = np.asarray(value)
    if array.shape != shape or not np.isfinite(array).all():
        _fail(name + ": invalid shape or nonfinite values")
    if kind == "rgb" and array.dtype != np.uint8:
        _fail(name + ": require uint8 RGB")
    if kind == "mask" and array.dtype != np.bool_:
        _fail(name + ": require boolean mask")
    if kind == "scalar" and (not np.issubdtype(array.dtype, np.floating)
                              or np.any((array < 0) | (array > 1))):
        _fail(name + ": require floating point [0,1] map")
    if kind == "depth" and not np.issubdtype(array.dtype, np.floating):
        _fail(name + ": require floating point depth")
    return array


def _blur(image, sigma):
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    weights = np.exp(-x * x / (2 * sigma * sigma))
    weights /= weights.sum()
    padded = np.pad(image, ((0, 0), (radius, radius)), mode="edge")
    result = sum(weight * padded[:, i:i + image.shape[1]] for i, weight in enumerate(weights))
    padded = np.pad(result, ((radius, radius), (0, 0)), mode="edge")
    return sum(weight * padded[i:i + image.shape[0], :] for i, weight in enumerate(weights))


def _morph(mask, radius, dilate=False):
    operation = ImageFilter.MaxFilter if dilate else ImageFilter.MinFilter
    # Explicit outside zeros keep image-border pixels out of a valid interior.
    padded = np.pad(mask, radius)
    result = np.asarray(Image.fromarray(padded.astype(np.uint8) * 255).filter(
        operation(2 * radius + 1))) > 0
    return result[radius:-radius, radius:-radius]


def _texture(source, generated, mask, minimum):
    region = _morph(mask, 12)
    if region.sum() < minimum:
        return None, "insufficient_interior_texture_support"
    luminance = np.array([.2126, .7152, .0722])
    source = _srgb_to_linear(source) @ luminance
    generated = _srgb_to_linear(generated) @ luminance
    if np.ptp(source[region]) < 1e-4:
        return None, "source_albedo_has_no_assessable_texture"
    scales = []
    for sigma in (1., 2., 4.):
        source_low, generated_low = _blur(source, sigma), _blur(generated, sigma)
        source_gradient = np.hypot(*np.gradient(source_low))
        generated_gradient = np.hypot(*np.gradient(generated_low))
        threshold = max(1e-4, float(np.quantile(source_gradient[region], .8)))
        source_edges = (source_gradient >= threshold) & region
        generated_edges = (generated_gradient >= threshold) & region
        if source_edges.sum() < 20:
            return None, "source_albedo_has_insufficient_multiscale_edges"
        precision = float((_morph(source_edges, 2, True) & generated_edges).sum()
                          / generated_edges.sum()) if generated_edges.any() else 0.
        recall = float((_morph(generated_edges, 2, True) & source_edges).sum() / source_edges.sum())
        source_high, generated_high = (source - source_low)[region], (generated - generated_low)[region]
        if source_high.std() < 1e-8:
            return None, "source_albedo_has_insufficient_high_frequency_variance"
        correlation = float(np.corrcoef(source_high, generated_high)[0, 1]) \
            if generated_high.std() >= 1e-8 else 0.
        scales.append({"sigma": sigma, "edge_threshold_linear": threshold,
                       "source_edge_count": int(source_edges.sum()),
                       "generated_edge_count": int(generated_edges.sum()),
                       "edge_precision": precision, "edge_recall": recall,
                       "high_frequency_correlation": correlation})
    return {"scales": scales, "valid_pixels": int(region.sum()),
            "albedo_edge_precision": min(row["edge_precision"] for row in scales),
            "albedo_edge_recall": min(row["edge_recall"] for row in scales),
            "albedo_high_frequency_correlation": min(row["high_frequency_correlation"] for row in scales)}, None


def evaluate(original_rgb, generated_rgb, object_mask, edit_mask, protected_mask,
             valid_mask, region_mask, original_maps, generated_maps,
             original_depth, generated_depth, thresholds):
    """Return JSON-safe QC; masks/maps/depth are aligned at 512x512.

    Output depth must be independently estimated by the caller. No segmentation
    IoU is claimed. Texture and Albedo scores inherit SuperMat ambiguities.
    """
    result = {"status": "invalid", "metrics": {}, "reasons": [], "training_eligible": False,
              "interpretation": "Estimated A/R/M and depth proxies; not PBR ground truth. RGB differences outside the edit mask are protection checks, not color leakage."}
    try:
        config = dict(thresholds or {})
        minimum = config.get("min_valid_pixels", 1000)
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 20:
            _fail("min_valid_pixels must be an integer >=20")
        for key in (*LIMITS, "min_material_direction_delta", "target_metallic", "target_roughness"):
            value = config.get(key)
            if value is not None and (isinstance(value, bool) or not np.isscalar(value)
                                      or not np.isfinite(value)):
                _fail(key + ": threshold must be finite or null")
        for key in ("target_metallic", "target_roughness"):
            if config.get(key) is not None and not 0 <= config[key] <= 1:
                _fail(key + ": target must be in [0,1]")
        shape = (512, 512)
        source = _array(original_rgb, "original_rgb", (*shape, 3), "rgb")
        generated = _array(generated_rgb, "generated_rgb", (*shape, 3), "rgb")
        masks = {name: _array(value, name, shape, "mask") for name, value in (
            ("object", object_mask), ("edit", edit_mask), ("protected", protected_mask),
            ("valid", valid_mask), ("region", region_mask))}
        obj, edit, protected, valid, region = (masks[key] for key in ("object", "edit", "protected", "valid", "region"))
        if not obj.any() or obj.all() or not edit.any() or not valid.any() or not region.any():
            _fail("empty required mask or missing object/background support")
        if np.any(edit & ~obj) or np.any(edit & protected) or np.any(region & ~obj):
            _fail("edit/region must lie in object; edit must exclude protected pixels")
        roi, material_roi = obj & edit & valid & ~protected, region & edit & valid & ~protected
        if min(int(roi.sum()), int(material_roi.sum())) < minimum:
            _fail("insufficient valid edit/material pixels")
        maps = []
        for name, values in (("original_maps", original_maps), ("generated_maps", generated_maps)):
            if not isinstance(values, dict) or not {"albedo", "roughness", "metallic"} <= set(values):
                _fail(name + ": missing albedo/roughness/metallic")
            maps.append({key: _array(values[key], name + "/" + key,
                                     (*shape, 3) if key == "albedo" else shape,
                                     "rgb" if key == "albedo" else "scalar")
                         for key in ("albedo", "roughness", "metallic")})
        source_maps, output_maps = maps
        depth_source = _array(original_depth, "original_depth", shape, "depth").astype(float)
        depth_output = _array(generated_depth, "generated_depth", shape, "depth").astype(float)
        if min(float(depth_source.std()), float(depth_output.std())) < 1e-8:
            _fail("depth has insufficient variation for affine alignment")
        metrics = result["metrics"]
        difference = np.abs(generated.astype(float) - source.astype(float)).mean(axis=2)
        metrics.update(background_mae255=float(difference[~obj].mean()),
                       protected_mae255=float(difference[protected].mean()) if protected.any() else None,
                       outside_edit_mae255=float(difference[~edit].mean()),
                       protected_pixels=int(protected.sum()), valid_pixels=int(roi.sum()))
        lab_source = linear_rgb_to_lab(_srgb_to_linear(source_maps["albedo"][roi]))
        lab_output = linear_rgb_to_lab(_srgb_to_linear(output_maps["albedo"][roi]))
        lab_difference = lab_output - lab_source
        delta_e = np.linalg.norm(lab_difference, axis=1)
        delta_ab = np.linalg.norm(lab_difference[:, 1:], axis=1)
        metrics.update(albedo_deltaE76_median=float(np.median(delta_e)),
                       albedo_deltaE76_p95=float(np.quantile(delta_e, .95)),
                       albedo_delta_ab_median=float(np.median(delta_ab)),
                       albedo_delta_ab_p95=float(np.quantile(delta_ab, .95)),
                       albedo_delta_L_abs_p95=float(np.quantile(np.abs(lab_difference[:, 0]), .95)))
        material = {}
        for channel in ("metallic", "roughness"):
            stats = {}
            for name, values in (("original", source_maps), ("generated", output_maps)):
                pixels = values[channel][material_roi]
                median = float(np.median(pixels))
                stats[name] = {"median": median, "mad": float(np.median(np.abs(pixels - median)))}
            stats["estimated_delta"] = stats["generated"]["median"] - stats["original"]["median"]
            stats["reference_target"] = config.get("target_" + channel)
            material[channel] = stats
        metrics["estimated_material"] = material
        texture, texture_reason = _texture(source_maps["albedo"], output_maps["albedo"], roi, minimum)
        metrics["estimated_albedo_texture"] = texture
        if texture:
            metrics.update({key: texture[key] for key in ("albedo_edge_precision", "albedo_edge_recall", "albedo_high_frequency_correlation")})
        else:
            metrics.update({key: None for key in ("albedo_edge_precision", "albedo_edge_recall", "albedo_high_frequency_correlation")})
        centered_output = depth_output - depth_output.mean()
        scale = float(np.mean(centered_output * (depth_source - depth_source.mean())) / np.mean(centered_output ** 2))
        offset = float(depth_source.mean() - scale * depth_output.mean())
        depth_range = float(np.quantile(depth_source, .95) - np.quantile(depth_source, .05))
        if depth_range < 1e-8:
            _fail("depth robust range insufficient")
        metrics.update(depth_alignment_scale=scale, depth_alignment_offset=offset,
                       aligned_depth_error=float(np.abs(scale * depth_output + offset - depth_source)[roi].mean() / depth_range),
                       depth_alignment="full-field least-squares scale/offset; error in fixed valid edit ROI")
        rejected, pending = [], []
        if scale <= 0:
            rejected.append("nonpositive_depth_alignment_scale")
        if texture_reason:
            pending.append(texture_reason)
        for threshold, (metric, direction) in LIMITS.items():
            if metric == "protected_mae255" and not protected.any():
                continue  # Explicitly empty protected region is not perfect preservation.
            limit, value = config.get(threshold), metrics[metric]
            if limit is None:
                pending.append("uncalibrated:" + threshold)
            elif value is None:
                pending.append("not_assessable:" + metric)
            elif (direction == "max" and value > limit) or (direction == "min" and value < limit):
                rejected.append("threshold_failed:" + threshold)
        for channel, stats in material.items():
            target, margin = stats["reference_target"], config.get("min_material_direction_delta")
            if target is None:
                continue
            if margin is None:
                pending.append("uncalibrated:min_material_direction_delta")
            elif margin < 0:
                _fail("min_material_direction_delta must be nonnegative")
            elif abs(target - stats["original"]["median"]) > margin:
                direction = np.sign(target - stats["original"]["median"])
                directed_delta = direction * stats["estimated_delta"]
                if directed_delta <= 0 or directed_delta < margin:
                    rejected.append("material_direction_failed:" + channel)
        if config.get("require_manual_review", True) and not config.get("manual_review_passed", False):
            pending.append("manual_review_required")
        result.update(status="rejected" if rejected else "pending_manual_review" if pending else "accepted",
                      reasons=rejected + pending, training_eligible=not (rejected or pending))
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, FloatingPointError) as error:
        result.update(status="invalid", reasons=[str(error)], training_eligible=False)
    # Finite inputs can still overflow during numerical operations. Never emit
    # JSON NaN/Infinity or accidentally approve an unassessable metric.
    try:
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError):
        result.update(status="invalid", metrics={}, reasons=["nonfinite_or_nonserializable_metric"], training_eligible=False)
    return result
