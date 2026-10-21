"""Descriptive image checks; none establishes physical BRDF accuracy or identity."""

import html
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def _dilate(mask, radius=2):
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(
        ImageFilter.MaxFilter(2 * radius + 1))) > 0


def _edges(rgb, region):
    gray = rgb.astype(float) @ np.array([0.2126, 0.7152, 0.0722])
    gy, gx = np.gradient(gray)
    gradient = np.hypot(gx, gy)
    threshold = max(8.0, float(np.quantile(gradient[region], 0.8)))
    return (gradient >= threshold) & region, gradient


def evaluate(original, raw, final, mask):
    """Evaluate aligned uint8 RGBs; resize raw separately, retain its original file."""
    original, raw, final, mask = map(np.asarray, (original, raw, final, mask))
    if (original.ndim != 3 or original.shape[2] != 3
            or any(image.shape != original.shape or image.dtype != np.uint8
                   for image in (original, raw, final))
            or mask.shape != original.shape[:2] or not set(np.unique(mask)) <= {0, 1, 255}):
        raise ValueError("require aligned uint8 RGB images and a binary mask")
    foreground = mask > 0
    if not foreground.any() or foreground.all():
        raise ValueError("mask must contain foreground and background")
    background = ~foreground
    metrics = {}
    for name, image in (("raw", raw), ("final", final)):
        difference = np.abs(image.astype(float) - original.astype(float))
        metrics[name + "_background"] = {
            "changed_pixels": int(np.any(difference > 0, axis=2)[background].sum()),
            "mean_absolute_error_255": float(difference[background].mean()),
            "pixel_count": int(background.sum())}

    # Saturation gating avoids assigning meaningful hue to gray reflections.
    source_hsv = np.asarray(Image.fromarray(original).convert("HSV"), dtype=float)
    final_hsv = np.asarray(Image.fromarray(final).convert("HSV"), dtype=float)
    chromatic = foreground & (source_hsv[..., 1] >= 51) & (final_hsv[..., 1] >= 51)
    hue_delta = np.abs(source_hsv[..., 0] - final_hsv[..., 0]) / 255 * 360
    hue_delta = np.minimum(hue_delta, 360 - hue_delta)[chromatic]
    metrics["color"] = {"chromatic_pixels": int(chromatic.sum()),
                        "hue_median_degrees": float(np.median(hue_delta)) if hue_delta.size else None,
                        "hue_p95_degrees": float(np.quantile(hue_delta, .95)) if hue_delta.size else None}
    interior = foreground & ~_dilate(background)
    region = interior if interior.any() else foreground
    source_edges, _ = _edges(original, region)
    raw_edges, _ = _edges(raw, region)
    boundary_band = _dilate(foreground) & _dilate(background)
    source_boundary_edges, _ = _edges(original, boundary_band)
    raw_boundary_edges, _ = _edges(raw, boundary_band)
    metrics["structure_proxy"] = {
        "evaluated_image": "raw_resized_before_background_composite",
        "edge_tolerance_pixels": 2,
        "source_edge_count": int(source_edges.sum()), "raw_edge_count": int(raw_edges.sum()),
        "source_edge_recall": float((_dilate(raw_edges) & source_edges).sum() / source_edges.sum())
        if source_edges.any() else None,
        "raw_edge_precision": float((_dilate(source_edges) & raw_edges).sum() / raw_edges.sum())
        if raw_edges.any() else None,
        "mask_boundary_band_source_edges": int(source_boundary_edges.sum()),
        "mask_boundary_band_edge_recall": float((_dilate(raw_boundary_edges) & source_boundary_edges).sum()
                                                / source_boundary_edges.sum())
        if source_boundary_edges.any() else None,
        "note": "Gradient agreement is confounded by changed reflections; no independent segmentation or IoU."}
    luminance = final.astype(float) @ np.array([.2126, .7152, .0722]) / 255
    _, gradient = _edges(final, region)
    metrics["highlight_proxy"] = {
        "luminance_p50_p90_p99": np.quantile(luminance[foreground], [.5, .9, .99]).tolist(),
        "gradient_mean_255": float(gradient[region].mean()),
        "note": "Descriptive RGB statistics, not calibrated roughness or metallic measurements."}
    flags = ["manual_review_required"]
    if metrics["final_background"]["changed_pixels"]:
        flags.append("final_background_changed")
    if metrics["raw_background"]["mean_absolute_error_255"] > 3:
        flags.append("raw_background_changed_before_compositing")
    if hue_delta.size and np.median(hue_delta) > 20:
        flags.append("inspect_color_drift")
    recall = metrics["structure_proxy"]["source_edge_recall"]
    if recall is not None and recall < .6:
        flags.append("inspect_structure_or_reflection_change")
    return {**metrics, "flags": flags, "manual_review_required": True, "training_eligible": False}


def write_review(output, originalpath, maskpath, rows):
    """Save six actual edited RGBs with relative controls, captions and QC flags."""
    output = Path(output)
    targets = [output / "contact_sheet.png", output / "index.html"]
    if any(path.exists() for path in targets):
        raise FileExistsError("review files already exist")
    combinations = {(row["metallic_state"], row["roughness_state"]) for row in rows}
    if (len(rows) != 6 or len(combinations) != 6
            or len({row["metallic_state"] for row in rows}) != 2
            or len({row["roughness_state"] for row in rows}) != 3):
        raise ValueError("review requires the complete two-by-three grid")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    sheet = Image.new("RGB", (1536, 1810), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), "MARBLE C / six RGB candidates / relative M/R controls / NOT training-approved", font=font, fill="black")
    draw.text((12, 32), "Image metrics require visual review; no physical parameter accuracy or identity guarantee.", font=font, fill="black")

    def panel(path, xy, size=(496, 496)):
        path = Path(path)
        path = path if path.is_absolute() else output / path
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail(size, Image.Resampling.LANCZOS)
            sheet.paste(image, xy)

    panel(originalpath, (8, 80), (400, 400))
    panel(maskpath, (520, 80), (400, 400))
    draw.text((8, 56), "Original input photograph", font=font, fill="black")
    draw.text((520, 56), "Input object mask (not predicted output segmentation)", font=font, fill="black")
    document = ["<h1>MARBLE six candidates: manual review pending</h1>",
                "<p>Relative controls, not precise PBR parameters. No subject training.</p>"]
    ordered = sorted(rows, key=lambda row: (float(row["metallic_control"]), float(row["roughness_control"])))
    for index, row in enumerate(ordered):
        x, y = (index % 3) * 512 + 8, (index // 3) * 650 + 500
        label = (f"{row['metallic_state']} M-control={row['metallic_control']} | "
                 f"{row['roughness_state']} R-control={row['roughness_control']}")
        draw.text((x, y), label, font=font, fill="black")
        panel(row["image"], (x, y + 28))
        lines = textwrap.wrap(row["caption"], 53) + textwrap.wrap(
            "; ".join(row["qc"]["flags"]), 53)
        for offset, line in enumerate(lines[:6]):
            draw.text((x, y + 528 + offset * 18), line, font=font, fill="black")
        document.append("<h2>" + html.escape(label) + "</h2><img width='512' src='" +
                        html.escape(str(row["image"]), quote=True) + "'><p>" +
                        html.escape(row["caption"]) + "</p><pre>" +
                        html.escape(str(row["qc"])) + "</pre><p>Raw output: " +
                        html.escape(str(row["raw_image"])) + "</p>")
    sheet.save(targets[0])
    targets[1].write_text("<!doctype html><meta charset='utf-8'>" + "\n".join(document), encoding="utf-8")
    return {"contact_sheet": str(targets[0]), "index": str(targets[1])}
