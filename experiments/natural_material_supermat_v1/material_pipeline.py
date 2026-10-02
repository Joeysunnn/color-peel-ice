"""Prepare one natural material sample and audit SuperMat scalar estimates."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def project_commit():
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def prepare(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    source = Image.open(args.image).convert("RGB")
    object_mask = Image.open(args.object_mask).convert("L")
    if source.size != object_mask.size:
        raise ValueError("Image and object mask dimensions differ")
    region = json.loads(Path(args.region).read_text())
    if region["source_size"] != list(source.size):
        raise ValueError("Region polygon was defined for another image size")
    polygon = Image.new("L", source.size)
    ImageDraw.Draw(polygon).polygon([tuple(xy) for xy in region["polygon_xy"]], fill=255)
    object_pixels = np.asarray(object_mask) >= 128
    polygon_pixels = np.asarray(polygon) >= 128
    region_pixels = object_pixels & polygon_pixels
    if args.erosion_px:
        region_pixels &= np.asarray(Image.fromarray((object_pixels * 255).astype("uint8")).filter(
            ImageFilter.MinFilter(args.erosion_px * 2 + 1))) >= 128
    if region_pixels.sum() < 1000:
        raise ValueError("Material region contains fewer than 1000 pixels")
    source.save(out / "source.png")
    Image.fromarray((object_pixels * 255).astype("uint8")).save(out / "object_mask.png")
    Image.fromarray((region_pixels * 255).astype("uint8")).save(out / "material_region.png")
    rgba = source.copy().convert("RGBA")
    rgba.putalpha(Image.fromarray((object_pixels * 255).astype("uint8")))
    rgba.save(out / "source_mailbox.png")
    gray = Image.new("RGB", source.size, (128, 128, 128))
    gray.paste(source, mask=rgba.getchannel("A"))
    gray.save(out / "source_gray_composite.png")
    save_json(out / "preparation.json", {
        "source_image": str(Path(args.image).resolve()),
        "source_image_sha256": digest(args.image),
        "source_object_mask": str(Path(args.object_mask).resolve()),
        "source_object_mask_sha256": digest(args.object_mask),
        "region_definition": region,
        "region_definition_sha256": digest(args.region),
        "erosion_px": args.erosion_px,
        "source_size": list(source.size),
        "object_pixel_count": int(object_pixels.sum()),
        "material_pixel_count": int(region_pixels.sum()),
        "supermat_input": "source_mailbox.png",
        "supermat_input_sha256": digest(out / "source_mailbox.png"),
        "gray_composite_sha256": digest(out / "source_gray_composite.png"),
        "project_git_commit": project_commit(),
        "pipeline_sha256": digest(__file__),
    })


def map_values(path, mask):
    image = Image.open(path).convert("L")
    if image.size != mask.size:
        raise ValueError(f"Map size differs from mask size: {path}")
    data = np.asarray(image, dtype=np.float64)[np.asarray(mask) >= 128] / 255.0
    if len(data) < 1000:
        raise ValueError(f"Material region too small for {path}")
    median = float(np.median(data))
    q05, q25, q75, q95 = [float(x) for x in np.quantile(data, [0.05, 0.25, 0.75, 0.95])]
    return {
        "median": median, "mad": float(np.median(np.abs(data - median))),
        "q05": q05, "q25": q25, "q75": q75, "q95": q95,
        "iqr": q75 - q25, "pixel_count": int(len(data)),
        "map_sha256": digest(path), "map_path": str(Path(path).resolve()),
    }


def aggregate(args):
    maps = Path(args.maps)
    mask = Image.open(args.region_mask).convert("L")
    roughness = map_values(maps / "roughness.png", mask)
    metallic = map_values(maps / "metallic.png", mask)
    result = {
        "schema_version": 1,
        "material_name": "visible_orange_painted_mailbox_finish",
        "roughness": roughness["median"],
        "metallic": metallic["median"],
        "aggregation": "median_of_8bit_supermat_maps_over_reviewed_material_region",
        "roughness_distribution": roughness,
        "metallic_distribution": metallic,
        "region_mask_sha256": digest(args.region_mask),
        "albedo_map_sha256": digest(maps / "albedo.png"),
        "supermat_run": str(maps.resolve()),
        "project_git_commit": project_commit(),
        "pipeline_sha256": digest(__file__),
    }
    save_json(args.output, result)


def roundtrip(args):
    canonical = json.loads(Path(args.canonical).read_text())
    entries = []
    for name in ("soft_front", "side_directional", "top_environment"):
        maps = Path(args.maps_root) / name / "input_rgba"
        mask = Image.open(Path(args.calibration_root) / name / "object_mask.png").convert("L")
        # Exclude silhouettes and antialiasing from the scalar comparison.
        mask = mask.filter(ImageFilter.MinFilter(11))
        r = map_values(maps / "roughness.png", mask)
        m = map_values(maps / "metallic.png", mask)
        entries.append({
            "light": name,
            "roughness_hat": r["median"], "metallic_hat": m["median"],
            "roughness_delta": r["median"] - canonical["roughness"],
            "metallic_delta": m["median"] - canonical["metallic"],
            "roughness_distribution": r, "metallic_distribution": m,
            "mask_sha256": digest(Path(args.calibration_root) / name / "object_mask.png"),
        })
    save_json(args.output, {
        "schema_version": 1,
        "canonical_sha256": digest(args.canonical),
        "project_git_commit": project_commit(),
        "pipeline_sha256": digest(__file__),
        "interpretation": "SuperMat-to-Cycles-to-SuperMat transfer diagnostic; same-estimator agreement is not ground truth",
        "results": entries,
    })


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--image", required=True)
    p.add_argument("--object-mask", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--erosion-px", type=int, default=5)
    p.add_argument("--output", required=True)
    p.set_defaults(func=prepare)
    p = sub.add_parser("aggregate")
    p.add_argument("--maps", required=True)
    p.add_argument("--region-mask", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=aggregate)
    p = sub.add_parser("roundtrip")
    p.add_argument("--canonical", required=True)
    p.add_argument("--maps-root", required=True)
    p.add_argument("--calibration-root", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=roundtrip)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
