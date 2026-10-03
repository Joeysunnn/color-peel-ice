"""Prepare a documented bowl crop and masks from the supplied wood spoon image."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Image.open(args.source).convert("RGB")
    pixels = np.asarray(source)
    foreground = (pixels.min(axis=2) < 235).astype("uint8") * 255
    bbox = Image.fromarray(foreground).getbbox()
    if bbox is None:
        raise ValueError("No spoon pixels found")
    x0, y0, x1, y1 = bbox
    width, height = x1 - x0, y1 - y0
    side = round(max(width * 1.4, height * 0.38))
    cx = (x0 + x1) // 2
    left = max(0, min(source.width - side, cx - side // 2))
    top = max(0, y0 - round(side * 0.03))
    box = (left, top, left + side, top + side)
    crop = source.crop(box).resize((512, 512), Image.Resampling.LANCZOS)
    mask = Image.fromarray(foreground).crop(box).resize((512, 512), Image.Resampling.NEAREST)
    mask = mask.point(lambda value: 255 if value >= 128 else 0)
    eroded = mask.filter(ImageFilter.MinFilter(17))
    # The bowl interior excludes the silhouette, handle, and hanging hole.
    bowl = Image.new("L", (512, 512))
    ImageDraw.Draw(bowl).ellipse((178, 95, 334, 326), fill=255)
    bowl = Image.fromarray(np.minimum(np.asarray(bowl), np.asarray(eroded)))
    if (np.asarray(bowl) >= 128).sum() < 1000:
        raise ValueError("Bowl material region too small")
    crop.save(args.output / "source.png")
    mask.save(args.output / "object_mask.png")
    bowl.save(args.output / "bowl_region.png")
    eroded.save(args.output / "eroded_object_region.png")
    rgba = crop.convert("RGBA")
    rgba.putalpha(mask)
    rgba.save(args.output / "supermat_input.png")
    (args.output / "preparation.json").write_text(json.dumps({
        "source": str(args.source.resolve()), "source_sha256": sha(args.source),
        "source_size": source.size, "segmentation": "min RGB < 235; resized nearest",
        "source_foreground_bbox": bbox, "source_crop_box": box,
        "supermat_input_sha256": sha(args.output / "supermat_input.png"),
        "bowl_region_sha256": sha(args.output / "bowl_region.png"),
        "eroded_object_region_sha256": sha(args.output / "eroded_object_region.png"),
        "script_sha256": sha(__file__),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
