"""Prepare a reproducible real-photo polished-metal positive control."""

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "assets" / "stainless_steel_soup_spoon_original.jpg"
OUT = HERE / "assets"
CROP = (900, 100, 2100, 1300)
SIZE = (512, 512)
OBJECT_POLYGON = [
    (302, 53), (330, 57), (354, 67), (376, 87), (392, 112),
    (402, 139), (408, 171), (411, 214), (410, 263), (402, 306),
    (389, 346), (370, 380), (345, 407), (314, 430), (275, 447),
    (235, 459), (204, 468), (179, 480), (163, 482), (145, 477),
    (138, 467), (143, 447), (139, 427), (128, 405), (121, 382),
    (114, 350), (110, 314), (111, 278), (119, 243), (132, 208),
    (149, 178), (171, 148), (193, 120), (221, 94), (246, 73),
    (272, 60),
]
REGION_POLYGON = [
    (295, 89), (331, 98), (358, 119), (375, 151), (381, 190),
    (380, 239), (371, 286), (352, 326), (325, 362), (290, 389),
    (247, 410), (207, 414), (173, 393), (150, 359), (139, 318),
    (141, 277), (152, 236), (172, 195), (197, 159), (228, 125),
    (261, 101),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask(points):
    image = Image.new("L", SIZE, 0)
    ImageDraw.Draw(image).polygon(points, fill=255)
    return image


def main():
    source = Image.open(SOURCE).convert("RGB")
    if source.size != (2448, 3264):
        raise ValueError(f"Unexpected Commons source size: {source.size}")
    crop = source.crop(CROP).resize(SIZE, Image.Resampling.LANCZOS)
    object_mask = mask(OBJECT_POLYGON)
    region_mask = mask(REGION_POLYGON)
    crop.save(OUT / "spoon_bowl_crop512.png")
    object_mask.save(OUT / "object_mask.png")
    region_mask.save(OUT / "material_region.png")
    rgba = crop.convert("RGBA")
    rgba.putalpha(object_mask)
    rgba.save(OUT / "input_rgba.png")
    overlay = crop.copy().convert("RGBA")
    tint = Image.new("RGBA", SIZE, (255, 0, 0, 75))
    overlay.alpha_composite(Image.composite(tint, Image.new("RGBA", SIZE), region_mask))
    overlay.save(OUT / "region_overlay.png")
    files = [SOURCE, OUT / "spoon_bowl_crop512.png", OUT / "object_mask.png",
             OUT / "material_region.png", OUT / "input_rgba.png"]
    provenance = {
        "source_page": "https://commons.wikimedia.org/wiki/File:Stainless_Steel_Soup_Spoon.jpg",
        "source_download": "https://upload.wikimedia.org/wikipedia/commons/d/dd/Stainless_Steel_Soup_Spoon.jpg",
        "author": "Paolomarco",
        "source_date": "2017-03-10",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "description": "Real photograph of a bare stainless-steel soup spoon; inner bowl is the measured material region.",
        "source_dimensions": list(source.size),
        "crop_xyxy": list(CROP),
        "output_dimensions": list(SIZE),
        "object_polygon_xy": OBJECT_POLYGON,
        "material_region_polygon_xy": REGION_POLYGON,
        "files_sha256": {p.name: sha256(p) for p in files},
    }
    (HERE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
