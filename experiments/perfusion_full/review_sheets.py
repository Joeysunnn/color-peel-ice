"""Create matched B0/full Perfusion sheets from immutable comparison images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def make_sheet(comparison: Path, baseline: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if json.loads((comparison / "provenance.json").read_text())["status"] != "succeeded":
        raise ValueError("full comparison is incomplete")
    output.mkdir(parents=True)
    manifest = [json.loads(line) for line in (comparison / "generation_status.jsonl").read_text().splitlines()]
    lookup = {(row["group"], row["condition"], row["seed"]): row
              for row in manifest}
    if len(lookup) != 27:
        raise ValueError("expected 27 full Perfusion images")
    for group in ("plain", "red", "blue"):
        canvas = Image.new("RGB", (4 * 256, 3 * 282), "white")
        draw = ImageDraw.Draw(canvas)
        for row_index, seed in enumerate((42, 43, 44)):
            for column, (arm, condition) in enumerate((
                    ("B0", "subject_only"), ("Full", "subject_only"),
                    ("B0", "subject_material"), ("Full", "subject_material"))):
                if arm == "B0":
                    path = baseline / "inference" / "images" / "B0" / group / (
                        f"B0__{group}__{condition}__seed{seed}.png")
                else:
                    path = Path(lookup[(group, condition, seed)]["image"])
                x, y = column * 256, row_index * 282
                if path.is_file():
                    with Image.open(path) as source:
                        canvas.paste(source.convert("RGB").resize((256, 256)), (x, y))
                else:
                    draw.text((x + 10, y + 100), "missing / filtered", fill="black")
                draw.text((x + 6, y + 260), f"{arm} {condition} seed{seed}", fill="black")
        canvas.save(output / f"{group}.png")
    material = Image.new("RGB", (3 * 256, 3 * 282), "white")
    draw = ImageDraw.Draw(material)
    for row_index, group in enumerate(("plain", "red", "blue")):
        for column, seed in enumerate((42, 43, 44)):
            path = Path(lookup[(group, "material_only", seed)]["image"])
            with Image.open(path) as source:
                material.paste(source.convert("RGB").resize((256, 256)),
                               (column * 256, row_index * 282))
            draw.text((column * 256 + 6, row_index * 282 + 260),
                      f"{group} Material seed{seed}", fill="black")
    material.save(output / "material_only.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_sheet(args.comparison, args.baseline, args.output)
