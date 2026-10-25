"""Comparison sheets from raw ZeST outputs and estimated quality metrics."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from experiments.zest_material_counterfactual_v1.run import read, load_plan
from experiments.sd15_token_local_kv_v1.contract import sha256


def preview(path, box=None):
    with Image.open(path) as image:
        image = image.convert("RGB")
        return image.crop(box) if box else image.copy()


def review(output):
    output = Path(output).resolve()
    plan = load_plan(output)
    manifest = read(output / "generation_manifest.json")
    generated = manifest["records"]
    label = "ZeST focused-reference diagnostic" if manifest.get("source_run") else "ZeST"
    scores = {}
    if (output / "quality/latest.json").exists():
        latest = read(output / "quality/latest.json")
        scores = {r["id"]: r for r in read(output / latest["report"])["records"]}
    folder = output / "reviews"
    folder.mkdir(exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
        small = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font, small = ImageFont.load_default(), ImageFont.load_default()
    for subject in plan["subjects"]:
        source = subject["source_files"]
        original = output / source["target"]
        with Image.open(output / source["object_mask"]) as mask:
            bbox = mask.getbbox()
        crop = (max(0, 2*bbox[0]-16), max(0, 2*bbox[1]-16), min(1024, 2*bbox[2]+16), min(1024, 2*bbox[3]+16))
        for stage in ("identity", "grid"):
            for variant in plan["variants"]:
                selected = [row for row in generated if row["subject_id"] == subject["id"] and row["stage"] == stage and row["variant"] == variant]
                if not selected:
                    continue
                width, height = 1080, 300
                sheet = Image.new("RGB", (width, 80 + height*len(selected)), "white")
                draw = ImageDraw.Draw(sheet)
                draw.text((10, 8), f"{label} {subject['id']} | {stage} / {variant} | seed42 | raw outputs | NO training", font=font, fill="black")
                for index, label in enumerate(("Original (same fixed crop)", "Actual material reference", "Generated (same fixed crop)", "Estimated QC; not ground truth")):
                    draw.text((index*270+8, 43), label, font=small, fill="black")
                for index, row in enumerate(selected):
                    y = 80 + index*height
                    reference = Path(row["reference_image"])
                    if not reference.is_absolute():
                        reference = output / reference
                    for column, path, box in ((0, original, crop), (1, reference, None), (2, output/row["raw_image"], crop)):
                        image = preview(path, box)
                        image.thumbnail((254, 240), Image.Resampling.LANCZOS)
                        sheet.paste(image, (column*270+8, y+25))
                    draw.text((8, y+3), row["reference_id"], font=small, fill="black")
                    score = scores.get(row["id"], {})
                    metrics = score.get("metrics", {})
                    material = metrics.get("estimated_material", {})
                    def value(key):
                        found = metrics.get(key)
                        return "n/a" if found is None else f"{found:.3f}"
                    target = row["reference_target"]
                    lines = [score.get("status", "not_assessed"), "Self-reference control" if target is None else
                             f"Reference M/R: {target['metallic']:.3f}/{target['roughness']:.3f}"]
                    for channel in ("metallic", "roughness"):
                        estimate = material.get(channel, {}).get("generated", {}).get("median")
                        lines.append(f"estimated {channel}: " + ("n/a" if estimate is None else f"{estimate:.3f}"))
                    lines += ["A DeltaE76 p95: " + value("albedo_deltaE76_p95"),
                              "A edge precision: " + value("albedo_edge_precision"),
                              "A edge recall: " + value("albedo_edge_recall"),
                              "depth error: " + value("aligned_depth_error"),
                              "background MAE: " + value("background_mae255"),
                              "Eligibility: dataset_manifest.json"]
                    for line_index, line in enumerate(lines):
                        # Keep potentially long dictionaries in the JSON report; labels fit their column.
                        draw.text((818, y+8+line_index*22), line[:33], font=small, fill="black")
                sheet.save(folder / f"{subject['id']}__{stage}__{variant}.png")
        comparisons = read(output / "config.json").get("comparisons", [])
        panels = [("Original mailbox", original)]
        for row in comparisons:
            if sha256(row["path"]) != row["sha256"]:
                raise ValueError("frozen MARBLE comparison changed")
            panels.append((row["label"], Path(row["path"])))
        panels += [(f"ZeST {row['variant']} / {row['reference_id']}", output/row["raw_image"])
                   for row in generated if row["stage"] == "identity" and row["subject_id"] == subject["id"]]
        sheet = Image.new("RGB", (1440, 65+330*((len(panels)+2)//3)), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), "MARBLE / ZeST diagnostics | settings differ; NOT an isolated architecture ablation", font=font, fill="black")
        for index, (label, path) in enumerate(panels):
            x, y = index%3*480+8, index//3*330+50
            image = preview(path, crop)
            image.thumbnail((460, 285), Image.Resampling.LANCZOS)
            draw.text((x, y), label, font=small, fill="black")
            sheet.paste(image, (x, y+28))
        sheet.save(folder / f"{subject['id']}__marble_comparison.png")
    if not (output / "bank/manifest.json").exists():
        return folder
    bank = read(output / "bank/manifest.json")
    sheet = Image.new("RGB", (1080, 90+360*4), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), "PBR reference bank | fixed geometry / Albedo / light / camera | ONLY M/R varies", fill="black", font=font)
    for index, row in enumerate(bank["records"]):
        x, y = index%3*360+8, index//3*360+60
        image = preview(row["image"])
        image.thumbnail((330, 300), Image.Resampling.LANCZOS)
        sheet.paste(image, (x, y+30))
        draw.text((x, y), f"M={row['metallic']:.3f} R={row['roughness']:.3f}", fill="black", font=font)
    sheet.save(folder / "reference_bank.png")
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(review(args.run))
