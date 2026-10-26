"""Display actual foreground-only conditions and compare outputs with the old scene-conditioned run."""

import argparse
import csv
from html import escape
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from experiments.sd15_token_local_kv_v1.contract import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def font():
    for name in ("DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, 16)
        except OSError:
            pass
    return ImageFont.load_default()


def panel(canvas, path, x, y, width, height, crop=None):
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        if crop:
            image = image.resize((1024, 1024), Image.Resampling.BICUBIC).crop(crop)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas.paste(image, (x + (width - image.width) // 2, y))


def inputs(output):
    """CPU-only preview before generation; the scene column is provenance, never model input."""
    output = Path(output).resolve()
    plan = read(output / "plan.json")
    colors = [item["name"] for item in plan["config"]["colors"]]
    canvas = Image.new("RGB", (1440, 100 + len(colors) * 390), "white")
    draw, face = ImageDraw.Draw(canvas), font()
    draw.multiline_text((10, 8), "Foreground-only MARBLE inputs | NO training | original scene is shown for provenance only\n"
        "Target keeps original canvas/position. Reference crop increases CLIP subject occupancy; this is a declared additional adaptation.", font=face, fill="black", spacing=5)
    for index, color in enumerate(colors):
        source = plan["sources"][color]
        y = 100 + index * 390
        columns = [(f"{color}: original scene\nNOT a model input", f"source/{color}/original_source512.png"),
                   ("Actual neutral Target\nused for DPT and initialization", source["target"]),
                   ("Actual cropped Reference\nused for CLIP / M/R MLP", source["reference"]),
                   ("Actual fixed edit Mask\nno rembg re-estimation", source["mask"])]
        for number, (label, relative) in enumerate(columns):
            x = number * 360 + 8
            draw.multiline_text((x, y), label, font=face, fill="black", spacing=4)
            panel(canvas, output / relative, x, y + 53, 344, 330)
    folder = output / "review"
    folder.mkdir(exist_ok=True)
    canvas.save(folder / "inputs.png")
    return folder / "inputs.png"


def review(output):
    output = Path(output).resolve()
    plan, generation = read(output / "plan.json"), read(output / "generation_manifest.json")
    colors = [item["name"] for item in plan["config"]["colors"]]
    rows, identities = generation["records"], generation["identity_records"]
    controls = [(m, r) for m in (-20, 20) for r in (-1, 0, 1)]
    key = lambda row: (row["color"], row["controls"]["metallic"], row["controls"]["roughness"])
    indexed = {key(row): row for row in rows}
    if (len(colors) != 5 or len(rows) != 30 or set(indexed) != {(c, m, r) for c in colors for m, r in controls}
            or len(identities) != 5 or {row["color"] for row in identities} != set(colors)):
        raise ValueError("review requires thirty complete M/R outputs and five no-MLP identities")
    by_identity = {row["color"]: row for row in identities}
    old_root = Path(plan["config"]["comparison_run"]).resolve()
    old = {key(row): row for row in read(old_root / "generation_manifest.json")["records"]}
    for root, records in ((output, rows + identities), (old_root, list(old.values()))):
        for row in records:
            if sha256(root / row["raw_image"]) != row["raw_sha256"]:
                raise ValueError("new or comparison raw image changed")
    dataset = read(output / "dataset/manifest.json")["records"]
    if len(dataset) != 35 or {row["id"] for row in dataset} != {row["id"] for row in rows} | {f"{c}__original" for c in colors}:
        raise ValueError("review must show all thirty-five actual candidates")
    for row in dataset:
        for name in ("image", "mask"):
            if sha256(output / "dataset" / row[name]) != row[name + "_sha256"]:
                raise ValueError("actual training candidate changed")
    candidates = {row["id"]: row for row in dataset}
    folder = inputs(output).parent
    face = font()
    overview = Image.new("RGB", (2240, 90 + 5 * 375), "white")
    draw = ImageDraw.Draw(overview)
    draw.multiline_text((10, 8), "35 ACTUAL candidates: five neutral-background recolors + thirty foreground-only MARBLE edits | NO training\n"
        "All captions are proposed control labels. Relative M/R are not physical PBR values; review quality before training.", font=face, fill="black", spacing=5)
    identity_sheet = Image.new("RGB", (1080, 70 + 5 * 390), "white")
    identity_draw = ImageDraw.Draw(identity_sheet)
    identity_draw.text((8, 8), "Identity diagnostics: common subject crop | no MLP still regenerates at strength1 | NO training", font=face, fill="black")
    gallery = ["<!doctype html><meta charset='utf-8'><title>Foreground-only MARBLE review</title>",
        "<h1>Foreground-only MARBLE: 30 edits + 5 base candidates</h1><p>No training. Review pending; no physical material or albedo validation is implied.</p>",
        "<p>Original scene content is excluded before model conditioning. Reference cropping also changes CLIP subject occupancy. "
        "Existing object reflections and model priors remain possible causes of unwanted patterns. The photographed mailbox is already truncated at the bottom.</p>",
        "<p><a href='inputs.png'>Actual model inputs and provenance</a> | <a href='identity_subject.png'>No-MLP identities</a> | "
        "<a href='../generation_manifest.json'>Generation manifest</a> | <a href='../dataset/manifest.json'>35-candidate manifest</a> | <a href='captions.csv'>Proposed captions</a></p>",
        "<a href='overview35.png'><img src='overview35.png' style='max-width:100%'></a>"]
    for index, color in enumerate(colors):
        source = plan["sources"][color]
        with Image.open(output / source["mask"]) as mask:
            box = mask.convert("L").getbbox()
        crop = (max(0, box[0] - 24), max(0, box[1] - 24), min(1024, box[2] + 24), min(1024, box[3] + 24))
        selected = [indexed[color, m, r] for m, r in controls]
        for column, row in enumerate([candidates[color + "__original"]] + [candidates[row["id"]] for row in selected]):
            x, y = column * 320 + 8, 90 + index * 375
            label = f"{color}: neutral base recolor" if column == 0 else f"{color}: M={controls[column - 1][0]:+d}, R={controls[column - 1][1]:+d}"
            draw.text((x, y), label, font=face, fill="black")
            panel(overview, output / "dataset" / row["image"], x, y + 35, 304, 330)
        canvas = Image.new("RGB", (1920, 1260), "white")
        detail = ImageDraw.Draw(canvas)
        detail.multiline_text((8, 8), f"{color} | actual foreground-only inputs + matched old/new subject crops | seed42 / 30 nominal steps / strength1\n"
            "Reference crop is an additional adaptation. OLD outputs used the scene during generation; their display crops do not remove that influence.", font=face, fill="black", spacing=5)
        conditions = [("Clean Target", source["target"]), ("Cropped clean Reference", source["reference"]), ("Fixed Mask", source["mask"]),
                      ("Fresh DPT, outside Mask=0", f"source/{color}/depth.png"), ("Actual grayscale initialization", f"source/{color}/initialization.png"),
                      ("No MLP identity (regenerated)", by_identity[color]["raw_image"])]
        for number, (label, relative) in enumerate(conditions):
            x = number * 320 + 8
            detail.text((x, 85), label, font=face, fill="black")
            panel(canvas, output / relative, x, 116, 304, 304)
        for line, (label, root, lookup) in enumerate((("OLD: background-conditioned output; display crop only", old_root, old),
                                                    ("NEW: foreground-only conditions; raw output crop", output, indexed))):
            y = 445 + line * 405
            detail.text((8, y), label, font=face, fill="black")
            for column, (m, r) in enumerate(controls):
                x = column * 320 + 8
                detail.text((x, y + 28), f"M={m:+d}, R={r:+d} (relative)", font=face, fill="black")
                panel(canvas, root / lookup[color, m, r]["raw_image"], x, y + 58, 304, 334, crop)
        canvas.save(folder / f"comparison_{color}.png")
        for column, (label, relative) in enumerate((("clean Target", source["target"]),
                ("gray initialization", f"source/{color}/initialization.png"), ("no-MLP raw output", by_identity[color]["raw_image"]))):
            x, y = column * 360 + 8, 70 + index * 390
            identity_draw.text((x, y), f"{color}: {label}", font=face, fill="black")
            panel(identity_sheet, output / relative, x, y + 30, 344, 350, crop)
        gallery.append(f"<h2>{escape(color)}</h2><p><a href='comparison_{color}.png'>Conditions and six matched old/new outputs</a> | "
            f"<a href='../{escape(by_identity[color]['raw_image'], quote=True)}'>No-MLP raw identity</a> | "
            f"<a href='../dataset/{escape(candidates[color + '__original']['image'], quote=True)}'>Actual neutral base candidate</a></p>")
        for row in selected:
            gallery.append(f"<p><a href='../{escape(row['raw_image'], quote=True)}'>{escape(row['id'])}: untouched raw1024</a> | "
                f"<a href='../dataset/{escape(candidates[row['id']]['image'], quote=True)}'>Actual512 candidate</a> — {escape(candidates[row['id']]['caption'])}</p>")
    overview.save(folder / "overview35.png")
    identity_sheet.save(folder / "identity_subject.png")
    keys = ("id", "color", "kind", "caption", "caption_status", "training_eligible", "image")
    with (folder / "captions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in keys} for row in dataset)
    (folder / "index.html").write_text("\n".join(gallery) + "\n", encoding="utf8")
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    print(review(args.run))
