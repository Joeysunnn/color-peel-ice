"""Review thirty independent MARBLE edits and their five actual recolor inputs."""

import argparse
import csv
from html import escape
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from experiments.sd15_token_local_kv_v1.contract import sha256
from experiments.subject_marble_color30_v1.data import verify_prepared


def review(output):
    output = Path(output).resolve()
    plan = verify_prepared(output)
    json.loads((output / "status.json").read_text(encoding="utf8"))
    rows = json.loads((output / "generation_manifest.json").read_text(encoding="utf8"))["records"]
    colors = [item["name"] for item in plan["config"]["colors"]]
    controls = [(m, r) for m in (-20, 20) for r in (-1, 0, 1)]
    indexed = {(row["color"], row["controls"]["metallic"], row["controls"]["roughness"]): row for row in rows}
    expected = {(color, m, r) for color in colors for m, r in controls}
    if (len(colors) != 5 or len(set(colors)) != 5 or len(rows) != 30
            or len({row["id"] for row in rows}) != 30 or set(indexed) != expected):
        raise ValueError("review requires all thirty unique five-color M/R outputs")

    def path(relative):
        resolved = (output / relative).resolve()
        if not resolved.is_relative_to(output):
            raise ValueError("review input must remain inside its run")
        return resolved

    for row in rows:
        if sha256(path(row["raw_image"])) != row["raw_sha256"]:
            raise ValueError("generated image hash differs")
    dataset = json.loads((output / "dataset/manifest.json").read_text(encoding="utf8"))["records"]
    if len(dataset) != 35 or {row["id"] for row in dataset} != {row["id"] for row in rows} | {f"{c}__original" for c in colors}:
        raise ValueError("review requires thirty generated and five original dataset candidates")
    for row in dataset:
        for key in ("image", "mask"):
            if sha256(path("dataset/" + row[key])) != row[key + "_sha256"]:
                raise ValueError("dataset candidate image or mask hash differs")
    candidates = {row["id"]: row for row in dataset}
    sources = plan["sources"]
    boxes = []
    for color in colors:
        source = sources[color]
        for key in ("original", "target", "reference", "mask"):
            location = path(source[key])
            if key + "_sha256" in source and sha256(location) != source[key + "_sha256"]:
                raise ValueError("source image hash differs")
        with Image.open(path(source["mask"])) as mask:
            bbox = mask.convert("L").resize((1024, 1024), Image.Resampling.NEAREST).getbbox()
        if bbox is None:
            raise ValueError("empty subject mask")
        boxes.append(bbox)
    crop = (max(0, min(box[0] for box in boxes) - 24), max(0, min(box[1] for box in boxes) - 24),
            min(1024, max(box[2] for box in boxes) + 24), min(1024, max(box[3] for box in boxes) + 24))
    folder = output / "review"
    folder.mkdir(exist_ok=True)
    font = ImageFont.load_default()
    for name in ("DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            font = ImageFont.truetype(name, 15)
            break
        except OSError:
            pass

    def panel(canvas, relative, x, y, width, height, box=None):
        with Image.open(path(relative)) as handle:
            image = handle.convert("RGB")
            if box:
                image = image.resize((1024, 1024), Image.Resampling.BICUBIC).crop(box)
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            canvas.paste(image, (x + (width - image.width) // 2, y))

    def text(draw, xy, value):
        draw.multiline_text(xy, value, font=font, fill="black", spacing=4)

    overview = Image.new("RGB", (1960, 80 + 5 * 340), "white")
    draw = ImageDraw.Draw(overview)
    text(draw, (10, 8), "MARBLE | actual35 candidate training images at512px | 30 generated + 5 original recolors | NO training\n"
         "Relative M/R controls, not physical PBR values. Original1024 outputs are linked separately; review pending.")
    caption_rows = [{key: row[key] for key in ("id", "color", "kind", "caption", "caption_status", "training_eligible")}
                    | {"image": "dataset/" + row["image"]} for row in dataset]
    gallery = ["<!doctype html><meta charset='utf-8'><title>MARBLE30 review</title>",
               "<h1>MARBLE: 30 candidates + 5 base images</h1><p>No training. Relative M/R controls; quality awaits user review.</p>",
               "<p><a href='../generation_manifest.json'>Generation manifest</a> | <a href='../plan.json'>Plan</a> | "
               "<a href='../dataset/manifest.json'>35-candidate manifest</a> | "
               "<a href='../dataset/images/'>35 candidate images</a> | <a href='captions.csv'>Proposed captions, not approved</a></p>",
               "<a href='overview35.png'><img src='overview35.png' style='max-width:100%'></a>"]
    for index, color in enumerate(colors):
        source = sources[color]
        selected = [indexed[(color, m, r)] for m, r in controls]
        y = 80 + index * 340
        text(draw, (8, y), f"{color}: original recolor\nincluded as one of five base images")
        panel(overview, "dataset/" + candidates[color + "__original"]["image"], 8, y + 56, 264, 274)
        for column, row in enumerate(selected, 1):
            m, r = row["controls"]["metallic"], row["controls"]["roughness"]
            x = column * 280 + 8
            text(draw, (x, y), f"{color}\nrelative M={m:+d}, R={r:+d}")
            panel(overview, "dataset/" + candidates[row["id"]]["image"], x, y + 56, 264, 274)
        for suffix, box in (("full", None), ("subject", crop)):
            canvas = Image.new("RGB", (1080, 1280), "white")
            detail = ImageDraw.Draw(canvas)
            text(detail, (8, 8), f"{color} | official gray initialization / strength1 | seed42 / 30steps | NO training\n"
                 "Relative controls, not physical PBR. Generated details require user review.")
            headers = [("Actual original recolor (base image)", source["original"]),
                       ("Target = self material reference", source["target"]),
                       ("Actual existing edit mask", source["mask"]),
                       ("DPT from this 1024 target", f"source/{color}/depth.png")]
            for column, (label, relative) in enumerate(headers):
                text(detail, (column * 270 + 8, 65), label)
                panel(canvas, relative, column * 270 + 8, 90, 254, 254)
            for position, row in enumerate(selected):
                x, y0 = (position % 3) * 360 + 8, 390 + (position // 3) * 440
                m, r = row["controls"]["metallic"], row["controls"]["roughness"]
                text(detail, (x, y0), f"relative M={m:+d}, R={r:+d} | raw1024" + (" / common crop" if box else ""))
                panel(canvas, row["raw_image"], x, y0 + 32, 344, 344, box)
            canvas.save(folder / f"six_{color}_{suffix}.png")
        gallery.append(f"<h2>{escape(color)}</h2><p><a href='six_{color}_full.png'>Six full images</a> | "
                       f"<a href='six_{color}_subject.png'>Six common subject crops</a> | "
                       f"<a href='../dataset/{escape(candidates[color + '__original']['image'], quote=True)}'>Actual original recolor candidate</a></p>")
        gallery.extend(f"<p><a href='../{escape(row['raw_image'], quote=True)}'>{escape(row['id'])}: raw1024</a> "
                       f"| <a href='../dataset/{escape(candidates[row['id']]['image'], quote=True)}'>512 candidate</a> "
                       f"— {escape(row['caption'])}</p>" for row in selected)
    overview.save(folder / "overview35.png")
    with (folder / "captions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(caption_rows[0]))
        writer.writeheader()
        writer.writerows(caption_rows)
    (folder / "index.html").write_text("\n".join(gallery) + "\n", encoding="utf8")
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    print(review(parser.parse_args().run))
