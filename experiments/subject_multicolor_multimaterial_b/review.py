"""Show the exact ten training RGB images and their surface-conditioned captions."""

import csv
import html
import json
import os
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw

from experiments.natural_texture_material_v1.probe_review import _font
from experiments.subject_multicolor_multimaterial_b.data import COLORS, MATERIALS, verify_dataset


def write_review(dataset, output):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    audit = verify_dataset(dataset)
    if output.exists() or output.is_relative_to(dataset):
        raise ValueError("Use a new review directory outside the training dataset")
    rows = [json.loads(line) for line in (dataset / "training_assets_manifest.jsonl").read_text().splitlines()]
    by_key = {(row["color"], row["material"]): row for row in rows}
    output.mkdir(parents=True)
    board = Image.new("RGB", (1920, 1110), "white")
    draw, font = ImageDraw.Draw(board), _font()
    headers = ["Experiment B: ACTUAL TRAINING IMAGES / five colors x metallic-matte / DATA PREVIEW ONLY",
        "Same RGB pixels and masks as reviewed source; red -> pink and magenta -> purple are label corrections only.",
        "Current SD1.5 balanced baseline already uses these ten RGB images; this is a surface-caption ablation.",
        "Matte images use the previously reviewed generated counterfactual; foreground detail is not a physical ground truth."]
    for i, line in enumerate(headers):
        draw.text((12, 8 + i * 19), line, font=font, fill="black")
    for col, color in enumerate(COLORS):
        for row_index, material in enumerate(MATERIALS):
            row = by_key[color, material]
            x, y = col * 384, 100 + row_index * 500
            draw.text((x + 8, y), f"{color} / {material} (source: {row['source_color']})", font=font, fill="black")
            with Image.open(row["image"]) as image:
                board.paste(image.resize((376, 376)), (x + 4, y + 24))
            for number, line in enumerate(textwrap.wrap(row["caption"], width=49)):
                draw.text((x + 8, y + 406 + number * 19), line, font=font, fill="black")
            draw.text((x + 8, y + 470), "RGB SHA256: " + row["image_sha256"][:20], font=font, fill="black")
    board.save(output / "training_pairs.jpg", quality=95)
    columns = ("id", "source_color", "color", "material", "caption", "image", "image_sha256", "mask", "mask_sha256")
    with (output / "training_captions.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    parts = ['<!doctype html><meta charset="utf-8"><title>Experiment B data review</title>',
        '<style>body{max-width:1500px;margin:30px auto;padding:0 20px;font:16px/1.6 system-ui}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:12px;vertical-align:top}code{overflow-wrap:anywhere}</style>',
        '<h1>Experiment B: Subject data and Caption review</h1>',
        '<p>Data preparation only. No training, checkpoint or generated inference grid exists in this run.</p>',
        '<p>All ten RGB images are byte-identical to reviewed originals. New assets use pink/purple labels; source IDs remain in provenance. Each image has exactly one matching caption and repaired binary mask.</p>',
        '<p>The existing SD1.5 Subject baseline already uses the same color x material RGB pairs. This study replaces glossy painted metal / matte plastic wording with symmetric metallic / matte surface captions, retaining color words. It is a caption ablation.</p>',
        '<p>Matte is a reviewed generated counterfactual; background pixels and mask are matched, while exact foreground geometry or BRDF is not established.</p>',
        '<p><a href="training_pairs.jpg">Open full contact sheet</a> | <a href="training_captions.csv">Ten actual training captions</a></p>',
        '<img src="training_pairs.jpg" alt="Actual five color by two material training images">',
        '<table><tr><th>Color</th><th>Metallic</th><th>Matte</th></tr>']
    for color in COLORS:
        parts.append(f'<tr><th>{html.escape(color)}</th>')
        for material in MATERIALS:
            row = by_key[color, material]
            href = html.escape(os.path.relpath(row["image"], output).replace(os.sep, "/"), quote=True)
            parts.append(f'<td><a href="{href}"><img width="256" src="{href}"></a><p>{html.escape(row["caption"])}</p>'
                f'<small>Original label: {html.escape(row["source_color"])}</small><p><code>{row["image_sha256"]}</code></p></td>')
        parts.append('</tr>')
    (output / "index.html").write_text("\n".join(parts) + '</table>', encoding="utf8")
    return {"contact_sheet": str(output / "training_pairs.jpg"), "captions": str(output / "training_captions.csv"),
            "index": str(output / "index.html"), "dataset_audit": audit}
