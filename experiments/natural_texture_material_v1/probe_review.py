"""Natural three-arm review with source photos and every actual RGB training image."""

import csv
import html
import itertools
import os
from pathlib import Path
import random
import textwrap

from PIL import Image, ImageDraw, ImageFont

from experiments.natural_texture_material_v1.dataset_protocol import SOURCES
from experiments.sd15_token_local_kv_v1.contract import sha256
from experiments.sd_native_comparison_v1.generate import write_json

METHODS = ("fullrank", "lora_r64")
BRANCHES = ("texture", "material")
WIDTH = 1152


def _font():
    for name in ("DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, 14)
        except OSError:
            pass
    return ImageFont.load_default()


def review_bundle(output, rows, protocol, references, contexts=None):
    output = Path(output)
    contexts = protocol["dataset"]["heldout_contexts"] if contexts is None else contexts
    expected = {(method, source + "_" + branch)
                for method, branch, source in itertools.product(METHODS, BRANCHES, SOURCES)}
    if set(references) != expected:
        raise ValueError("expected twelve natural Texture/Material training references")
    for (method, concept), reference in references.items():
        count = 16 if reference["branch"] == "texture" else 8
        if (reference["method"] != method or reference["concept_id"] != concept
                or concept != reference["source_id"] + "_" + reference["branch"]
                or len(reference["images"]) != count
                or len({item["id"] for item in reference["images"]}) != count):
            raise ValueError("actual training reference identity or image count differs")
        for item in reference["images"]:
            if sha256(Path(item["image"])) != item["image_sha256"]:
                raise ValueError("actual training RGB changed before review")
    by_key = {(row["method"], row["concept_id"], row["context"], row["seed"], row["condition"]): row
              for row in rows}
    if len(rows) != 288 or len(by_key) != 288:
        raise ValueError("expected 288 distinct three-arm natural probe rows")
    sheets, groups, links = [], [], []
    (output / "contact_sheets").mkdir()
    font = _font()
    for method, branch, source, seed in itertools.product(METHODS, BRANCHES, SOURCES,
                                                          protocol["inference"]["seeds"]):
        concept = source + "_" + branch
        reference = references[method, concept]
        learned = "learned_" + branch
        conditions = ("base", "literal", learned)
        token, opposite = ("T", "Material") if branch == "texture" else ("M", "Texture")
        metadata = [f"{method} / {source} / {branch} / seed {seed} / checkpoint step {reference['step']}",
                    "Columns 1 and 2: native SD1.5, no concept-specific fine-tuning or training references.",
                    f"Column 3 only: learned <{token}*>; all {len(reference['images'])} ACTUAL RGB TRAINING IMAGES below.",
                    f"No {opposite} prompt or {opposite} adapter. Source photo at left is not a training render.",
                    ("Fixed extracted texture; varied shape, color, material and lighting." if branch == "texture"
                     else f"Fixed R={reference['source']['roughness']:.6f}, M={reference['source']['metallic']:.6f}; "
                          "uniform albedo, no texture; varied shape, color and lighting."),
                    f"Training run: {reference['run']}", f"Checkpoint: {reference['checkpoint']}",
                    f"Dataset manifest: {reference['dataset_manifest']}",
                    "PNDM / 100 sampling steps / CFG 3.5 / 512 px / fp16 / safety checker enabled"]
        metadata = [line for value in metadata for line in textwrap.wrap(value, 139)]
        thumb_rows = (len(reference["images"]) + 7) // 8
        header_height = 12 + len(metadata) * 19
        grid_top = header_height + thumb_rows * 133 + 38
        sheet = Image.new("RGB", (WIDTH, grid_top + len(contexts) * 466), "white")
        draw = ImageDraw.Draw(sheet)
        for number, line in enumerate(metadata):
            draw.text((10, 8 + number * 19), line, font=font, fill="black")
        with Image.open(reference["source"]["input"]) as image:
            rgba = image.convert("RGBA")
            image = Image.alpha_composite(Image.new("RGBA", rgba.size, (128, 128, 128, 255)), rgba).convert("RGB")
            image.thumbnail((122, 122))
            sheet.paste(image, (10, header_height))
        draw.text((10, header_height + 124), "Source photo", font=font, fill="black")
        for index, item in enumerate(reference["images"]):
            x, top = 144 + index % 8 * 126, header_height + index // 8 * 133
            with Image.open(item["image"]) as image:
                image = image.convert("RGB")
                image.thumbnail((118, 112))
                sheet.paste(image, (x, top))
            draw.text((x, top + 113), f"Train {index + 1:02d}", font=font, fill="black")
        labels = ("Native SD1.5", "Literal " + branch, "Learned <" + token + "*>")
        for column, label in enumerate(labels):
            draw.text((10 + column * 384, grid_top - 26), label, font=font, fill="black")
        for line, context in enumerate(contexts):
            for column, condition in enumerate(conditions):
                row = by_key[method, concept, context["id"], seed, condition]
                x, top = 10 + column * 384, grid_top + line * 466
                with Image.open(output / row["image_path"]) as image:
                    sheet.paste(image.convert("RGB").resize((360, 360)), (x, top))
                draw.text((x, top + 364), context["id"], font=font, fill="black")
                for number, prompt in enumerate(textwrap.wrap(row["prompt"], 44)):
                    draw.text((x, top + 383 + number * 18), prompt, font=font, fill="black")
                if row["status"] == "safety_filtered":
                    draw.text((x, top + 444), "SAFETY FILTERED", font=font, fill="red")
        path = f"contact_sheets/{method}__{concept}__seed{seed}.jpg"
        sheet.save(output / path, quality=90)
        sheets.append(path)
    for (method, concept), reference in references.items():
        groups.append({**reference, "conditions": {"base": None, "literal": None,
                       "learned_" + reference["branch"]: "images"}})
        links.append(f"<h2>{html.escape(method)} / {html.escape(concept)}: learned column training RGB</h2>"
                     f"<p>{html.escape(str(reference['dataset_manifest']))}</p>")
        for number, item in enumerate(reference["images"], 1):
            href = html.escape(os.path.relpath(item["image"], output).replace(os.sep, "/"), quote=True)
            links.append(f'<figure><a href="{href}"><img width="128" src="{href}"></a>'
                         f'<figcaption>Train {number:02d}: {html.escape(item["id"])}<br>'
                         f'{html.escape(item.get("prompt", item.get("caption", "")))}<br>'
                         f'{html.escape(item["image"])}<br>{item["image_sha256"]}</figcaption></figure>')
    write_json(output / "training_references.json", {"groups": groups,
               "native_columns": "Native and literal: no concept-specific training; foundation pretraining is not enumerated."})
    shuffled = list(rows)
    random.Random(protocol["seed"]).shuffle(shuffled)
    key, cards = [], []
    with (output / "blinded_ratings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["blind_id", "texture_pattern", "material_appearance",
                                                    "shape", "color", "lighting", "background", "notes"])
        writer.writeheader()
        for index, row in enumerate(shuffled):
            blind_id = f"sample_{index:04d}"
            key.append({"blind_id": blind_id, **row})
            if row["status"] != "safety_filtered":
                writer.writerow({"blind_id": blind_id})
                cards.append(f'<figure><img width="256" src="{html.escape(row["image_path"], quote=True)}">'
                             f'<figcaption>{blind_id}</figcaption></figure>')
    write_json(output / "blinded_key.json", {"samples": key})
    (output / "blinded_review.html").write_text('<!doctype html><meta charset="utf-8"><title>Blinded review</title>'
        '<p>Rate visible attributes in blinded_ratings.csv before opening blinded_key.json. '
        'Safety-filtered samples are excluded; no training references are displayed.</p>' + "".join(cards), encoding="utf-8")
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Natural T/M probe</title>'
        '<p>Columns: native SD1.5, literal attribute, learned attribute. Texture has no Material prompt/adapter; '
        'Material has no Texture prompt/adapter. Training RGB references apply only to the learned column.</p>'
        '<p>No automatic disentanglement score or accuracy claim.</p><a href="blinded_review.html">Blinded review</a>'
        + "".join(f'<p><a href="{path}">{path}</a></p>' for path in sheets) + "".join(links), encoding="utf-8")
    return sheets
