"""Three-arm Texture review with the actual RGB training-image references."""

import csv
import html
import itertools
import os
from pathlib import Path
import random
import textwrap

from PIL import Image, ImageDraw

from experiments.sd15_token_local_kv_v1.contract import sha256
from experiments.sd_native_comparison_v1.generate import write_json
from experiments.texture_material_disentangled_v1 import infer

CONDITIONS = ("base", "literal", "learned_texture")
LABELS = ("Native SD1.5 / no concept training", "Literal texture / no concept training",
          "Learned <T*> / training RGB below")


def review_bundle(output, rows, protocol, references):
    output = Path(output)
    (output / "contact_sheets").mkdir()
    expected = set(itertools.product(infer.METHODS, infer.target_ids(protocol, "texture")))
    if set(references) != expected:
        raise ValueError("expected only the four Texture training references")
    for key, reference in references.items():
        if ((reference["method"], reference["texture"]) != key or len(reference["images"]) != 16
                or len({item["id"] for item in reference["images"]}) != 16):
            raise ValueError("Texture training reference identity or image count differs")
        for item in reference["images"]:
            if sha256(Path(item["image"])) != item["image_sha256"]:
                raise ValueError("actual training RGB changed before review")
    by_key = {(row["method"], row["texture"], row["context"], row["seed"], row["condition"]): row
              for row in rows}
    sheets, groups, links = [], [], []
    for method, texture, seed in itertools.product(infer.METHODS, infer.target_ids(protocol, "texture"),
                                                   protocol["inference"]["seeds"]):
        reference = references[method, texture]
        metadata = ["REFERENCE FOR COLUMN 3 ONLY: all 16 actual training RGB images (not extracted maps).",
                    "Fixed texture; 2 shapes x 2 colors x 2 lights x 2 nuisance materials. No Material adapter.",
                    f"Run: {Path(reference['run']).name} / checkpoint step {reference['step']}",
                    f"Dataset manifest: {reference['dataset_manifest']}",
                    f"Manifest SHA256: {reference['dataset_manifest_sha256']}"]
        metadata = [line for text in metadata for line in textwrap.wrap(text, 112)]
        sheet = Image.new("RGB", (816, 70 + 13 * len(metadata) + 240 + 20 + 4 * 294), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f"{method} / {texture} / seed{seed} / step{rows[0]['training_steps']}", fill="black")
        for column, label in enumerate(LABELS):
            for line, text in enumerate(textwrap.wrap(label, 34)):
                draw.text((8 + 272 * column, 32 + line * 13), text, fill="black")
        y = 70
        for line in metadata:
            draw.text((8, y), line, fill="black")
            y += 13
        for index, item in enumerate(reference["images"]):
            x, top = 8 + index % 8 * 102, y + index // 8 * 116
            with Image.open(item["image"]) as image:
                sheet.paste(image.convert("RGB").resize((94, 94)), (x, top))
            draw.text((x, top + 96), f"Train {index + 1:02d}", fill="black")
        y += 240
        for column, condition in enumerate(CONDITIONS):
            draw.text((8 + column * 272, y), condition, fill="black")
        y += 20
        for line, context in enumerate(protocol["dataset"]["heldout_contexts"]):
            for column, condition in enumerate(CONDITIONS):
                row = by_key[method, texture, context["id"], seed, condition]
                x, top = 8 + column * 272, y + line * 294
                with Image.open(output / row["image_path"]) as image:
                    sheet.paste(image.convert("RGB").resize((256, 256)), (x, top))
                draw.text((x, top + 258), context["id"], fill="black")
                if row["status"] == "safety_filtered":
                    draw.text((x, top + 272), "SAFETY FILTERED", fill="red")
        path = f"contact_sheets/{method}__{texture}__seed{seed}.jpg"
        sheet.crop((0, 0, 816, y + 4 * 294)).save(output / path, quality=90)
        sheets.append(path)
    for (method, texture), reference in references.items():
        groups.append({**reference, "conditions": {"base": None, "literal": None,
                       "learned_texture": "images"}})
        links.append(f"<h2>{html.escape(method)} / {html.escape(texture)}: learned column training RGB</h2>")
        links.append(f"<p>{html.escape(str(reference['dataset_manifest']))}<br>"
                     f"SHA256: {reference['dataset_manifest_sha256']}</p>")
        for number, item in enumerate(reference["images"], 1):
            href = html.escape(os.path.relpath(item["image"], output).replace(os.sep, "/"), quote=True)
            links.append(f'<figure><a href="{href}"><img width="128" src="{href}"></a>'
                         f'<figcaption>Train {number:02d}: {html.escape(item["id"])}<br>'
                         f'{html.escape(item.get("prompt", ""))}<br>{item["image_sha256"]}</figcaption></figure>')
    write_json(output / "training_references.json", {"groups": groups,
               "native_columns": "Base and literal use native SD1.5; no concept-specific training or adapters."})
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
    (output / "blinded_review.html").write_text('<!doctype html><meta charset="utf-8"><title>Texture review</title>'
        '<p>Rate visible texture and material in blinded_ratings.csv before opening blinded_key.json.</p>'
        + "".join(cards), encoding="utf-8")
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Texture probe</title>'
        '<p>Columns: native SD1.5, literal texture, learned T. No material prompts or Material adapter.</p>'
        '<p>Training references belong only to the learned column. No automatic disentanglement score.</p>'
        '<a href="blinded_review.html">Blinded review</a>' + "".join(
            f'<p><a href="{path}">{path}</a></p>' for path in sheets) + "".join(links), encoding="utf-8")
    return sheets
