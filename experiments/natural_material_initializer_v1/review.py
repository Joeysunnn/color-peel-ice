"""Wood Material initializer comparison with all actual RGB training references."""

import html
import itertools
import os
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw

from experiments.natural_texture_material_v1.probe_review import _font
from experiments.sd15_token_local_kv_v1.contract import sha256
from experiments.sd_native_comparison_v1.generate import write_json

METHODS = ("fullrank", "lora_r64")
CONTEXTS = ("chair", "mug", "vase", "sofa")
CONDITIONS = ("base", "literal", "learned_metal_init", "learned_material_init")
CONCEPT = "wood_spoon_material"


def review_bundle(output, rows, references):
    output = Path(output)
    if set(references) != set(METHODS):
        raise ValueError("expected both Wood Material training references")
    for method, reference in references.items():
        images = reference["images"]
        if (reference["method"] != method or reference["concept_id"] != CONCEPT
                or reference["branch"] != "material" or reference["source_id"] != "wood_spoon"
                or len(images) != 8 or len({item["id"] for item in images}) != 8
                or len({item["image"] for item in images}) != 8):
            raise ValueError("actual training reference identity or image count differs")
        for item in images:
            if sha256(Path(item["image"])) != item["image_sha256"]:
                raise ValueError("actual training RGB changed before review")
            with Image.open(item["image"]) as image:
                if image.mode != "RGB":
                    raise ValueError("actual training image must be RGB")
    first, second = (references[method] for method in METHODS)
    if [(item["image"], item["image_sha256"]) for item in first["images"]] != [
            (item["image"], item["image_sha256"]) for item in second["images"]]:
        raise ValueError("both methods must reference the same eight training RGB images")
    by_key = {(row["method"], row["concept_id"], row["context"], row["seed"], row["condition"]): row
              for row in rows}
    expected = set(itertools.product(METHODS, (CONCEPT,), CONTEXTS, (42, 43), CONDITIONS))
    if len(rows) != 64 or set(by_key) != expected:
        raise ValueError("expected 64 distinct Wood Material initializer transfer rows")
    (output / "contact_sheets").mkdir()
    font, sheets, groups, links = _font(), [], [], []
    for method, seed in itertools.product(METHODS, (42, 43)):
        reference = references[method]
        metadata = [f"{method} / Wood Material initializer transfer / step {reference['step']} / seed {seed}",
            "Columns: native SD1.5 / literal wood / learned <M*> initialized with metal / learned <M*> initialized with material.",
            "Both learned columns use the SAME eight ACTUAL RGB TRAINING IMAGES below. Source photo is not a training render.",
            "Native and literal columns: no concept-specific training. No Texture prompt or Texture adapter.",
            f"Fixed R={reference['source']['roughness']:.6f}, M={reference['source']['metallic']:.6f}; uniform albedo, no texture.",
            f"Old metal-init run: {reference['baseline_run']}",
            f"Old metal-init checkpoint: {reference['baseline_checkpoint']}",
            f"New material-init run: {reference['run']}",
            f"New material-init checkpoint: {reference['checkpoint']}",
            f"Actual training dataset manifest: {reference['dataset_manifest']}",
            "PNDM / 100 sampling steps / CFG 3.5 / 512 px / fp16 / safety checker enabled"]
        metadata = [line for value in metadata for line in textwrap.wrap(value, 185)]
        header = 12 + len(metadata) * 19
        grid_top = header + 170
        sheet = Image.new("RGB", (1536, grid_top + len(CONTEXTS) * 466), "white")
        draw = ImageDraw.Draw(sheet)
        for number, line in enumerate(metadata):
            draw.text((10, 8 + number * 19), line, font=font, fill="black")
        with Image.open(reference["source"]["input"]) as image:
            rgba = image.convert("RGBA")
            image = Image.alpha_composite(Image.new("RGBA", rgba.size, (128, 128, 128, 255)), rgba).convert("RGB")
            image.thumbnail((128, 128))
            sheet.paste(image, (10, header))
        draw.text((10, header + 130), "Source photo", font=font, fill="black")
        for index, item in enumerate(reference["images"]):
            x = 154 + index * 140
            with Image.open(item["image"]) as image:
                image = image.copy()
                image.thumbnail((128, 128))
                sheet.paste(image, (x, header))
            draw.text((x, header + 130), f"Train {index + 1:02d}", font=font, fill="black")
        for column, label in enumerate(("Native SD1.5", "Literal wood", "Learned: metal init", "Learned: material init")):
            draw.text((10 + column * 384, grid_top - 24), label, font=font, fill="black")
        for line, context in enumerate(CONTEXTS):
            for column, condition in enumerate(CONDITIONS):
                row = by_key[method, CONCEPT, context, seed, condition]
                x, top = 10 + column * 384, grid_top + line * 466
                with Image.open(output / row["image_path"]) as image:
                    sheet.paste(image.convert("RGB").resize((360, 360)), (x, top))
                draw.text((x, top + 364), context, font=font, fill="black")
                for number, prompt in enumerate(textwrap.wrap(row["prompt"], 44)):
                    draw.text((x, top + 383 + number * 18), prompt, font=font, fill="black")
                if row["status"] == "safety_filtered":
                    draw.text((x, top + 444), "SAFETY FILTERED", font=font, fill="red")
        path = f"contact_sheets/{method}__{CONCEPT}__seed{seed}.jpg"
        sheet.save(output / path, quality=90)
        sheets.append(path)
    for method, reference in references.items():
        groups.append({**reference, "conditions": {condition: ("images" if condition.startswith("learned_") else None)
                                                  for condition in CONDITIONS}})
        links.append(f"<h2>{html.escape(method)}: shared training RGB for both learned columns</h2>")
        for number, item in enumerate(reference["images"], 1):
            href = html.escape(os.path.relpath(item["image"], output).replace(os.sep, "/"), quote=True)
            links.append(f'<figure><a href="{href}"><img width="128" src="{href}"></a>'
                         f'<figcaption>Train {number:02d}: {html.escape(item["id"])}<br>'
                         f'{html.escape(item.get("prompt", item.get("caption", "")))}<br>'
                         f'{html.escape(item["image"])}<br>{item["image_sha256"]}</figcaption></figure>')
    write_json(output / "training_references.json", {"groups": groups})
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Wood Material initializer transfer</title>'
        '<p>Columns: native SD1.5, literal wood, metal initialization, material initialization. '
        'Both learned columns share the eight actual training RGB images shown. Source photo is separate. '
        'Native and literal have no concept-specific training. No Texture prompt or adapter. '
        'No automatic disentanglement score or accuracy claim.</p>'
        + "".join(f'<p><a href="{path}">{path}</a></p>' for path in sheets) + "".join(links), encoding="utf-8")
    return sheets
