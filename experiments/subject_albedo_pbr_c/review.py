"""Show extracted maps and proposed parameters without implying RGB renders."""

import html
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


def _font(size=16):
    for name in ("DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _path(output, value):
    path = Path(value)
    return path if path.is_absolute() else output / path


def _thumbnail(sheet, path, box):
    left, top, width, height = box
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        sheet.paste(image, (left + (width - image.width) // 2,
                            top + (height - image.height) // 2))


def _text(draw, position, text, width, font, fill="black"):
    x, y = position
    for line in textwrap.wrap(str(text), width):
        draw.text((x, y), line, font=font, fill=fill)
        y += 22


def write_review(output, rows):
    """Write three map-review PNGs and an HTML index into an existing run."""
    output = Path(output)
    colors = list(dict.fromkeys(row["color"] for row in rows))
    groups = {color: [row for row in rows if row["color"] == color] for color in colors}
    if (len(rows) != 30 or len(colors) != 5 or any(len(group) != 6 for group in groups.values())
            or any(row.get("render_status") != "not_rendered"
                   or row.get("training_eligible") is not False
                   or row.get("rgb_image") is not None for row in rows)):
        raise ValueError("review requires thirty unrendered, training-ineligible parameter proposals")
    combinations = {(row["metallic_state"], row["roughness_state"]) for row in rows}
    if (len(combinations) != 6
            or len({row["metallic_state"] for row in rows}) != 2
            or len({row["roughness_state"] for row in rows}) != 3
            or any({(row["metallic_state"], row["roughness_state"]) for row in group} != combinations
                   for group in groups.values())
            or any(len({row["maps"]["albedo"]["sha256"] for row in group}) != 1
                   for group in groups.values())):
        raise ValueError("review requires the complete five-color by two-metallic by three-roughness grid")
    font, title = _font(16), _font(21)
    source = output / "source"

    extraction = Image.new("RGB", (1400, 425), "white")
    draw = ImageDraw.Draw(extraction)
    draw.text((12, 8), "Experiment C: source photograph and extracted maps", font=title, fill="black")
    draw.text((12, 40), "A/M/R are SuperMat estimates, not measured ground truth. Maps are not RGB counterfactual renders.",
              font=font, fill="black")
    with Image.open(source / "object_mask.png") as image:
        mask = image.convert("L")
    with Image.open(source / "input.png") as image:
        image = image.convert("RGB").resize(mask.size, Image.Resampling.LANCZOS)
    overlay = Image.composite(Image.new("RGB", image.size, (240, 30, 30)), image,
                              mask.point(lambda value: round(value * 0.3)))
    panels = [("Source photograph", "original.jpg"), ("Input with red object-mask overlay", None),
              ("Extracted Albedo (A)", "albedo.png"), ("Extracted Metallic (M)", "metallic.png"),
              ("Extracted Roughness (R)", "roughness.png")]
    for column, (label, name) in enumerate(panels):
        x = column * 280 + 10
        _text(draw, (x, 76), label, 27, font)
        if name is None:
            overlay.thumbnail((260, 260), Image.Resampling.LANCZOS)
            extraction.paste(overlay, (x + (260 - overlay.width) // 2, 125))
        else:
            _thumbnail(extraction, source / name, (x, 125, 260, 260))
    extraction.save(output / "extraction_grid.png")

    albedo = Image.new("RGB", (1600, 475), "white")
    draw = ImageDraw.Draw(albedo)
    draw.text((12, 8), "Five color variants: ALBEDO MAPS, NOT RGB RENDERS", font=title, fill="black")
    draw.text((12, 40), "Hue-edited spatial reflectance estimates only; no geometry, lighting or BRDF rendering is implied.",
              font=font, fill="black")
    for column, color in enumerate(colors):
        row = groups[color][0]
        x = column * 320 + 10
        draw.text((x, 82), f"{color}; hue {row['hue_degrees']:g} deg", font=font, fill="black")
        _thumbnail(albedo, _path(output, row["maps"]["albedo"]["path"]), (x, 115, 300, 300))
        draw.text((x, 435), "ALBEDO / not a training RGB image", font=font, fill="darkred")
    albedo.save(output / "albedo_grid.png")

    parameters = Image.new("RGB", (1440, 1525), "white")
    draw = ImageDraw.Draw(parameters)
    draw.text((12, 8), "30 PARAMETER PROPOSALS - NOT RENDERED - NOT ELIGIBLE FOR TRAINING", font=title, fill="darkred")
    draw.text((12, 43), "Five colors x two metallic values x three roughness values. Thumbnails show Albedo; swatches preview M/R.",
              font=font, fill="black")
    draw.text((12, 72), "Original-subject geometry/normal, camera alignment and lighting inputs are missing; no RGB result exists.",
              font=font, fill="black")
    ordered = sorted(groups[colors[0]], key=lambda row: (row["metallic"], row["roughness"]))
    keys = [(row["metallic_state"], row["roughness_state"]) for row in ordered]
    for line, color in enumerate(colors):
        by_key = {(row["metallic_state"], row["roughness_state"]): row for row in groups[color]}
        for column, key in enumerate(keys):
            row = by_key[key]
            x, y = column * 240 + 8, 112 + line * 280
            draw.rectangle((x - 3, y - 3, x + 227, y + 271), outline="#bbbbbb")
            draw.text((x, y), str(color), font=font, fill="black")
            draw.text((x, y + 24), f"M={row['metallic']:.3f} / R={row['roughness']:.3f}", font=font, fill="black")
            _thumbnail(parameters, _path(output, row["maps"]["albedo"]["path"]), (x, y + 50, 220, 145))
            for number, channel in enumerate(("metallic", "roughness")):
                offset = number * 112
                preview = _path(output, row["maps"][channel]["preview"])
                with Image.open(preview) as image:
                    parameters.paste(image.convert("RGB").resize((100, 24)), (x + offset, y + 200))
                draw.text((x + offset, y + 228), "M preview" if number == 0 else "R preview", font=font, fill="black")
            draw.text((x, y + 249), "Not rendered / no training", font=font, fill="darkred")
    parameters.save(output / "parameter_grid.png")

    names = ["extraction_grid.png", "albedo_grid.png", "parameter_grid.png"]
    links = [("parameter_manifest.jsonl", "All 30 proposed captions, parameter values and map hashes"),
             ("protocol.json", "Protocol"), ("source_locks.json", "Immutable source hashes"),
             ("renderer_readiness.json", "Renderer readiness and missing inputs"), ("status.json", "Run status")]
    page = ['<!doctype html><meta charset="utf-8"><title>Experiment C: map preparation only</title>',
            '<style>body{font:16px sans-serif;max-width:1500px;margin:24px auto;padding:0 12px}'
            'img{max-width:100%;height:auto}td,th{border:1px solid #bbb;padding:8px}table{border-collapse:collapse}</style>',
            '<h1>Experiment C: extracted maps and proposed PBR parameters</h1>',
            '<p><strong>No counterfactual RGB images have been rendered. None of these 30 proposals is eligible for training.</strong></p>',
            '<p>The source photograph is an actual input. Albedo, metallic and roughness panels are maps, '
            'and the parameter grid is a review of map combinations. An Albedo map is not a relit RGB photograph.</p>',
            '<p>SuperMat A/R/M alone does not provide original-mailbox geometry, reliable surface normals, '
            'camera alignment or scene lighting. These inputs and their surface correspondence must be supplied '
            'and checked before claiming original-subject PBR rendering. A generic sphere or cube is not the original mailbox.</p>',
            '<ul>' + ''.join(f'<li><a href="{name}">{html.escape(label)}</a></li>' for name, label in links) + '</ul>']
    page.extend(f'<h2>{html.escape(name)}</h2><a href="{name}"><img src="{name}"></a>' for name in names)
    page.append('<h2>Proposed captions only; not training records</h2><table><tr><th>Color</th>'
                '<th>Metallic</th><th>Roughness</th><th>Proposed caption</th></tr>')
    page.extend('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in
                (row["color"], row["metallic"], row["roughness"], row["proposed_caption"])) + '</tr>' for row in rows)
    page.append('</table>')
    (output / "index.html").write_text("\n".join(page) + "\n", encoding="utf-8")
    return names + ["index.html"]
