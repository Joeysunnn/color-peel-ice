"""Validate and stage the six natural-image counterfactual datasets; never train."""

import argparse
import collections
import html
import json
from pathlib import Path
import shutil
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from experiments.natural_texture_material_v1 import dataset_protocol as schema
from experiments.texture_material_disentangled_v1.data import _input_path, write_json
from src.train.instance_mask_utils import load_latent_instance_mask, pair_instance_images_and_masks


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def check_sources(protocol):
    if any(schema.sha256(path) != digest for path, digest in protocol["source_locks"].items()):
        raise ValueError("Extraction source changed")


def validate_renders(root):
    manifest_path = root / "manifest.json"
    manifest = read(manifest_path)
    protocol = read(manifest["protocol_path"])
    digest = schema.protocol_sha256(protocol)
    if (manifest.get("status") != "complete" or manifest.get("protocol_sha256") != digest
            or manifest.get("source_locks") != protocol["source_locks"]):
        raise ValueError("Incomplete render manifest or changed protocol/source locks")
    plan = read(manifest["plan_path"])
    if (schema.sha256(manifest["plan_path"]) != manifest["plan_sha256"]
            or plan.get("protocol_sha256") != digest or plan.get("records") != schema.records(protocol)):
        raise ValueError("Changed source render plan")
    check_sources(protocol)
    expected = {row["id"]: row for row in schema.records(protocol)}
    rows = manifest["records"]
    if len(rows) != 72 or len(rows) != len(expected) or {row["id"] for row in rows} != set(expected):
        raise ValueError("Render records must match the exact 72-image natural factorial")
    validated, masks = {}, collections.defaultdict(list)
    for row in rows:
        if any(row.get(key) != value for key, value in expected[row["id"]].items()):
            raise ValueError(f'Changed factorial condition: {row["id"]}')
        material = protocol["materials"][row["material_id"]]
        color = protocol["dataset"]["colors_linear_rgb"][row["color"]]
        shader = {"roughness": material["roughness"], "metallic": material["metallic"],
                  "base_color_linear_rgb": color, "normal_bump_displacement": False,
                  "shader": "Principled BSDF", "ior": 1.5, "specular_ior_level": 0.5,
                  "coat_weight": 0.0, "transmission_weight": 0.0,
                  "roughness_input_linked": False, "metallic_input_linked": False,
                  "normal_input_linked": False, "base_color_input_linked": row["branch"] == "texture"}
        if row["branch"] == "texture":
            shader.update(texture_projection="UV", uv_repeats=1, texture_colorspace="Non-Color",
                          texture_image_nodes=1)
            if (not row.get("uv_mapping") or len(row.get("uv_sha256", "")) != 64
                    or schema.sha256(row["texture_path"]) != row["texture_sha256"]):
                raise ValueError("Changed or missing extracted Texture/UV evidence")
        else:
            if "texture_id" in row or "texture_path" in row:
                raise ValueError("Uniform Material cannot contain texture assets")
            shader.update(albedo_mode="uniform", albedo_level=0.5, texture_image_nodes=0,
                          uniform_base_color_linear_rgba=[0.5 * value for value in color] + [1.0])
        renderer = {"cycles_samples": protocol["renderer"]["cycles_samples"],
                    "resolution": [protocol["renderer"]["resolution"]] * 2,
                    "blender_version": [4, 2, 11], "render_seed": row["seed"],
                    "object_scale": protocol["dataset"]["object_scale"]}
        if any(row.get(key) != value for key, value in {**shader, **renderer}.items()):
            raise ValueError(f'Changed shader/material/renderer: {row["id"]}')
        metadata = _input_path(root, row["metadata"])
        if (schema.sha256(metadata) != row["metadata_sha256"] or read(metadata) !=
                {key: value for key, value in row.items() if key not in ("metadata", "metadata_sha256")}):
            raise ValueError("Changed render metadata or disagreement with manifest")
        paths = {key: _input_path(root, row[key]) for key in ("image", "mask", "input_rgba")}
        for key, path in paths.items():
            if schema.sha256(path) != row[key + "_sha256"]:
                raise ValueError(f'Changed render {key}: {row["id"]}')
            with Image.open(path) as opened:
                size = protocol["renderer"]["resolution"]
                if opened.size != (size, size) or (key == "image" and opened.mode != "RGB"):
                    raise ValueError(f'Wrong render dimensions/mode: {row["id"]}')
                if key == "mask":
                    pixels = np.asarray(opened.convert("L"))
                    if not (pixels >= 128).any():
                        raise ValueError(f'Empty object mask: {row["id"]}')
                    context = tuple(row[name] for name in ("shape", "color", "light", "seed"))
                    masks[context].append(pixels.tobytes())
        validated[row["id"]] = paths
    if any(any(value != values[0] for value in values) for values in masks.values()):
        raise ValueError("Target/branch swaps changed paired foreground masks")
    return protocol, rows, validated, manifest_path


def font(size):
    for path in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def contact_sheet(output, concept, assets, protocol):
    """Show source evidence separately from every actual training RGB render."""
    first = assets[0]
    source, branch = protocol["sources"][first["source_id"]], first["branch"]
    cell, header, stride = 300, 370, 370
    sheet = Image.new("RGB", (cell * 4, header + stride * ((len(assets) + 3) // 4)), "white")
    draw, small = ImageDraw.Draw(sheet), font(16)
    draw.text((12, 8), f"Phase 2 | {concept} | {len(assets)} actual training RGB images", font=font(22), fill="black")
    draw.text((12, 40), "Dataset prepared; training NOT started. original and rank64 will use these SAME images.", font=small, fill="black")
    natural = Image.open(source["input"]).convert("RGBA")
    background = Image.new("RGBA", natural.size, (128, 128, 128, 255))
    natural = Image.alpha_composite(background, natural).convert("RGB")
    sheet.paste(ImageOps.contain(natural, (160, 160)), (12, 92))
    draw.text((12, 68), "Reference source", font=small, fill="black")
    if branch == "texture":
        patch = ImageOps.contain(Image.open(source["texture"]).convert("RGB"), (160, 160))
        sheet.paste(patch, (200, 92))
        draw.text((200, 68), "Reference Texture", font=small, fill="black")
        description = "Fixed: extracted grayscale albedo Texture; UV repeats=1, Non-Color.\nVaried: geometry, color, nuisance R/M, lighting.\nNo spatial roughness/metallic map, bump, normal or displacement."
    else:
        description = (f'Fixed: roughness={source["roughness"]:.6f}; metallic={source["metallic"]:.6f}.\n'
                       "Uniform albedo = 0.5 x color; NO texture image nodes.\nVaried: geometry, color, lighting.\nNo source albedo, spatial R/M, bump, normal or displacement.")
    draw.multiline_text((390, 92), description, font=small, fill="black", spacing=10)
    path_lines = textwrap.fill(str(output / concept / "training_assets_manifest.jsonl"), width=130)
    draw.multiline_text((12, 255), "Source / patch above are references, not training RGB.\nActual RGB list / captions / SHA256:\n" + path_lines, font=small, fill="black", spacing=5)
    for index, row in enumerate(assets):
        x, y = (index % 4) * cell, header + (index // 4) * stride
        with Image.open(row["image"]) as opened:
            sheet.paste(opened.convert("RGB").resize((288, 288)), (x + 6, y))
        label = (f'{index + 1:02d} | {row["shape"]} | {row["color"]}\nlight={row["light"]}\n'
                 f'R={row["roughness"]:.4f} M={row["metallic"]:.4f} | seed={row["seed"]}')
        draw.multiline_text((x + 6, y + 292), label, font=small, fill="black", spacing=4)
    path = output / "contact_sheets" / f"{concept}.jpg"
    sheet.save(path, quality=92)
    return path


def review_bundle(output, protocol, all_assets):
    (output / "contact_sheets").mkdir()
    references, sections = {}, []
    for concept, assets in all_assets.items():
        first = assets[0]
        sheet = contact_sheet(output, concept, assets, protocol)
        references[concept] = {"source_id": first["source_id"], "branch": first["branch"],
            "status": "dataset_prepared_training_not_started", "planned_methods": ["fullrank", "lora_r64"],
            "shared_images_for_both_methods": True, "image_count": len(assets),
            "source": protocol["sources"][first["source_id"]],
            "training_assets_manifest": str(output / concept / "training_assets_manifest.jsonl"),
            "images": [{key: row[key] for key in ("id", "image", "image_sha256", "prompt", "shape", "color", "light", "roughness", "metallic", "seed")} for row in assets]}
        links = "".join(f'<li><a href="{html.escape(Path(row["image"]).relative_to(output).as_posix())}">'
                        f'{html.escape(row["id"])}</a> - {html.escape(row["prompt"])}</li>' for row in assets)
        relative = sheet.relative_to(output).as_posix()
        sections.append(f'<h2>{html.escape(concept)}: {len(assets)} actual training RGB images</h2>'
                        f'<a href="{relative}"><img src="{relative}" style="max-width:100%"></a><ul>{links}</ul>')
    write_json(output / "training_references.json", references)
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><h1>Phase 2 counterfactual datasets</h1>'
        '<p>Dataset prepared; training NOT started. Contact sheets show ALL actual training RGB images. '
        'Source photo and extracted Texture are reference evidence, not substituted training images. '
        'original and rank64 share each corresponding dataset.</p>'
        '<p><a href="audit.json">Audit</a> | <a href="training_references.json">Training references</a></p>' +
        "\n".join(sections), encoding="utf8")


def stage(renderroot, output):
    renderroot, output = Path(renderroot).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    protocol, rows, validated, manifest_path = validate_renders(renderroot)
    output.mkdir(parents=True)
    write_json(output / "protocol.json", protocol)
    datasets, all_assets = {}, {}
    for concept in sorted({row["concept_id"] for row in rows}):
        folder = output / concept
        assets, concepts = [], []
        for row in sorted((row for row in rows if row["concept_id"] == concept), key=lambda row: row["id"]):
            image_dir, mask_dir = folder / "images" / row["id"], folder / "masks" / row["id"]
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            image, mask = image_dir / "image.png", mask_dir / "image.png"
            shutil.copyfile(validated[row["id"]]["image"], image)
            with Image.open(validated[row["id"]]["mask"]) as opened:
                Image.fromarray((np.asarray(opened.convert("L")) >= 128).astype("uint8") * 255).save(mask)
            if pair_instance_images_and_masks(image_dir, mask_dir) != [(image, mask)]:
                raise ValueError("Training loader cannot pair staged image/mask")
            size = protocol["renderer"]["resolution"]
            load_latent_instance_mask(mask, size, size // 8)
            prompt = schema.caption(row, protocol)
            concepts.append({"instance_prompt": [prompt], "instance_data_dir": str(image_dir), "instance_mask_dir": str(mask_dir)})
            assets.append({**row, "source_image": str(validated[row["id"]]["image"]),
                           "source_mask": str(validated[row["id"]]["mask"]),
                           "source_image_sha256": row["image_sha256"], "source_mask_sha256": row["mask_sha256"],
                           "image": str(image), "mask": str(mask), "prompt": prompt,
                           "image_sha256": schema.sha256(image), "mask_sha256": schema.sha256(mask)})
        write_json(folder / "concepts.json", concepts)
        ledger = folder / "training_assets_manifest.jsonl"
        ledger.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in assets), encoding="utf8")
        first = assets[0]
        receipt = {"status": "verified", "phase": 2, "branch": first["branch"], "role": first["branch"],
            "source_id": first["source_id"], "concept_id": concept,
            "token": "<T*>" if first["branch"] == "texture" else "<M*>",
            "sample_count": len(assets), "image_count": len(assets), "training_started": False,
            "protocol_sha256": schema.protocol_sha256(protocol), "concepts_sha256": schema.sha256(folder / "concepts.json"),
            "training_assets_manifest_sha256": schema.sha256(ledger), "source_render_manifest_sha256": schema.sha256(manifest_path),
            "mask_staging": "threshold raw renderer L mask >=128 to strict 0/255; matching image stem"}
        write_json(folder / "dataset_receipt.json", receipt)
        datasets[concept] = {**receipt, "path": str(folder), "dataset_receipt_sha256": schema.sha256(folder / "dataset_receipt.json")}
        all_assets[concept] = assets
    check_sources(protocol)
    review_bundle(output, protocol, all_assets)
    audit = {"status": "complete", "phase": 2, "stage": "dataset_only", "training_started": False,
        "protocol_sha256": schema.protocol_sha256(protocol), "train_images": 72, "texture_images": 48, "material_images": 24,
        "eval_images": 0, "datasets": datasets, "source_render_manifest": str(manifest_path),
        "source_render_manifest_sha256": schema.sha256(manifest_path), "paired_target_masks_identical": True,
        "balanced_nuisance_distributions": True, "training_references_sha256": schema.sha256(output / "training_references.json"),
        "interpretation": "Dataset/control integrity only; not a learned disentanglement result"}
    write_json(output / "audit.json", audit)
    verify_staging(output)
    return audit


def verify_staging(root):
    root = Path(root).resolve()
    audit, protocol = read(root / "audit.json"), read(root / "protocol.json")
    if audit.get("status") != "complete" or audit.get("protocol_sha256") != schema.protocol_sha256(protocol):
        raise ValueError("Incomplete staging audit or changed protocol")
    check_sources(protocol)
    expected = collections.defaultdict(dict)
    for row in schema.records(protocol):
        expected[row["concept_id"]][row["id"]] = row
    if set(audit["datasets"]) != set(expected):
        raise ValueError("Staging concept membership changed")
    for concept, conditions in expected.items():
        folder = root / concept
        receipt = read(folder / "dataset_receipt.json")
        if (schema.sha256(folder / "dataset_receipt.json") != audit["datasets"][concept]["dataset_receipt_sha256"]
                or receipt.get("status") != "verified" or receipt.get("image_count") != len(conditions)
                or receipt.get("protocol_sha256") != audit["protocol_sha256"]
                or schema.sha256(folder / "concepts.json") != receipt["concepts_sha256"]
                or schema.sha256(folder / "training_assets_manifest.jsonl") != receipt["training_assets_manifest_sha256"]):
            raise ValueError("Changed staging receipt or metadata")
        rows = [json.loads(line) for line in (folder / "training_assets_manifest.jsonl").read_text(encoding="utf8").splitlines()]
        if len(rows) != len(conditions) or {row["id"] for row in rows} != set(conditions):
            raise ValueError("Changed staging sample membership")
        concepts, files = [], {"images": set(), "masks": set()}
        for row in rows:
            if any(row.get(key) != value for key, value in conditions[row["id"]].items()):
                raise ValueError("Changed staged factorial condition")
            image, mask = (folder / name / row["id"] / "image.png" for name in ("images", "masks"))
            if (Path(row["image"]) != image or Path(row["mask"]) != mask
                    or schema.sha256(image) != row["image_sha256"] or schema.sha256(mask) != row["mask_sha256"]
                    or row["prompt"] != schema.caption(row, protocol)):
                raise ValueError("Changed staged RGB, mask or caption")
            if pair_instance_images_and_masks(image.parent, mask.parent) != [(image, mask)]:
                raise ValueError("Training-loader image/mask pairing failed")
            size = protocol["renderer"]["resolution"]
            load_latent_instance_mask(mask, size, size // 8)
            files["images"].add(image)
            files["masks"].add(mask)
            concepts.append({"instance_prompt": [schema.caption(row, protocol)], "instance_data_dir": str(image.parent), "instance_mask_dir": str(mask.parent)})
        if read(folder / "concepts.json") != concepts:
            raise ValueError("Changed concept sampling groups")
        for name, members in files.items():
            if {path for path in (folder / name).rglob("*") if path.is_file()} != members:
                raise ValueError("Unexpected staged files")
    if schema.sha256(root / "training_references.json") != audit["training_references_sha256"]:
        raise ValueError("Changed training references")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = stage(args.renders, args.output)
    print(json.dumps({key: audit[key] for key in ("status", "phase", "train_images", "training_started")}))


if __name__ == "__main__":
    main()
