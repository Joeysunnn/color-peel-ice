"""Validate and stage independent uniform-albedo material counterfactual data."""

import argparse
import collections
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from experiments.material_uniform_albedo_v1.protocol import (
    caption, protocol_sha256, read_protocol, records,
)
from experiments.texture_material_disentangled_v1.data import _input_path, sha256, write_json
from src.train.instance_mask_utils import load_latent_instance_mask, pair_instance_images_and_masks


def make_plan(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    plan = {"status": "complete", "protocol_sha256": protocol_sha256(),
            "train_images": 16, "eval_images": 0, "records": records()}
    write_json(output / "plan.json", plan)
    return plan


def _validate_renders(root):
    protocol = read_protocol()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    if manifest.get("status") != "complete" or manifest.get("protocol_sha256") != protocol_sha256():
        raise ValueError("Incomplete render manifest or changed protocol")
    expected = {row["id"]: row for row in records()}
    rows = manifest["records"]
    if len(rows) != len(expected) or {row["id"] for row in rows} != set(expected):
        raise ValueError("Render records must match the exact sixteen-image material factorial")
    validated, mask_pairs = {}, collections.defaultdict(list)
    for row in rows:
        if any(row.get(key) != value for key, value in expected[row["id"]].items()):
            raise ValueError(f'Changed factorial condition: {row["id"]}')
        if "texture_path" in row or "texture_id" in row:
            raise ValueError("Uniform material records cannot contain texture assets")
        material = protocol["materials"][row["material_id"]]
        color = protocol["dataset"]["colors_linear_rgb"][row["color"]]
        shader = {"roughness": material["roughness"], "metallic": material["metallic"],
                  "base_color_linear_rgb": color,
                  "uniform_base_color_linear_rgba": [0.5 * value for value in color] + [1.0],
                  "texture_image_nodes": 0, "normal_bump_displacement": False,
                  "shader": "Principled BSDF", "ior": 1.5, "specular_ior_level": 0.5,
                  "coat_weight": 0.0, "transmission_weight": 0.0}
        if any(row.get(key) != value for key, value in shader.items()):
            raise ValueError(f'Changed uniform shader/material: {row["id"]}')
        render = {"cycles_samples": protocol["renderer"]["cycles_samples"],
                  "resolution": [protocol["renderer"]["resolution"]] * 2,
                  "blender_version": [4, 2, 11], "render_seed": row["seed"],
                  "object_scale": protocol["dataset"]["object_scale"]}
        if any(row.get(key) != value for key, value in render.items()):
            raise ValueError(f'Changed renderer settings: {row["id"]}')
        metadata = _input_path(root, row["metadata"])
        if sha256(metadata) != row["metadata_sha256"]:
            raise ValueError("Changed render metadata")
        recorded = json.loads(metadata.read_text(encoding="utf8"))
        if recorded != {key: value for key, value in row.items() if key not in ("metadata", "metadata_sha256")}:
            raise ValueError("Render metadata disagrees with manifest")
        paths = {key: _input_path(root, row[key]) for key in ("image", "mask", "input_rgba")}
        size = protocol["renderer"]["resolution"]
        for key, path in paths.items():
            if sha256(path) != row[key + "_sha256"]:
                raise ValueError(f'Changed render {key}: {row["id"]}')
            with Image.open(path) as opened:
                if opened.size != (size, size):
                    raise ValueError(f'Wrong render dimensions: {row["id"]}')
                if key == "mask":
                    pixels = np.asarray(opened.convert("L"))
                    if not (pixels >= 128).any():
                        raise ValueError(f'Empty object mask: {row["id"]}')
                    context = tuple(row[key] for key in ("shape", "color", "light", "seed"))
                    mask_pairs[context].append(pixels.tobytes())
        validated[row["id"]] = paths
    if any(len(masks) != 2 or masks[0] != masks[1] for masks in mask_pairs.values()):
        raise ValueError("Material swaps changed paired foreground masks")
    return rows, validated, manifest_path


def stage(renderroot, output):
    renderroot, output = Path(renderroot).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    rows, validated, manifest_path = _validate_renders(renderroot)
    output.mkdir(parents=True)
    contacts = Image.new("RGB", (160 * 8, 190 * 2), "white")
    draw = ImageDraw.Draw(contacts)
    datasets = {}
    for concept_index, concept in enumerate(read_protocol()["materials"]):
        folder = output / concept
        assets, concepts = [], []
        chosen = sorted((row for row in rows if row["concept_id"] == concept), key=lambda row: row["id"])
        for index, row in enumerate(chosen):
            image_dir, mask_dir = folder / "images" / row["id"], folder / "masks" / row["id"]
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            image, mask = image_dir / "image.png", mask_dir / "image.png"
            shutil.copyfile(validated[row["id"]]["image"], image)
            with Image.open(validated[row["id"]]["mask"]) as source:
                binary = (np.asarray(source.convert("L")) >= 128).astype("uint8") * 255
                Image.fromarray(binary).save(mask)
            if pair_instance_images_and_masks(image_dir, mask_dir) != [(image, mask)]:
                raise ValueError("Staged image and mask cannot be paired by training loader")
            size = read_protocol()["renderer"]["resolution"]
            load_latent_instance_mask(mask, size, size // 8)
            prompt = caption(row)
            concepts.append({"instance_prompt": [prompt], "instance_data_dir": str(image_dir),
                             "instance_mask_dir": str(mask_dir)})
            assets.append({**row, "prompt": prompt, "image": str(image), "mask": str(mask),
                           "source_image_sha256": row["image_sha256"],
                           "source_mask_sha256": row["mask_sha256"],
                           "image_sha256": sha256(image), "mask_sha256": sha256(mask)})
            with Image.open(image) as opened:
                contacts.paste(opened.convert("RGB").resize((160, 160)), (index * 160, concept_index * 190))
            draw.text((index * 160 + 2, concept_index * 190 + 161),
                      f'{concept}\n{row["shape"]} {row["color"]}', fill="black")
        write_json(folder / "concepts.json", concepts)
        ledger = folder / "training_assets_manifest.jsonl"
        ledger.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in assets), encoding="utf8")
        receipt = {"status": "verified", "phase": 1, "branch": "material", "role": "material",
                   "token": "<M*>", "concept_id": concept, "sample_count": len(assets),
                   "image_count": len(assets), "protocol_sha256": protocol_sha256(),
                   "concepts_sha256": sha256(folder / "concepts.json"),
                   "training_assets_manifest_sha256": sha256(ledger),
                   "source_render_manifest_sha256": sha256(manifest_path),
                   "albedo_mode": "uniform", "albedo_level": 0.5,
                   "mask_staging": "threshold raw renderer L mask >=128 to strict 0/255; matching image stem"}
        write_json(folder / "dataset_receipt.json", receipt)
        datasets[concept] = {"path": str(folder), "dataset_receipt_sha256": sha256(folder / "dataset_receipt.json"),
                             **receipt}
    contacts.save(output / "training_contact_sheet.jpg", quality=90)
    audit = {"status": "complete", "protocol_sha256": protocol_sha256(), "train_images": 16,
             "eval_images": 0, "datasets": datasets, "balanced_nuisance_distributions": True,
             "paired_target_masks_identical": True, "albedo_mode": "uniform", "albedo_level": 0.5,
             "source_render_manifest_sha256": sha256(manifest_path),
             "interpretation": "Dataset/control integrity only; not a learned disentanglement result"}
    write_json(output / "audit.json", audit)
    verify_staging(output)
    return audit


def verify_staging(root):
    root = Path(root).resolve()
    protocol = read_protocol()
    audit = json.loads((root / "audit.json").read_text(encoding="utf8"))
    if audit.get("status") != "complete" or audit.get("protocol_sha256") != protocol_sha256():
        raise ValueError("Staging audit is incomplete or belongs to another protocol")
    expected = collections.defaultdict(dict)
    for row in records():
        expected[row["concept_id"]][row["id"]] = row
    if set(audit["datasets"]) != set(expected):
        raise ValueError("Staging concept membership changed")
    for concept, conditions in expected.items():
        folder = root / concept
        receipt_path = folder / "dataset_receipt.json"
        if sha256(receipt_path) != audit["datasets"][concept]["dataset_receipt_sha256"]:
            raise ValueError(f"Changed dataset receipt: {concept}")
        receipt = json.loads(receipt_path.read_text(encoding="utf8"))
        concepts_path, ledger = folder / "concepts.json", folder / "training_assets_manifest.jsonl"
        if (receipt.get("status") != "verified" or receipt.get("protocol_sha256") != protocol_sha256()
                or receipt.get("image_count") != len(conditions)
                or receipt["concepts_sha256"] != sha256(concepts_path)
                or receipt["training_assets_manifest_sha256"] != sha256(ledger)):
            raise ValueError(f"Changed staging metadata: {concept}")
        rows = [json.loads(line) for line in ledger.read_text(encoding="utf8").splitlines()]
        if len(rows) != len(conditions) or {row["id"] for row in rows} != set(conditions):
            raise ValueError(f"Changed staging sample membership: {concept}")
        expected_concepts, expected_files = [], {"images": set(), "masks": set()}
        for row in rows:
            if any(row.get(key) != value for key, value in conditions[row["id"]].items()):
                raise ValueError(f'Changed staging condition: {row["id"]}')
            image = folder / "images" / row["id"] / "image.png"
            mask = folder / "masks" / row["id"] / "image.png"
            if (Path(row["image"]) != image or Path(row["mask"]) != mask
                    or sha256(image) != row["image_sha256"] or sha256(mask) != row["mask_sha256"]
                    or row["prompt"] != caption(row)):
                raise ValueError(f'Changed staged image, mask or caption: {row["id"]}')
            if pair_instance_images_and_masks(image.parent, mask.parent) != [(image, mask)]:
                raise ValueError(f'Training-loader image/mask pairing failed: {row["id"]}')
            size = protocol["renderer"]["resolution"]
            load_latent_instance_mask(mask, size, size // 8)
            expected_files["images"].add(image)
            expected_files["masks"].add(mask)
            expected_concepts.append({"instance_prompt": [caption(row)], "instance_data_dir": str(image.parent),
                                      "instance_mask_dir": str(mask.parent)})
        if json.loads(concepts_path.read_text(encoding="utf8")) != expected_concepts:
            raise ValueError(f"Changed concept sampling groups: {concept}")
        for name, members in expected_files.items():
            if {path for path in (folder / name).rglob("*") if path.is_file()} != members:
                raise ValueError(f"Unexpected staged files: {concept}/{name}")
    return audit


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--output", type=Path, required=True)
    staging = sub.add_parser("stage")
    staging.add_argument("--renders", type=Path, required=True)
    staging.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = make_plan(args.output) if args.command == "plan" else stage(args.renders, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
