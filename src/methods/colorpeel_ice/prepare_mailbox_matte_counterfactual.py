"""Preview and stage a controlled matte-surface ablation of the frozen mailbox base-5 images."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = REPO_ROOT / "experiments/subject_material_composition_v1/protocols/mailbox_matte_counterfactual_v1.json"
COLORS = ("red", "green", "cyan", "blue", "magenta")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def protocol(path: Path):
    value = read_json(path)
    source = REPO_ROOT / value["source_training_protocol"]
    if (value.get("schema") != "mailbox_matte_counterfactual/v1"
            or sha256(source) != value["source_training_protocol_sha256"]
            or value.get("colors") != list(COLORS)
            or value.get("transform") != {
                "method": "masked_lab_highlight_suppression/v1",
                "gaussian_sigma_pixels": 16.0, "highlight_floor_l8": 3.0,
                "suppression_strength": 1.0, "edge_transition_pixels": 5.0,
            }
            or value.get("training") != {
                "subject_token": "<S*>", "initializer_token": "mailbox",
                "architecture": "token_local_kv", "steps_per_branch": 5000,
                "caa_weight": 0.0, "matte_only_rows": 5, "balanced_rows": 10,
            }):
        raise ValueError("mailbox matte protocol differs")
    return value


def verified_source(value: dict, run_root: Path):
    root = (run_root / value["source_root_relative_to_COLORPEEL_RUN_ROOT"]).resolve()
    concepts_path, manifest_path = root / "concepts.json", root / "staging_manifest.json"
    if (sha256(concepts_path) != value["source_concepts_sha256"]
            or sha256(manifest_path) != value["source_staging_manifest_sha256"]):
        raise ValueError("mailbox base-5 source provenance differs")
    old = read_json(REPO_ROOT / value["source_training_protocol"])
    concepts = read_json(concepts_path)
    if len(concepts) != 5 or old["training_data"]["image_names"] != list(COLORS):
        raise ValueError("mailbox base-5 source coverage differs")
    result = []
    for color, concept in zip(COLORS, concepts):
        item = root / color
        image, mask = item / "images/image.png", item / "masks/image.png"
        if (concept != {
                "instance_prompt": [f"a photo of <S*> mailbox in {color} color"],
                "instance_data_dir": str(item / "images"),
                "instance_mask_dir": str(item / "masks"),
            } or sha256(image) != old["training_data"]["expected_image_sha256"][color]
                or sha256(mask) != old["source_pilot"]["repaired_mask_sha256"]):
            raise ValueError(f"mailbox base-5 source differs: {color}")
        result.append((color, image, mask))
    return result


def suppress_highlights(rgb: np.ndarray, binary_mask: np.ndarray, settings: dict):
    import cv2

    if (rgb.shape != (512, 512, 3) or rgb.dtype != np.uint8
            or binary_mask.shape != (512, 512)
            or set(np.unique(binary_mask).tolist()) != {0, 255}):
        raise ValueError("mailbox image or repaired mask differs")
    inside = binary_mask == 255
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0].astype(np.float32)
    local = cv2.GaussianBlur(lightness, (0, 0), settings["gaussian_sigma_pixels"])
    excess = np.maximum(lightness - local - settings["highlight_floor_l8"], 0.0)
    distance = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 5)
    edge_weight = np.minimum(distance / settings["edge_transition_pixels"], 1.0)
    reduction = settings["suppression_strength"] * excess * edge_weight
    changed = reduction >= 0.5
    adjusted = lab.copy()
    adjusted[:, :, 0] = np.clip(np.rint(lightness - reduction), 0, 255).astype(np.uint8)
    converted = cv2.cvtColor(adjusted, cv2.COLOR_LAB2RGB)
    output = rgb.copy()
    output[inside & changed] = converted[inside & changed]
    if not np.array_equal(output[~inside], rgb[~inside]):
        raise ValueError("matte edit changed background")
    return output, {
        "foreground_pixels": int(inside.sum()),
        "changed_foreground_pixels": int((np.any(output != rgb, axis=2) & inside).sum()),
        "outside_mask_changed_pixels": 0,
        "max_l8_reduction": float(reduction.max()),
        "mean_positive_l8_reduction": float(reduction[reduction > 0].mean()) if np.any(reduction > 0) else 0.0,
    }


def preview(protocol_path: Path, run_root: Path, output_dir: Path):
    value = protocol(protocol_path)
    sources = verified_source(value, run_root)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    board = Image.new("RGB", (5 * 512, 2 * 548), "white")
    draw = ImageDraw.Draw(board)
    records = []
    for index, (color, image_path, mask_path) in enumerate(sources):
        rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        matte, metrics = suppress_highlights(rgb, mask, value["transform"])
        if metrics["changed_foreground_pixels"] == 0:
            raise ValueError(f"no highlight suppression for {color}")
        matte_path = output_dir / f"{color}_matte.png"
        Image.fromarray(matte, mode="RGB").save(matte_path)
        board.paste(Image.fromarray(rgb), (index * 512, 32))
        board.paste(Image.fromarray(matte), (index * 512, 580))
        draw.text((index * 512 + 8, 8), f"{color}: original metal", fill="black")
        draw.text((index * 512 + 8, 556), f"{color}: highlight suppressed", fill="black")
        records.append({"color": color, "source_image_sha256": sha256(image_path),
                        "source_mask_sha256": sha256(mask_path),
                        "matte_image": matte_path.name, "matte_image_sha256": sha256(matte_path),
                        "metrics": metrics})
    board.save(output_dir / "preview_contact_sheet.jpg", quality=92)
    write_json(output_dir / "preview_manifest.json", {
        "schema": "mailbox_matte_preview/v1", "protocol_sha256": sha256(protocol_path),
        "source_concepts_sha256": value["source_concepts_sha256"], "records": records,
    })
    write_json(output_dir.parent / "preview_review_template.json", {
        "verdict": "pending", "reviewer": "", "reviewed_at": "",
        "preview_manifest_sha256": sha256(output_dir / "preview_manifest.json"),
    })
    return records


def validated_preview(protocol_path: Path, run_root: Path, preview_dir: Path):
    value = protocol(protocol_path)
    sources = verified_source(value, run_root)
    manifest = read_json(preview_dir / "preview_manifest.json")
    if (manifest.get("schema") != "mailbox_matte_preview/v1"
            or manifest.get("protocol_sha256") != sha256(protocol_path)
            or manifest.get("source_concepts_sha256") != value["source_concepts_sha256"]
            or len(manifest.get("records", [])) != 5):
        raise ValueError("matte preview manifest differs")
    for (color, source, mask), record in zip(sources, manifest["records"]):
        matte = preview_dir / record["matte_image"]
        source_rgb = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
        source_mask = np.asarray(Image.open(mask).convert("L"), dtype=np.uint8)
        expected, metrics = suppress_highlights(source_rgb, source_mask, value["transform"])
        if (record["color"] != color or record["source_image_sha256"] != sha256(source)
                or record["source_mask_sha256"] != sha256(mask)
                or record["matte_image"] != f"{color}_matte.png"
                or record["matte_image_sha256"] != sha256(matte)
                or record["metrics"] != metrics
                or not np.array_equal(np.asarray(Image.open(matte).convert("RGB")), expected)):
            raise ValueError(f"matte preview artifact differs: {color}")
    return sources, manifest


def stage(protocol_path: Path, run_root: Path, preview_dir: Path,
          review_path: Path, output_dir: Path):
    sources, manifest = validated_preview(protocol_path, run_root, preview_dir)
    review = read_json(review_path)
    if (review.get("verdict") != "pass" or not review.get("reviewer")
            or not review.get("reviewed_at")
            or review.get("preview_manifest_sha256") != sha256(preview_dir / "preview_manifest.json")):
        raise ValueError("matte preview has no matching pass review")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    matte_only, balanced, assets = [], [], []
    for (color, original, mask), record in zip(sources, manifest["records"]):
        for material, source in (("metal", original), ("matte", preview_dir / record["matte_image"])):
            item = output_dir / color / material
            image_dir, mask_dir = item / "images", item / "masks"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir()
            image_target, mask_target = image_dir / "image.png", mask_dir / "image.png"
            shutil.copy2(source, image_target)
            shutil.copy2(mask, mask_target)
            assets.append({"color": color, "material": material,
                           "image": str(image_target.resolve()), "image_sha256": sha256(image_target),
                           "mask": str(mask_target.resolve()), "mask_sha256": sha256(mask_target)})
            baseline_caption = f"a photo of <S*> mailbox in {color} color"
            if material == "matte":
                matte_only.append({"instance_prompt": [baseline_caption],
                                   "instance_data_dir": str(image_dir.resolve()),
                                   "instance_mask_dir": str(mask_dir.resolve())})
            balanced.append({"instance_prompt": [baseline_caption + (
                " made of glossy painted metal" if material == "metal" else " made of matte plastic")],
                "instance_data_dir": str(image_dir.resolve()),
                "instance_mask_dir": str(mask_dir.resolve())})
    write_json(output_dir / "matte_only_concepts.json", matte_only)
    write_json(output_dir / "balanced_concepts.json", balanced)
    (output_dir / "training_assets_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in assets), encoding="utf-8")
    write_json(output_dir / "staging_provenance.json", {
        "protocol_sha256": sha256(protocol_path),
        "preview_manifest_sha256": sha256(preview_dir / "preview_manifest.json"),
        "review_sha256": sha256(review_path),
        "matte_only_concepts_sha256": sha256(output_dir / "matte_only_concepts.json"),
        "balanced_concepts_sha256": sha256(output_dir / "balanced_concepts.json"),
        "training_assets_manifest_sha256": sha256(output_dir / "training_assets_manifest.jsonl"),
        "matte_only_rows": 5, "balanced_rows": 10,
    })
    return matte_only, balanced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "stage"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--review-record", type=Path)
    args = parser.parse_args()
    if args.command == "preview":
        print(f"{len(preview(args.protocol, args.run_root, args.output_dir))} matte preview images")
    else:
        if args.preview_dir is None or args.review_record is None:
            parser.error("stage requires --preview-dir and --review-record")
        matte, balanced = stage(args.protocol, args.run_root, args.preview_dir,
                                args.review_record, args.output_dir)
        print(f"{len(matte)} matte-only and {len(balanced)} balanced rows")


if __name__ == "__main__":
    main()
