"""Copy the five reviewed matte previews into a fresh caption-locked staging set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


COLORS = ("red", "green", "cyan", "blue", "magenta")
CAPTION_COLORS = ("red", "green", "cyan", "blue", "purple")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_protocol(protocol: dict) -> None:
    assets = protocol.get("assets", [])
    if (protocol.get("schema") != "lora_kv_subject_matte5_training_source/v1"
            or len(assets) != 5
            or tuple(item.get("source_color") for item in assets) != COLORS
            or tuple(item.get("caption_color") for item in assets) != CAPTION_COLORS
            or any(item.get("prompt") !=
                   f"a photo of <S*> mailbox in {item['caption_color']} color"
                   for item in assets)):
        raise ValueError("five-matte training-source protocol differs")


def stage(protocol_path: Path, run_root: Path, output: Path) -> dict:
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    source = (run_root / protocol["source_root_relative_to_COLORPEEL_RUN_ROOT"]).resolve()
    locks = {
        "preview_manifest_sha256": source / "preview" / "preview_manifest.json",
        "preview_review_sha256": source / "preview_review.json",
        "source_assets_manifest_sha256": (
            source / "caption_aligned_staged" / "training_assets_manifest.jsonl"),
        "source_staging_provenance_sha256": (
            source / "caption_aligned_staged" / "staging_provenance.json"),
    }
    for key, path in locks.items():
        if sha256(path) != protocol[key]:
            raise ValueError(f"reviewed source hash differs: {key}")
    review = read_json(source / "preview_review.json")
    if (review.get("verdict") != "pass"
            or review.get("preview_manifest_sha256") != protocol["preview_manifest_sha256"]):
        raise ValueError("reviewed matte preview authorization differs")
    preview = read_json(source / "preview" / "preview_manifest.json")
    preview_records = {row["color"]: row for row in preview.get("records", [])}
    source_assets = [json.loads(line) for line in locks[
        "source_assets_manifest_sha256"].read_text(encoding="utf-8").splitlines()]
    matte_assets = {(row["color"], row["material"]): row for row in source_assets
                    if row.get("material") == "matte"}
    if set(preview_records) != set(COLORS) or len(matte_assets) != 5:
        raise ValueError("reviewed five-matte source grid differs")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    concepts, rows = [], []
    for item in protocol["assets"]:
        color = item["source_color"]
        source_image = (source / item["preview_image"]).resolve()
        source_mask = (source / item["mask"]).resolve()
        preview_record = preview_records[color]
        source_asset = matte_assets[(color, "matte")]
        if (sha256(source_image) != item["image_sha256"]
                or preview_record.get("matte_image_sha256") != item["image_sha256"]
                or source_asset.get("image_sha256") != item["image_sha256"]
                or Path(source_asset["mask"]).resolve() != source_mask
                or sha256(source_mask) != item["mask_sha256"]
                or source_asset.get("mask_sha256") != item["mask_sha256"]):
            raise ValueError(f"reviewed matte asset differs: {color}")
        image_dir = output / color / "images"
        mask_dir = output / color / "masks"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir()
        image = image_dir / "image.png"
        mask = mask_dir / "image.png"
        shutil.copy2(source_image, image)
        shutil.copy2(source_mask, mask)
        if sha256(image) != item["image_sha256"] or sha256(mask) != item["mask_sha256"]:
            raise ValueError(f"staged matte asset copy differs: {color}")
        concepts.append({
            "instance_data_dir": str(image_dir.resolve()),
            "instance_mask_dir": str(mask_dir.resolve()),
            "instance_prompt": [item["prompt"]],
        })
        rows.append({
            "source_color": color, "caption_color": item["caption_color"],
            "prompt": item["prompt"], "image": str(image.resolve()),
            "image_sha256": sha256(image), "mask": str(mask.resolve()),
            "mask_sha256": sha256(mask), "source_image": str(source_image),
            "source_mask": str(source_mask),
        })
    concepts_path = output / "concepts.json"
    assets_path = output / "training_assets_manifest.jsonl"
    write_json(concepts_path, concepts)
    assets_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    provenance = {
        "schema": "lora_kv_subject_matte5_staging/v1",
        "source_root": str(source), "source_protocol_sha256": sha256(protocol_path),
        **{key: sha256(path) for key, path in locks.items()},
        "concepts_sha256": sha256(concepts_path),
        "training_assets_manifest_sha256": sha256(assets_path), "row_count": 5,
    }
    write_json(output / "staging_provenance.json", provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_root = Path(os.environ["COLORPEEL_RUN_ROOT"]).resolve()
    result = stage(args.protocol, run_root, args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
