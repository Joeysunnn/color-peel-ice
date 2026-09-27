#!/usr/bin/env python3
"""Copy reviewed mailbox assets and align only their training color captions."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.colorpeel_ice.prepare_mailbox_matte_counterfactual import read_json, sha256, write_json


COLORS = ("red", "green", "cyan", "blue", "magenta")
LABELS = ("pink", "green", "cyan", "blue", "purple")
MATERIALS = ("metal", "matte")


def stage(protocol_path: Path, source_dir: Path, output_dir: Path):
    protocol = read_json(protocol_path)
    mapping = protocol.get("caption_color_by_source_color")
    if (protocol.get("schema") != "mailbox_caption_alignment/v1"
            or mapping != dict(zip(COLORS, LABELS))
            or protocol.get("training_prompt_template") != "a photo of <S*> mailbox in {color} color"
            or sha256(source_dir / "staging_provenance.json") != protocol["source_staging_provenance_sha256"]):
        raise ValueError("mailbox caption alignment protocol or reviewed source differs")
    old_assets = [json.loads(line) for line in (source_dir / "training_assets_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(old_assets) != 10 or [(row["color"], row["material"]) for row in old_assets] != [
            (color, material) for color in COLORS for material in MATERIALS]:
        raise ValueError("reviewed mailbox asset rows differ")
    for name in ("matte_only", "balanced"):
        if sha256(source_dir / f"{name}_concepts.json") != protocol["source_concepts_sha256"][name]:
            raise ValueError(f"reviewed {name} concepts differ")
    for row in old_assets:
        for field in ("image", "mask"):
            path = Path(row[field])
            if not path.resolve().is_relative_to(source_dir.resolve()) or sha256(path) != row[f"{field}_sha256"]:
                raise ValueError(f"reviewed mailbox {field} differs")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    assets, matte_only, balanced = [], [], []
    for old in old_assets:
        color, material = old["color"], old["material"]
        folder = output_dir / color / material
        image_dir, mask_dir = folder / "images", folder / "masks"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir()
        image, mask = image_dir / "image.png", mask_dir / "image.png"
        shutil.copy2(old["image"], image)
        shutil.copy2(old["mask"], mask)
        row = {"color": color, "material": material, "image": str(image.resolve()),
               "image_sha256": sha256(image), "mask": str(mask.resolve()), "mask_sha256": sha256(mask)}
        if row["image_sha256"] != old["image_sha256"] or row["mask_sha256"] != old["mask_sha256"]:
            raise ValueError("mailbox asset copy differs")
        assets.append(row)
        caption = protocol["training_prompt_template"].format(color=mapping[color])
        concept = {"instance_prompt": [caption], "instance_data_dir": str(image_dir.resolve()),
                   "instance_mask_dir": str(mask_dir.resolve())}
        if material == "matte":
            matte_only.append(concept)
        balanced.append({**concept, "instance_prompt": [caption + (
            " made of glossy painted metal" if material == "metal" else " made of matte plastic")]})
    write_json(output_dir / "matte_only_concepts.json", matte_only)
    write_json(output_dir / "balanced_concepts.json", balanced)
    (output_dir / "training_assets_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in assets), encoding="utf-8")
    write_json(output_dir / "staging_provenance.json", {
        "caption_protocol_sha256": sha256(protocol_path),
        "source_staging_provenance_sha256": sha256(source_dir / "staging_provenance.json"),
        "matte_only_concepts_sha256": sha256(output_dir / "matte_only_concepts.json"),
        "balanced_concepts_sha256": sha256(output_dir / "balanced_concepts.json"),
        "training_assets_manifest_sha256": sha256(output_dir / "training_assets_manifest.jsonl"),
        "matte_only_rows": 5, "balanced_rows": 10,
    })
    return matte_only, balanced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    matte, balanced = stage(args.protocol, args.source_dir, args.output_dir)
    print(f"{len(matte)} matte-only and {len(balanced)} balanced caption-aligned rows")


if __name__ == "__main__":
    main()
