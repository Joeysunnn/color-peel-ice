"""Stage train-only rendered images and binary masks for one material token."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.grid / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete" or manifest["image_count"] != 96:
        raise ValueError("A complete 96-image grid is required")
    args.output.mkdir(parents=True, exist_ok=False)
    groups = {}
    rows = []
    held_out = []
    for record in manifest["records"]:
        folder = args.grid / record["shape"] / record["color"] / record["light"] / record["view"]
        image, mask = folder / "image.png", folder / "object_mask.png"
        if sha(image) != record["image_sha256"] or sha(mask) != record["mask_sha256"]:
            raise ValueError(f"Grid file hash mismatch: {folder}")
        if record["split"] == "held_out_shape":
            held_out.append(record)
            continue
        key = f'{record["shape"]}_{record["color"]}'
        image_dir = args.output / "images" / key
        mask_dir = args.output / "masks" / key
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        stem = f'{record["light"]}__{record["view"]}'
        image_out, mask_out = image_dir / f"{stem}.png", mask_dir / f"{stem}.png"
        if image_out.exists():
            raise FileExistsError(image_out)
        shutil.copyfile(image, image_out)
        binary = Image.open(mask).convert("L").point(lambda value: 255 if value >= 128 else 0)
        if binary.size != (512, 512) or not binary.getbbox():
            raise ValueError(f"Invalid object mask: {mask}")
        binary.save(mask_out)
        prompt = f'a photo of a {record["color"]} {record["shape"]} made of <M*>'
        groups[key] = {"instance_prompt": [prompt], "instance_data_dir": str(image_dir.resolve()),
                       "instance_mask_dir": str(mask_dir.resolve())}
        rows.append({"source": str(folder), "shape": record["shape"],
                     "color": record["color"], "light": record["light"],
                     "view": record["view"], "prompt": prompt,
                     "image": str(image_out.resolve()), "image_sha256": sha(image_out),
                     "mask": str(mask_out.resolve()), "mask_sha256": sha(mask_out)})
    if len(rows) != 72 or len(held_out) != 24 or len(groups) != 12:
        raise ValueError(f"Unexpected split: {len(rows)} train, {len(held_out)} holdout")
    (args.output / "concepts.json").write_text(json.dumps(
        [groups[key] for key in sorted(groups)], indent=2) + "\n")
    (args.output / "training_assets_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    (args.output / "staging_manifest.json").write_text(json.dumps({
        "material_id": manifest["material_id"], "train_images": len(rows),
        "held_out_cone_images": len(held_out), "concept_groups": len(groups),
        "source_grid_manifest_sha256": sha(manifest_path),
        "concepts_sha256": sha(args.output / "concepts.json"),
        "training_assets_manifest_sha256": sha(args.output / "training_assets_manifest.jsonl"),
        "script_sha256": sha(__file__),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
