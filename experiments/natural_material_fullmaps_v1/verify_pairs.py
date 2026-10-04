"""Verify that full-map grids differ only by material for paired conditions."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


MATERIALS = ("mailbox", "metal_spoon", "wood_spoon")
CONDITION_KEYS = ("index", "shape", "light", "view", "azimuth_degrees",
                  "camera_height_offset", "seed", "split", "object_scale",
                  "camera", "lights", "cycles_samples", "texture_projection",
                  "texture_repeats")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    grids = {}
    for name in MATERIALS:
        root = args.run / "grids" / name
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "complete" or manifest["image_count"] != 96:
            raise ValueError(f"Incomplete grid: {name}")
        keys = set()
        for row in manifest["records"]:
            key = (row["shape"], row["light"], row["view"])
            if key in keys:
                raise ValueError(f"Duplicate condition: {name}/{key}")
            keys.add(key)
            folder = root / row["shape"] / row["light"] / row["view"]
            if sha(folder / "image.png") != row["image_sha256"]:
                raise ValueError(f"Image hash mismatch: {name}/{key}")
            if sha(folder / "object_mask.png") != row["mask_sha256"]:
                raise ValueError(f"Mask hash mismatch: {name}/{key}")
        grids[name] = (root, manifest, {(
            row["shape"], row["light"], row["view"]): row for row in manifest["records"]})
    cases = set(grids[MATERIALS[0]][2])
    if len(cases) != 96 or any(set(grids[name][2]) != cases for name in MATERIALS):
        raise ValueError("Paired condition sets differ")
    for key in sorted(cases):
        reference = grids[MATERIALS[0]][2][key]
        masks = []
        for name in MATERIALS:
            root, _, rows = grids[name]
            row = rows[key]
            if any(row[field] != reference[field] for field in CONDITION_KEYS):
                raise ValueError(f"Nonmaterial metadata differs: {name}/{key}")
            mask = root / row["shape"] / row["light"] / row["view"] / "object_mask.png"
            masks.append(Image.open(mask).convert("L").tobytes())
        if masks[0] != masks[1] or masks[0] != masks[2]:
            raise ValueError(f"Object masks differ: {key}")
    args.output.write_text(json.dumps({
        "status": "complete", "paired_cases": len(cases),
        "material_variants": len(MATERIALS),
        "train_cases": sum(grids[MATERIALS[0]][2][key]["split"] == "train" for key in cases),
        "held_out_cone_cases": sum(grids[MATERIALS[0]][2][key]["split"] == "held_out_shape"
                                   for key in cases),
        "grid_manifest_sha256": {name: sha(grids[name][0] / "manifest.json")
                                 for name in MATERIALS},
        "script_sha256": sha(Path(__file__)),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
