"""Verify that all counterfactual triples differ only in the material settings."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


MATERIALS = ("mailbox", "metal_spoon", "wood_spoon")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.run_root / "paired_counterfactual_manifest.jsonl"
    if out.exists():
        raise FileExistsError(out)
    manifests = {name: json.loads((args.run_root / "grids" / name / "manifest.json").read_text())
                 for name in MATERIALS}
    if any(manifest["status"] != "complete" or manifest["image_count"] != 96
           for manifest in manifests.values()):
        raise ValueError("All three grids must contain 96 complete renders")
    common = ("shape", "color", "light", "view", "seed", "split", "base_color_linear_rgba",
              "object_scale", "world_strength", "camera", "lights", "cycles_samples")
    with out.open("w") as handle:
        for index in range(96):
            records = {name: manifests[name]["records"][index] for name in MATERIALS}
            first = records[MATERIALS[0]]
            for name, record in records.items():
                if any(record[key] != first[key] for key in common):
                    raise ValueError(f"Counterfactual control differs: {index}, {name}")
            folders = {name: args.run_root / "grids" / name / first["shape"] /
                       first["color"] / first["light"] / first["view"] for name in MATERIALS}
            masks = [np.asarray(Image.open(folder / "object_mask.png").convert("L")) >= 128
                     for folder in folders.values()]
            if not all(np.array_equal(masks[0], mask) for mask in masks[1:]):
                raise ValueError(f"Counterfactual object mask differs: {index}")
            pair = {key: first[key] for key in common}
            pair["index"] = index
            pair["material_variants"] = {
                name: {"roughness": record["roughness"], "metallic": record["metallic"],
                       "image": str(folders[name] / "image.png"),
                       "image_sha256": sha(folders[name] / "image.png")}
                for name, record in records.items()}
            handle.write(json.dumps(pair, sort_keys=True) + "\n")
    (args.run_root / "paired_counterfactual_summary.json").write_text(json.dumps({
        "status": "complete", "paired_cases": 96, "material_variants": len(MATERIALS),
        "train_cases": 72, "held_out_cone_cases": 24,
        "pair_manifest_sha256": sha(out), "script_sha256": sha(__file__),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
