"""Verify the scale-only metal-spoon grid and write its token-local train config."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--scale-run", type=Path, required=True)
    args = parser.parse_args()
    old, new = args.source_run.resolve(), args.scale_run.resolve()
    old_grid, new_grid = read(old / "grid_config.json"), read(
        Path(__file__).with_name("grid_config.json"))
    if old_grid.get("object_scale") != 2.0 or new_grid.get("object_scale") != 1.3:
        raise ValueError("Unexpected source or target object scale")
    if {k: v for k, v in old_grid.items() if k != "object_scale"} != {
            k: v for k, v in new_grid.items() if k != "object_scale"}:
        raise ValueError("Grid configurations differ beyond object scale")

    original = read(old / "grids/metal_spoon/manifest.json")
    scaled = read(new / "grid_metal_spoon/manifest.json")
    if (scaled["status"] != "complete" or scaled["image_count"] != 96
            or scaled["material_id"] != "metal_spoon"):
        raise ValueError("Scale grid is incomplete")
    for key in ("maps_manifest", "profile", "base_scene", "script"):
        if original["input_sha256"][key] != scaled["input_sha256"][key]:
            raise ValueError(f"Renderer input changed: {key}")
    if scaled["input_sha256"]["grid_config"] != sha(
            Path(__file__).with_name("grid_config.json")):
        raise ValueError("Rendered grid uses a different scale config")
    fixed = ("index", "shape", "light", "view", "azimuth_degrees",
             "camera_height_offset", "seed", "split", "material_id",
             "texture_sha256", "texture_projection", "texture_repeats",
             "cycles_samples")
    if len(original["records"]) != len(scaled["records"]):
        raise ValueError("Grid lengths differ")
    for source_row, target_row in zip(original["records"], scaled["records"]):
        if any(source_row[k] != target_row[k] for k in fixed):
            raise ValueError(f"Non-scale grid condition changed: {source_row['index']}")
        if target_row["object_scale"] != 1.3:
            raise ValueError("Wrong scale in rendered metadata")

    staging = new / "staging/metal_spoon"
    stage = read(staging / "staging_manifest.json")
    if stage["train_images"] != 72 or stage["held_out_cone_images"] != 24:
        raise ValueError("Unexpected train/holdout split")
    if stage["source_grid_manifest_sha256"] != sha(
            new / "grid_metal_spoon/manifest.json"):
        raise ValueError("Staging does not match scale grid")
    concepts = read(staging / "concepts.json")
    if sorted(group["instance_prompt"][0] for group in concepts) != [
            f"a photo of a {shape} made of <M*>" for shape in ("cube", "cylinder", "sphere")]:
        raise ValueError("Shape-specific captions changed")
    rows = [json.loads(line) for line in (
        staging / "training_assets_manifest.jsonl").read_text().splitlines()]
    if len(rows) != 72:
        raise ValueError("Incomplete staged assets")
    for row in rows:
        if (sha(Path(row["image"])) != row["image_sha256"]
                or sha(Path(row["mask"])) != row["mask_sha256"]):
            raise ValueError("Staged image or mask changed")

    source = read(old / "configs/metal_spoon.json")
    config = json.loads(json.dumps(source))
    config["status"] = "authorized_scale_only_diagnostic"
    config["run"] = {"study": "natural_material_scale_ablation_v1",
                     "variant": "metal_spoon_scale13_token_local_kv_5000", "seed": 42}
    config["args"]["concepts_list"] = str(staging / "concepts.json")
    config["data_manifest"] = str(staging / "training_assets_manifest.jsonl")
    config["source_grid_manifest_sha256"] = sha(
        new / "grid_metal_spoon/manifest.json")
    config["staging_manifest_sha256"] = sha(staging / "staging_manifest.json")
    config["ablation_source"] = {
        "source_config": str(old / "configs/metal_spoon.json"),
        "source_config_sha256": sha(old / "configs/metal_spoon.json"),
        "changed_factor": "grid_config.object_scale: 2.0 -> 1.3",
    }
    expected_args = {**source["args"], "concepts_list": config["args"]["concepts_list"]}
    if config["args"] != expected_args or config["environment"] != source["environment"]:
        raise ValueError("Training settings changed")
    out = new / "configs"
    out.mkdir(exist_ok=False)
    write(out / "metal_spoon.json", config)
    write(out / "audit.json", {
        "source_config_sha256": sha(old / "configs/metal_spoon.json"),
        "new_config_sha256": sha(out / "metal_spoon.json"),
        "source_grid_manifest_sha256": sha(old / "grids/metal_spoon/manifest.json"),
        "scale_grid_manifest_sha256": sha(new / "grid_metal_spoon/manifest.json"),
        "staging_manifest_sha256": sha(staging / "staging_manifest.json"),
        "verified_training_pairs": len(rows),
        "changed_factor": "object_scale_only",
    })
    print(out / "metal_spoon.json")


if __name__ == "__main__":
    main()
