"""Verify that a full-map Material LoRA config changes only the adapter."""

import argparse
import hashlib
import json
import os
from pathlib import Path


SOURCE_ROOT = "run_20261004_fullmaps_threeway_lighting_v2"
MATERIALS = ("mailbox", "metal_spoon", "wood_spoon")
SOURCE_RUN_PREFIX = "20261004-114311__natural_material_fullmaps_v1__"
SOURCE_RUN_SUFFIX = "__241c32a__42"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expanded(value):
    return Path(os.path.expandvars(value)).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    material = config["run"]["variant"].removesuffix("_fullmaps_token_local_kv_lora_r4_5000")
    if material not in MATERIALS or config["run"] != {
            "study": "natural_material_fullmaps_lora_v1",
            "variant": f"{material}_fullmaps_token_local_kv_lora_r4_5000", "seed": 42}:
        raise ValueError("Unexpected Material LoRA study or variant")
    source = config["fullmap_source"]
    root = expanded("${COLORPEEL_RUN_ROOT}") / "natural_material_fullmaps_v1"
    source_config_path = root / SOURCE_ROOT / "configs" / f"{material}.json"
    if expanded(source["config"]) != source_config_path:
        raise ValueError("Source config is not the fixed full-map run")
    source_config = json.loads(source_config_path.read_text())
    if source_config["run"] != {
            "study": "natural_material_fullmaps_v1",
            "variant": f"{material}_fullmaps_token_local_kv_5000", "seed": 42}:
        raise ValueError("Source config differs from the completed full-map study")
    source_run = expanded(source["training_run"])
    if source_run != root / (SOURCE_RUN_PREFIX +
                             f"{material}_fullmaps_token_local_kv_5000" +
                             SOURCE_RUN_SUFFIX):
        raise ValueError("Source training run is not the reviewed full-map run")
    source_manifest_path = source_run / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    if (source_manifest.get("status") != "succeeded"
            or source_manifest.get("returncode") != 0
            or source_manifest.get("run") != source_config["run"]
            or source_manifest.get("data_manifest") != source_config["data_manifest"]
            or Path(source_manifest.get("config_source", "")).resolve() != source_config_path):
        raise ValueError("Source full-map training is not complete")
    expected = dict(source_config["args"])
    if expected.pop("token_local_kv") is not True:
        raise ValueError("Source is not original token-local K/V training")
    expected.update(material_lora_mode="token_local_kv", material_lora_rank=4,
                    material_lora_alpha=4)
    if (expanded(config["args"]["concepts_list"]) !=
            expanded(source_config["args"]["concepts_list"])
            or expanded(config["data_manifest"]) != expanded(source_config["data_manifest"])):
        raise ValueError("LoRA config selects different training data")
    expected["concepts_list"] = config["args"]["concepts_list"]
    if config["args"] != expected:
        raise ValueError("LoRA config changes more than the adapter")
    staging = root / SOURCE_ROOT / "staging" / material
    staging_manifest = staging / "staging_manifest.json"
    if sha(staging_manifest) != source_config["staging_manifest_sha256"]:
        raise ValueError("Staging manifest hash differs")
    details = json.loads(staging_manifest.read_text())
    grid_manifest = root / SOURCE_ROOT / "grids" / material / "manifest.json"
    if (sha(grid_manifest) != source_config["source_grid_manifest_sha256"]
            or details["source_grid_manifest_sha256"] != sha(grid_manifest)):
        raise ValueError("Source full-map grid hash differs")
    assets_path = expanded(config["data_manifest"])
    concepts_path = expanded(config["args"]["concepts_list"])
    if (assets_path != staging / "training_assets_manifest.jsonl"
            or concepts_path != staging / "concepts.json"
            or sha(assets_path) != details["training_assets_manifest_sha256"]
            or sha(concepts_path) != details["concepts_sha256"]):
        raise ValueError("Staged concepts or asset manifest differs")
    records = [json.loads(line) for line in assets_path.read_text().splitlines()]
    if (len(records) != 72 or details["train_images"] != 72
            or details["held_out_cone_images"] != 24
            or {shape: sum(row["shape"] == shape for row in records)
                for shape in ("sphere", "cube", "cylinder")} !=
            {"sphere": 24, "cube": 24, "cylinder": 24}):
        raise ValueError("Full-map data split differs")
    for record in records:
        for name in ("image", "mask"):
            path = Path(record[name]).resolve()
            if not path.is_relative_to(staging) or sha(path) != record[f"{name}_sha256"]:
                raise ValueError(f"Staged {name} hash differs: {path}")
    result = {
        "status": "verified", "material_id": material, "train_images": len(records),
        "source_training_run": str(source_run),
        "source_training_manifest_sha256": sha(source_manifest_path),
        "source_config_sha256": sha(source_config_path),
        "staging_manifest_sha256": sha(staging_manifest),
        "training_assets_manifest_sha256": sha(assets_path),
        "lora_config_sha256": sha(args.config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
