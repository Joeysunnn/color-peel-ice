"""Prepare a caption-only ablation from completed full-map staging."""

import argparse
import hashlib
import json
from pathlib import Path


CAPTION = "a photo of an object made of <M*>"
SHAPES = ("cube", "cylinder", "sphere")
STUDY = "natural_material_caption_ablation_v1"
MATERIALS = ("mailbox", "metal_spoon", "wood_spoon")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True,
                        help="Completed full-map grid run containing configs/ and staging/")
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh ablation staging directory; must not exist")
    parser.add_argument("--material-id", choices=MATERIALS, default="metal_spoon")
    args = parser.parse_args()
    source = args.source_run.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    material = args.material_id
    source_config_path = source / "configs" / f"{material}.json"
    source_staging = source / "staging" / material
    source_concepts_path = source_staging / "concepts.json"
    source_assets_path = source_staging / "training_assets_manifest.jsonl"
    source_config = json.loads(source_config_path.read_text())
    source_concepts = json.loads(source_concepts_path.read_text())
    rows = [json.loads(line) for line in source_assets_path.read_text().splitlines()]
    if (source_config["run"] != {"study": "natural_material_fullmaps_v1",
                                  "variant": f"{material}_fullmaps_token_local_kv_5000",
                                  "seed": 42}
            or Path(source_config["args"]["concepts_list"]).resolve() != source_concepts_path
            or Path(source_config["data_manifest"]).resolve() != source_assets_path
            or len(rows) != 72 or len(source_concepts) != 3):
        raise ValueError("Unexpected source training configuration or staging")
    if source_config["args"].get("token_local_kv") is not True:
        raise ValueError("Source is not token-local K/V")

    by_shape = {}
    for concept in source_concepts:
        image_dir = Path(concept["instance_data_dir"]).resolve()
        mask_dir = Path(concept["instance_mask_dir"]).resolve()
        shape = image_dir.name
        if (shape not in SHAPES or shape in by_shape
                or image_dir != source_staging / "images" / shape
                or mask_dir != source_staging / "masks" / shape
                or concept["instance_prompt"] != [f"a photo of a {shape} made of <M*>"]):
            raise ValueError(f"Unexpected source concept: {concept}")
        by_shape[shape] = concept
    if set(by_shape) != set(SHAPES):
        raise ValueError("Missing source shape group")

    counts = {shape: 0 for shape in SHAPES}
    new_rows = []
    for row in rows:
        shape = row["shape"]
        image = Path(row["image"]).resolve()
        mask = Path(row["mask"]).resolve()
        if (shape not in by_shape
                or image.parent != Path(by_shape[shape]["instance_data_dir"])
                or mask.parent != Path(by_shape[shape]["instance_mask_dir"])
                or row["prompt"] != f"a photo of a {shape} made of <M*>"
                or sha256(image) != row["image_sha256"]
                or sha256(mask) != row["mask_sha256"]):
            raise ValueError(f"Source asset mismatch: {row}")
        counts[shape] += 1
        new_rows.append({**row, "prompt": CAPTION})
    if any(count != 24 for count in counts.values()):
        raise ValueError(f"Unexpected shape balance: {counts}")

    concepts = [{**by_shape[shape], "instance_prompt": [CAPTION]} for shape in SHAPES]
    config = json.loads(json.dumps(source_config))
    config["status"] = "authorized_caption_only_diagnostic"
    config["run"] = {"study": STUDY,
                     "variant": f"{material}_generic_caption_token_local_kv_5000", "seed": 42}
    config["args"]["concepts_list"] = str(output / "concepts.json")
    config["data_manifest"] = str(output / "training_assets_manifest.jsonl")
    config["ablation_source"] = {
        "source_config": str(source_config_path),
        "source_config_sha256": sha256(source_config_path),
        "source_concepts_sha256": sha256(source_concepts_path),
        "source_assets_sha256": sha256(source_assets_path),
        "source_staging_manifest_sha256": sha256(source_staging / "staging_manifest.json"),
        "changed_factor": "instance_prompt_only",
    }

    output.mkdir(parents=True)
    write_json(output / "concepts.json", concepts)
    (output / "training_assets_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in new_rows))
    write_json(output / "train.json", config)
    write_json(output / "audit.json", {
        "source": config["ablation_source"],
        "image_and_mask_files_reused_without_copy": True,
        "shape_counts": counts,
        "source_to_ablation_row_differences": ["prompt"],
        "source_to_ablation_config_differences": [
            "status", "run.study", "run.variant", "args.concepts_list", "data_manifest",
            "ablation_source"],
        "source_image_and_mask_hashes_verified": len(rows),
        "concepts_sha256": sha256(output / "concepts.json"),
        "training_assets_manifest_sha256": sha256(output / "training_assets_manifest.jsonl"),
        "train_config_sha256": sha256(output / "train.json"),
    })
    print(output)


if __name__ == "__main__":
    main()
