"""Verify frozen Texture snapshots and identify their actual RGB training images."""

import json
from pathlib import Path

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.texture_material_disentangled_v1 import data, infer
from scripts.launch.colorpeel_run import read_config

STEPS = (500, 750, 1000)


def _training_reference(selection, step, locks, staging):
    run = Path(selection["run"])
    config_path, manifest_path = run / "config.yaml", run / "manifest.json"
    # These paths must already be bound to the selected checkpoint's provenance.
    if any(str(path) not in selection["hashes"] for path in (config_path, manifest_path)):
        raise ValueError("Texture selection lacks its frozen training config or manifest")
    config = read_config(config_path)
    concepts_path = Path(config["args"]["concepts_list"])
    ledger = Path(config["data_manifest"])
    folder = concepts_path.parent
    if (not concepts_path.is_absolute() or concepts_path.name != "concepts.json"
            or folder.name != selection["concept_id"]
            or ledger != folder / "training_assets_manifest.jsonl"
            or read_json(manifest_path).get("data_manifest") != str(ledger)):
        raise ValueError("Texture training dataset paths disagree")
    root = folder.parent
    if root not in staging:
        staging[root] = data.verify_staging(root)
    audit = staging[root]
    receipt_path = folder / "dataset_receipt.json"
    receipt = read_json(receipt_path)
    if (receipt.get("branch") != "texture" or receipt.get("concept_id") != selection["concept_id"]
            or receipt.get("token") != "<T*>" or receipt.get("image_count") != 16
            or receipt.get("training_assets_manifest_sha256") != sha256(ledger)
            or receipt.get("concepts_sha256") != sha256(concepts_path)
            or audit["datasets"][selection["concept_id"]]["dataset_receipt_sha256"] != sha256(receipt_path)):
        raise ValueError("Texture training dataset receipt differs")
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 16 or len({row["id"] for row in rows}) != 16:
        raise ValueError("expected sixteen distinct Texture training images")
    groups, images = [], []
    for row in rows:
        image, mask = Path(row["image"]), Path(row["mask"])
        if (row.get("branch") != "texture" or row.get("concept_id") != selection["concept_id"]
                or row.get("split") != "train" or row.get("texture_id") != selection["concept_id"]
                or row.get("prompt", "").count("<T*>") != 1 or "<M*>" in row["prompt"]
                or image != folder / "images" / row["id"] / "image.png"
                or mask != folder / "masks" / row["id"] / "image.png"
                or sha256(image) != row["image_sha256"] or sha256(mask) != row["mask_sha256"]):
            raise ValueError("Texture reference is not the verified RGB training image/mask")
        groups.append({"instance_prompt": [row["prompt"]], "instance_data_dir": str(image.parent),
                       "instance_mask_dir": str(mask.parent)})
        images.append({key: row[key] for key in
                       ("id", "image", "image_sha256", "prompt", "shape", "color", "light", "material_id")})
        locks.update({str(image): row["image_sha256"], str(mask): row["mask_sha256"]})
    if read_json(concepts_path) != groups:
        raise ValueError("Texture references differ from trained concept sampling groups")
    for path in (root / "audit.json", receipt_path, concepts_path, ledger, config_path, manifest_path):
        locks[str(path)] = sha256(path)
    return {"method": selection["method"], "texture": selection["concept_id"], "step": step,
            "run": str(run), "checkpoint": selection["checkpoint"], "dataset_manifest": str(ledger),
            "dataset_manifest_sha256": sha256(ledger), "images": images}


def verify_source(source, step):
    """Return only Texture selections, while validating the original eight-run index."""
    source = Path(source)
    if step not in STEPS or read_json(source / "status.json").get("status") != "execution_verified":
        raise ValueError("expected the completed source step experiment")
    index_path = source / f"step{step}/training_index.json"
    provenance_path = source / f"step{step}/inference/provenance.json"
    index, provenance, protocol = read_json(index_path), read_json(provenance_path), data.read_protocol()
    infer.verify_inputs(index, protocol, step)
    if (index.get("training_steps") != step or index.get("source_training_steps") != 1000
            or provenance.get("status") != "succeeded" or provenance.get("training_steps") != step
            or provenance.get("sampling") != protocol["inference"]
            or provenance.get("training_index_sha256") != sha256(index_path)):
        raise ValueError("source step or sampling provenance differs")
    selections = [item for item in index["runs"] if item["branch"] == "texture"]
    expected = {(method, texture) for method in infer.METHODS for texture in infer.target_ids(protocol, "texture")}
    if (len(selections) != 4 or {(item["method"], item["concept_id"]) for item in selections} != expected):
        raise ValueError("expected exactly four Texture concept/method selections")
    locks = {str(path): sha256(path) for path in (source / "status.json", index_path, provenance_path)}
    for item in index["runs"]:
        locks.update(item["hashes"])
    references, staging = {}, {}
    for item in selections:
        adaptation = read_json(Path(item["checkpoint"]) / "adaptation_config.json")
        mode = "token_local_kv" if item["method"] == "fullrank" else "lora_subject_kv"
        if (item.get("token") != "<T*>" or adaptation.get("modifier_tokens") != ["<T*>"]
                or adaptation.get("adaptation_mode") != mode
                or (item["method"] == "lora_r64" and any(adaptation.get(key) != value for key, value in
                    {"rank": 64, "alpha": 64, "mode": "token_local_kv"}.items()))):
            raise ValueError("Texture adapter token, mode or rank differs")
        references[item["method"], item["concept_id"]] = _training_reference(item, step, locks, staging)
    return index, selections, locks, references
