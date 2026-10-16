"""Frozen SD1.4 recipes with a separate, local SD1.5 base model."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path


STUDY = "sd15_token_local_kv_v1"
ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "protocol.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_sha256():
    # Git may normalize Windows line endings; lock parsed protocol semantics.
    data = json.dumps(read_json(PROTOCOL), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def build_config(recipe, smoke=False):
    protocol = read_json(PROTOCOL)
    source = protocol["recipes"][recipe]["source_config"]
    args = deepcopy(source["args"])
    args["pretrained_model_name_or_path"] = protocol["model"]["root"]
    if smoke:
        args.update(max_train_steps=2, checkpoint_steps=[2])
    environment = deepcopy(source.get("environment", {}))
    # GPU selection belongs to the invocation, not to an old occupied device.
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DIFFUSERS_OFFLINE="1")
    return {
        "schema_version": 1, "status": "authorized_sd15_migration", "stage": "train",
        "run": {"study": STUDY, "variant": recipe + ("_smoke2" if smoke else ""), "seed": 42},
        "environment": environment, "data_manifest": source["data_manifest"],
        "sd15_migration": {"recipe": recipe, "smoke": smoke, "protocol_sha256": protocol_sha256()},
        "args": args,
    }


def verify_directory_members(concepts, records, staged=False):
    """The trainer enumerates directories, so reject unmanifested input files."""
    for concept in concepts:
        for field, directory_key in (("image", "instance_data_dir"), ("mask", "instance_mask_dir")):
            directory = Path(concept[directory_key]).resolve()
            record_key = "staged_" + field if staged else field
            expected = {Path(row[record_key]).resolve() for row in records
                        if Path(row[record_key]).resolve().parent == directory}
            if not expected or set(directory.iterdir()) != expected:
                raise ValueError("training directory has unmanifested or missing files")


def validate_config(config, environment, verify_assets=True):
    """Reject recipe drift, then verify the original read-only input assets."""
    selection = config.get("sd15_migration", {})
    recipe, smoke = selection.get("recipe"), selection.get("smoke")
    protocol = read_json(PROTOCOL)
    if recipe not in protocol["recipes"] or type(smoke) is not bool:
        raise ValueError("unknown SD1.5 migration recipe or smoke mode")
    if config != build_config(recipe, smoke):
        raise ValueError("SD1.5 recipe differs from its frozen source except base/smoke dose")
    if not verify_assets:
        return {"recipe": recipe, "assets_verified": False}

    from scripts.launch.colorpeel_run import expand_value, validate_material_pilot_train_inputs
    from experiments.perfusion_subject_pilot.data_contract import verify_training_data

    record = protocol["recipes"][recipe]
    source = record["source_config"]
    concepts = Path(expand_value(source["args"]["concepts_list"], environment)).resolve()
    manifest = Path(expand_value(source["data_manifest"], environment)).resolve()
    if record["data_kind"] == "balanced_subject":
        lock = source.get("subject_source", source.get("lora_source"))
        verify_training_data("balanced_aligned", concepts, manifest,
                             lock["concepts_sha256"], lock["asset_manifest_sha256"])
        rows = 10
    elif record["data_kind"] == "controlled_material":
        validate_material_pilot_train_inputs(source, environment)
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        verify_directory_members(read_json(concepts), records, staged=True)
        rows = 72
    else:
        lock = source["source_lock"]
        staging = concepts.parent
        checks = {concepts: lock["concepts_sha256"],
                  manifest: lock["training_assets_manifest_sha256"],
                  staging / "staging_manifest.json": lock["staging_manifest_sha256"]}
        if any(sha256(path) != expected for path, expected in checks.items()):
            raise ValueError("full-map material source hash differs")
        details = read_json(staging / "staging_manifest.json")
        grid = staging.parent.parent / "grids" / lock["material_id"] / "manifest.json"
        if (sha256(grid) != lock["source_grid_manifest_sha256"]
                or details["source_grid_manifest_sha256"] != lock["source_grid_manifest_sha256"]
                or details["train_images"] != 72 or details["held_out_cone_images"] != 24):
            raise ValueError("full-map grid or split differs")
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        if len(records) != 72 or {shape: sum(r["shape"] == shape for r in records)
                                 for shape in ("sphere", "cube", "cylinder")} != dict.fromkeys(
                                     ("sphere", "cube", "cylinder"), 24):
            raise ValueError("full-map material must retain 72 training images")
        expected_concepts = []
        for shape in ("cube", "cylinder", "sphere"):
            expected_concepts.append({"instance_prompt": [f"a photo of a {shape} made of <M*>"],
                                      "instance_data_dir": str(staging / "images" / shape),
                                      "instance_mask_dir": str(staging / "masks" / shape)})
        if read_json(concepts) != expected_concepts:
            raise ValueError("full-map captions or concept directories differ")
        for row in records:
            for field in ("image", "mask"):
                path = Path(row[field]).resolve()
                if not path.is_relative_to(staging) or sha256(path) != row[field + "_sha256"]:
                    raise ValueError("full-map image/mask differs")
        verify_directory_members(expected_concepts, records)
        rows = len(records)
    model = Path(protocol["model"]["root"])
    receipt = read_json(model / "model_source.json")
    if (receipt["model"] != protocol["model"]
            or any(sha256(model / name) != value for name, value in receipt["files"].items())):
        raise ValueError("prepared SD1.5 model differs from its immutable receipt")
    return {"recipe": recipe, "assets_verified": True, "row_count": rows,
            "concepts_sha256": sha256(concepts), "manifest_sha256": sha256(manifest),
            "model_receipt_sha256": sha256(model / "model_source.json"),
            "protocol_sha256": protocol_sha256()}
