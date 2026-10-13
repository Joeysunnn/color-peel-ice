from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.perfusion_subject_pilot import data_contract
from scripts.launch import colorpeel_run


ROOT = Path(__file__).parents[3]
BASELINE = (ROOT / "experiments" / "subject_material_composition_v1" / "configs"
            / "mailbox_subject_balanced_caption_aligned_5000.yaml")
CONFIG = (ROOT / "experiments" / "subject_token_local_kv_balanced_step_ablation_v1"
          / "configs" / "train.yaml")
PROTOCOL = (ROOT / "experiments" / "subject_token_local_kv_balanced_step_ablation_v1"
            / "protocols" / "training_v1.json")
STEPS = [300, 500, 700, 1000, 2000]


def config_environment(config: dict) -> dict[str, str]:
    return {key: str(value) for key, value in config["environment"].items()}


def test_training_config_changes_only_dose_and_checkpoint_plan(monkeypatch):
    monkeypatch.setattr(
        data_contract,
        "verify_training_data",
        lambda cohort, concepts, manifest, concepts_hash, manifest_hash: {
            "cohort": cohort,
            "concepts_sha256": concepts_hash,
            "asset_manifest_sha256": manifest_hash,
            "row_count": 10,
        },
    )
    baseline = colorpeel_run.read_config(BASELINE)
    config = colorpeel_run.read_config(CONFIG)
    expected_args = deepcopy(baseline["args"])
    expected_args.update(max_train_steps=2000, checkpoint_steps=STEPS)

    assert config["run"] == {
        "study": "subject_token_local_kv_balanced_step_ablation_v1",
        "variant": "balanced_aligned_token_local_kv_steps300_500_700_1000_2000",
        "seed": 42,
    }
    assert config["data_manifest"] == baseline["data_manifest"]
    assert config["args"] == expected_args
    assert config["args"]["token_local_kv"] is True
    assert "subject_lora_mode" not in config["args"]

    result = colorpeel_run.validate_subject_token_local_kv_balanced_step_train_inputs(
        config, config_environment(config))
    assert result == {
        "cohort": "balanced_aligned",
        "concepts_sha256": config["subject_source"]["concepts_sha256"],
        "asset_manifest_sha256": config["subject_source"]["asset_manifest_sha256"],
        "row_count": 10,
    }

    command = colorpeel_run.build_command(config, Path("run"), config_environment(config))
    start = command.index("--checkpoint_steps")
    assert command[start + 1:start + 6] == [str(step) for step in STEPS]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["args"].__setitem__("max_train_steps", 5000),
         "only the dose"),
        (lambda value: value["args"].__setitem__("checkpoint_steps", [300, 2000]),
         "only the dose"),
        (lambda value: value["args"].__setitem__("kv_learning_rate", 5e-5),
         "only the dose"),
        (lambda value: value["subject_source"].__setitem__("concepts_sha256", "0" * 64),
         "source lock"),
        (lambda value: value["training_protocol"].__setitem__("sha256", "0" * 64),
         "protocol lock"),
        (lambda value: value["environment"].__setitem__(
            "COLORPEEL_MAILBOX_MATTE_ROOT", "/tmp/different-assets"),
         "protocol differs"),
    ],
)
def test_launcher_rejects_training_or_source_drift(monkeypatch, mutate, message):
    monkeypatch.setattr(data_contract, "verify_training_data", lambda *args: {})
    config = colorpeel_run.read_config(CONFIG)
    mutate(config)
    with pytest.raises(ValueError, match=message):
        colorpeel_run.validate_subject_token_local_kv_balanced_step_train_inputs(
            config, config_environment(config))


def test_protocol_and_balanced_ten_row_assets_are_locked(tmp_path):
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    config = colorpeel_run.read_config(CONFIG)
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == config["training_protocol"]["sha256"]
    assert protocol["training_data"]["row_count"] == 10
    assert protocol["training"]["checkpoint_steps"] == STEPS
    assert protocol["subject"] == {
        "modifier_token": "<S*>",
        "initializer_token": "mailbox",
        "adaptation_mode": "token_local_kv",
        "lora": False,
    }

    captions = {
        "red": "pink",
        "green": "green",
        "cyan": "cyan",
        "blue": "blue",
        "magenta": "purple",
    }
    concepts = []
    records = []
    for color in captions:
        for material in ("metal", "matte"):
            item = tmp_path / color / material
            image_dir, mask_dir = item / "images", item / "masks"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir()
            image, mask = image_dir / "image.png", mask_dir / "image.png"
            image.write_bytes(f"{color}-{material}-image".encode())
            mask.write_bytes(f"{color}-{material}-mask".encode())
            suffix = "glossy painted metal" if material == "metal" else "matte plastic"
            concepts.append({
                "instance_prompt": [
                    f"a photo of <S*> mailbox in {captions[color]} color made of {suffix}"],
                "instance_data_dir": str(image_dir),
                "instance_mask_dir": str(mask_dir),
            })
            records.append({
                "color": color,
                "material": material,
                "image": str(image),
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "mask": str(mask),
                "mask_sha256": hashlib.sha256(mask.read_bytes()).hexdigest(),
            })

    concepts_path = tmp_path / "balanced_concepts.json"
    manifest_path = tmp_path / "training_assets_manifest.jsonl"
    concepts_path.write_text(json.dumps(concepts), encoding="utf-8")
    manifest_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    result = data_contract.verify_training_data(
        "balanced_aligned", concepts_path, manifest_path,
        hashlib.sha256(concepts_path.read_bytes()).hexdigest(),
        hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    assert result["row_count"] == 10

    (tmp_path / "red" / "metal" / "images" / "image.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="training image or mask differs"):
        data_contract.verify_training_data(
            "balanced_aligned", concepts_path, manifest_path,
            hashlib.sha256(concepts_path.read_bytes()).hexdigest(),
            hashlib.sha256(manifest_path.read_bytes()).hexdigest())
