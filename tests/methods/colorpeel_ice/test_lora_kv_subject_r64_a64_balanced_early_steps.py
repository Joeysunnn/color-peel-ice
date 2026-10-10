from copy import deepcopy
from pathlib import Path

import pytest

from experiments.lora_kv_subject_r64_a64_balanced_early_steps_v1 import evaluate
from experiments.perfusion_subject_pilot import data_contract
from scripts.launch import colorpeel_run
from src.train.checkpoint_utils import should_save_checkpoint, validate_checkpoint_plan


ROOT = Path(__file__).parents[3]
SOURCE = (ROOT / "experiments" / "lora_kv_subject_r64_a64_lr_step_ablation_v1"
          / "configs" / "balanced_aligned_token_local_kv_r64_a64_kvlr5em5_3000.yaml")
CONFIG = (ROOT / "experiments" / "lora_kv_subject_r64_a64_balanced_early_steps_v1"
          / "configs" / "train.yaml")
PROTOCOL = (ROOT / "experiments" / "lora_kv_subject_r64_a64_balanced_early_steps_v1"
            / "protocols" / "inference_v1.json")


def test_early_config_changes_only_identity_and_step_plan(monkeypatch):
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
    source = colorpeel_run.read_config(SOURCE)
    config = colorpeel_run.read_config(CONFIG)
    assert config["run"] == {
        "study": "lora_kv_subject_r64_a64_balanced_early_steps_v1",
        "variant": "balanced_aligned_token_local_kv_r64_a64_kvlr5em5_steps300_500_700",
        "seed": 42,
    }
    assert config["status"] == "authorized_lora_subject_r64_a64_balanced_early_steps"
    assert config["data_manifest"] == source["data_manifest"]
    assert config["lora_source"] == source["lora_source"]
    expected_args = deepcopy(source["args"])
    expected_args["max_train_steps"] = 700
    expected_args["checkpoint_steps"] = [300, 500, 700]
    assert config["args"] == expected_args
    result = colorpeel_run.validate_lora_subject_train_inputs(
        config, {key: str(value) for key, value in config["environment"].items()})
    assert result["row_count"] == 10


def test_early_checkpoint_plan_is_exact():
    steps = [300, 500, 700]
    explicit = validate_checkpoint_plan(steps, 500, 700)
    assert [step for step in range(1, 701)
            if should_save_checkpoint(step, explicit, 500)] == steps
    invalid = colorpeel_run.read_config(CONFIG)
    invalid["args"]["checkpoint_steps"] = [300, 700]
    with pytest.raises(ValueError, match="must match"):
        colorpeel_run.validate_lora_subject_train_inputs(
            invalid, {key: str(value) for key, value in invalid["environment"].items()})


def test_inference_protocol_locks_four_conditions_and_600_rows():
    protocol = evaluate.read_json(PROTOCOL)
    evaluate.validate_protocol(protocol)
    assert protocol["subject"]["snapshot_steps"] == [300, 500, 700]
    assert tuple(protocol["materials"]) == ("metal_spoon", "wood_spoon")
    comparison = [
        row for step in evaluate.SUBJECT_STEPS
        for row in evaluate.comparison_rows(protocol, step)
    ]
    assert len(comparison) == 180
    assert len({row["id"] for row in comparison}) == 180
    assert {row["condition"] for row in comparison} == set(evaluate.CONDITIONS)
    assert {row["subject_step"] for row in comparison} == {300, 500, 700}

    source = [
        {
            "id": f"source-{index}",
            "color": f"prompt-{index // 5}",
            "prompt": f"a photo of <S*> in scene {index // 5}",
            "seed": 42 + index % 5,
            "num_inference_steps": 100,
            "guidance_scale": 3.5,
        }
        for index in range(140)
    ]
    transfer = [
        row for step in evaluate.SUBJECT_STEPS
        for row in evaluate.transfer_rows(source, step)
    ]
    assert len(transfer) == 420
    assert len({row["id"] for row in transfer}) == 420
    assert len(comparison) + len(transfer) == 600
