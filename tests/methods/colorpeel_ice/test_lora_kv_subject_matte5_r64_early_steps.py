from copy import deepcopy
from pathlib import Path

import pytest

from experiments.lora_kv_subject_matte5_r64_v1 import evaluate
from scripts.launch import colorpeel_run
from src.train.checkpoint_utils import (
    should_save_checkpoint,
    tracker_safe_config,
    validate_checkpoint_plan,
)


ROOT = Path(__file__).parents[3]
OLD_CONFIG = (ROOT / "experiments" / "lora_kv_subject_matte5_r64_v1"
              / "configs" / "train.json")
NEW_CONFIG = (ROOT / "experiments" / "lora_kv_subject_matte5_r64_early_steps_v1"
              / "configs" / "train.json")
CHECKPOINT_STEPS = [300, 500, 700]
OLD_PROTOCOL = (ROOT / "experiments" / "lora_kv_subject_matte5_r64_v1"
                / "protocols" / "inference_v1.json")
NEW_PROTOCOL = (ROOT / "experiments" / "lora_kv_subject_matte5_r64_early_steps_v1"
                / "protocols" / "inference_v1.json")


def test_explicit_checkpoint_plan_saves_only_early_steps():
    explicit = validate_checkpoint_plan(CHECKPOINT_STEPS, 1000, 700)
    assert explicit == frozenset(CHECKPOINT_STEPS)
    assert [step for step in range(1, 701)
            if should_save_checkpoint(step, explicit, 1000)] == CHECKPOINT_STEPS
    assert tracker_safe_config({"checkpoint_steps": CHECKPOINT_STEPS}) \
        == {"checkpoint_steps": "300,500,700"}
    assert [step for step in range(1, 3001)
            if should_save_checkpoint(step, None, 1000)] == [1000, 2000, 3000]


@pytest.mark.parametrize(
    "steps", [[], [0, 300], [-1, 300], [300, 300], [500, 300], [300, 701]])
def test_explicit_checkpoint_plan_rejects_invalid_steps(steps):
    with pytest.raises(ValueError, match="checkpoint_steps"):
        validate_checkpoint_plan(steps, 1000, 700)
    with pytest.raises(ValueError, match="checkpointing_steps"):
        validate_checkpoint_plan(None, 0, 700)


def test_early_step_config_changes_only_run_identity_and_step_plan():
    old = colorpeel_run.read_config(OLD_CONFIG)
    new = colorpeel_run.read_config(NEW_CONFIG)
    assert new["run"] == {
        "study": "lora_kv_subject_matte5_r64_early_steps_v1",
        "variant": "mailbox_matte5_token_local_kv_r64_a64_kvlr5em5_steps300_500_700",
        "seed": 42,
    }
    assert new["status"] == "authorized_lora_subject_matte5_r64_early_steps"
    assert new["data_manifest"] == old["data_manifest"]
    assert new["matte5_source"] == old["matte5_source"]
    assert new["environment"] == old["environment"]
    expected_args = deepcopy(old["args"])
    expected_args["max_train_steps"] = 700
    expected_args.pop("checkpointing_steps")
    expected_args["checkpoint_steps"] = CHECKPOINT_STEPS
    assert new["args"] == expected_args
    assert colorpeel_run.argument_tokens(
        {"checkpoint_steps": CHECKPOINT_STEPS}, {}) \
        == ["--checkpoint_steps", "300", "500", "700"]


def test_early_inference_protocol_changes_only_subject_source_and_steps():
    old = evaluate.read_json(OLD_PROTOCOL)
    new = evaluate.read_json(NEW_PROTOCOL)
    evaluate.validate_protocol(old)
    evaluate.validate_protocol(new)
    for field in ("base_model", "materials", "comparison", "transfer", "safety_checker"):
        assert new[field] == old[field]
    assert new["subject"]["snapshot_steps"] == CHECKPOINT_STEPS
    assert set(new["subject"]["source_snapshot_sha256"]) == {"300", "500", "700"}
    for field in ("mode", "rank", "alpha", "kv_learning_rate"):
        assert new["subject"][field] == old["subject"][field]


def test_early_inference_matrix_has_600_unique_rows():
    protocol = evaluate.read_json(NEW_PROTOCOL)
    comparison = [row for step in CHECKPOINT_STEPS
                  for row in evaluate.comparison_rows(protocol, step)]
    source = [
        {
            "id": f"source-{index}", "color": f"prompt-{index // 5}",
            "prompt": f"a photo of <S*> in scene {index // 5}",
            "seed": 42 + index % 5, "num_inference_steps": 100,
            "guidance_scale": 3.5,
        }
        for index in range(140)
    ]
    transfer = [row for step in CHECKPOINT_STEPS
                for row in evaluate.transfer_rows(source, step)]
    assert len(comparison) == 180
    assert len(transfer) == 420
    assert len(comparison) + len(transfer) == 600
    assert len({row["id"] for row in comparison}) == 180
    assert len({row["id"] for row in transfer}) == 420
    with pytest.raises(ValueError, match="unsupported Subject snapshot"):
        evaluate.comparison_rows(protocol, 1000)
