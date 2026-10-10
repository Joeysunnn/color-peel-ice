from copy import deepcopy
from pathlib import Path

import pytest

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
