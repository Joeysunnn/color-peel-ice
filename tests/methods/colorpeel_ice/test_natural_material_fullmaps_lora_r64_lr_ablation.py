import json
from pathlib import Path

import pytest

from scripts.launch import colorpeel_run
from experiments.natural_material_fullmaps_lora_r64_a64_lr_step_ablation_v1.evaluate import (
    composition_rows,
    transfer_rows,
    validate_protocol,
)
from src.train.checkpoint_utils import (
    should_save_checkpoint,
    tracker_safe_config,
    validate_checkpoint_plan,
)


ROOT = Path(__file__).parents[3]
EXPERIMENT = ROOT / "experiments" / "natural_material_fullmaps_lora_r64_a64_lr_step_ablation_v1"
CHECKPOINT_STEPS = [600, 1000, 2000, 3000]


def test_exact_checkpoint_plan_preserves_legacy_interval():
    original = {"checkpoint_steps": CHECKPOINT_STEPS, "max_train_steps": 3000}
    assert tracker_safe_config(original) == {
        "checkpoint_steps": "600,1000,2000,3000", "max_train_steps": 3000}
    assert original["checkpoint_steps"] == CHECKPOINT_STEPS
    explicit = validate_checkpoint_plan(CHECKPOINT_STEPS, 1000, 3000)
    assert explicit == frozenset(CHECKPOINT_STEPS)
    assert [step for step in range(1, 3001)
            if should_save_checkpoint(step, explicit, 1000)] == CHECKPOINT_STEPS
    assert [step for step in range(1, 3001)
            if should_save_checkpoint(step, None, 1000)] == [1000, 2000, 3000]


@pytest.mark.parametrize("steps", [[0, 600], [-1, 600], [600, 600], [1000, 600], [600, 3001]])
def test_exact_checkpoint_plan_rejects_invalid_steps(steps):
    with pytest.raises(ValueError, match="checkpoint"):
        validate_checkpoint_plan(steps, 1000, 3000)
    with pytest.raises(ValueError, match="checkpointing_steps"):
        validate_checkpoint_plan(None, 0, 3000)


def test_six_rank64_material_configs_are_locked():
    configs = sorted((EXPERIMENT / "configs").glob("*.json"))
    assert len(configs) == 6
    observed = set()
    for path in configs:
        config = colorpeel_run.read_config(path)
        args = config["args"]
        material = config["source_lock"]["material_id"]
        observed.add((material, args["kv_learning_rate"]))
        assert args["learning_rate"] == 1.0e-5
        assert args["material_lora_mode"] == "token_local_kv"
        assert args["material_lora_rank"] == args["material_lora_alpha"] == 64
        assert args["max_train_steps"] == 3000
        assert args["checkpoint_steps"] == CHECKPOINT_STEPS
        assert "checkpointing_steps" not in args
        assert colorpeel_run.argument_tokens(
            {"checkpoint_steps": args["checkpoint_steps"]}, {}) == [
                "--checkpoint_steps", "600", "1000", "2000", "3000"]
    assert observed == {
        (material, learning_rate)
        for material in ("mailbox", "metal_spoon", "wood_spoon")
        for learning_rate in (1.0e-5, 5.0e-5)
    }


def test_inference_protocol_locks_360_rows_per_trajectory():
    protocol = json.loads(
        (EXPERIMENT / "protocols" / "inference_v1.json").read_text(encoding="utf-8"))
    validate_protocol(protocol)
    for material in ("mailbox", "metal_spoon", "wood_spoon"):
        runtime = dict(protocol, material_id=material)
        transfers = transfer_rows(runtime)
        compositions = composition_rows(runtime)
        assert len(transfers) == 180
        assert len(compositions) == 180
        assert len({row["id"] for row in transfers + compositions}) == 360
        assert {row["material_step"] for row in transfers + compositions} == {
            600, 1000, 2000, 3000}
        assert {row["subject_step"] for row in compositions} == {1000, 2000, 3000}
        assert all(row["prompt"].count("<S*>") == 1
                   and row["prompt"].count("<M*>") == 1
                   for row in compositions)
        assert all("<S*>" not in row["prompt"] for row in transfers)
        assert all(row["prompt"].count("<M*>") == int(row["arm"] == "token")
                   for row in transfers)
