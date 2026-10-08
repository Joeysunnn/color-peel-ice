from copy import deepcopy
from pathlib import Path

import pytest

from experiments.lora_kv_subject_step_ablation_v1.evaluate import (
    ARMS,
    comparison_rows,
    read_json,
    validate_protocol,
)
from experiments.perfusion_subject_pilot import data_contract
from scripts.launch import colorpeel_run
from src.train.checkpoint_utils import (
    should_save_checkpoint,
    tracker_safe_config,
    validate_checkpoint_plan,
)


ROOT = Path(__file__).parents[3]
EXPERIMENT = ROOT / "experiments" / "lora_kv_subject_r64_a64_lr_step_ablation_v1"
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


def test_launcher_locks_eight_rank64_learning_rate_configs(monkeypatch):
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
    configs = sorted((EXPERIMENT / "configs").glob("*.yaml"))
    assert len(configs) == 8
    observed = set()
    for path in configs:
        config = colorpeel_run.read_config(path)
        args = config["args"]
        observed.add((args["subject_lora_mode"], args["kv_learning_rate"]))
        assert args["learning_rate"] == 1.0e-5
        assert args["subject_lora_rank"] == args["subject_lora_alpha"] == 64
        assert args["checkpoint_steps"] == CHECKPOINT_STEPS
        assert "checkpointing_steps" not in args
        result = colorpeel_run.validate_lora_subject_train_inputs(
            config, {key: str(value) for key, value in config["environment"].items()})
        assert result["row_count"] == 10
        tokens = colorpeel_run.argument_tokens(
            {"checkpoint_steps": args["checkpoint_steps"]}, {})
        assert tokens == ["--checkpoint_steps", "600", "1000", "2000", "3000"]
    assert observed == {(mode, lr) for mode in ARMS for lr in (1.0e-5, 5.0e-5)}

    invalid = deepcopy(colorpeel_run.read_config(configs[0]))
    invalid["args"]["kv_learning_rate"] = 5.0e-5
    with pytest.raises(ValueError, match="cohort, mode or status"):
        colorpeel_run.validate_lora_subject_train_inputs(
            invalid, {key: str(value) for key, value in invalid["environment"].items()})
    invalid = deepcopy(colorpeel_run.read_config(configs[0]))
    invalid["args"]["checkpoint_steps"] = [600, 1000, 3000]
    with pytest.raises(ValueError, match="must match"):
        colorpeel_run.validate_lora_subject_train_inputs(
            invalid, {key: str(value) for key, value in invalid["environment"].items()})


@pytest.mark.parametrize(
    ("filename", "learning_rate"),
    [("comparison_kvlr1em5_v1.json", 1.0e-5),
     ("comparison_kvlr5em5_v1.json", 5.0e-5)],
)
def test_rank64_evaluation_protocols_are_locked(filename, learning_rate):
    protocol = read_json(EXPERIMENT / "protocols" / filename)
    validate_protocol(protocol)
    assert protocol["lora"] == {"rank": 64, "alpha": 64.0}
    assert protocol["snapshot_steps"] == CHECKPOINT_STEPS
    for mode in ARMS:
        rows = comparison_rows(protocol, mode)
        assert len(rows) == 180
        assert {row["step"] for row in rows} == set(CHECKPOINT_STEPS)
    invalid = deepcopy(protocol)
    invalid["lora"]["alpha"] = 8.0
    with pytest.raises(ValueError, match="unexpected"):
        validate_protocol(invalid)
