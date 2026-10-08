from copy import deepcopy
from pathlib import Path

import pytest

from experiments.perfusion_subject_pilot import data_contract
from experiments.lora_kv_subject_step_ablation_v1.evaluate import (
    ARMS,
    comparison_rows,
    read_json,
    validate_protocol,
)
from scripts.launch import colorpeel_run


ROOT = Path(__file__).parents[3]
EXPERIMENT = ROOT / "experiments" / "lora_kv_subject_alpha8_step_ablation_v1"


def test_alpha8_evaluation_protocol_is_locked():
    protocol = read_json(EXPERIMENT / "protocols" / "comparison_v1.json")
    validate_protocol(protocol)
    assert protocol["lora"] == {"rank": 4, "alpha": 8.0}
    for mode in ARMS:
        rows = comparison_rows(protocol, mode)
        assert len(rows) == 135
        assert {row["step"] for row in rows} == {1000, 2000, 3000}
        assert {row["condition"] for row in rows} == {
            "subject_only", "subject_literal_material", "subject_material_token"}

    invalid = deepcopy(protocol)
    invalid["lora"]["alpha"] = 4.0
    with pytest.raises(ValueError, match="unexpected"):
        validate_protocol(invalid)


def test_launcher_authorizes_only_locked_alpha8_configs(monkeypatch):
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
    assert len(configs) == 4
    assert {
        colorpeel_run.read_config(path)["args"]["subject_lora_mode"]
        for path in configs
    } == {"full_kv", "token_local_kv", "full_v", "token_local_v"}

    for path in configs:
        config = colorpeel_run.read_config(path)
        assert config["environment"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert config["args"]["subject_lora_rank"] == 4
        assert config["args"]["subject_lora_alpha"] == 8
        assert config["args"]["max_train_steps"] == 3000
        assert config["args"]["checkpointing_steps"] == 1000
        result = colorpeel_run.validate_lora_subject_train_inputs(
            config, {key: str(value) for key, value in config["environment"].items()})
        assert result == {
            "cohort": "balanced_aligned",
            "concepts_sha256": config["lora_source"]["concepts_sha256"],
            "asset_manifest_sha256": config["lora_source"]["asset_manifest_sha256"],
            "row_count": 10,
        }

    invalid = deepcopy(colorpeel_run.read_config(configs[0]))
    invalid["args"]["subject_lora_alpha"] = 4
    with pytest.raises(ValueError, match="must match"):
        colorpeel_run.validate_lora_subject_train_inputs(
            invalid, {key: str(value) for key, value in invalid["environment"].items()})

    invalid = deepcopy(colorpeel_run.read_config(configs[0]))
    invalid["run"]["variant"] = invalid["run"]["variant"].replace("_a8", "")
    with pytest.raises(ValueError, match="cohort, mode or status"):
        colorpeel_run.validate_lora_subject_train_inputs(
            invalid, {key: str(value) for key, value in invalid["environment"].items()})
