from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.lora_kv_subject_step_ablation_v1.attention import (
    install_subject_material_lora_kv,
)
from experiments.lora_kv_subject_step_ablation_v1.evaluate import (
    ARMS,
    checked_output_dir,
    comparison_rows,
    read_json,
    validate_protocol,
)
from experiments.lora_kv_subject_v1.attention import (
    SubjectLoraKVAttnProcessor,
    subject_lora_state_dict,
)
from experiments.perfusion_subject_pilot import data_contract
from scripts.launch import colorpeel_run


ROOT = Path(__file__).parents[3]
EXPERIMENT = ROOT / "experiments" / "lora_kv_subject_step_ablation_v1"


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_k = nn.Linear(3, 4, bias=False)
        self.to_v = nn.Linear(3, 4, bias=False)


class FakeUnet(nn.Module):
    def __init__(self, mode, rank=2, alpha=2):
        super().__init__()
        self.attn2 = FakeAttention()
        self.attn_processors = {
            "attn1.processor": SubjectLoraKVAttnProcessor(mode=mode, rank=rank, alpha=alpha),
            "attn2.processor": SubjectLoraKVAttnProcessor(4, 3, mode, rank, alpha),
        }

    def set_attn_processor(self, processors):
        self.attn_processors = processors


def state(mode, k_up, v_up):
    source = FakeUnet(mode)
    processor = source.attn_processors["attn2.processor"]
    with torch.no_grad():
        processor.to_v_lora.up.weight.fill_(v_up)
        if hasattr(processor, "to_k_lora"):
            processor.to_k_lora.up.weight.fill_(k_up)
    return subject_lora_state_dict(source)


@pytest.mark.parametrize("mode", ARMS)
def test_all_subject_modes_compose_with_token_local_material(mode):
    torch.manual_seed(7)
    subject = state(mode, 0.3, 0.2)
    material = state("token_local_kv", 0.5, 0.4)
    target = FakeUnet(mode)
    install_subject_material_lora_kv(target, subject, material, mode, rank=2, alpha=2)
    processor = target.attn_processors["attn2.processor"]
    assert all(not parameter.requires_grad for parameter in processor.parameters())

    hidden = torch.randn(1, 4, 3)
    subject_mask = torch.tensor([[False, True, False, False]])
    material_mask = torch.tensor([[False, False, True, False]])
    key, value = processor.project_kv(
        target.attn2, hidden, {"subject": subject_mask, "material": material_mask})
    base_k, base_v = target.attn2.to_k(hidden), target.attn2.to_v(hidden)
    subject_k = processor.to_k_lora(hidden) if hasattr(processor, "to_k_lora") else 0
    subject_v = processor.to_v_lora(hidden)
    material_k = processor.material_to_k_lora(hidden)
    material_v = processor.material_to_v_lora(hidden)

    expected_k = base_k.clone()
    expected_v = base_v.clone()
    if mode == "full_kv":
        expected_k += subject_k
    elif mode == "token_local_kv":
        expected_k[:, 1] += subject_k[:, 1]
    if mode.startswith("full"):
        expected_v += subject_v
    else:
        expected_v[:, 1] += subject_v[:, 1]
    expected_k[:, 2] += material_k[:, 2]
    expected_v[:, 2] += material_v[:, 2]
    assert torch.allclose(key, expected_k)
    assert torch.allclose(value, expected_v)


def test_composition_rejects_bad_masks_and_checkpoint_keys():
    subject = state("token_local_v", 0.0, 0.2)
    material = state("token_local_kv", 0.5, 0.4)
    target = FakeUnet("token_local_v")
    install_subject_material_lora_kv(target, subject, material, "token_local_v", rank=2, alpha=2)
    processor = target.attn_processors["attn2.processor"]
    hidden = torch.randn(1, 3, 3)
    mask = torch.tensor([[False, True, False]])
    with pytest.raises(ValueError, match="separate"):
        processor.project_kv(target.attn2, hidden, mask)
    with pytest.raises(ValueError, match="disjoint"):
        processor.project_kv(target.attn2, hidden, {"subject": mask, "material": mask})
    broken = dict(material)
    broken.pop(next(iter(broken)))
    with pytest.raises(ValueError, match="missing or mismatched"):
        install_subject_material_lora_kv(
            FakeUnet("token_local_v"), subject, broken, "token_local_v", rank=2, alpha=2)


def test_protocol_rows_and_material_lock():
    protocol = read_json(EXPERIMENT / "protocols" / "comparison_v1.json")
    validate_protocol(protocol)
    assert protocol["sampling"]["seeds"] == [42, 43, 44, 45, 46]
    assert protocol["material_checkpoint"]["weights_sha256"] == (
        "9713ae80319aabb7ec08faa4349cfa2e03918bab915228bfa447afea6d7f170b")
    assert protocol["material_checkpoint"]["adaptation_config_sha256"] == (
        "cefde5e7bf5a623a7d682646e8d7f07de28c5339453ff399efe916a2fbe32280")
    for mode in ARMS:
        rows = comparison_rows(protocol, mode)
        assert len(rows) == 135
        assert {row["step"] for row in rows} == {1000, 2000, 3000}
        assert {row["condition"] for row in rows} == {
            "subject_only", "subject_literal_material", "subject_material_token"}
        assert all(row["prompt"].count("<M*>") ==
                   int(row["condition"] == "subject_material_token") for row in rows)
    changed = deepcopy(protocol)
    changed["sampling"]["guidance_scale"] = 7.5
    with pytest.raises(ValueError, match="unexpected"):
        validate_protocol(changed)
    changed = deepcopy(protocol)
    changed["material_checkpoint"]["weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="unexpected"):
        validate_protocol(changed)


def test_output_directory_must_not_be_inside_training_runs(tmp_path):
    subject = tmp_path / "subject-run"
    material = tmp_path / "material-run"
    with pytest.raises(ValueError, match="must not be inside"):
        checked_output_dir(subject / "evaluation" / "new", [subject, material])
    output = tmp_path / "evaluations" / "new"
    assert checked_output_dir(output, [subject, material]) == output.resolve()


def test_launcher_authorizes_only_locked_3000_step_configs(monkeypatch):
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
    for path in configs:
        config = colorpeel_run.read_config(path)
        result = colorpeel_run.validate_lora_subject_train_inputs(
            config, {key: str(value) for key, value in config["environment"].items()})
        assert result == {
            "cohort": "balanced_aligned",
            "concepts_sha256": config["lora_source"]["concepts_sha256"],
            "asset_manifest_sha256": config["lora_source"]["asset_manifest_sha256"],
            "row_count": 10,
        }
    invalid = deepcopy(colorpeel_run.read_config(configs[0]))
    invalid["args"]["max_train_steps"] = 2000
    with pytest.raises(ValueError, match="must match"):
        colorpeel_run.validate_lora_subject_train_inputs(
            invalid, {key: str(value) for key, value in invalid["environment"].items()})
    invalid = deepcopy(colorpeel_run.read_config(configs[0]))
    invalid["run"]["seed"] = 43
    with pytest.raises(ValueError, match="cohort, mode or status"):
        colorpeel_run.validate_lora_subject_train_inputs(
            invalid, {key: str(value) for key, value in invalid["environment"].items()})
