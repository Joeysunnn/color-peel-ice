import pytest
import torch
from torch import nn

from experiments.lora_kv_subject_v1.attention import (
    SubjectLoraKVAttnProcessor, install_subject_lora_kv, subject_lora_state_dict,
)


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


def active_processor(mode, rank=2, alpha=2):
    attn = FakeAttention()
    attn.to_k.requires_grad_(False)
    attn.to_v.requires_grad_(False)
    processor = SubjectLoraKVAttnProcessor(4, 3, mode, rank, alpha)
    with torch.no_grad():
        processor.to_v_lora.up.weight.fill_(0.2)
        if hasattr(processor, "to_k_lora"):
            processor.to_k_lora.up.weight.fill_(0.3)
    return attn, processor


@pytest.mark.parametrize("mode", ["full_kv", "token_local_kv", "full_v", "token_local_v"])
def test_projection_scope_and_frozen_k(mode):
    torch.manual_seed(1)
    attn, processor = active_processor(mode)
    hidden = torch.randn(1, 4, 3)
    subject = torch.tensor([[False, True, False, False]])
    key, value = processor.project_kv(attn, hidden, subject)
    base_k, base_v = attn.to_k(hidden), attn.to_v(hidden)
    ordinary = ~subject[0]
    if mode.startswith("token_local"):
        assert torch.equal(key[:, ordinary], base_k[:, ordinary])
        assert torch.equal(value[:, ordinary], base_v[:, ordinary])
    else:
        assert not torch.equal(value[:, ordinary], base_v[:, ordinary])
    if mode.endswith("_v"):
        assert torch.equal(key, base_k)
        assert not hasattr(processor, "to_k_lora")
    else:
        assert not torch.equal(key[:, subject[0]], base_k[:, subject[0]])
    assert not torch.equal(value[:, subject[0]], base_v[:, subject[0]])


def test_zero_initialization_alpha_scaling_and_gradients():
    torch.manual_seed(2)
    attn = FakeAttention()
    attn.to_k.requires_grad_(False)
    attn.to_v.requires_grad_(False)
    processor = SubjectLoraKVAttnProcessor(4, 3, "full_kv", rank=2, alpha=4)
    hidden = torch.randn(1, 3, 3)
    assert torch.equal(processor.project_kv(attn, hidden)[0], attn.to_k(hidden))
    assert torch.equal(processor.project_kv(attn, hidden)[1], attn.to_v(hidden))
    with torch.no_grad():
        processor.to_k_lora.up.weight.fill_(0.5)
        processor.to_v_lora.up.weight.fill_(0.5)
    key, value = processor.project_kv(attn, hidden)
    assert torch.allclose(key - attn.to_k(hidden), processor.to_k_lora.up(
        processor.to_k_lora.down(hidden)) * 2)
    (key.sum() + value.sum()).backward()
    assert attn.to_k.weight.grad is None and attn.to_v.weight.grad is None
    assert all(parameter.grad is not None for parameter in processor.parameters())


def test_local_requires_boolean_mask_and_absent_token_is_exact_base():
    attn, processor = active_processor("token_local_v")
    hidden = torch.randn(1, 3, 3)
    with pytest.raises(ValueError, match="Subject mask"):
        processor.project_kv(attn, hidden)
    with pytest.raises(ValueError, match="Boolean"):
        processor.project_kv(attn, hidden, torch.zeros(1, 3))
    key, value = processor.project_kv(attn, hidden, torch.zeros(1, 3, dtype=torch.bool))
    assert torch.equal(key, attn.to_k(hidden))
    assert torch.equal(value, attn.to_v(hidden))


@pytest.mark.parametrize("mode", ["full_kv", "token_local_kv", "full_v", "token_local_v"])
def test_material_composition_and_checkpoint_round_trip(mode):
    torch.manual_seed(3)
    source = FakeUnet(mode)
    processor = source.attn_processors["attn2.processor"]
    with torch.no_grad():
        processor.to_v_lora.up.weight.fill_(0.2)
        if hasattr(processor, "to_k_lora"):
            processor.to_k_lora.up.weight.fill_(0.3)
    state = subject_lora_state_dict(source)
    assert all("material_delta" not in name for name in state)
    assert (not any("to_k_lora" in name for name in state)) == mode.endswith("_v")
    material = {
        "attn1.processor": {},
        "attn2.processor.delta_k.weight": torch.full((4, 3), 0.1),
        "attn2.processor.delta_v.weight": torch.full((4, 3), 0.2),
    }
    target = FakeUnet(mode)
    target.attn2.load_state_dict(source.attn2.state_dict())
    install_subject_lora_kv(target, state, mode, rank=2, alpha=2, material_state=material)
    assert set(subject_lora_state_dict(target)) == set(state)
    assert all(torch.equal(state[key], value) for key, value in subject_lora_state_dict(target).items())
    installed = target.attn_processors["attn2.processor"]
    assert all(not p.requires_grad for p in installed.material_delta_k.parameters())
    hidden = torch.randn(1, 4, 3)
    subject = torch.tensor([[False, True, False, False]])
    material_mask = torch.tensor([[False, False, True, False]])
    masks = {"subject": subject, "material": material_mask}
    key, value = installed.project_kv(target.attn2, hidden, masks)
    subject_key, subject_value = processor.project_kv(source.attn2, hidden, subject)
    expected_key = subject_key.clone()
    expected_value = subject_value.clone()
    expected_key[:, 2] += installed.material_delta_k(hidden)[:, 2]
    expected_value[:, 2] += installed.material_delta_v(hidden)[:, 2]
    assert torch.allclose(key, expected_key)
    assert torch.allclose(value, expected_value)
    with pytest.raises(ValueError, match="disjoint"):
        installed.project_kv(target.attn2, hidden, {"subject": subject, "material": subject})
    broken = dict(state)
    broken.pop(next(iter(broken)))
    with pytest.raises(ValueError, match="missing or mismatched"):
        install_subject_lora_kv(FakeUnet(mode), broken, mode, rank=2, alpha=2)
