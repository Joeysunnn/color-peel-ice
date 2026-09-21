import sys
from pathlib import Path

import torch
from torch import nn


TRAIN_ROOT = Path(__file__).parents[3] / "src" / "train"
sys.path.insert(0, str(TRAIN_ROOT))
from custom_attention.attention_processor_custom import TokenLocalKVAttnProcessor, build_modifier_token_mask


class FakeAttention:
    def __init__(self):
        self.to_k = nn.Linear(3, 4, bias=False)
        self.to_v = nn.Linear(3, 4, bias=False)


def test_modifier_mask_marks_only_all_explicit_subject_token_positions():
    input_ids = torch.tensor([[1, 2, 99, 4, 99, 0]])
    assert torch.equal(build_modifier_token_mask(input_ids, [99]), torch.tensor([[False, False, True, False, True, False]]))


def test_non_subject_positions_are_exactly_base_kv_and_subject_residual_is_active():
    torch.manual_seed(0)
    attn = FakeAttention()
    processor = TokenLocalKVAttnProcessor(hidden_size=4, cross_attention_dim=3)
    hidden = torch.randn(1, 5, 3)
    mask = torch.tensor([[False, False, True, False, False]])
    with torch.no_grad():
        processor.delta_k.weight.fill_(0.5)
        processor.delta_v.weight.fill_(-0.25)
    key, value = processor.project_kv(attn, hidden, mask)
    assert torch.allclose(key[:, ~mask[0]], attn.to_k(hidden)[:, ~mask[0]], atol=0, rtol=0)
    assert torch.allclose(value[:, ~mask[0]], attn.to_v(hidden)[:, ~mask[0]], atol=0, rtol=0)
    assert not torch.allclose(key[:, mask[0]], attn.to_k(hidden)[:, mask[0]])
    assert not torch.allclose(value[:, mask[0]], attn.to_v(hidden)[:, mask[0]])


def test_prompt_without_subject_token_reduces_to_base_kv_and_residual_parameters_receive_gradients():
    torch.manual_seed(1)
    attn = FakeAttention()
    attn.to_k.weight.requires_grad_(False)
    attn.to_v.weight.requires_grad_(False)
    processor = TokenLocalKVAttnProcessor(hidden_size=4, cross_attention_dim=3)
    hidden = torch.randn(1, 4, 3)
    empty = torch.zeros((1, 4), dtype=torch.bool)
    key, value = processor.project_kv(attn, hidden, empty)
    assert torch.allclose(key, attn.to_k(hidden), atol=0, rtol=0)
    assert torch.allclose(value, attn.to_v(hidden), atol=0, rtol=0)
    active = torch.tensor([[False, True, False, False]])
    (processor.project_kv(attn, hidden, active)[0].sum() + processor.project_kv(attn, hidden, active)[1].sum()).backward()
    assert processor.delta_k.weight.grad is not None and processor.delta_v.weight.grad is not None
    assert attn.to_k.weight.grad is None and attn.to_v.weight.grad is None


def test_token_local_state_dict_round_trip_preserves_residuals():
    source = TokenLocalKVAttnProcessor(hidden_size=4, cross_attention_dim=3)
    target = TokenLocalKVAttnProcessor(hidden_size=4, cross_attention_dim=3)
    with torch.no_grad():
        source.delta_k.weight.fill_(0.2)
        source.delta_v.weight.fill_(-0.3)
    target.load_state_dict(source.state_dict())
    assert torch.equal(target.delta_k.weight, source.delta_k.weight)
    assert torch.equal(target.delta_v.weight, source.delta_v.weight)
