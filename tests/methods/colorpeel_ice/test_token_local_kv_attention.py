import sys
from pathlib import Path

import torch
from torch import nn


TRAIN_ROOT = Path(__file__).parents[3] / "src" / "train"
sys.path.insert(0, str(TRAIN_ROOT))
from custom_attention.attention_processor_custom import (
    DualTokenLocalKVAttnProcessor, FullColorMaterialKVAttnProcessor,
    TokenLocalKVAttnProcessor, build_modifier_token_mask,
)
from src.methods.colorpeel_ice.dual_token_local_kv import install_dual_token_local_kv


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


def test_dual_token_local_kv_matches_each_single_branch_and_base_positions():
    torch.manual_seed(2)
    attn = FakeAttention()
    color = TokenLocalKVAttnProcessor(hidden_size=4, cross_attention_dim=3)
    material = TokenLocalKVAttnProcessor(hidden_size=4, cross_attention_dim=3)
    dual = DualTokenLocalKVAttnProcessor(hidden_size=4, cross_attention_dim=3)
    with torch.no_grad():
        color.delta_k.weight.fill_(0.1)
        color.delta_v.weight.fill_(0.2)
        material.delta_k.weight.fill_(-0.3)
        material.delta_v.weight.fill_(-0.4)
        dual.delta_k.weight.copy_(color.delta_k.weight)
        dual.delta_v.weight.copy_(color.delta_v.weight)
        dual.material_delta_k.weight.copy_(material.delta_k.weight)
        dual.material_delta_v.weight.copy_(material.delta_v.weight)
    hidden = torch.randn(1, 5, 3)
    c = torch.tensor([[False, True, False, False, False]])
    m = torch.tensor([[False, False, False, True, False]])
    zero = torch.zeros_like(c)
    assert all(torch.allclose(a, b, atol=0, rtol=0) for a, b in zip(
        dual.project_kv(attn, hidden, {"color": c, "material": zero}),
        color.project_kv(attn, hidden, c),
    ))
    assert all(torch.allclose(a, b, atol=0, rtol=0) for a, b in zip(
        dual.project_kv(attn, hidden, {"color": zero, "material": m}),
        material.project_kv(attn, hidden, m),
    ))
    key, value = dual.project_kv(attn, hidden, {"color": c, "material": m})
    ordinary = ~(c | m)
    assert torch.equal(key[:, ordinary[0]], attn.to_k(hidden)[:, ordinary[0]])
    assert torch.equal(value[:, ordinary[0]], attn.to_v(hidden)[:, ordinary[0]])


def test_dual_token_local_kv_rejects_overlapping_masks():
    dual = DualTokenLocalKVAttnProcessor(hidden_size=4, cross_attention_dim=3)
    mask = torch.tensor([[False, True, False]])
    try:
        dual.project_kv(FakeAttention(), torch.randn(1, 3, 3), {"color": mask, "material": mask})
    except ValueError as error:
        assert "disjoint" in str(error)
    else:
        raise AssertionError("overlapping color and material positions were accepted")


def test_dual_token_local_kv_accepts_separate_subject_and_material_masks():
    dual = DualTokenLocalKVAttnProcessor(hidden_size=4, cross_attention_dim=3, primary_label="subject")
    subject = torch.tensor([[False, True, False]])
    material = torch.tensor([[False, False, True]])
    key, value = dual.project_kv(
        FakeAttention(), torch.randn(1, 3, 3), {"subject": subject, "material": material})
    assert key.shape == value.shape == (1, 3, 4)


def test_install_dual_token_local_kv_preserves_both_independent_weights():
    class FakeUnet:
        device = torch.device("cpu")
        dtype = torch.float32

        def __init__(self):
            self.attn_processors = {
                "block.attn1.processor": TokenLocalKVAttnProcessor(),
                "block.attn2.processor": TokenLocalKVAttnProcessor(4, 3),
            }

        def set_attn_processor(self, processors):
            self.attn_processors = processors

    unet = FakeUnet()
    color = unet.attn_processors["block.attn2.processor"]
    with torch.no_grad():
        color.delta_k.weight.fill_(0.2)
        color.delta_v.weight.fill_(0.3)
    material = {
        "block.attn1.processor": {},
        "block.attn2.processor.delta_k.weight": torch.full((4, 3), -0.4),
        "block.attn2.processor.delta_v.weight": torch.full((4, 3), -0.5),
    }
    install_dual_token_local_kv(unet, material)
    dual = unet.attn_processors["block.attn2.processor"]
    assert torch.equal(dual.delta_k.weight, color.delta_k.weight)
    assert torch.equal(dual.delta_v.weight, color.delta_v.weight)
    assert torch.equal(dual.material_delta_k.weight, material["block.attn2.processor.delta_k.weight"])
    assert torch.equal(dual.material_delta_v.weight, material["block.attn2.processor.delta_v.weight"])


def test_archived_full_color_is_preserved_with_material_only_at_its_token():
    torch.manual_seed(3)
    attn = FakeAttention()
    mixed = FullColorMaterialKVAttnProcessor(hidden_size=4, cross_attention_dim=3)
    with torch.no_grad():
        mixed.delta_k.weight.fill_(0.2)
        mixed.delta_v.weight.fill_(-0.3)
    hidden = torch.randn(1, 4, 3)
    empty = torch.zeros((1, 4), dtype=torch.bool)
    material = torch.tensor([[False, False, True, False]])
    archived = (mixed.color_k(hidden), mixed.color_v(hidden))
    assert all(torch.equal(actual, expected) for actual, expected in zip(
        mixed.project_kv(attn, hidden, empty), archived))
    key, value = mixed.project_kv(attn, hidden, material)
    assert torch.equal(key[:, ~material[0]], archived[0][:, ~material[0]])
    assert torch.equal(value[:, ~material[0]], archived[1][:, ~material[0]])
    assert not torch.equal(key[:, material[0]], archived[0][:, material[0]])
    assert not torch.equal(value[:, material[0]], archived[1][:, material[0]])
