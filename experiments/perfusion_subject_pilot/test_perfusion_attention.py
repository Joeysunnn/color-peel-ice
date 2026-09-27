"""Mechanism checks for the isolated Perfusion-style Subject pilot."""

import torch
from torch import nn

from experiments.perfusion_subject_pilot.perfusion_attention import (
    PerfusionSubjectAttnProcessor,
    PerfusionSubjectMaterialAttnProcessor,
    install_subject_material,
)


class DummyAttention:
    def __init__(self):
        self.to_k = nn.Linear(3, 2, bias=False)
        self.to_v = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.to_k.weight.copy_(torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
            self.to_v.weight.copy_(torch.tensor([[0., 1., 0.], [0., 0., 1.]]))
        self.to_k.requires_grad_(False)
        self.to_v.requires_grad_(False)


def test_subject_key_is_fixed_and_value_edit_is_rank_one_at_subject_only():
    attention = DummyAttention()
    processor = PerfusionSubjectAttnProcessor(2, 3)
    with torch.no_grad():
        processor.value_a.copy_(torch.tensor([2., -1.]))
        processor.value_b.copy_(torch.tensor([1., 0., 1.]))
    states = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
    reference = torch.tensor([10., 11., 12.])
    mask = torch.tensor([[False, True, False]])
    key, value = processor.project_kv(attention, states, mask, reference)

    base_k, base_v = attention.to_k(states), attention.to_v(states)
    assert torch.equal(key[0, 0], base_k[0, 0])
    assert torch.equal(key[0, 2], base_k[0, 2])
    assert torch.equal(key[0, 1], attention.to_k(reference))
    assert torch.equal(value[0, 0], base_v[0, 0])
    assert torch.equal(value[0, 2], base_v[0, 2])
    assert torch.allclose(value[0, 1] - base_v[0, 1],
                          processor.value_a * (states[0, 1] @ processor.value_b))
    assert torch.linalg.matrix_rank(torch.outer(processor.value_a, processor.value_b)) == 1

    value.sum().backward()
    assert processor.value_a.grad is not None
    assert processor.value_b.grad is not None
    assert attention.to_k.weight.grad is None
    assert attention.to_v.weight.grad is None
    assert not hasattr(processor, "delta_k")


def test_material_uses_unchanged_local_kv_and_disjoint_mask():
    attention = DummyAttention()
    processor = PerfusionSubjectMaterialAttnProcessor(2, 3)
    with torch.no_grad():
        processor.value_a.copy_(torch.tensor([1., 2.]))
        processor.value_b.copy_(torch.tensor([1., 0., 0.]))
        processor.material_delta_k.weight.fill_(0.5)
        processor.material_delta_v.weight.fill_(0.25)
    states = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
    masks = {"subject": torch.tensor([[True, False, False]]),
             "material": torch.tensor([[False, True, False]])}
    reference = torch.tensor([10., 11., 12.])
    key, value = processor.project_kv(attention, states, masks, reference)
    assert torch.equal(key[0, 0], attention.to_k(reference))
    assert torch.equal(key[0, 1], attention.to_k(states)[0, 1]
                       + processor.material_delta_k(states)[0, 1])
    assert torch.equal(value[0, 1], attention.to_v(states)[0, 1]
                       + processor.material_delta_v(states)[0, 1])
    assert torch.equal(key[0, 2], attention.to_k(states)[0, 2])
    assert torch.equal(value[0, 2], attention.to_v(states)[0, 2])


def test_bad_or_overlapping_masks_fail_closed():
    attention = DummyAttention()
    processor = PerfusionSubjectMaterialAttnProcessor(2, 3)
    states = torch.zeros(1, 2, 3)
    reference = torch.zeros(3)
    bad = {"subject": torch.tensor([[True, False]]),
           "material": torch.tensor([[True, False]])}
    try:
        processor.project_kv(attention, states, bad, reference)
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping masks were accepted")


def test_installer_rejects_extra_state_and_freezes_selected_material():
    name = "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor"

    class DummyUnet:
        device = torch.device("cpu")
        dtype = torch.float32

        def __init__(self):
            self.attn_processors = {name: object(), "attn1.processor": object()}

        def set_attn_processor(self, processors):
            self.attn_processors = processors

    subject_state = {f"{name}.value_a": torch.ones(2), f"{name}.value_b": torch.ones(3)}
    material_state = {"attn1.processor": {}, f"{name}.delta_k.weight": torch.ones(2, 3),
                      f"{name}.delta_v.weight": torch.ones(2, 3)}
    unet = DummyUnet()
    install_subject_material(unet, subject_state, material_state)
    assert all(not parameter.requires_grad for parameter in unet.attn_processors[name].parameters())
    try:
        install_subject_material(DummyUnet(), {**subject_state, "extra": torch.ones(1)}, material_state)
    except ValueError:
        pass
    else:
        raise AssertionError("extra Subject state was accepted")
