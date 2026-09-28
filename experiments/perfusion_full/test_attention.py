"""Checks for the full-sequence and multi-concept Perfusion equations."""

import pytest
import torch
from torch import nn

from experiments.perfusion_full.attention import FullPerfusionAttnProcessor


class DummyAttention:
    def __init__(self):
        self.to_k = nn.Linear(3, 2, bias=False)
        self.to_v = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.to_k.weight.copy_(torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
            self.to_v.weight.copy_(torch.tensor([[0., 1., 0.], [0., 0., 1.]]))
        self.to_k.requires_grad_(False)
        self.to_v.requires_grad_(False)


def test_single_concept_matches_equation_three_at_every_position():
    attention = DummyAttention()
    processor = FullPerfusionAttnProcessor(
        2, 3, key_outputs=torch.tensor([[3., 4.]]),
        value_outputs=torch.tensor([[5., 6.]]),
    )
    states = torch.tensor([[[1., 2., 3.], [0., 1., 0.], [2., 0., 0.]]])
    anchor = torch.tensor([[1., 0., 0.]])
    inv_cov = torch.diag(torch.tensor([2., 3., 4.]))
    key, value = processor.project_kv(attention, states, inv_cov, anchor, .75, .1)

    similarity = 2 * states[..., 0]
    energy = torch.tensor(2.)
    orthogonal = states - anchor * (similarity / energy).unsqueeze(-1)
    gate = torch.sigmoid(((similarity / energy) - .75) / .1).unsqueeze(-1)
    assert torch.allclose(key, attention.to_k(orthogonal) + gate * processor.key_outputs)
    assert torch.allclose(value, attention.to_v(orthogonal) + gate * processor.value_outputs)
    assert torch.any(torch.abs(key[0, 2] - attention.to_k(orthogonal)[0, 2]) > .001)
    # The update is not confined to the concept token.

    value.sum().backward()
    assert processor.value_outputs.grad is not None
    assert processor.key_outputs.grad is None
    assert attention.to_k.weight.grad is None
    assert attention.to_v.weight.grad is None


def test_two_concepts_use_joint_metric_projection_and_individual_gates():
    attention = DummyAttention()
    processor = FullPerfusionAttnProcessor(
        2, 3, key_outputs=torch.tensor([[1., 2.], [3., 4.]]),
        value_outputs=torch.tensor([[5., 6.], [7., 8.]]),
    )
    states = torch.tensor([[[1., 2., 3.], [2., 1., 1.], [0., 3., 2.]]])
    anchors = torch.tensor([[1., 0., 0.], [1., 1., 0.]])
    inv_cov = torch.tensor([[2., .2, 0.], [.2, 3., 0.], [0., 0., 4.]])
    key, value = processor.project_kv(
        attention, states, inv_cov, anchors,
        beta=torch.tensor([.6, .75]), tau=torch.tensor([.1, .15]),
    )

    # Appendix B: whiten with Cholesky, QR the joint concept span, unwhiten.
    chol = torch.linalg.cholesky(inv_cov)
    whitened_anchors = chol.T @ anchors.T
    basis = torch.linalg.qr(whitened_anchors, mode="reduced").Q
    whitened_states = states @ chol
    projected = (whitened_states @ basis) @ basis.T
    orthogonal = states - torch.linalg.solve_triangular(
        chol.T, projected.transpose(1, 2), upper=True,
    ).transpose(1, 2)
    similarities = states @ inv_cov @ anchors.T
    energies = torch.diagonal(anchors @ inv_cov @ anchors.T)
    gates = torch.sigmoid(
        (similarities / energies - torch.tensor([.6, .75])) / torch.tensor([.1, .15])
    )
    assert torch.allclose(orthogonal @ inv_cov @ anchors.T,
                          torch.zeros(1, 3, 2), atol=1e-5)
    assert torch.allclose(key, attention.to_k(orthogonal) + gates @ processor.key_outputs,
                          atol=1e-5)
    assert torch.allclose(value, attention.to_v(orthogonal) + gates @ processor.value_outputs,
                          atol=1e-5)


def test_bad_joint_anchors_fail_and_self_attention_has_no_edit_parameters():
    processor = FullPerfusionAttnProcessor()
    assert len(list(processor.parameters())) == 0
    attention = DummyAttention()
    edit = FullPerfusionAttnProcessor(
        2, 3, key_outputs=torch.zeros(2, 2), value_outputs=torch.zeros(2, 2),
    )
    with pytest.raises(ValueError, match="independent"):
        edit.project_kv(
            attention, torch.ones(1, 2, 3), torch.eye(3),
            torch.tensor([[1., 0., 0.], [2., 0., 0.]]), .75, .1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA needed for autocast check")
def test_metric_projection_keeps_float32_under_cuda_autocast():
    attention = DummyAttention()
    attention.to_k.to("cuda")
    attention.to_v.to("cuda")
    processor = FullPerfusionAttnProcessor(
        2, 3, key_outputs=torch.tensor([[3., 4.]]),
        value_outputs=torch.tensor([[5., 6.]]),
    ).to("cuda")
    with torch.cuda.amp.autocast():
        key, value = processor.project_kv(
            attention, torch.ones(1, 3, 3, device="cuda"),
            torch.eye(3, device="cuda"), torch.tensor([[1., 0., 0.]], device="cuda"),
            .75, .1,
        )
    assert key.isfinite().all() and value.isfinite().all()
