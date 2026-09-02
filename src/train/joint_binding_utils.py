"""Loss helpers for the versioned two-object joint-binding experiment.

The attention guidance follows ICE Stage Two: average the modifier-token
attention for one object, compare it with that object's instance mask using
entropically regularized Wasserstein distance, and apply weight ``1e-5``.
The pure-PyTorch Sinkhorn step keeps this repository independent of the ICE
checkout while preserving the published implementation's mathematics.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F


ICE_ATTENTION_WEIGHT = 1e-5
ICE_SINKHORN_REGULARIZATION = 1e-3
ICE_SINKHORN_ITERATIONS = 200


def balanced_instance_masked_mse(
    per_pixel_mse: torch.Tensor, instance_masks: torch.Tensor
) -> torch.Tensor:
    """Average independently normalized reconstruction loss over instances."""
    if per_pixel_mse.ndim != 4 or instance_masks.ndim != 4:
        raise ValueError("per-pixel loss and masks must be BCHW tensors")
    if per_pixel_mse.shape[0] != instance_masks.shape[0] or per_pixel_mse.shape[-2:] != instance_masks.shape[-2:]:
        raise ValueError("per-pixel loss and masks must share batch and spatial dimensions")
    denominators = instance_masks.sum(dim=(2, 3))
    if torch.any(denominators <= 0):
        raise ValueError("every instance mask must contain foreground pixels")
    losses = (per_pixel_mse.unsqueeze(1) * instance_masks.unsqueeze(2)).sum(dim=(2, 3, 4))
    return (losses / denominators).mean()


def modifier_group_positions(
    input_ids: torch.Tensor, token_id_groups: Sequence[Sequence[int]]
) -> list[list[int]]:
    """Resolve exactly one prompt position for every token in each object bundle."""
    if input_ids.ndim != 1:
        raise ValueError("input_ids must describe one prompt")
    positions = []
    for group in token_id_groups:
        if len(group) < 2:
            raise ValueError("each modifier-token group must contain at least two tokens")
        current = []
        for token_id in group:
            matches = (input_ids == int(token_id)).nonzero(as_tuple=False).reshape(-1)
            if matches.numel() != 1:
                raise ValueError(
                    f"joint-binding prompt must contain token id {token_id} exactly once; found {matches.numel()}"
                )
            current.append(int(matches.item()))
        positions.append(current)
    return positions


def grouped_caa_loss(attention_maps: torch.Tensor, position_groups: Sequence[Sequence[int]]) -> torch.Tensor:
    """Compute ColorPeel CAA within each object bundle and average bundles."""
    losses = []
    for positions in position_groups:
        token_attention = attention_maps[:, :, list(positions)].reshape(-1, len(positions)).t()
        pair_mask = torch.tril(
            torch.ones((len(positions), len(positions)), device=attention_maps.device, dtype=torch.bool),
            diagonal=-1,
        )
        similarities = F.cosine_similarity(token_attention[:, :, None], token_attention.t()[None, :, :])
        losses.append(1 - similarities[pair_mask].mean())
    if not losses:
        raise ValueError("at least one modifier-token group is required")
    return torch.stack(losses).mean()


def _sinkhorn_step(
    distance: torch.Tensor,
    log_target: torch.Tensor,
    log_source_scaling: torch.Tensor,
    regularization: float,
) -> torch.Tensor:
    return log_target - torch.logsumexp(
        -distance / regularization + log_source_scaling[:, :, None], dim=1
    )


class _SinkhornOT(torch.autograd.Function):
    """Pure-PyTorch form of the Sinkhorn autograd function used by ICE."""

    @staticmethod
    def forward(ctx, source, target, distance, regularization, iterations):
        source_size, target_size = distance.shape
        log_source = source.log()
        log_target = target.log()
        log_u = torch.full_like(source, -math.log(source_size))
        log_v = torch.full_like(target, -math.log(target_size))
        for _ in range(int(iterations)):
            log_v = _sinkhorn_step(distance, log_target, log_u, regularization)
            log_u = _sinkhorn_step(distance.t(), log_source, log_v, regularization)
        distances = (
            -_sinkhorn_step(-distance.log() + distance / regularization, -log_v, log_u, 1.0)
        ).logsumexp(1).exp()
        ctx.log_u = log_u
        ctx.log_v = log_v
        ctx.regularization = regularization
        return distances

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.regularization
        return (
            grad_output[:, None] * ctx.log_u * scale,
            grad_output[:, None] * ctx.log_v * scale,
            None,
            None,
            None,
        )


def _grid_distance(size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    axis = torch.arange(size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coordinates = torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=1)
    return torch.cdist(coordinates, coordinates, p=2)


def ice_wasserstein_attention_loss(
    attention_maps: torch.Tensor,
    position_groups: Sequence[Sequence[int]],
    instance_masks: torch.Tensor,
) -> torch.Tensor:
    """Return the unweighted ICE mask-to-attention loss averaged over objects."""
    if attention_maps.ndim != 3 or attention_maps.shape[0] != attention_maps.shape[1]:
        raise ValueError("attention_maps must have square HWT shape")
    if instance_masks.ndim != 3 or instance_masks.shape[0] != len(position_groups):
        raise ValueError("one instance mask is required for every modifier-token group")
    size = attention_maps.shape[0]
    resized_masks = F.interpolate(
        instance_masks.unsqueeze(0).float(), size=(size, size)
    ).squeeze(0)
    distance = _grid_distance(size, device=attention_maps.device, dtype=torch.float32)
    losses = []
    for mask, positions in zip(resized_masks, position_groups):
        source = mask.reshape(-1).float()
        target = attention_maps[:, :, list(positions)].mean(dim=-1).reshape(-1).float()
        if source.sum() <= 0 or target.sum() <= 0:
            raise ValueError("mask and attention distributions must have positive mass")
        source = source / source.sum()
        target = target / target.sum()
        losses.append(
            _SinkhornOT.apply(
                source.unsqueeze(0),
                target.unsqueeze(0),
                distance,
                ICE_SINKHORN_REGULARIZATION,
                ICE_SINKHORN_ITERATIONS,
            ).mean()
        )
    return torch.stack(losses).mean()


def cross_object_attention_mass(
    attention_maps: torch.Tensor,
    position_groups: Sequence[Sequence[int]],
    instance_masks: torch.Tensor,
) -> torch.Tensor:
    """Report attention mass landing on the other instance; never used as loss."""
    size = attention_maps.shape[0]
    masks = F.interpolate(instance_masks.unsqueeze(0).float(), size=(size, size)).squeeze(0)
    values = []
    for index, positions in enumerate(position_groups):
        attention = attention_maps[:, :, list(positions)].mean(dim=-1)
        attention = attention / attention.sum()
        other = masks[1 - index]
        values.append((attention * other).sum())
    return torch.stack(values).mean()
