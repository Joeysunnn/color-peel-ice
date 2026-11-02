"""Inference-only scaling of the existing Texture residual layers."""

import torch
from torch import nn


class ScaledTextureResidual(nn.Module):
    def __init__(self, layer, scale):
        super().__init__()
        self.layer, self.scale = layer, scale

    def forward(self, hidden_states):
        value = self.layer(hidden_states)
        if self.scale == 1:
            return value
        return torch.zeros_like(value) if self.scale == 0 else value * self.scale


def scale_texture_residuals(unet, key_scale, value_scale):
    if key_scale not in (0, 0.25, 0.5, 1) or value_scale not in (0, 0.25, 0.5, 1):
        raise ValueError('require one of the declared Texture scales')
    targets = []
    for processor in unet.attn_processors.values():
        for field, scale in (('k_residuals', key_scale), ('v_residuals', value_scale)):
            layers = getattr(processor, field)
            if layers:
                layer = layers['texture']
                if isinstance(layer, ScaledTextureResidual):
                    raise ValueError('Texture scaling must be installed once on a fresh processor')
                targets.append((layers, layer, scale))
    for layers, layer, scale in targets:
        if scale != 1:  # Leave the existing scale1 execution path completely unchanged.
            layers['texture'] = ScaledTextureResidual(layer, scale).eval()
