"""Compose the archived full-K/V color checkpoint with token-local material."""

from __future__ import annotations

import torch

from custom_attention.attention_processor_custom import (
    CustomDiffusionAttnProcessor, FullColorMaterialKVAttnProcessor,
)


def install_full_color_material_kv(unet, material_state: dict[str, torch.Tensor]) -> None:
    """The archived color K/V processors must already be loaded on the UNet."""
    expected = set()
    combined = {}
    for name, color in unet.attn_processors.items():
        if not isinstance(color, CustomDiffusionAttnProcessor) or color.train_q_out:
            raise ValueError(f"old color processor differs: {name}")
        mixed = FullColorMaterialKVAttnProcessor(color.hidden_size, color.cross_attention_dim)
        if color.cross_attention_dim is None:
            expected.add(name)
            if color.train_kv or material_state.get(name) != {}:
                raise ValueError(f"self-attention entry differs: {name}")
        else:
            if not color.train_kv:
                raise ValueError(f"cross-attention color K/V missing: {name}")
            for suffix, source, target in (
                ("delta_k.weight", material_state.get(f"{name}.delta_k.weight"), mixed.delta_k.weight),
                ("delta_v.weight", material_state.get(f"{name}.delta_v.weight"), mixed.delta_v.weight),
            ):
                expected.add(f"{name}.{suffix}")
                if not isinstance(source, torch.Tensor) or source.shape != target.shape:
                    raise ValueError(f"material adapter differs: {name}.{suffix}")
                with torch.no_grad():
                    target.copy_(source)
            with torch.no_grad():
                mixed.color_k.weight.copy_(color.to_k_custom_diffusion.weight)
                mixed.color_v.weight.copy_(color.to_v_custom_diffusion.weight)
        combined[name] = mixed.to(device=unet.device, dtype=unet.dtype)
    if set(material_state) != expected:
        raise ValueError("material adapter has missing or unexpected attention layers")
    unet.set_attn_processor(combined)
