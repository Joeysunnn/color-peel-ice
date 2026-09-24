"""Install two frozen, independently learned token-local K/V adapters for inference."""

from __future__ import annotations

import torch

from custom_attention.attention_processor_custom import (
    DualTokenLocalKVAttnProcessor, TokenLocalKVAttnProcessor,
)


def install_dual_token_local_kv(unet, material_state: dict[str, torch.Tensor], primary_label: str = "color") -> None:
    """The UNet must already have the primary adapter loaded via load_attn_procs."""
    expected_keys = set()
    combined = {}
    for name, color in unet.attn_processors.items():
        if not isinstance(color, TokenLocalKVAttnProcessor):
            raise ValueError(f"color processor is not token-local: {name}")
        dual = DualTokenLocalKVAttnProcessor(color.hidden_size, color.cross_attention_dim, primary_label)
        if color.cross_attention_dim is None:
            expected_keys.add(name)
            if material_state.get(name) != {}:
                raise ValueError(f"material self-attention entry differs: {name}")
        else:
            for suffix, color_layer, material_layer in (
                ("delta_k.weight", color.delta_k, dual.material_delta_k),
                ("delta_v.weight", color.delta_v, dual.material_delta_v),
            ):
                key = f"{name}.{suffix}"
                expected_keys.add(key)
                value = material_state.get(key)
                if not isinstance(value, torch.Tensor) or value.shape != material_layer.weight.shape:
                    raise ValueError(f"material adapter weight differs: {key}")
                with torch.no_grad():
                    material_layer.weight.copy_(value)
            with torch.no_grad():
                dual.delta_k.weight.copy_(color.delta_k.weight)
                dual.delta_v.weight.copy_(color.delta_v.weight)
        combined[name] = dual.to(device=unet.device, dtype=unet.dtype)
    if set(material_state) != expected_keys:
        raise ValueError("material adapter has missing or unexpected attention layers")
    unet.set_attn_processor(combined)
