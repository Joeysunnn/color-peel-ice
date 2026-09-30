"""Compose independently trained token-local Subject and Material K/V LoRAs."""

from __future__ import annotations

import torch

from experiments.lora_kv_subject_v1.attention import SubjectLoraKVAttnProcessor
from src.train.custom_attention.attention_processor_custom import LoRALinearLayer


class DualTokenLocalLoraKVAttnProcessor(SubjectLoraKVAttnProcessor):
    def __init__(self, hidden_size=None, cross_attention_dim=None, rank=4, alpha=4):
        super().__init__(hidden_size, cross_attention_dim, "token_local_kv", rank, alpha)
        self.material_key_scale = 1.0
        self.material_value_scale = 1.0
        if cross_attention_dim is not None:
            self.material_to_k_lora = LoRALinearLayer(cross_attention_dim, hidden_size, rank, alpha)
            self.material_to_v_lora = LoRALinearLayer(cross_attention_dim, hidden_size, rank, alpha)

    def project_kv(self, attn, encoder_hidden_states, modifier_token_mask=None):
        if not isinstance(modifier_token_mask, dict) or set(modifier_token_mask) != {"subject", "material"}:
            raise ValueError("composition requires separate Subject and Material masks")
        subject_mask = modifier_token_mask["subject"]
        material_mask = modifier_token_mask["material"]
        for label, mask in (("subject", subject_mask), ("material", material_mask)):
            if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                    or mask.shape != encoder_hidden_states.shape[:2]):
                raise ValueError(f"{label} mask must be Boolean and match text positions")
        if torch.any(subject_mask & material_mask):
            raise ValueError("Subject and Material masks must be disjoint")

        base_k = attn.to_k(encoder_hidden_states)
        base_v = attn.to_v(encoder_hidden_states)
        subject_gate = subject_mask.to(device=base_k.device).unsqueeze(-1)
        material_gate = material_mask.to(device=base_k.device).unsqueeze(-1)
        key = torch.where(subject_gate, base_k + self.to_k_lora(encoder_hidden_states), base_k)
        value = torch.where(subject_gate, base_v + self.to_v_lora(encoder_hidden_states), base_v)
        key = torch.where(material_gate,
                          key + self.material_key_scale * self.material_to_k_lora(encoder_hidden_states),
                          key)
        value = torch.where(material_gate,
                            value + self.material_value_scale * self.material_to_v_lora(encoder_hidden_states),
                            value)
        return key, value


def install_dual_token_local_lora_kv(unet, subject_state, material_state, rank=4, alpha=4) -> None:
    """Load two complete LoRA states without altering the base UNet weights."""
    for label, state in (("Subject", subject_state), ("Material", material_state)):
        if (not isinstance(state, dict)
                or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items())):
            raise ValueError(f"{label} checkpoint must be a tensor state dictionary")
    processors = {}
    expected_subject = set()
    expected_material = set()
    for name in unet.attn_processors:
        if name.endswith("attn1.processor"):
            processors[name] = DualTokenLocalLoraKVAttnProcessor(rank=rank, alpha=alpha)
            continue
        if not name.endswith("attn2.processor"):
            raise ValueError(f"unexpected attention processor: {name}")
        attn = unet.get_submodule(name.removesuffix(".processor"))
        hidden_size, cross_attention_dim = attn.to_k.weight.shape
        if attn.to_v.weight.shape != (hidden_size, cross_attention_dim):
            raise ValueError(f"base K/V shape differs: {name}")
        processor = DualTokenLocalLoraKVAttnProcessor(hidden_size, cross_attention_dim, rank, alpha)
        local_state = processor.state_dict()
        with torch.no_grad():
            for key, parameter in local_state.items():
                is_material = key.startswith("material_")
                source_key = key.removeprefix("material_") if is_material else key
                full_key = f"{name}.{source_key}"
                source = material_state if is_material else subject_state
                value = source.get(full_key)
                if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                    raise ValueError(f"missing or mismatched attention tensor: {full_key}")
                parameter.copy_(value)
                (expected_material if is_material else expected_subject).add(full_key)
            processor.load_state_dict(local_state)
        processors[name] = processor.to(device=attn.to_k.weight.device, dtype=attn.to_k.weight.dtype)
        processor.requires_grad_(False)
    if set(subject_state) != expected_subject:
        raise ValueError("Subject checkpoint has unexpected attention tensors")
    if set(material_state) != expected_material:
        raise ValueError("Material checkpoint has unexpected attention tensors")
    unet.set_attn_processor(processors)
