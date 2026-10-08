"""Compose any Subject LoRA K/V arm with the fixed token-local Material LoRA."""

from __future__ import annotations

import torch

from experiments.lora_kv_subject_v1.attention import SubjectLoraKVAttnProcessor
from src.train.custom_attention.attention_processor_custom import LoRALinearLayer


class SubjectMaterialLoraKVAttnProcessor(SubjectLoraKVAttnProcessor):
    def __init__(self, hidden_size=None, cross_attention_dim=None, subject_mode="full_kv",
                 rank=4, alpha=4, *, material_rank=None, material_alpha=None):
        material_rank = rank if material_rank is None else material_rank
        material_alpha = alpha if material_alpha is None else material_alpha
        super().__init__(hidden_size, cross_attention_dim, subject_mode, rank, alpha)
        if cross_attention_dim is not None:
            self.material_to_k_lora = LoRALinearLayer(
                cross_attention_dim, hidden_size, material_rank, material_alpha)
            self.material_to_v_lora = LoRALinearLayer(
                cross_attention_dim, hidden_size, material_rank, material_alpha)

    def project_kv(self, attn, encoder_hidden_states, modifier_token_mask=None):
        if not isinstance(modifier_token_mask, dict) or set(modifier_token_mask) != {"subject", "material"}:
            raise ValueError("composition requires separate Subject and Material masks")
        subject_mask = modifier_token_mask["subject"]
        material_mask = modifier_token_mask["material"]
        for label, mask in (("Subject", subject_mask), ("Material", material_mask)):
            if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                    or mask.shape != encoder_hidden_states.shape[:2]):
                raise ValueError(f"{label} mask must be Boolean and match text positions")
        if torch.any(subject_mask & material_mask):
            raise ValueError("Subject and Material masks must be disjoint")

        base_k = attn.to_k(encoder_hidden_states)
        base_v = attn.to_v(encoder_hidden_states)
        subject_gate = subject_mask.to(device=base_k.device).unsqueeze(-1)
        material_gate = material_mask.to(device=base_k.device).unsqueeze(-1)

        key = base_k
        if hasattr(self, "to_k_lora"):
            subject_key = base_k + self.to_k_lora(encoder_hidden_states)
            key = (torch.where(subject_gate, subject_key, base_k)
                   if self.mode.startswith("token_local") else subject_key)
        subject_value = base_v + self.to_v_lora(encoder_hidden_states)
        value = (torch.where(subject_gate, subject_value, base_v)
                 if self.mode.startswith("token_local") else subject_value)

        key = torch.where(
            material_gate, key + self.material_to_k_lora(encoder_hidden_states), key)
        value = torch.where(
            material_gate, value + self.material_to_v_lora(encoder_hidden_states), value)
        return key, value


def install_subject_material_lora_kv(unet, subject_state, material_state, subject_mode,
                                     rank=4, alpha=4, *, material_rank=None,
                                     material_alpha=None) -> None:
    """Load independent Subject and Material LoRAs without changing base weights."""
    material_rank = rank if material_rank is None else material_rank
    material_alpha = alpha if material_alpha is None else material_alpha
    for label, state in (("Subject", subject_state), ("Material", material_state)):
        if (not isinstance(state, dict)
                or not all(isinstance(key, str) and isinstance(value, torch.Tensor)
                           for key, value in state.items())):
            raise ValueError(f"{label} checkpoint must be a tensor state dictionary")
    processors = {}
    expected_subject = set()
    expected_material = set()
    for name in unet.attn_processors:
        if name.endswith("attn1.processor"):
            processors[name] = SubjectMaterialLoraKVAttnProcessor(
                subject_mode=subject_mode, rank=rank, alpha=alpha,
                material_rank=material_rank, material_alpha=material_alpha)
            continue
        if not name.endswith("attn2.processor"):
            raise ValueError(f"unexpected attention processor: {name}")
        attn = unet.get_submodule(name.removesuffix(".processor"))
        hidden_size, cross_attention_dim = attn.to_k.weight.shape
        if attn.to_v.weight.shape != (hidden_size, cross_attention_dim):
            raise ValueError(f"base K/V shape differs: {name}")
        processor = SubjectMaterialLoraKVAttnProcessor(
            hidden_size, cross_attention_dim, subject_mode, rank, alpha,
            material_rank=material_rank, material_alpha=material_alpha)
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
        processors[name] = processor.to(
            device=attn.to_k.weight.device, dtype=attn.to_k.weight.dtype)
        processors[name].requires_grad_(False)
    if set(subject_state) != expected_subject:
        raise ValueError("Subject checkpoint has unexpected attention tensors")
    if set(material_state) != expected_material:
        raise ValueError("Material checkpoint has unexpected attention tensors")
    unet.set_attn_processor(processors)
