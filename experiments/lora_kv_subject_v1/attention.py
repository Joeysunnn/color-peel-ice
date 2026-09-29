"""Subject K/V LoRA ablations, with optional frozen token-local Material editing."""

from __future__ import annotations

import math

import torch
from torch import nn

from src.train.custom_attention.attention_processor_custom import LoRALinearLayer


MODES = {"full_kv", "token_local_kv", "full_v", "token_local_v"}
WEIGHT_NAME = "pytorch_lora_kv_weights.bin"


class SubjectLoraKVAttnProcessor(nn.Module):
    """Apply Subject LoRA to all text positions or only explicit <S*> positions."""

    def __init__(self, hidden_size=None, cross_attention_dim=None, mode="full_kv", rank=4, alpha=4,
                 with_material=False):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown Subject LoRA mode: {mode}")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")
        if not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA alpha must be positive and finite")
        self.mode = mode
        self.rank = rank
        self.alpha = alpha
        self.with_material = with_material
        if cross_attention_dim is not None:
            if hidden_size is None:
                raise ValueError("cross-attention requires hidden_size")
            if mode in {"full_kv", "token_local_kv"}:
                self.to_k_lora = LoRALinearLayer(cross_attention_dim, hidden_size, rank, alpha)
            self.to_v_lora = LoRALinearLayer(cross_attention_dim, hidden_size, rank, alpha)
            if with_material:
                self.material_delta_k = nn.Linear(cross_attention_dim, hidden_size, bias=False)
                self.material_delta_v = nn.Linear(cross_attention_dim, hidden_size, bias=False)
                self.material_delta_k.requires_grad_(False)
                self.material_delta_v.requires_grad_(False)

    def project_kv(self, attn, encoder_hidden_states, modifier_token_mask=None):
        base_k = attn.to_k(encoder_hidden_states)
        base_v = attn.to_v(encoder_hidden_states)
        if not hasattr(self, "to_v_lora"):
            raise ValueError("project_kv requires a cross-attention processor")

        material_mask = None
        subject_mask = modifier_token_mask
        if isinstance(modifier_token_mask, dict):
            if set(modifier_token_mask) != {"subject", "material"}:
                raise ValueError("composition requires subject and material masks")
            subject_mask = modifier_token_mask["subject"]
            material_mask = modifier_token_mask["material"]
            if subject_mask is None or material_mask is None:
                raise ValueError("composition requires separate Boolean subject and material masks")
        if self.with_material and material_mask is None:
            raise ValueError("Material adapter requires separate subject and material masks")
        if material_mask is not None and not self.with_material:
            raise ValueError("Material mask was supplied without a Material adapter")

        for label, mask in (("subject", subject_mask), ("material", material_mask)):
            if mask is not None and (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                                     or mask.shape != encoder_hidden_states.shape[:2]):
                raise ValueError(f"{label} mask must be Boolean and match text positions")
        if material_mask is not None and torch.any(subject_mask & material_mask):
            raise ValueError("Subject and Material masks must be disjoint")
        if self.mode.startswith("token_local") and subject_mask is None:
            raise ValueError("token-local Subject LoRA requires a Subject mask")

        subject_gate = None
        if self.mode.startswith("token_local"):
            subject_gate = subject_mask.to(device=base_k.device).unsqueeze(-1)
        if hasattr(self, "to_k_lora"):
            delta_k = self.to_k_lora(encoder_hidden_states)
            key = base_k + delta_k if subject_gate is None else torch.where(subject_gate, base_k + delta_k, base_k)
        else:
            key = base_k
        delta_v = self.to_v_lora(encoder_hidden_states)
        value = base_v + delta_v if subject_gate is None else torch.where(subject_gate, base_v + delta_v, base_v)

        if material_mask is not None:
            material_gate = material_mask.to(device=base_k.device).unsqueeze(-1)
            key = torch.where(material_gate, key + self.material_delta_k(encoder_hidden_states), key)
            value = torch.where(material_gate, value + self.material_delta_v(encoder_hidden_states), value)
        return key, value

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 modifier_token_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        else:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            key, value = self.project_kv(attn, encoder_hidden_states, modifier_token_mask)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        attn.attn_probs = attention_probs
        output = attn.batch_to_head_dim(torch.bmm(attention_probs, value))
        return attn.to_out[1](attn.to_out[0](output))


def subject_lora_state_dict(unet) -> dict[str, torch.Tensor]:
    """Serialize only Subject LoRA tensors, including when Material is installed."""
    return {
        f"{name}.{key}": value.detach().cpu()
        for name, processor in unet.attn_processors.items()
        for key, value in processor.state_dict().items()
        if key.startswith(("to_k_lora.", "to_v_lora."))
    }


def install_subject_lora_kv(unet, state, mode, rank, alpha, material_state=None) -> None:
    """Install validated Subject LoRA and optional selected Material on a base UNet."""
    if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                               for k, v in state.items()):
        raise ValueError("Subject checkpoint must be a tensor state dictionary")
    if material_state is not None and not isinstance(material_state, dict):
        raise ValueError("Material checkpoint must be a state dictionary")
    processors = {}
    expected_subject = set()
    expected_material = set()
    for name in unet.attn_processors:
        if name.endswith("attn1.processor"):
            processors[name] = SubjectLoraKVAttnProcessor(mode=mode, rank=rank, alpha=alpha)
            if material_state is not None:
                if material_state.get(name) != {}:
                    raise ValueError(f"Material self-attention entry differs: {name}")
                expected_material.add(name)
            continue
        if not name.endswith("attn2.processor"):
            raise ValueError(f"unexpected attention processor: {name}")
        attn = unet.get_submodule(name.removesuffix(".processor"))
        hidden_size, cross_attention_dim = attn.to_k.weight.shape
        if attn.to_v.weight.shape != (hidden_size, cross_attention_dim):
            raise ValueError(f"base K/V shape differs: {name}")
        processor = SubjectLoraKVAttnProcessor(
            hidden_size, cross_attention_dim, mode, rank, alpha, with_material=material_state is not None,
        )
        local_state = processor.state_dict()
        with torch.no_grad():
            for key, parameter in local_state.items():
                full_key = f"{name}.{key}"
                source = material_state if key.startswith("material_delta_") else state
                value = source.get(full_key)
                if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                    raise ValueError(f"missing or mismatched attention tensor: {full_key}")
                parameter.copy_(value)
                (expected_material if source is material_state else expected_subject).add(full_key)
            processor.load_state_dict(local_state)
        processors[name] = processor.to(device=attn.to_k.weight.device, dtype=attn.to_k.weight.dtype)
        if material_state is not None:
            processor.material_delta_k.requires_grad_(False)
            processor.material_delta_v.requires_grad_(False)
    if set(state) != expected_subject:
        raise ValueError("Subject checkpoint has unexpected attention tensors")
    if material_state is not None and set(material_state) != expected_material:
        raise ValueError("Material checkpoint has unexpected attention tensors")
    unet.set_attn_processor(processors)
