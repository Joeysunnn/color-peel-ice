"""Mailbox Key-Locking and token-position-only rank-1 Subject Value editing."""

from __future__ import annotations

import torch
from torch import nn


REFERENCE_PROMPT = "a photo of a mailbox"
WEIGHT_NAME = "pytorch_perfusion_subject_weights.bin"
REFERENCE_NAME = "mailbox_key_reference.pt"


def mailbox_key_reference(text_encoder, tokenizer) -> torch.Tensor:
    """Freeze the contextual mailbox encoding used by every base cross-attention Key."""
    mailbox_ids = tokenizer.encode("mailbox", add_special_tokens=False)
    if len(mailbox_ids) != 1:
        raise ValueError("mailbox must be one tokenizer token")
    ids = tokenizer(
        REFERENCE_PROMPT, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(text_encoder.get_input_embeddings().weight.device)
    positions = (ids[0] == mailbox_ids[0]).nonzero(as_tuple=False).flatten()
    if positions.numel() != 1:
        raise ValueError("reference prompt must contain exactly one mailbox token")
    was_training = text_encoder.training
    text_encoder.eval()
    with torch.no_grad():
        reference = text_encoder(ids)[0][0, positions.item()].detach().clone()
    text_encoder.train(was_training)
    return reference


class PerfusionSubjectAttnProcessor(nn.Module):
    """Use a fixed superclass Key and a rank-1 Value edit only at <S*>."""

    def __init__(self, hidden_size=None, cross_attention_dim=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        if cross_attention_dim is not None:
            self.value_a = nn.Parameter(torch.zeros(hidden_size))
            self.value_b = nn.Parameter(torch.empty(cross_attention_dim))
            nn.init.normal_(self.value_b, std=0.01)

    def project_kv(self, attn, encoder_hidden_states, modifier_token_mask, key_reference):
        if modifier_token_mask is None or key_reference is None:
            raise ValueError("Perfusion Subject requires a token mask and frozen mailbox reference")
        if (modifier_token_mask.shape != encoder_hidden_states.shape[:2]
                or modifier_token_mask.dtype != torch.bool):
            raise ValueError("Subject mask must be Boolean and match text positions")
        if key_reference.ndim != 1 or key_reference.numel() != encoder_hidden_states.shape[-1]:
            raise ValueError("mailbox reference must be one contextual text encoding")
        subject = modifier_token_mask.to(device=encoder_hidden_states.device).unsqueeze(-1)
        base_k = attn.to_k(encoder_hidden_states)
        base_v = attn.to_v(encoder_hidden_states)
        reference = key_reference.to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
        locked_k = attn.to_k(reference).view(1, 1, -1)
        rank_one_v = (encoder_hidden_states @ self.value_b).unsqueeze(-1) * self.value_a
        return torch.where(subject, locked_k, base_k), base_v + subject * rank_one_v

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 modifier_token_mask=None, key_reference=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        else:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            key, value = self.project_kv(attn, encoder_hidden_states, modifier_token_mask, key_reference)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        attn.attn_probs = attention_probs
        output = attn.batch_to_head_dim(torch.bmm(attention_probs, value))
        return attn.to_out[1](attn.to_out[0](output))


class PerfusionSubjectMaterialAttnProcessor(PerfusionSubjectAttnProcessor):
    """Add the existing frozen token-local Material K/V at <M*> positions."""

    def __init__(self, hidden_size=None, cross_attention_dim=None):
        super().__init__(hidden_size, cross_attention_dim)
        if cross_attention_dim is not None:
            self.material_delta_k = nn.Linear(cross_attention_dim, hidden_size, bias=False)
            self.material_delta_v = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def project_kv(self, attn, encoder_hidden_states, modifier_token_mask, key_reference):
        if not isinstance(modifier_token_mask, dict) or set(modifier_token_mask) != {"subject", "material"}:
            raise ValueError("Subject and Material require separate token masks")
        subject = modifier_token_mask["subject"]
        material = modifier_token_mask["material"]
        if (subject.shape != encoder_hidden_states.shape[:2]
                or material.shape != subject.shape or subject.dtype != torch.bool
                or material.dtype != torch.bool or torch.any(subject & material)):
            raise ValueError("Subject and Material masks must be disjoint Boolean text masks")
        key, value = super().project_kv(attn, encoder_hidden_states, subject, key_reference)
        material_gate = material.to(device=encoder_hidden_states.device).unsqueeze(-1)
        return (key + material_gate * self.material_delta_k(encoder_hidden_states),
                value + material_gate * self.material_delta_v(encoder_hidden_states))


def subject_state_dict(unet) -> dict[str, torch.Tensor]:
    return {
        f"{name}.{key}": value.detach().cpu()
        for name, processor in unet.attn_processors.items()
        for key, value in processor.state_dict().items()
    }


def install_subject_material(unet, subject_state: dict, material_state: dict) -> None:
    """Install immutable P1 Subject and selected Material parameters on a fresh base UNet."""
    processors = {}
    subject_keys = set()
    material_keys = set()
    for name in unet.attn_processors:
        if name.endswith("attn1.processor"):
            if material_state.get(name) != {}:
                raise ValueError(f"Material self-attention entry differs: {name}")
            material_keys.add(name)
            processor = PerfusionSubjectMaterialAttnProcessor()
        else:
            a_key, b_key = f"{name}.value_a", f"{name}.value_b"
            mk_key, mv_key = f"{name}.delta_k.weight", f"{name}.delta_v.weight"
            a, b = subject_state.get(a_key), subject_state.get(b_key)
            mk, mv = material_state.get(mk_key), material_state.get(mv_key)
            if not all(isinstance(value, torch.Tensor) for value in (a, b, mk, mv)):
                raise ValueError(f"Missing Subject or Material layer: {name}")
            if (a.ndim != 1 or b.ndim != 1 or mk.shape != (a.numel(), b.numel())
                    or mv.shape != mk.shape):
                raise ValueError(f"Subject or Material layer shape differs: {name}")
            processor = PerfusionSubjectMaterialAttnProcessor(a.numel(), b.numel())
            with torch.no_grad():
                processor.value_a.copy_(a)
                processor.value_b.copy_(b)
                processor.material_delta_k.weight.copy_(mk)
                processor.material_delta_v.weight.copy_(mv)
            subject_keys.update((a_key, b_key))
            material_keys.update((mk_key, mv_key))
        processors[name] = processor.to(device=unet.device, dtype=unet.dtype).requires_grad_(False)
    if set(subject_state) != subject_keys or set(material_state) != material_keys:
        raise ValueError("Subject or Material state has unexpected attention layers")
    unet.set_attn_processor(processors)
