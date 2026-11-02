"""Route only Texture residual inputs; retain the original joint attention path."""

import torch
from torch import nn

from experiments.sd21_mailbox_sct_v1.attention import SCTTokenLocalKVAttnProcessor, TOKENS


class TextureContextRoutingProcessor(nn.Module):
    def __init__(self, original):
        super().__init__()
        if not isinstance(original, SCTTokenLocalKVAttnProcessor) or not original.k_residuals:
            raise ValueError('routing requires an installed original SCT cross-attention processor')
        self.original = original
        self.training = original.training

    def project_kv(self, attn, encoder_hidden_states, modifier_token_mask,
                   texture_encoder_hidden_states=None):
        if texture_encoder_hidden_states is None:
            return self.original.project_kv(attn, encoder_hidden_states, modifier_token_mask)
        if (not isinstance(texture_encoder_hidden_states, torch.Tensor)
                or texture_encoder_hidden_states.shape != encoder_hidden_states.shape
                or texture_encoder_hidden_states.dtype != encoder_hidden_states.dtype
                or texture_encoder_hidden_states.device != encoder_hidden_states.device):
            raise ValueError('Texture reference hidden must match joint hidden shape, dtype and device')
        if not isinstance(modifier_token_mask, dict) or set(modifier_token_mask) != set(TOKENS):
            raise ValueError('require separate S/C/T masks, including an optional all-zero Texture mask')
        masks = list(modifier_token_mask.values())
        if any(not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
               or mask.shape != encoder_hidden_states.shape[:2] for mask in masks):
            raise ValueError('S/C/T gates must be Boolean and match joint text positions')
        if encoder_hidden_states.shape[0] != 2 or any(bool(mask[0].any()) for mask in masks):
            raise ValueError('Q1 requires CFG negative-first batch2 with all negative gates zero')
        if any(bool((masks[i] & masks[j]).any()) for i in range(3) for j in range(i + 1, 3)):
            raise ValueError('S/C/T gates must remain disjoint')
        key, value = attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)
        for branch in TOKENS:
            residual_input = texture_encoder_hidden_states if branch == 'texture' else encoder_hidden_states
            gate = modifier_token_mask[branch].to(device=key.device).unsqueeze(-1)
            key = torch.where(gate, key + self.original.k_residuals[branch](residual_input), key)
            value = torch.where(gate, value + self.original.v_residuals[branch](residual_input), value)
        return key, value

    def forward(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                modifier_token_mask=None, texture_encoder_hidden_states=None, **kwargs):
        if texture_encoder_hidden_states is None or encoder_hidden_states is None:
            # Keep the baseline/true-SC operation order exactly as installed originally.
            return self.original(attn, hidden_states, encoder_hidden_states=encoder_hidden_states,
                                 attention_mask=attention_mask, modifier_token_mask=modifier_token_mask, **kwargs)
        if attn.norm_cross:
            raise ValueError('this fixed SD2.1 Q1 diagnostic requires norm_cross=None')
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)
        key, value = self.project_kv(attn, encoder_hidden_states, modifier_token_mask,
                                     texture_encoder_hidden_states=texture_encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        attn.attn_probs = attention_probs
        output = attn.batch_to_head_dim(torch.bmm(attention_probs, value))
        return attn.to_out[1](attn.to_out[0](output))


def install_context_routing(unet):
    """Wrap existing cross processors without loading, copying or merging weights."""
    processors = {}
    for name, original in unet.attn_processors.items():
        if not isinstance(original, SCTTokenLocalKVAttnProcessor):
            raise ValueError('install context routing once, immediately after original SCT installation')
        if name.endswith('attn2.processor'):
            processors[name] = TextureContextRoutingProcessor(original)
        elif name.endswith('attn1.processor'):
            processors[name] = original
        else:
            raise ValueError(f'unexpected attention processor: {name}')
    unet.set_attn_processor(processors)
