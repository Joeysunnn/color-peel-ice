# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-token K/V adapters on the native SDXL UNet attention interface.

The SDPA forward adapts Diffusers v0.35.1 AttnProcessor2_0; only the
cross-attention K/V projection gains the existing ColorPeel residual.
https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/models/attention_processor.py
"""

import torch
from torch import nn
from torch.nn import functional as F


class NativeSelfAttentionProcessor:
    """Consume the cross-attention-only gate, delegate self-attention unchanged."""

    def __init__(self, delegate):
        self.delegate = delegate

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, modifier_token_mask=None):
        return self.delegate(attn, hidden_states, encoder_hidden_states=encoder_hidden_states,
                             attention_mask=attention_mask, temb=temb)


class TokenLocalKVProcessor(nn.Module):
    def __init__(self, hidden_size, cross_attention_dim, method, rank=64, alpha=64):
        super().__init__()
        self.method = method
        if method == "fullrank":
            self.delta_k = nn.Linear(cross_attention_dim, hidden_size, bias=False)
            self.delta_v = nn.Linear(cross_attention_dim, hidden_size, bias=False)
            nn.init.zeros_(self.delta_k.weight)
            nn.init.zeros_(self.delta_v.weight)
        elif method == "lora_r64":
            if rank <= 0 or rank > min(hidden_size, cross_attention_dim) or alpha <= 0:
                raise ValueError("invalid LoRA rank or alpha")
            self.scale = alpha / rank
            self.k_down = nn.Linear(cross_attention_dim, rank, bias=False)
            self.k_up = nn.Linear(rank, hidden_size, bias=False)
            self.v_down = nn.Linear(cross_attention_dim, rank, bias=False)
            self.v_up = nn.Linear(rank, hidden_size, bias=False)
            nn.init.normal_(self.k_down.weight, std=1 / rank)
            nn.init.normal_(self.v_down.weight, std=1 / rank)
            nn.init.zeros_(self.k_up.weight)
            nn.init.zeros_(self.v_up.weight)
        else:
            raise ValueError("method must be fullrank or lora_r64")

    def project_kv(self, attn, encoder_hidden_states, modifier_token_mask):
        if (not isinstance(modifier_token_mask, torch.Tensor)
                or modifier_token_mask.dtype != torch.bool
                or modifier_token_mask.shape != encoder_hidden_states.shape[:2]):
            raise ValueError("modifier_token_mask must be Boolean [batch, text positions]")
        base_k = attn.to_k(encoder_hidden_states)
        base_v = attn.to_v(encoder_hidden_states)
        inputs = encoder_hidden_states.to(next(self.parameters()).dtype)
        if self.method == "fullrank":
            delta_k, delta_v = self.delta_k(inputs), self.delta_v(inputs)
        else:
            delta_k = self.k_up(self.k_down(inputs)) * self.scale
            delta_v = self.v_up(self.v_down(inputs)) * self.scale
        gate = modifier_token_mask.to(base_k.device).unsqueeze(-1)
        # The historical adapters do not detach the contextual text states.
        key = torch.where(gate, base_k + delta_k.to(base_k.dtype), base_k)
        value = torch.where(gate, base_v + delta_v.to(base_v.dtype), base_v)
        return key, value

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, modifier_token_mask=None):
        if encoder_hidden_states is None:
            raise ValueError("install token-local adapters only on cross-attention")
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key, value = self.project_kv(attn, encoder_hidden_states, modifier_token_mask)
        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        )
        output = output.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
        output = attn.to_out[1](attn.to_out[0](output))
        if input_ndim == 4:
            output = output.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            output = output + residual
        return output / attn.rescale_output_factor


def install_token_local_kv(unet, method, rank=64, alpha=64):
    """Freeze the stock UNet, retain its self-attention, install FP32 adapters."""
    unet.requires_grad_(False)
    processors = dict(unet.attn_processors)
    installed = {}
    for name in processors:
        if name.endswith("attn1.processor"):
            if not isinstance(processors[name], NativeSelfAttentionProcessor):
                processors[name] = NativeSelfAttentionProcessor(processors[name])
            continue
        if not name.endswith("attn2.processor"):
            raise ValueError(f"unexpected SDXL attention module: {name}")
        attn = unet.get_submodule(name.removesuffix(".processor"))
        hidden_size, cross_attention_dim = attn.to_k.weight.shape
        if attn.to_v.weight.shape != attn.to_k.weight.shape:
            raise ValueError(f"K/V dimensions differ: {name}")
        processor = TokenLocalKVProcessor(hidden_size, cross_attention_dim, method, rank, alpha)
        installed[name] = processor.to(device=attn.to_k.weight.device, dtype=torch.float32)
        processors[name] = installed[name]
    if not installed:
        raise ValueError("UNet has no cross-attention adapters")
    unet.set_attn_processor(processors)
    return installed


def adapter_parameters(unet):
    return [parameter for processor in unet.attn_processors.values()
            if isinstance(processor, TokenLocalKVProcessor) for parameter in processor.parameters()]


def adapter_state_dict(unet):
    return {f"{name}.{key}": value.detach().cpu().clone()
            for name, processor in unet.attn_processors.items()
            if isinstance(processor, TokenLocalKVProcessor)
            for key, value in processor.state_dict().items()}


def load_adapter_state_dict(unet, state):
    expected = {f"{name}.{key}": value
                for name, processor in unet.attn_processors.items()
                if isinstance(processor, TokenLocalKVProcessor)
                for key, value in processor.state_dict().items()}
    if set(state) != set(expected):
        raise ValueError("adapter tensor names do not match the installed method and UNet")
    for key, target in expected.items():
        if (not isinstance(state[key], torch.Tensor) or state[key].shape != target.shape
                or not torch.isfinite(state[key]).all()):
            raise ValueError(f"invalid adapter tensor: {key}")
    for name, processor in unet.attn_processors.items():
        if isinstance(processor, TokenLocalKVProcessor):
            processor.load_state_dict({key: state[f"{name}.{key}"] for key in processor.state_dict()})
