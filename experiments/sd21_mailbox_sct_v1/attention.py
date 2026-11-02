"""Three independently trained Subject/Color/Texture K/V adapters for SD2.1."""

import torch
from torch import nn

from src.train.custom_attention.attention_processor_custom import LoRALinearLayer, TokenLocalKVAttnProcessor

TOKENS = {'subject': '<S*>', 'color': '<C*>', 'texture': '<T*>'}
METHODS = ('fullrank', 'lora_r64')


class SCTTokenLocalKVAttnProcessor(TokenLocalKVAttnProcessor):
    def __init__(self, hidden_size=None, cross_attention_dim=None, method='fullrank'):
        if method not in METHODS:
            raise ValueError('require matching fullrank or rank64 adapters')
        super().__init__()
        self.hidden_size, self.cross_attention_dim = hidden_size, cross_attention_dim
        self.k_residuals, self.v_residuals = nn.ModuleDict(), nn.ModuleDict()
        if cross_attention_dim is not None:
            for branch in TOKENS:
                for layers in (self.k_residuals, self.v_residuals):
                    layers[branch] = (nn.Linear(cross_attention_dim, hidden_size, bias=False)
                                      if method == 'fullrank' else
                                      LoRALinearLayer(cross_attention_dim, hidden_size, rank=64, network_alpha=64))

    def project_kv(self, attn, encoder_hidden_states, modifier_token_mask):
        if not isinstance(modifier_token_mask, dict) or set(modifier_token_mask) != set(TOKENS):
            raise ValueError('require separate subject, color and texture masks')
        masks = list(modifier_token_mask.values())
        if any(not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
               or mask.shape != encoder_hidden_states.shape[:2] for mask in masks):
            raise ValueError('S/C/T masks must be Boolean and match text positions')
        if any(torch.any(masks[i] & masks[j]) for i in range(3) for j in range(i + 1, 3)):
            raise ValueError('S/C/T masks must be disjoint')
        key, value = attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)
        for branch in TOKENS:
            gate = modifier_token_mask[branch].to(device=key.device).unsqueeze(-1)
            key = torch.where(gate, key + self.k_residuals[branch](encoder_hidden_states), key)
            value = torch.where(gate, value + self.v_residuals[branch](encoder_hidden_states), value)
        return key, value


def install_sct_token_local_kv(unet, states, method):
    """Install three complete states without merging or changing the base weights."""
    if method not in METHODS or set(states) != set(TOKENS):
        raise ValueError('require three named states and one matching adapter method')
    if any(not isinstance(state, dict) for state in states.values()):
        raise ValueError('adapter states must be dictionaries')
    processors, expected = {}, set()
    for name in unet.attn_processors:
        if name.endswith('attn1.processor'):
            if method == 'fullrank':
                expected.add(name)
                if any(not isinstance(state.get(name), dict) or state[name] for state in states.values()):
                    raise ValueError('fullrank self-attention entries must be empty')
            processors[name] = SCTTokenLocalKVAttnProcessor(method=method)
            continue
        if not name.endswith('attn2.processor'):
            raise ValueError(f'unexpected attention processor: {name}')
        attn = unet.get_submodule(name.removesuffix('.processor'))
        hidden_size, cross_attention_dim = attn.to_k.weight.shape
        if attn.to_v.weight.shape != (hidden_size, cross_attention_dim):
            raise ValueError(f'base K/V shapes differ: {name}')
        processor = SCTTokenLocalKVAttnProcessor(hidden_size, cross_attention_dim, method)
        for branch, state in states.items():
            for projection in ('k', 'v'):
                layer = getattr(processor, projection + '_residuals')[branch]
                selected = {}
                prefix = f'delta_{projection}' if method == 'fullrank' else f'to_{projection}_lora'
                for suffix, parameter in layer.state_dict().items():
                    key = f'{name}.{prefix}.{suffix}'
                    expected.add(key)
                    tensor = state.get(key)
                    if not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape:
                        raise ValueError(f'missing or mismatched {branch} attention tensor: {key}')
                    selected[suffix] = tensor
                layer.load_state_dict(selected)
        processor.to(device=attn.to_k.weight.device, dtype=attn.to_k.weight.dtype)
        processor.requires_grad_(False)
        processor.eval()
        processors[name] = processor
    if any(set(state) != expected for state in states.values()):
        raise ValueError('S/C/T checkpoint has missing or unexpected attention entries')
    unet.set_attn_processor(processors)


def attention_masks(pipe, prompt, guidance=3.5):
    ids = pipe.tokenizer(prompt, padding='max_length', max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors='pt').input_ids.to(pipe.unet.device)
    masks = {}
    for branch, token in TOKENS.items():
        if token not in pipe.tokenizer.get_vocab() or prompt.count(token) != 1:
            raise ValueError('S/C/T prompt requires each learned token exactly once')
        mask = ids == pipe.tokenizer.convert_tokens_to_ids(token)
        if int(mask.sum()) != 1:
            raise ValueError('S/C/T modifier was repeated or truncated')
        if guidance > 1:
            mask = torch.cat([torch.zeros_like(mask), mask])
        masks[branch] = mask
    return {'modifier_token_mask': masks}
