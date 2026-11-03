"""Four independent SD2.1 rank64 K/V adapters; inference with explicit CFG gates."""

import torch
from torch import nn

from src.train.custom_attention.attention_processor_custom import LoRALinearLayer, TokenLocalKVAttnProcessor

TOKENS = {'subject': '<S*>', 'color': '<C*>', 'material': '<M*>', 'texture': '<T*>'}


class FourBranchLoraKVAttnProcessor(TokenLocalKVAttnProcessor):
    def __init__(self, hidden_size=None, cross_attention_dim=None):
        super().__init__()
        self.k_residuals, self.v_residuals = nn.ModuleDict(), nn.ModuleDict()
        if cross_attention_dim is not None:
            if cross_attention_dim != 1024:
                raise ValueError('four-branch inference requires SD2.1 text width1024')
            for branch in TOKENS:
                for layers in (self.k_residuals, self.v_residuals):
                    layers[branch] = LoRALinearLayer(1024, hidden_size, rank=64, network_alpha=64)

    def project_kv(self, attn, encoder_hidden_states, modifier_token_mask):
        if not isinstance(modifier_token_mask, dict) or set(modifier_token_mask) != set(TOKENS):
            raise ValueError('require four separate subject/color/material/texture masks')
        masks = list(modifier_token_mask.values())
        if (encoder_hidden_states.shape[0] != 2 or any(
                not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                or mask.shape != encoder_hidden_states.shape[:2] for mask in masks)):
            raise ValueError('four-branch CFG requires Boolean masks matching two text batches')
        if any(bool(mask[0].any()) or int(mask[1].sum()) > 1 for mask in masks):
            raise ValueError('CFG negative gates must be zero; each positive branch has at most one token')
        if any(torch.any(masks[i] & masks[j]) for i in range(4) for j in range(i + 1, 4)):
            raise ValueError('four branch masks must be disjoint')
        key, value = attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)
        for branch in TOKENS:
            gate = modifier_token_mask[branch].to(device=key.device).unsqueeze(-1)
            key = torch.where(gate, key + self.k_residuals[branch](encoder_hidden_states), key)
            value = torch.where(gate, value + self.v_residuals[branch](encoder_hidden_states), value)
        return key, value


def install_four_branch_lora(unet, states):
    """Validate all four complete exports before replacing any processor."""
    if not isinstance(states, dict) or set(states) != set(TOKENS):
        raise ValueError('require exactly four named LoRA states')
    if any(not isinstance(state, dict) or not state for state in states.values()):
        raise ValueError('LoRA states must be nonempty dictionaries')
    expected, processors = set(), {}
    for name in unet.attn_processors:
        if name.endswith('attn1.processor'):
            processors[name] = FourBranchLoraKVAttnProcessor()
            continue
        if not name.endswith('attn2.processor'):
            raise ValueError(f'unexpected attention processor: {name}')
        attn = unet.get_submodule(name.removesuffix('.processor'))
        hidden, width = attn.to_k.weight.shape
        if width != 1024 or attn.to_v.weight.shape != (hidden, width):
            raise ValueError(f'base SD2.1 K/V dimensions differ: {name}')
        processor = FourBranchLoraKVAttnProcessor(hidden, width)
        for branch, state in states.items():
            for projection in ('k', 'v'):
                layer = getattr(processor, projection + '_residuals')[branch]
                selected = {}
                for suffix, parameter in layer.state_dict().items():
                    key = f'{name}.to_{projection}_lora.{suffix}'
                    expected.add(key)
                    value = state.get(key)
                    if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                            or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
                        raise ValueError(f'missing, mismatched or nonfinite FP32 {branch} tensor: {key}')
                    selected[suffix] = value
                layer.load_state_dict(selected)
        processor.to(device=attn.to_k.weight.device, dtype=attn.to_k.weight.dtype)
        if any(not bool(torch.isfinite(parameter).all()) for parameter in processor.parameters()):
            raise ValueError('LoRA weights overflowed inference dtype')
        processor.requires_grad_(False)
        processor.eval()
        processors[name] = processor
    if not expected or any(set(state) != expected for state in states.values()):
        raise ValueError('four-branch export has missing or unexpected attention tensors')
    unet.set_attn_processor(processors)


def attention_masks(pipe, prompt):
    """Build four disjoint gates; refuse CLIP truncation or token expansion."""
    tokenizer = pipe.tokenizer
    if (tokenizer.model_max_length != 77 or pipe.maybe_convert_prompt(prompt, tokenizer) != prompt
            or len(tokenizer.encode(prompt, add_special_tokens=True)) > 77):
        raise ValueError('require an unexpanded prompt fitting all77 CLIP positions')
    ids = tokenizer(prompt, padding='max_length', max_length=77,
                    truncation=False, return_tensors='pt').input_ids.to(pipe.unet.device)
    masks, token_ids = {}, []
    for branch, token in TOKENS.items():
        if token not in tokenizer.get_vocab() or prompt.count(token) != 1:
            raise ValueError('four-branch prompt requires each learned token exactly once')
        token_id = tokenizer.convert_tokens_to_ids(token)
        token_ids.append(token_id)
        mask = ids == token_id
        if int(mask.sum()) != 1:
            raise ValueError('four-branch token must occupy exactly one text position')
        masks[branch] = torch.cat((torch.zeros_like(mask), mask))
    if len(set(token_ids)) != 4:
        raise ValueError('four learned tokens must have distinct IDs')
    return {'modifier_token_mask': masks}
