"""Exact CFG gates and same-position S/C initializer context for Texture only."""

from experiments.sd21_prompt_order_v1.conditioning import initializer_ids, tensor_sha256


def replace_sc_ids(ids, positions, initializers):
    """Pure ID operation: replace two entries without changing T or token positions."""
    if len(ids) != 77 or set(positions) != {'subject', 'color', 'texture'}:
        raise ValueError('require the full77-position SCT encoding')
    if len(set(positions.values())) != 3 or any(type(pos) is not int or not 0 <= pos < 77 for pos in positions.values()):
        raise ValueError('require three distinct learned-token positions')
    result = list(ids)
    for branch in ('subject', 'color'):
        values = initializers[branch]
        if len(values) != 1 or type(values[0]) is not int or values[0] == ids[positions[branch]]:
            raise ValueError('initializer must be one ordinary ID different from the learned token')
        result[positions[branch]] = values[0]
    return result


def prepare_conditioning(pipe, prompt, condition):
    import torch

    if condition not in ('baseline_sct', 'true_SC', 't_residual_initializer'):
        raise ValueError('unknown context-routing condition')
    tokens = {'subject': '<S*>', 'color': '<C*>', 'texture': '<T*>'}
    if (pipe.tokenizer.model_max_length != 77 or pipe.text_encoder.training
            or pipe.maybe_convert_prompt(prompt, pipe.tokenizer) != prompt
            or len(pipe.tokenizer.encode(prompt, add_special_tokens=True)) > 77):
        raise ValueError('require eval CLIP77 and an unexpanded, untruncated prompt')
    device = pipe.unet.device
    encoded = pipe.tokenizer(prompt, padding='max_length', max_length=77, truncation=True, return_tensors='pt')
    positions, masks = {}, {}
    for branch, token in tokens.items():
        positive = encoded.input_ids == pipe.tokenizer.convert_tokens_to_ids(token)
        required = 0 if condition == 'true_SC' and branch == 'texture' else 1
        if prompt.count(token) != required or int(positive.sum()) != required:
            raise ValueError('prompt token presence differs from this condition')
        positions[branch] = int(positive[0].nonzero().item()) if required else None
        masks[branch] = torch.cat((torch.zeros_like(positive), positive)).to(device)
    kwargs = {'modifier_token_mask': masks}
    use_mask = bool(getattr(pipe.text_encoder.config, 'use_attention_mask', False))

    def encode(ids, attention_mask):
        arguments = {'attention_mask': attention_mask.to(device)} if use_mask else {}
        return pipe.text_encoder(ids.to(device), **arguments)[0]

    with torch.no_grad():
        main = pipe._encode_prompt(prompt, device, 1, True)
        direct = encode(encoded.input_ids, encoded.attention_mask)
        negative = pipe.tokenizer('', padding='max_length', max_length=77, truncation=True, return_tensors='pt')
        negative_hidden = encode(negative.input_ids, negative.attention_mask)
        if (tuple(main.shape) != (2, 77, 1024) or not torch.equal(main[1:], direct)
                or not torch.equal(main[:1], negative_hidden)):
            raise ValueError('direct CLIP encoding differs from actual pipeline CFG conditioning')
        record = {'prompt': prompt, 'condition': condition, 'main_input_ids': encoded.input_ids.tolist(),
            'negative_input_ids': negative.input_ids.tolist(), 'modifier_positions': positions,
            'cfg_masks': {branch: mask.cpu().tolist() for branch, mask in masks.items()},
            'main_hidden_sha256': tensor_sha256(main), 'main_hidden_shape': list(main.shape),
            'main_hidden_dtype': str(main.dtype), 'pipeline_hidden_equal': True,
            'text_encoder_use_attention_mask': use_mask, 'texture_auxiliary': None}
        if condition == 't_residual_initializer':
            initializers = initializer_ids(pipe.tokenizer)
            if not initializers['same_length_replacement_available']:
                raise ValueError('Q1 requires single-token mailbox/orange initializers; no BPE-piece substitution')
            auxiliary_ids = encoded.input_ids.clone()
            auxiliary_ids[0] = torch.tensor(replace_sc_ids(encoded.input_ids[0].tolist(), positions, initializers['ids']))
            positive_hidden = encode(auxiliary_ids, encoded.attention_mask).to(dtype=main.dtype)
            auxiliary = torch.cat((main[:1], positive_hidden))
            if not bool(torch.isfinite(auxiliary).all()) or not torch.equal(auxiliary[:1], main[:1]):
                raise ValueError('Q1 auxiliary conditioning is nonfinite or changed the unconditional hidden')
            kwargs['texture_encoder_hidden_states'] = auxiliary
            texture_delta = positive_hidden[0, positions['texture']].float() - main[1, positions['texture']].float()
            record['texture_auxiliary'] = {'input_ids': auxiliary_ids.tolist(),
                'cfg_input_ids': torch.cat((negative.input_ids, auxiliary_ids)).tolist(),
                'attention_mask': encoded.attention_mask.tolist(), 'initializer_ids': initializers['ids'],
                'hidden_sha256': tensor_sha256(auxiliary), 'hidden_shape': list(auxiliary.shape),
                'hidden_dtype': str(auxiliary.dtype), 'negative_hidden_unchanged': True,
                'texture_hidden_max_abs_change': float(texture_delta.abs().max()),
                'texture_hidden_l2_change': float(texture_delta.norm()),
                'route': 'Texture K/V residual input only; base K/V and Subject/Color residuals keep main hidden.'}
    return kwargs, record
