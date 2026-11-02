"""Read-only CLIP diagnostics for the two fixed prompt orders."""

import hashlib


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def initializer_ids(tokenizer):
    result = {name: tokenizer.encode(word, add_special_tokens=False)
              for name, word in (('subject', 'mailbox'), ('color', 'orange'))}
    return {'ids': result, 'same_length_replacement_available': all(len(ids) == 1 for ids in result.values()),
            'strategy': 'Original trainer single_token_initializer_id rejects multi-token initializers; '
                        'this diagnostic records and skips that counterfactual instead of selecting a piece.'}


def tokenize(pipe, prompt, tokens):
    converted = pipe.maybe_convert_prompt(prompt, pipe.tokenizer)
    if converted != prompt or len(pipe.tokenizer.encode(prompt, add_special_tokens=True)) > 77:
        raise ValueError('require unexpanded, untruncated single-vector CLIP77 prompts')
    encoded = pipe.tokenizer(prompt, padding='max_length', max_length=77, truncation=True, return_tensors='pt')
    positions = {}
    for branch, token in tokens.items():
        matches = (encoded.input_ids[0] == pipe.tokenizer.convert_tokens_to_ids(token)).nonzero().flatten()
        if prompt.count(token) != 1 or matches.numel() != 1:
            raise ValueError('each prompt must retain each learned token exactly once')
        positions[branch] = int(matches.item())
    return encoded, positions


def diagnose(pipe, body, scenes, tokens):
    import torch

    if pipe.tokenizer.model_max_length != 77 or pipe.text_encoder.training:
        raise ValueError('require the actual CLIP77 text encoder in evaluation mode')
    device = pipe.unet.device
    use_mask = bool(getattr(pipe.text_encoder.config, 'use_attention_mask', False))

    def encode_ids(ids, attention_mask):
        kwargs = {'attention_mask': attention_mask.to(device)} if use_mask else {}
        return pipe.text_encoder(ids.to(device), **kwargs)[0]

    prompts = {'body': body}
    prompts.update({f'P0_suffix/{scene}': body + ' ' + suffix for scene, suffix in scenes.items()})
    prompts.update({f'P1_prefix/{scene}': suffix + ' ' + body for scene, suffix in scenes.items()})
    hidden, inputs, positions, records = {}, {}, {}, {}
    initializers = initializer_ids(pipe.tokenizer)
    with torch.no_grad():
        for name, prompt in prompts.items():
            encoded, slots = tokenize(pipe, prompt, tokens)
            h = encode_ids(encoded.input_ids, encoded.attention_mask)
            if tuple(h.shape[:2]) != (1, 77):
                raise ValueError('unexpected CLIP token hidden shape')
            # Verify that these are the hidden states the installed pipeline uses.
            actual = pipe._encode_prompt(prompt, device, 1, False)
            if actual.shape != h.shape or not torch.equal(actual, h):
                raise ValueError('direct CLIP encoding differs from the pipeline conditioning path')
            hidden[name], inputs[name], positions[name] = h, encoded, slots
            records[name] = {'prompt': prompt, 'input_ids': encoded.input_ids.tolist(),
                             'attention_mask': encoded.attention_mask.tolist(), 'modifier_positions': slots,
                             'hidden_shape': list(h.shape), 'hidden_dtype': str(h.dtype),
                             'hidden_sha256': tensor_sha256(h), 'pipeline_hidden_equal': True}
        suffix_changes, neutralized = {}, {}
        for scene in scenes:
            name = f'P0_suffix/{scene}'
            prefix_ids = inputs[name].input_ids[0, :positions['body']['texture'] + 1]
            if (positions[name] != positions['body']
                    or not torch.equal(prefix_ids, inputs['body'].input_ids[0, :len(prefix_ids)])):
                raise ValueError('suffix changed the shared prefix before Texture')
            suffix_changes[scene] = {branch: float((hidden[name][0, slot] - hidden['body'][0, slot]).abs().max())
                                     for branch, slot in positions[name].items()}
            if initializers['same_length_replacement_available']:
                ids = inputs[name].input_ids.clone()
                for branch in ('subject', 'color'):
                    ids[0, positions[name][branch]] = initializers['ids'][branch][0]
                h = encode_ids(ids, inputs[name].attention_mask)
                slot = positions[name]['texture']
                delta = h[0, slot].float() - hidden[name][0, slot].float()
                neutralized[scene] = {'status': 'measured', 'input_ids': ids.tolist(),
                                      'texture_position': slot, 'texture_hidden_max_abs': float(delta.abs().max()),
                                      'texture_hidden_l2': float(delta.norm()), 'hidden_sha256': tensor_sha256(h)}
            else:
                neutralized[scene] = {'status': 'skipped_multi_token_initializer', 'initializer_ids': initializers['ids']}
        negative = pipe.tokenizer('', padding='max_length', max_length=77, truncation=True, return_tensors='pt')
    return {'prompts': records, 'suffix_modifier_hidden_max_abs_vs_body': suffix_changes,
            'same_length_SC_initializer_counterfactual': neutralized, 'initializers': initializers,
            'negative_input_ids': negative.input_ids.tolist(), 'text_encoder_use_attention_mask': use_mask,
            'cross_attention_norms': {name: str(pipe.unet.get_submodule(name.removesuffix('.processor')).norm_cross)
                                      for name in pipe.unet.attn_processors if name.endswith('attn2.processor')},
            'interpretation': 'Observed token hidden differences; prompt order also changes position and syntax. '
                              'Unchanged modifier hidden does not prevent scene tokens from conditioning the UNet.'}


def validate_gates(ids, gates, tokenizer, tokens):
    import torch

    masks = gates['modifier_token_mask']
    if set(masks) != set(tokens):
        raise ValueError('missing S/C/T CFG gates')
    for branch, token in tokens.items():
        expected = ids == tokenizer.convert_tokens_to_ids(token)
        gate = masks[branch]
        if (gate.dtype != torch.bool or tuple(gate.shape) != (2, 77) or bool(gate[0].any())
                or not torch.equal(gate[1:].cpu(), expected.cpu()) or int(expected.sum()) != 1):
            raise ValueError('prompt/conditional gate mismatch or nonzero unconditional gate')


def capture_initial_input(unet, captured):
    """Observe the actual first denoiser input without changing generator use."""
    def hook(module, args):
        if captured:
            return
        if len(args) < 2:
            raise ValueError('the pinned pipeline must pass UNet sample/timestep positionally')
        sample, timestep = args[:2]
        captured.update(sha256=tensor_sha256(sample[:1]), shape=list(sample.shape), dtype=str(sample.dtype),
                        timestep=float(timestep), measurement='first_UNet_input_after_scheduler_scaling')
    return unet.register_forward_pre_hook(hook)
