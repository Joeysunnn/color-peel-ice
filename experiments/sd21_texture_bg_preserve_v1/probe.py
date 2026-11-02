"""One actual T400 training batch; sequential gradient controls; no optimizer step."""

import argparse
import gc
import os
from pathlib import Path
import pickle
import random
import sys

from experiments.sd21_texture_bg_preserve_v1 import contract
from experiments.sd21_mailbox_sct_v1 import run as sct


def probe(audit, output):
    import numpy as np
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import AutoTokenizer, CLIPTextModel
    from experiments.lora_kv_subject_v1.attention import install_subject_lora_kv
    from src.train.texture_background_preservation import (
        cache_reference_conditioning, reference_prediction, background_preservation_loss, gradient_report)

    sys.path.insert(0, str(sct.ROOT / 'src/train'))
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel
    from train_colorpeel import CustomDiffusionDataset, collate_fn

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sct.st.shared.snapshots.check_locks(audit['source_locks'])
    sct.st.require_idle_gpu0()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = str(contract.MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(model, subfolder='tokenizer', use_fast=False, local_files_only=True)
    encoder = CLIPTextModel.from_pretrained(model, subfolder='text_encoder', local_files_only=True).to('cuda:0')
    initializer = tokenizer.encode('texture', add_special_tokens=False)
    if len(initializer) != 1 or tokenizer.add_tokens('<T*>') != 1:
        raise ValueError('probe requires unmodified SD2.1 tokenizer and one texture initializer')
    encoder.resize_token_embeddings(len(tokenizer))
    token_id = tokenizer.convert_tokens_to_ids('<T*>')
    with torch.no_grad():
        encoder.get_input_embeddings().weight[token_id].copy_(encoder.get_input_embeddings().weight[initializer[0]])
    config = contract.original.build_config('texture', 'lora_r64')['args']
    concepts = sct.st.read_json(config['concepts_list'])
    dataset = CustomDiffusionDataset(concepts, tokenizer, size=512, mask_size=64,
        center_crop=config.get('center_crop', False), with_prior_preservation=False, hflip=False)
    if len(dataset) != 16:
        raise ValueError('probe requires original Texture16')
    captions = [caption for row in concepts for caption in row['instance_prompt']]
    cache_ids = tokenizer(captions, truncation=True, padding='max_length', max_length=77,
                          return_tensors='pt').input_ids

    def fingerprint(module):
        # Only GPU0 is visible; do not initialize or inspect SDXL devices.
        return {'python': pickle.dumps(random.getstate()), 'numpy': pickle.dumps(np.random.get_state()),
                'torch_cpu': torch.get_rng_state().clone(), 'torch_cuda0': torch.cuda.get_rng_state(0).clone(),
                'modes': [child.training for child in module.modules()]}

    def unchanged(before, module):
        after = fingerprint(module)
        return {key: bool(torch.equal(value, after[key])) if isinstance(value, torch.Tensor)
                else value == after[key] for key, value in before.items()}

    before = fingerprint(encoder)
    reference = cache_reference_conditioning(encoder, cache_ids, token_id, initializer[0])
    cache_restored = unchanged(before, encoder)
    source = audit['texture_sources']['400']
    snapshot = Path(source['run']) / 'checkpoints/checkpoint-400'
    encoder.load_state_dict(torch.load(snapshot / 'pytorch_model_1.bin', map_location='cpu'))
    encoder.requires_grad_(False)
    embedding = encoder.get_input_embeddings().weight
    embedding.requires_grad_(True)
    encoder.train()
    unet = UNet2DConditionModel.from_pretrained(model, subfolder='unet', local_files_only=True).to('cuda:0')
    unet.requires_grad_(False)
    install_subject_lora_kv(unet, torch.load(Path(source['checkpoint']) / source['weight'], map_location='cpu'),
                            'token_local_kv', 64, 64)
    unet.train()
    if any(parameter.dtype != torch.float32 for module in (encoder, unet) for parameter in module.parameters()):
        raise ValueError('real training probe requires FP32 base, encoder and adapters')
    # Match the original recipe. Reentrant checkpointing is incompatible with
    # autograd.grad in the pinned PyTorch1.13 stack; release each control graph.
    groups = {f'{projection}_{direction}': [] for projection in ('K', 'V') for direction in ('up', 'down')}
    for name, parameter in unet.named_parameters():
        if parameter.requires_grad:
            matched = [key for key in groups if f'.to_{key[0].lower()}_lora.{key[2:]}.' in name]
            if len(matched) != 1:
                raise ValueError('unexpected trainable probe parameter')
            groups[matched[0]].append(parameter)
    if any(len(group) != 16 for group in groups.values()):
        raise ValueError('require all16 LoRA K/V up/down groups')
    batch = collate_fn([dataset[0]], False, modifier_token_ids=[token_id])
    batch = {key: value.to('cuda:0') for key, value in batch.items()}
    vae = AutoencoderKL.from_pretrained(model, subfolder='vae', local_files_only=True).to('cuda:0')
    vae.requires_grad_(False)
    if any(parameter.dtype != torch.float32 for parameter in vae.parameters()):
        raise ValueError('real training probe requires FP32 VAE')
    with torch.no_grad():
        latents = vae.encode(batch['pixel_values']).latent_dist.sample() * vae.config.scaling_factor
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    scheduler = DDPMScheduler.from_pretrained(model, subfolder='scheduler', local_files_only=True)
    noise = torch.randn_like(latents)
    timestep = torch.tensor([500], device='cuda:0', dtype=torch.long)
    noisy = scheduler.add_noise(latents, noise, timestep)
    if scheduler.config.prediction_type == 'epsilon':
        target = noise
    elif scheduler.config.prediction_type == 'v_prediction':
        target = scheduler.get_velocity(latents, noise, timestep)
    else:
        raise ValueError('unexpected training scheduler prediction type')

    def clear_graphs():
        for module in unet.modules():
            if hasattr(module, 'attn_probs'):
                module.attn_probs = None
        gc.collect()
        torch.cuda.empty_cache()

    reports = {}
    class_prompt = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True).replace('<T*>', 'texture')
    for name, prompt in (('class_only', class_prompt), ('unconditional', '')):
        ids = tokenizer(prompt, truncation=True, padding='max_length', max_length=77, return_tensors='pt').input_ids.to('cuda:0')
        if bool((ids == token_id).any()):
            raise ValueError('negative control includes Texture token')
        prediction = unet(noisy, timestep, encoder_hidden_states=encoder(ids)[0],
            cross_attention_kwargs={'modifier_token_mask': torch.zeros_like(ids, dtype=torch.bool)}).sample
        loss = (prediction.float() - target).square().mean()
        reports[name] = gradient_report(loss, groups, embedding, token_id, retain_graph=False)
        reports[name].update(prompt=prompt, input_ids=ids.cpu().tolist())
        del loss, prediction, ids
        clear_graphs()
    before = fingerprint(unet)
    teacher = reference_prediction(unet, noisy, timestep, batch['input_ids'], reference)
    teacher_restored = unchanged(before, unet)
    prediction = unet(noisy, timestep, encoder_hidden_states=encoder(batch['input_ids'])[0],
        cross_attention_kwargs={'modifier_token_mask': batch['input_ids'] == token_id}).sample
    foreground_loss = (((prediction.float() - target.float()).square() * batch['mask']).sum((1, 2, 3))
                       / batch['mask'].sum((1, 2, 3))).mean()
    reports['foreground'] = gradient_report(foreground_loss, groups, embedding, token_id, retain_graph=True)
    loss, background = background_preservation_loss(prediction, teacher, batch['mask'])
    reports['background'] = gradient_report(loss, groups, embedding, token_id, retain_graph=False)
    ratios = {key: reports['background']['adapter_group_norms'][key] / norm if norm else None
              for key, norm in reports['foreground']['adapter_group_norms'].items()}
    foreground_token_norm = reports['foreground']['modifier_embedding_norm']
    ratios['texture_embedding'] = reports['background']['modifier_embedding_norm'] / foreground_token_norm if foreground_token_norm else None
    controls_zero = all(all(value == 0 for value in reports[name]['adapter_group_norms'].values())
        and reports[name]['modifier_embedding_norm'] == 0 for name in ('class_only', 'unconditional'))
    nonzero = (any(value > 0 for value in reports['background']['adapter_group_norms'].values())
               and reports['background']['modifier_embedding_norm'] > 0)
    passed = (controls_zero and nonzero and not teacher.requires_grad and all(cache_restored.values())
        and all(teacher_restored.values()) and all(row['finite'] and row['ordinary_embedding_norm_after_mask'] == 0
                                                 for row in reports.values()))
    result = {'passed': passed, 'gradient_reports': reports, 'teacher_requires_grad': teacher.requires_grad,
        'negative_controls_zero': controls_zero, 'background_gradients_nonzero': nonzero,
        'cache_state_restored': cache_restored, 'teacher_state_restored': teacher_restored,
        'background_loss': float(loss.detach()), 'background_pixels': background.sum().item(),
        'foreground_loss': float(foreground_loss.detach()), 'background_to_foreground_gradient_norm_ratios': ratios,
        'reference': reference.audit(), 'timestep': 500, 'probe_seed': 42, 'dtype': 'float32',
        'gradient_checkpointing': False, 'optimizer_step_performed': False,
        'data_item': {'image': str(dataset.instance_images_path[0][0]),
                      'mask': str(dataset.instance_images_path[0][2]), 'input_ids': batch['input_ids'].cpu().tolist()},
        'source_pair': source, 'inference_git': sct.st.git_info()}
    sct.write(Path(output), result)
    sct.st.shared.snapshots.check_locks(audit['source_locks'])
    if not passed:
        raise ValueError('real T400 background gradient audit failed; training is blocked')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    probe(sct.st.read_json(args.audit), args.output)
