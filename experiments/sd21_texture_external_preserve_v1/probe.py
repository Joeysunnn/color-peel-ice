"""Actual T400 foreground and external regularizer gradients, including checkpoint backward."""

import argparse
import gc
import os
from pathlib import Path
import pickle
import random
import sys

from experiments.sd21_texture_external_preserve_v1 import contract
from experiments.sd21_mailbox_sct_v1 import run as sct


def probe(audit, output):
    import numpy as np
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import AutoTokenizer, CLIPTextModel
    from experiments.lora_kv_subject_v1.attention import install_subject_lora_kv
    from src.train.texture_background_preservation import gradient_report
    from src.train.texture_external_preservation import ExternalTexturePreservation

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
    preservation = ExternalTexturePreservation(audit['preservation_assets']['manifest_path'], tokenizer, encoder, token_id, initializer[0])
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

    prediction = unet(noisy, timestep, encoder_hidden_states=encoder(batch['input_ids'])[0],
        cross_attention_kwargs={'modifier_token_mask': batch['input_ids'] == token_id}).sample
    foreground_loss = (((prediction.float() - target.float()).square() * batch['mask']).sum((1, 2, 3))
                       / batch['mask'].sum((1, 2, 3))).mean()
    fg_value = float(foreground_loss.detach())
    foreground = gradient_report(foreground_loss, groups, embedding, token_id, retain_graph=False)
    del prediction, foreground_loss
    clear_graphs()
    before = fingerprint(unet)
    with preservation.loss_context(unet, vae, scheduler, step=401, device='cuda:0', dtype=torch.float32) as (loss, metrics):
        external = gradient_report(loss, groups, embedding, token_id, retain_graph=False)
        raw_loss = float(loss.detach())
    restored = unchanged(before, unet)
    ratios = {key: contract.ARMS['preservation'] * external['adapter_group_norms'][key] / value
              if value else None for key, value in foreground['adapter_group_norms'].items()}
    clear_graphs()
    # The real trainer uses reentrant checkpointing. Detached text must not silently
    # disable adapter gradients in this path; exercise actual backward as well.
    unet.enable_gradient_checkpointing()
    unet.zero_grad(set_to_none=True)
    encoder.zero_grad(set_to_none=True)
    before = fingerprint(unet)
    with preservation.loss_context(unet, vae, scheduler, step=401, device='cuda:0', dtype=torch.float32) as (loss2, checkpoint_metrics):
        (contract.ARMS['preservation'] * loss2).backward()
        repeated_loss_equal = float(loss2.detach()) == raw_loss
    checkpoint_restored = unchanged(before, unet)
    checkpoint_norms = {key: sum(float(parameter.grad.detach().double().square().sum())
                        for parameter in values if parameter.grad is not None) ** 0.5
                        for key, values in groups.items()}
    base_gradient_absent = all(parameter.grad is None for parameter in unet.parameters() if not parameter.requires_grad)
    embedding_gradient_absent = all(parameter.grad is None for parameter in encoder.parameters())
    vae_gradient_absent = all(parameter.grad is None for parameter in vae.parameters())
    passed = (foreground['finite'] and external['finite'] and
        all(value > 0 for value in external['adapter_group_norms'].values()) and
        all(value > 0 and np.isfinite(value) for value in checkpoint_norms.values()) and
        external['modifier_embedding_norm'] == 0 and external['ordinary_embedding_norm_before_mask'] == 0 and
        not metrics['teacher_requires_grad'] and not metrics['hidden_requires_grad'] and
        all(cache_restored.values()) and all(restored.values()) and all(checkpoint_restored.values()) and
        repeated_loss_equal and base_gradient_absent and embedding_gradient_absent and vae_gradient_absent)
    result = {'passed': bool(passed), 'foreground_loss': fg_value, 'preservation_loss': raw_loss,
        'foreground_gradients': foreground, 'preservation_gradients': external,
        'weighted_preservation_to_foreground_gradient_ratios': ratios,
        'checkpoint_backward_norms': checkpoint_norms, 'checkpoint_repeat_loss_equal': repeated_loss_equal,
        'base_gradient_absent': base_gradient_absent, 'embedding_gradient_absent': embedding_gradient_absent,
        'vae_gradient_absent': vae_gradient_absent, 'cache_state_restored': cache_restored,
        'rng_state_restored': restored, 'checkpoint_rng_state_restored': checkpoint_restored,
        'preservation_metrics': metrics, 'checkpoint_preservation_metrics': checkpoint_metrics,
        'reference': preservation.audit(), 'probe_seed': 42, 'dtype': 'float32',
        'optimizer_step_performed': False, 'source_pair': source, 'inference_git': sct.st.git_info()}
    sct.write(Path(output), result)
    if not passed:
        raise ValueError('real external preservation gradient probe failed; training blocked')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    probe(sct.st.read_json(args.audit), args.output)
