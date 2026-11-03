"""Audit learned S/C/T gradients without updating weights; training remains fresh."""

import argparse
import gc
import os
from pathlib import Path
import pickle
import random
import sys


def probe(experiment):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import numpy as np
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import AutoTokenizer, CLIPTextModel
    from experiments.lora_kv_subject_v1.attention import install_subject_lora_kv
    from experiments.sd21_mailbox_sct_v1 import run as sct
    from experiments.sd21_tokenverse_isolation_v1 import contract
    from src.train.concept_isolation import ConceptIsolation, clear_attention_caches
    from src.train.texture_background_preservation import gradient_report

    sys.path.insert(0, str(sct.ROOT / 'src/train'))
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel
    from train_colorpeel import CustomDiffusionDataset, collate_fn

    experiment = Path(experiment).resolve()
    output = experiment / 'gradient_probe.json'
    if output.exists():
        raise ValueError('gradient probe output must be new')
    audit = sct.st.read_json(experiment / 'source_audit.json')
    revision = sct.st.git_info()
    if revision != audit['new_training_git']:
        raise ValueError('probe checkout differs from the new training checkout')
    result = {'passed': False, 'status': 'running', 'branches': {}, 'expected_samples': 9,
        'sample_indices': '[0, n//2, n-1]', 'probe_seed': 42, 'dtype': 'float32', 'gpu': 0,
        'isolation_weight': 1.0, 'optimizer_step_performed': False, 'inference_git': revision,
        'purpose': 'Original learned pairs provide nonzero-gradient probes only; both training arms start from the base.'}
    sct.write(output, result)

    def fingerprint(modules):
        return {'python': pickle.dumps(random.getstate()), 'numpy': pickle.dumps(np.random.get_state()),
            'torch_cpu': torch.get_rng_state().clone(), 'torch_cuda0': torch.cuda.get_rng_state(0).clone(),
            'modes': {name: [child.training for child in module.modules()] for name, module in modules.items()}}

    def restored(before, modules):
        after = fingerprint(modules)
        return {key: bool(torch.equal(value, after[key])) if isinstance(value, torch.Tensor)
                else value == after[key] for key, value in before.items()}

    def valid_gradients(report):
        return (report['finite'] and report['modifier_embedding_norm'] > 0
                and report['ordinary_embedding_norm_after_mask'] == 0
                and all(value > 0 for value in report['adapter_group_norms'].values()))

    def backward_report(groups, embedding, token_id):
        norms, finite = {}, True
        for name, parameters in groups.items():
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            finite = finite and all(bool(torch.isfinite(value).all()) for value in gradients)
            norms[name] = sum(float(value.detach().double().square().sum()) for value in gradients) ** 0.5
        gradient = embedding.grad
        if gradient is None:
            return {'adapter_group_norms': norms, 'modifier_embedding_norm': 0,
                'ordinary_embedding_norm_before_mask': 0, 'ordinary_embedding_norm_after_mask': 0, 'finite': False}
        finite = finite and bool(torch.isfinite(gradient).all())
        ordinary = torch.ones(gradient.shape[0], dtype=torch.bool, device=gradient.device)
        ordinary[token_id] = False
        before, modifier = float(gradient[ordinary].norm()), float(gradient[token_id].norm())
        gradient[ordinary] = 0  # Apply the trainer's row mask to actual backward gradients.
        return {'adapter_group_norms': norms, 'modifier_embedding_norm': modifier,
            'ordinary_embedding_norm_before_mask': before, 'ordinary_embedding_norm_after_mask': float(gradient[ordinary].norm()),
            'finite': finite}

    try:
        sct.st.require_idle_gpu0()
        locks = dict(audit['source_locks'])
        for branch in contract.BRANCHES:
            sct.st.source._merge(locks, audit['sources'][branch]['lora_r64']['hashes'])
        sct.st.shared.snapshots.check_locks(locks)
        base = sct.st.shared.verify_model(contract.MODEL_DIR)['model']
        sct.st.require_idle_gpu0()
        for branch in contract.BRANCHES:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            source = audit['sources'][branch]['lora_r64']
            maximum, token = contract.STEPS[branch][-1], contract.TOKENS[branch]
            if (source.get('step') != maximum or source.get('steps') != maximum
                    or source.get('branch') != branch or source.get('token') != token
                    or source.get('method') != 'lora_r64' or source.get('base_model') != base
                    or source.get('embedding_dim') != 1024
                    or source.get('paired_embedding_finite') is not True or source.get('status') != 'verified'):
                raise ValueError('probe requires the audited original learned S1000/C150/T1000 pairs')
            config = contract.build_config(branch, 'isolation', audit['bank']['manifest_path'])['args']
            concepts = sct.st.read_json(config['concepts_list'])
            if any(not isinstance(row['instance_prompt'], list) or len(row['instance_prompt']) != 1
                   or not isinstance(row['instance_prompt'][0], str) for row in concepts):
                raise ValueError('each original concept must have exactly one fixed caption')
            prompts = [row['instance_prompt'][0] for row in concepts]
            model = str(contract.MODEL_DIR)
            tokenizer = AutoTokenizer.from_pretrained(model, subfolder='tokenizer', use_fast=False, local_files_only=True)
            initializer = tokenizer.encode(contract.INITIALIZERS[branch], add_special_tokens=False)
            if len(initializer) != 1 or tokenizer.add_tokens(token) != 1:
                raise ValueError('require a fresh tokenizer and one ordinary initializer ID')
            token_id = tokenizer.convert_tokens_to_ids(token)
            encoder = CLIPTextModel.from_pretrained(model, subfolder='text_encoder', local_files_only=True).to('cuda:0')
            encoder.resize_token_embeddings(len(tokenizer))
            encoder.requires_grad_(False)
            embedding = encoder.get_input_embeddings().weight
            embedding.requires_grad_(True)
            saved = torch.load(Path(source['checkpoint']) / (token + '.bin'), map_location='cpu')
            if set(saved) != {token} or tuple(saved[token].squeeze().shape) != (1024,) or not bool(torch.isfinite(saved[token]).all()):
                raise ValueError('learned probe token must be one finite1024-vector')
            with torch.no_grad():
                embedding[token_id].copy_(saved[token].reshape(1024))
            del saved
            encoder.train()
            unet = UNet2DConditionModel.from_pretrained(model, subfolder='unet', local_files_only=True).to('cuda:0')
            unet.requires_grad_(False)
            install_subject_lora_kv(unet, torch.load(Path(source['checkpoint']) / source['weight'], map_location='cpu'),
                                    'token_local_kv', 64, 64)
            unet.train()
            vae = AutoencoderKL.from_pretrained(model, subfolder='vae', local_files_only=True).to('cuda:0')
            vae.requires_grad_(False)
            vae.eval()
            modules = {'unet': unet, 'text_encoder': encoder, 'vae': vae}
            if any(parameter.dtype != torch.float32 for module in modules.values() for parameter in module.parameters()):
                raise ValueError('all probe model parameters must remain FP32')
            groups = {f'{projection}_{direction}': [] for projection in ('K', 'V') for direction in ('up', 'down')}
            for name, parameter in unet.named_parameters():
                if parameter.requires_grad:
                    matches = [key for key in groups if f'.to_{key[0].lower()}_lora.{key[2:]}.' in name]
                    if len(matches) != 1:
                        raise ValueError('unexpected trainable UNet parameter')
                    groups[matches[0]].append(parameter)
            if any(len(values) != 16 for values in groups.values()):
                raise ValueError('require sixteen projections in each LoRA K/V up/down group')
            dataset = CustomDiffusionDataset(concepts, tokenizer, size=512, mask_size=64,
                center_crop=config.get('center_crop', False), hflip=config.get('hflip', False),
                aug=config.get('aug', False), with_prior_preservation=False)
            if len(dataset) != audit['datasets'][branch]['rows'] or len(dataset) < 3:
                raise ValueError('probe dataset count differs from the audited original training dataset')
            isolation = ConceptIsolation(audit['bank']['manifest_path'], tokenizer, token_id, initializer[0], prompts, maximum, seed=42)
            isolation.bind_text_encoder(encoder)
            reference = isolation.audit()
            scheduled = sorted(random.Random(9042).sample(range(1, maximum + 1), maximum // 2))
            if reference['isolation_steps'] != scheduled or not all(isolation.enabled(step) for step in scheduled[:3]):
                raise ValueError('isolation schedule differs from the preregistered exact50% policy')
            scheduler = DDPMScheduler.from_pretrained(model, subfolder='scheduler', local_files_only=True)
            branch_result = {'passed': False, 'source_pair': {key: source[key] for key in
                ('branch', 'method', 'step', 'training_steps', 'token', 'checkpoint', 'weight',
                 'derived_adapter_sha256', 'derived_token_sha256')},
                'reference': reference, 'samples': []}
            result['branches'][branch] = branch_result
            for sample_number, (index, step) in enumerate(zip((0, len(dataset) // 2, len(dataset) - 1), scheduled[:3])):
                torch.cuda.reset_peak_memory_stats(0)
                image_path, captions, _, _ = dataset.instance_images_path[index]
                prompt = captions[0]
                batch = collate_fn([dataset[index]], False, modifier_token_ids=[token_id])
                if not all(isinstance(value, torch.Tensor) for value in batch.values()):
                    raise ValueError('probe requires the original tensor-only dataset mode')
                batch = {key: value.to('cuda:0') for key, value in batch.items()}
                ids = tokenizer(prompt, padding='max_length', max_length=77, truncation=True, return_tensors='pt').input_ids.to('cuda:0')
                if not torch.equal(ids, batch['input_ids']) or int((ids == token_id).sum()) != 1:
                    raise ValueError('actual sample caption differs from its fixed dataset caption')
                with torch.no_grad():
                    latent = vae.encode(batch['pixel_values']).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latent)
                timestep = torch.tensor([500], device='cuda:0', dtype=torch.long)
                noisy = scheduler.add_noise(latent, noise, timestep)
                if scheduler.config.prediction_type == 'epsilon':
                    target = noise
                elif scheduler.config.prediction_type == 'v_prediction':
                    target = scheduler.get_velocity(latent, noise, timestep)
                else:
                    raise ValueError('unexpected original scheduler prediction type')
                prediction = unet(noisy, timestep, encoder_hidden_states=encoder(ids)[0],
                                  cross_attention_kwargs={'modifier_token_mask': ids == token_id}).sample
                foreground_loss = (((prediction.float() - target.float()).square() * batch['mask']).sum((1, 2, 3))
                                   / batch['mask'].sum((1, 2, 3))).mean()
                foreground_value = float(foreground_loss.detach())
                foreground = gradient_report(foreground_loss, groups, embedding, token_id, retain_graph=False)
                del prediction, foreground_loss, target, noisy, noise, latent
                clear_attention_caches(unet)
                gc.collect()
                torch.cuda.empty_cache()
                before = fingerprint(modules)
                with isolation.loss_context(unet, vae, encoder, scheduler, batch['pixel_values'], prompt, step,
                                            'cuda:0', torch.float32) as (loss, metrics):
                    auxiliary = gradient_report(loss, groups, embedding, token_id, retain_graph=False)
                    auxiliary_value = float(loss.detach())
                auxiliary_restored = restored(before, modules)
                del loss
                gc.collect()
                torch.cuda.empty_cache()
                sample = {'dataset_index': index, 'image_path': str(image_path), 'prompt': prompt, 'isolation_step': step,
                    'foreground_loss': foreground_value, 'foreground_gradients': foreground,
                    'isolation_loss': auxiliary_value, 'isolation_gradients': auxiliary, 'metrics': metrics,
                    'rng_and_modes_restored': auxiliary_restored,
                    'isolation_to_foreground_gradient_ratios': {key: auxiliary['adapter_group_norms'][key] / value if value else None
                        for key, value in foreground['adapter_group_norms'].items()},
                    'embedding_gradient_ratio': auxiliary['modifier_embedding_norm'] / foreground['modifier_embedding_norm']
                        if foreground['modifier_embedding_norm'] else None,
                    'checkpoint_backward': None}
                passed = (np.isfinite(foreground_value) and np.isfinite(auxiliary_value)
                    and valid_gradients(foreground) and valid_gradients(auxiliary) and all(auxiliary_restored.values())
                    and metrics['concept_isolation_right_pixels'] == 4096 and metrics['concept_isolation_applied'] is True
                    and metrics['concept_isolation_step'] == step and metrics['concept_isolation_seed'] == 200042 + step
                    and metrics['concept_isolation_concept_prompt'] == prompt
                    and metrics['teacher_requires_grad'] is False and metrics['teacher_hidden_requires_grad'] is False
                    and metrics['student_hidden_requires_grad'] is True)
                if sample_number == 0:
                    unet.enable_gradient_checkpointing()
                    encoder.gradient_checkpointing_enable()
                    unet.zero_grad(set_to_none=True)
                    encoder.zero_grad(set_to_none=True)
                    before = fingerprint(modules)
                    with isolation.loss_context(unet, vae, encoder, scheduler, batch['pixel_values'], prompt, step,
                                                'cuda:0', torch.float32) as (repeated, repeated_metrics):
                        repeated.backward()  # Fixed lambda1; no optimizer step.
                        repeated_value = float(repeated.detach())
                    checkpoint_restored = restored(before, modules)
                    gradients = backward_report(groups, embedding, token_id)
                    base_frozen = all(parameter.grad is None for module in (unet, encoder) for parameter in module.parameters()
                                      if not parameter.requires_grad)
                    vae_frozen = all(parameter.grad is None for parameter in vae.parameters())
                    checkpoint_passed = (valid_gradients(gradients) and all(checkpoint_restored.values())
                        and repeated_value == auxiliary_value and repeated_metrics == metrics and base_frozen and vae_frozen)
                    sample['checkpoint_backward'] = {'passed': checkpoint_passed, 'loss_equal': repeated_value == auxiliary_value,
                        'gradients': gradients, 'rng_and_modes_restored': checkpoint_restored,
                        'frozen_unet_encoder_gradients_absent': base_frozen, 'vae_gradients_absent': vae_frozen,
                        'unet_and_text_encoder_checkpointing': True}
                    passed = passed and checkpoint_passed
                    unet.zero_grad(set_to_none=True)
                    encoder.zero_grad(set_to_none=True)
                    unet.disable_gradient_checkpointing()
                    encoder.gradient_checkpointing_disable()
                    del repeated
                sample.update(passed=bool(passed), peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
                branch_result['samples'].append(sample)
                sct.write(output, result)
                if not passed:
                    raise ValueError(f'{branch} sample{index} failed the isolation gradient/RNG probe')
                del batch, ids
                clear_attention_caches(unet)
                gc.collect()
                torch.cuda.empty_cache()
            branch_result.update(passed=True, teacher_verification=isolation.verify_teacher(encoder))
            sct.write(output, result)
            del groups, embedding, modules, encoder, unet, vae, isolation, dataset
            gc.collect()
            torch.cuda.empty_cache()
        sct.st.shared.snapshots.check_locks(locks)
        if sct.st.git_info() != revision or sum(len(row['samples']) for row in result['branches'].values()) != 9:
            raise ValueError('probe checkout changed or nine samples were not completed')
        result.update(passed=True, status='succeeded')
    except BaseException as error:
        result.update(passed=False, status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        sct.write(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, type=Path)
    probe(parser.parse_args().experiment)
