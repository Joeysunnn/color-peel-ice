"""Compose reviewed Mailbox center11 Subject1500 with existing wood Texture600."""

import argparse
import json
import os
from pathlib import Path

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd15_token_local_kv_v1.evaluate import attention_masks
from experiments.sd21_mailbox_center11_v1 import contract as subject_contract
from experiments.sd21_mailbox_center11_v1.run import require_idle_gpu0
from experiments.sd21_wood_branches_v1 import composition as shared
from experiments.sd21_wood_subject27_v1 import composition_source as source
from experiments.wood_spoon_composition_v1 import inference as review_helpers
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
SUBJECT_PARENT = Path('/home/r12user5/Documents/Jiawei/colorpeel-runs/sd21_mailbox_center11_v1/'
                      '20261102-172717__center11_both1500__87d6e18__42')
SUBJECT_ROOT = SUBJECT_PARENT / 'experiment'
SUBJECT_GIT = {'commit': '87d6e18ad340d2c930010c8e79db4e163a08a05f',
               'branch': 'codex/sd21-mailbox-center11-1500'}
METHODS, SEEDS = ('fullrank', 'lora_r64'), tuple(range(30, 41))
PROMPT = 'a photo of <S*> mailbox with <T*> texture'
write = review_helpers.write


def prompt_rows(method):
    if method not in METHODS:
        raise ValueError('require matching fullrank or rank64 methods')
    return [{'id': f'{method}__mailbox_S1500_wood_T600__seed{seed}', 'method': method,
        'branch': 'texture', 'seed': seed, 'prompt': PROMPT, 'subject_step': 1500,
        'attribute_step': 600, 'subject_images': 11, 'texture_source': 'wood_spoon_texture',
        'image_path': f'images/seed{seed}.png'} for seed in SEEDS]


def prepare():
    revision = git_info()
    completion = read_json(SUBJECT_PARENT / 'monitor/status.json')
    if completion['status'] != 'completed' or completion['attempts']:
        raise ValueError('require completed original center11 queue; review recovery locations separately')
    os.environ[subject_contract.DATASET_ENV] = str(SUBJECT_ROOT / 'dataset')
    model = shared.verify_model()
    locks = shared.snapshots.model_locks(subject_contract.MODEL_DIR, model)
    subject_data = subject_contract.verify_dataset('subject')
    texture_data = shared.contract.verify_dataset('texture')
    if subject_data['rows'] != 11 or texture_data['rows'] != 16:
        raise ValueError('require approved center11 Subject and original wood Texture16')
    sources = {'subject': {}, 'texture': {}}
    for data in (subject_data, texture_data):
        source._merge(locks, data['source_locks'])
    for method in METHODS:
        for branch, root, step, training_git, recipe in (
            ('subject', SUBJECT_ROOT, 1500, SUBJECT_GIT, subject_contract),
            ('texture', source.TEXTURE_ROOT, 600, source.TEXTURE_GIT, shared.contract)):
            item, selected = source._pair(root, branch, method, step, training_git,
                                           model['model'], recipe_contract=recipe)
            sources[branch][method] = item
            source._merge(locks, selected)
    if git_info() != revision:
        raise ValueError('inference checkout changed during preparation')
    return {'inference_git': revision, 'training_gits': {'subject': SUBJECT_GIT, 'texture': source.TEXTURE_GIT},
        'sources': sources, 'references': {'subject': subject_data['references'], 'texture': texture_data['references']},
        'source_locks': locks, 'prompt': PROMPT, 'subject_images': 11, 'subject_step': 1500,
        'texture_step': 600, 'texture_source': 'wood_spoon_texture'}


def generate(audit, output, method, execute):
    rows = prompt_rows(method)
    output.mkdir(parents=True)
    refs, copies = review_helpers.references(audit, 'texture', output, subject_images=11)
    state = {'status': 'running' if execute else 'dry_run', 'method': method, 'branch': 'texture',
        'literal_subject': False, 'subject_step': 1500, 'attribute_step': 600, 'subject_images': 11,
        'texture_source': 'wood_spoon_texture', 'prompt': PROMPT, 'expected_images': len(rows),
        'generated_images': 0, 'sampling': {**shared.inference.SAMPLING, 'seeds': list(SEEDS)},
        'inference_git': audit['inference_git'], 'training_gits': audit['training_gits'],
        'internal_adapter_slots': {'subject': '<S*>', 'material': '<T*>'}, 'retraining': False}
    generated = []
    (output / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
    write(output / 'status.json', state)
    try:
        if execute:
            import torch
            pipe = shared.load_pipeline(audit['sources']['subject'][method], audit['sources']['texture'][method], 'cuda:0')
            with (output / 'generation_status.jsonl').open('x', encoding='utf8') as ledger, torch.no_grad():
                for row in rows:
                    result = pipe(PROMPT, num_inference_steps=100, guidance_scale=3.5, width=512, height=512,
                        generator=torch.Generator(device='cuda:0').manual_seed(row['seed']),
                        cross_attention_kwargs=attention_masks(pipe, PROMPT, ['<S*>', '<T*>'], 3.5))
                    if (result.nsfw_content_detected is None or len(result.nsfw_content_detected) != 1
                            or len(result.images) != 1 or result.images[0].mode != 'RGB' or result.images[0].size != (512, 512)):
                        raise ValueError('require one512 RGB image with a safety result')
                    path = output / row['image_path']
                    path.parent.mkdir(exist_ok=True)
                    with path.open('xb') as stream:
                        result.images[0].save(stream, format='PNG')
                    generated.append({**row, 'status': 'safety_filtered' if result.nsfw_content_detected[0] else 'ok',
                        'image_sha256': sha256(path), 'subject_adapter_loaded': True, 'texture_adapter_loaded': True})
                    ledger.write(json.dumps(generated[-1]) + '\n')
                    ledger.flush()
                    state['generated_images'] = len(generated)
                    write(output / 'status.json', state)
            del pipe
            torch.cuda.empty_cache()
        shared.snapshots.check_locks(copies)
        state.update(status='succeeded' if execute else 'dry_run_verified', generated_images=len(generated))
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}', generated_images=len(generated))
        raise
    finally:
        state['reference_copy_hashes'] = copies
        write(output / 'status.json', state)
        write(output / 'provenance.json', state)
        review_helpers.review(output, refs, generated or rows, state)
    return state


def run(output, execute=False):
    output = resolve_run_dir(Path(output))
    if (output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(SUBJECT_PARENT)
            or output.is_relative_to(source.TEXTURE_ROOT)):
        raise ValueError('require a fresh external output separate from the source experiments')
    # GPU1/2/3 remain reserved for the running SDXL queue.
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if execute:
        require_idle_gpu0()
    audit = prepare()
    if execute:
        require_idle_gpu0()
    output.mkdir(parents=True)
    write(output / 'source_audit.json', audit)
    state = {'status': 'running' if execute else 'dry_run', 'git': audit['inference_git'],
        'subject_step': 1500, 'texture_step': 600, 'subject': 'mailbox', 'texture_source': 'wood_spoon_texture',
        'expected_images': 22, 'generated_images': 0, 'gpu': 0, 'methods': {}, 'retraining': False}
    write(output / 'status.json', state)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>SD2.1 Mailbox S1500 + wood T600</h1>'
        '<p>11 Subject images +16 wood Texture images; seed30–40; PNDM100/CFG3.5/512px.</p>'
        + ''.join(f'<p><a href="methods/{m}/index.html">{m}: results and actual training references</a></p>' for m in METHODS)
        + '<table><tr><th>seed</th><th>fullrank</th><th>LoRA rank64</th></tr>'
        + ''.join(f'<tr><th>{seed}</th>' + ''.join(f'<td><a href="methods/{m}/images/seed{seed}.png">'
            f'<img loading="lazy" width="384" src="methods/{m}/images/seed{seed}.png"></a></td>' for m in METHODS)
            + '</tr>' for seed in SEEDS) + '</table>', encoding='utf8')
    try:
        for method in METHODS:
            state['methods'][method] = generate(audit, output / 'methods' / method, method, execute)
            state['generated_images'] += state['methods'][method]['generated_images']
            write(output / 'status.json', state)
        shared.snapshots.check_locks(audit['source_locks'])
        if git_info() != audit['inference_git']:
            raise ValueError('inference checkout changed')
        state['status'] = 'succeeded' if execute else 'dry_run_verified'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(output / 'status.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    run(args.output, args.execute)
