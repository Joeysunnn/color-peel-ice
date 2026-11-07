"""Mailbox M150/M200/T300/T500: user-confirmed transfer prompts, seeds20-60."""

import argparse
import html
import json
import os
from pathlib import Path
import re

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd21_qwen_initializers_v1 import contract
from experiments.sd21_qwen_initializers_v1.run import ENVIRONMENT, require_idle
from experiments.sd21_qwen_midpoint_transfer_v1.run import _merge_locks, collect, run_group, validate_output
from experiments.sd21_wood_branches_v1 import snapshots
from experiments.subject_multicolor_multimaterial_b.training import _write
from experiments.paper_subject_token_v1.run import OBJECT, SAMPLING
from scripts.launch.colorpeel_run import git_info

STEPS = {'material': (150, 200), 'texture': (300, 500)}
SEEDS = tuple(range(20, 61))
OBJECTS = ('chair', 'mug', 'vase', 'sofa')
TRAINING_RUN = '20261107__mt_vlm__16c1e7e__42'
SOURCE_GIT = {'branch': 'codex/paper-mailbox-mt-v1',
              'commit': '16c1e7e12bc99c7d58489bef683e342cfbbf2024'}
CANONICAL_SHA = '644a77fe664bc2d9b0c94ada86259ce5a810a530580e754f1b23a40a82906be5'


def plan(branch, step):
    if branch not in STEPS or type(step) is not int or step not in STEPS[branch]:
        raise ValueError('select a requested Material or Texture checkpoint')
    suffix = 'with a <M*> surface' if branch == 'material' else 'with <T*> texture'
    return [{'id': f'{branch}{step}__{obj}__seed{seed}', 'case': obj, 'seed': seed,
             'prompt': f'a photo of a {obj} {suffix}', 'mode': branch + '_only',
             'branch_steps': {branch: step}, 'negative_prompt': '', 'guidance_scale': 3.5,
             'image_path': f'images/{obj}/seed{seed}.png'} for obj in OBJECTS for seed in SEEDS]


def validate_selection(item, branch, step, source, source_audit, manifest, model):
    plan(branch, step)
    expected = {'branch': branch, 'step': step, 'steps': step, 'training_steps': STEPS[branch][-1],
                'method': 'lora_r64', 'token': contract.TOKENS[branch], 'embedding_dim': 1024,
                'base_model': model, 'status': 'verified', 'paired_embedding_finite': True,
                'weight': 'pytorch_lora_kv_weights.bin'}
    if any(item.get(key) != value for key, value in expected.items()):
        raise ValueError('wrong branch, step, base model or paired checkpoint')
    directory = Path(item['checkpoint'])
    if (not directory.is_absolute() or directory.resolve() != (Path(source) / 'ckpts' / f'step{step}').resolve()
            or any(str(directory / name) not in item['hashes']
                   for name in (item['weight'], item['token'] + '.bin'))):
        raise ValueError('checkpoint must be the hash-locked export of this training run')
    path = Path(source_audit['initializer_manifest'])
    entry, provenance = manifest['branches'][branch], item['initializer_provenance']
    expected = {'initializer_manifest': str(path), 'initializer_manifest_sha256': sha256(path),
                'initializer': entry['initializer'], 'initializer_token_id': entry['initializer_token_id']}
    if (any(provenance.get(key) != value for key, value in expected.items())
            or not Path(provenance['initializer_audit']).is_absolute()
            or item['hashes'].get(provenance['initializer_audit']) != provenance['initializer_audit_sha256']):
        raise ValueError('initializer provenance differs from the actual training export')


def training_source(object_root, branch, training_run):
    if (branch not in STEPS or not isinstance(training_run, str)
            or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', training_run) is None):
        raise ValueError('training run must be one safe directory name')
    root = Path(object_root).resolve() / 'training'
    source = (root / training_run / 'object_003' / branch).resolve()
    if not source.is_relative_to(root):
        raise ValueError('training source escapes the object training directory')
    return source


def prepare(object_root, branch, step, gpu, *, training_run=TRAINING_RUN,
            source_git=SOURCE_GIT, source_image_sha=CANONICAL_SHA):
    import torch
    from transformers import CLIPTokenizer
    from experiments.sd21_wood_branches_v1.prepare_model import verify_model
    from src.train.initializer_token_utils import load_initializer_manifest

    rows, object_root = plan(branch, step), Path(object_root).resolve()
    identity = read_json(object_root / 'object.json')
    if (object_root.name != OBJECT or identity['object_key'] != OBJECT
            or identity['source_image_sha256'] != contract.SOURCE_SHA
            or Path(identity['inference']).resolve().parent != object_root / 'infer'):
        raise ValueError('require the existing Mailbox object and Subject inference parent')
    source = training_source(object_root, branch, training_run)
    revision = git_info()
    output = validate_output(object_root / 'infer' /
        f'20261107__{branch}{step}_transfer__{revision["commit"][:7]}__seeds20-60', source)
    paths = {name: source / name for name in ('status.json', 'prepared/prepared.json',
                                            'prepared/source_audit.json', 'ckpts/index.json')}
    status, prepared, prior, exports = [read_json(path) for path in paths.values()]
    if (status['status'] != 'trained' or status['completed_steps'] != STEPS[branch][-1]
            or status['checkpoint_steps'] != list(STEPS[branch])
            or prepared['git'] != source_git or prior['new_training_git'] != source_git
            or exports['git'] != source_git or exports['smoke']
            or prepared['source_audit_sha256'] != sha256(paths['prepared/source_audit.json'])
            or prior['object']['id'] != 'object_003'
            or prior['object']['source_image_sha256'] != source_image_sha
            or prior['branches'] != [branch] or prior['steps'] != {branch: STEPS[branch][-1]}
            or prior['checkpoint_steps'] != {branch: list(STEPS[branch])}):
        raise ValueError('source completion, producer Git, object or checkpoint schedule differs')
    model = verify_model(contract.MODEL_DIR)
    if prior['base_model'] != model['model']:
        raise ValueError('inference base differs from the training base')
    tokenizer = CLIPTokenizer.from_pretrained(str(contract.MODEL_DIR), subfolder='tokenizer', local_files_only=True)
    manifest_path = Path(prior['initializer_manifest'])
    manifest = load_initializer_manifest(manifest_path, tokenizer, contract.MODEL_DIR)
    if manifest != prior['initializers']:
        raise ValueError('initializer manifest differs from training')
    item = exports['steps'][str(step)]
    validate_selection(item, branch, step, source, prior, manifest, model['model'])
    receipt = Path(item['checkpoint']) / 'receipt.json'
    if read_json(receipt) != item:
        raise ValueError('step receipt differs from the paired export index')
    locks = dict(prior['source_locks'])
    for extra in (exports['source_locks'], item['hashes'], snapshots.model_locks(contract.MODEL_DIR, model),
                  {str(path): sha256(path) for path in (*paths.values(), receipt, manifest_path)}):
        _merge_locks(locks, extra)
    snapshots.check_locks(locks)
    weights = torch.load(Path(item['checkpoint']) / item['weight'], map_location='cpu')
    shapes, empty = snapshots.adapter_schema('lora_r64')
    snapshots.shared._check_state(torch, weights, shapes, empty, branch + str(step))
    token = item['token']
    vector = torch.load(Path(item['checkpoint']) / (token + '.bin'), map_location='cpu')
    if (any(value.dtype != torch.float32 or not bool(torch.isfinite(value.half()).all()) for value in weights.values())
            or not isinstance(vector, dict) or set(vector) != {token}
            or not isinstance(vector[token], torch.Tensor) or vector[token].shape != (1024,)
            or vector[token].dtype != torch.float32 or not bool(torch.isfinite(vector[token].half()).all())):
        raise ValueError('require paired FP32 weights/vector finite in FP16')
    if tokenizer.add_tokens([token]) != 1 or any(
            len(tokenizer.encode(row['prompt'], add_special_tokens=True)) > 77 for row in rows):
        raise ValueError('unexpected token inventory or truncated prompt')
    if git_info() != revision:
        raise ValueError('checkout changed during preparation')
    output.mkdir(parents=True)
    audit = {'object_key': OBJECT, 'source_training_git': source_git, 'source_experiment': str(source),
             'inference_git': revision, 'selected_exports': {branch: item}, 'source_locks': locks,
             'sampling': SAMPLING, 'physical_gpu': gpu, 'fresh_training': False}
    _write(output / 'source_audit.json', audit)
    (output / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
    return output, rows, audit


def gallery(output, rows, generated, state, group=None):
    branch, step = next(iter(rows[0]['branch_steps'].items()))
    folder, prefix = (output / 'groups' / group, '../../') if group else (output, '')
    folder.mkdir(parents=True, exist_ok=True)
    records = {row['id']: row for row in generated}
    body = [f'<h1>{OBJECT}: {branch}{step}</h1>',
            f'<p>{state["status"]}: {len(generated)}/{len(rows)}. Seeds20-60; CFG3.5; PNDM100; 512px.</p>',
            '<p>All seeds and safety-filtered outputs are retained for user review.</p>']
    for obj in OBJECTS:
        selected = [row for row in rows if row['case'] == obj]
        body.append('<h2>' + html.escape(selected[0]['prompt']) + '</h2><div class="grid">')
        for row in selected:
            record = records.get(row['id'])
            body.append(f'<figure><figcaption>seed{row["seed"]}</figcaption>')
            if record:
                link = prefix + row['image_path']
                body.append(f'<a href="{link}"><img loading="lazy" width="192" src="{link}"></a>'
                            f'<small>{record["status"]}</small>')
            else:
                body.append('<p>pending</p>')
            body.append('</figure>')
        body.append('</div>')
    (folder / 'index.html').write_text('<!doctype html><meta charset="utf-8"><style>'
        'body{font:16px sans-serif}.grid{display:flex;flex-wrap:wrap}figure{width:192px;margin:8px}'
        'small{display:block}</style>' + ''.join(body), encoding='utf8')


def run(object_root, branch, step, gpu, *, training_run=TRAINING_RUN,
        source_git=SOURCE_GIT, source_image_sha=CANONICAL_SHA):
    if type(gpu) is not int or gpu < 0:
        raise ValueError('require a nonnegative physical GPU')
    os.environ.update(ENVIRONMENT, CUDA_VISIBLE_DEVICES=str(gpu))
    require_idle([gpu])
    output, rows, audit = prepare(object_root, branch, step, gpu, training_run=training_run,
                                  source_git=source_git, source_image_sha=source_image_sha)
    state = {'status': 'running', 'expected_images': len(rows), 'generated_images': 0,
             'inference_git': audit['inference_git'], 'branch': branch, 'step': step, 'gpu': gpu,
             'fresh_training': False, 'quality_verified': False}
    generated, mode = [], branch + '_only'
    _write(output / 'status.json', state)
    gallery(output, rows, generated, state)
    print(json.dumps({'output': str(output), 'expected_images': len(rows)}), flush=True)
    try:
        require_idle([gpu])
        run_group(output, mode, rows, audit, gpu, render_gallery=gallery)
        generated = collect(output, rows)
        if (len(generated) != len(rows) or any(row['loaded_branches'] != [branch]
                or row['loaded_source_group'] != mode for row in generated)
                or read_json(output / 'groups' / mode / 'status.json')['status'] != 'succeeded'
                or git_info() != audit['inference_git']):
            raise ValueError('inference incomplete or loaded sources differ')
        snapshots.check_locks(audit['source_locks'])
        state['status'] = 'succeeded'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        # Preserve partial output evidence if generation or final verification failed.
        if not generated:
            try:
                generated = collect(output, rows)
            except Exception as error:
                state['collection_error'] = f'{type(error).__name__}: {error}'
        state.update(generated_images=len(generated),
                     safety_filtered_images=sum(row['nsfw_content_detected'] for row in generated))
        _write(output / 'status.json', state)
        gallery(output, rows, generated, state)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object-root', type=Path, required=True)
    parser.add_argument('--branch', choices=STEPS, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    run(args.object_root, args.branch, args.step, args.gpu)
