"""S1000/C150/M150/T300 composition, exact category-free prompts, seeds20-60."""

import argparse
import html
import json
import os
from pathlib import Path

from experiments.paper_subject_token_v1.run import OBJECT, SAMPLING, copy_checkpoint
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd21_qwen_initializers_v1 import contract
from experiments.sd21_qwen_initializers_v1.run import ENVIRONMENT, require_idle
from experiments.sd21_qwen_midpoint_transfer_v1.protocol import SOURCE, SOURCE_GIT
from experiments.sd21_qwen_midpoint_transfer_v1.run import _merge_locks, collect, run_group, validate_output
from experiments.sd21_wood_branches_v1 import snapshots
from experiments.subject_multicolor_multimaterial_b.training import _write
from scripts.launch.colorpeel_run import git_info

STEPS = {'subject': 1000, 'color': 150, 'material': 150, 'texture': 300}
SEEDS = tuple(range(20, 61))
BODY = 'a photo of <S*> in <C*> color with a <M*> surface with <T*> texture'
PROMPTS = {case: BODY + suffix for case, suffix in (
    ('plain', ''), ('street', ' in the street'), ('snow', ' in the snow'),
    ('flowers', ' surrounded by flowers'))}
INFER_TAGS = {
    'subject': '20261107__subject1000_no_category__a9ff5a5__seeds20-60',
    'material': '20261107__material150_transfer__aa87dd4__seeds20-60',
    'texture': '20261107__texture300_transfer__ad3a393__seeds20-60',
}
PRODUCERS = {'subject': SOURCE_GIT, 'color': SOURCE_GIT,
    'material': {'branch': 'codex/paper-mailbox-mt-v1',
                 'commit': '16c1e7e12bc99c7d58489bef683e342cfbbf2024'},
    'texture': {'branch': 'codex/paper-mailbox-legacy-texture-v1',
                'commit': 'ad3a393a200d3f8c164d7b99c76311ac3a44bd15'}}


def plan():
    return [{'id': f'scmt__{case}__seed{seed}', 'case': case, 'seed': seed,
             'prompt': prompt, 'mode': 'scmt', 'branch_steps': dict(STEPS),
             'negative_prompt': '', 'guidance_scale': 3.5,
             'image_path': f'images/{case}/seed{seed}.png'}
            for case, prompt in PROMPTS.items() for seed in SEEDS]


def checkpoint_paths(root):
    return {'subject': root / 'ckpts/subject/cb169c6/step1000',
            'color': SOURCE / 'formal/color/ckpts/step150',
            'material': root / 'training/20261107__mt_vlm__16c1e7e__42/object_003/material/ckpts/step150',
            'texture': root / 'training/20261107__texture_legacy_t16__ad3a393__42/object_003/texture/ckpts/step300'}


def validate_export(item, branch, expected_path, model):
    expected = {'branch': branch, 'step': STEPS[branch], 'steps': STEPS[branch],
                'method': 'lora_r64', 'token': contract.TOKENS[branch], 'embedding_dim': 1024,
                'base_model': model, 'status': 'verified', 'paired_embedding_finite': True,
                'weight': 'pytorch_lora_kv_weights.bin'}
    if any(item.get(key) != value for key, value in expected.items()):
        raise ValueError('wrong branch, checkpoint step, model or paired export')
    directory = Path(item['checkpoint'])
    if not directory.is_absolute() or directory.resolve() != expected_path.resolve():
        raise ValueError('checkpoint is not the user-selected producer export')
    receipt = read_json(directory / 'receipt.json')
    # Subject has an audited relocation; its producer receipt remains unchanged.
    for key, value in receipt.items():
        if key not in ('checkpoint', 'path', 'hashes') and item.get(key) != value:
            raise ValueError('paired checkpoint differs from producer receipt')
    for name in (item['weight'], item['token'] + '.bin'):
        if str(directory / name) not in item['hashes']:
            raise ValueError('paired adapter and embedding must both be hash-locked')
    if any(item['hashes'].get(path) != digest for path, digest in receipt['hashes'].items()):
        raise ValueError('producer hashes were removed or changed')


def prepare(root, gpu):
    import torch
    from transformers import CLIPTokenizer
    from experiments.sd21_wood_branches_v1.prepare_model import verify_model
    from src.train.initializer_token_utils import load_initializer_manifest

    root = Path(root).resolve()
    identity = read_json(root / 'object.json')
    if (root.name != OBJECT or identity['object_key'] != OBJECT
            or identity['source_image_sha256'] != contract.SOURCE_SHA):
        raise ValueError('require the existing paper Mailbox object')
    revision, rows = git_info(), plan()
    output = validate_output(root / 'infer' /
        f'20261107__composition_S1000_C150_M150_T300__{revision["commit"][:7]}__seeds20-60', SOURCE)
    model = verify_model(contract.MODEL_DIR)
    tokenizer = CLIPTokenizer.from_pretrained(str(contract.MODEL_DIR), subfolder='tokenizer', local_files_only=True)
    locks, selected = snapshots.model_locks(contract.MODEL_DIR, model), {}
    origins = {}
    for branch, tag in INFER_TAGS.items():
        path = root / 'infer' / tag / 'source_audit.json'
        prior = read_json(path)
        if (prior['object_key'] != OBJECT or prior['source_training_git'] != PRODUCERS[branch]
                or set(prior['selected_exports']) != {branch} or prior['sampling'] != SAMPLING):
            raise ValueError('single-branch audit has the wrong object, producer or sampling')
        selected[branch] = prior['selected_exports'][branch]
        origins[branch] = str(path)
        _merge_locks(locks, prior['source_locks'])
        _merge_locks(locks, {str(path): sha256(path)})
    paths = [SOURCE / name for name in ('source_audit.json', 'exports.json', 'status.json')]
    prior, exports, status = [read_json(path) for path in paths]
    if (prior['new_training_git'] != SOURCE_GIT or status['status'] != 'succeeded'
            or status['new_training_git'] != SOURCE_GIT or prior['base_model'] != model['model']):
        raise ValueError('Color source producer or base model differs')
    manifest_path = Path(prior['initializer_manifest'])
    manifest = load_initializer_manifest(manifest_path, tokenizer, contract.MODEL_DIR)
    if manifest != prior['initializers'] or manifest['source']['image_sha256'] != contract.SOURCE_SHA:
        raise ValueError('Color initializer manifest differs from frozen Mailbox source')
    selected['color'] = exports['color']['150']
    provenance = selected['color']['initializer_provenance']
    expected = {'initializer_manifest': str(manifest_path),
                'initializer_manifest_sha256': sha256(manifest_path),
                'initializer': manifest['branches']['color']['initializer'],
                'initializer_token_id': manifest['branches']['color']['initializer_token_id']}
    if (any(provenance.get(key) != value for key, value in expected.items())
            or prior['source_locks'].get(provenance['initializer_audit']) != provenance['initializer_audit_sha256']):
        raise ValueError('Color initializer provenance differs')
    _merge_locks(locks, prior['source_locks'])
    _merge_locks(locks, {str(path): sha256(path) for path in (*paths, manifest_path)})
    origins['color'] = str(paths[1])
    selected = {branch: selected[branch] for branch in STEPS}
    shapes, empty = snapshots.adapter_schema('lora_r64')
    for branch, item in selected.items():
        validate_export(item, branch, checkpoint_paths(root)[branch], model['model'])
        _merge_locks(locks, item['hashes'])
        receipt = Path(item['checkpoint']) / 'receipt.json'
        _merge_locks(locks, {str(receipt): sha256(receipt)})
    snapshots.check_locks(locks)
    for branch, item in selected.items():
        weights = torch.load(Path(item['checkpoint']) / item['weight'], map_location='cpu')
        snapshots.shared._check_state(torch, weights, shapes, empty, branch)
        token = item['token']
        vector = torch.load(Path(item['checkpoint']) / (token + '.bin'), map_location='cpu')
        if (any(value.dtype != torch.float32 or not bool(torch.isfinite(value.half()).all()) for value in weights.values())
                or not isinstance(vector, dict) or set(vector) != {token}
                or not isinstance(vector[token], torch.Tensor) or vector[token].shape != (1024,)
                or vector[token].dtype != torch.float32 or not bool(torch.isfinite(vector[token].half()).all())):
            raise ValueError('paired adapter/vector must be FP32 and finite in FP16')
        del weights, vector
    if tokenizer.add_tokens(list(contract.TOKENS.values())) != 4 or any(
            len(tokenizer.encode(row['prompt'], add_special_tokens=True)) > 77 for row in rows):
        raise ValueError('unexpected token inventory or truncated prompt')
    if git_info() != revision:
        raise ValueError('checkout changed during preparation')
    color_copy = root / 'ckpts/color/cb169c6/step150'
    if not color_copy.resolve().is_relative_to(root) or color_copy.exists():
        raise ValueError('require a fresh object-local Color copy destination')
    selected['color'], copies = copy_checkpoint(selected['color'], color_copy)
    _merge_locks(locks, selected['color']['hashes'])
    output.mkdir(parents=True)
    audit = {'object_key': OBJECT, 'inference_git': revision, 'source_training_gits': PRODUCERS,
             'source_audits': origins, 'selected_exports': selected, 'source_locks': locks,
             'color_checkpoint_copies': copies, 'sampling': SAMPLING, 'physical_gpu': gpu,
             'fresh_training': False, 'branch_steps': STEPS, 'base_model': model['model']}
    _write(output / 'source_audit.json', audit)
    (output / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
    return output, rows, audit


def gallery(output, rows, generated, state, group=None):
    folder, prefix = (output / 'groups' / group, '../../') if group else (output, '')
    folder.mkdir(parents=True, exist_ok=True)
    records = {row['id']: row for row in generated}
    body = [f'<h1>{OBJECT}: S1000/C150/M150/T300</h1>',
            f'<p>{state["status"]}: {len(generated)}/{len(rows)}. Seeds20-60; PNDM100; CFG3.5; 512px.</p>']
    for case, prompt in PROMPTS.items():
        body.append('<h2>' + html.escape(prompt) + '</h2><div class="grid">')
        for row in (row for row in rows if row['case'] == case):
            body.append(f'<figure><figcaption>seed{row["seed"]}</figcaption>')
            if row['id'] in records:
                link = prefix + row['image_path']
                body.append(f'<a href="{link}"><img loading="lazy" width="192" src="{link}"></a>'
                            f'<small>{records[row["id"]]["status"]}</small>')
            else:
                body.append('<p>pending</p>')
            body.append('</figure>')
        body.append('</div>')
    (folder / 'index.html').write_text('<!doctype html><meta charset="utf-8"><style>'
        'body{font:16px sans-serif}.grid{display:flex;flex-wrap:wrap}figure{width:192px;margin:8px}'
        'small{display:block}</style>' + ''.join(body), encoding='utf8')


def run(root, gpu):
    if gpu < 0:
        raise ValueError('require a nonnegative physical GPU')
    os.environ.update(ENVIRONMENT, CUDA_VISIBLE_DEVICES=str(gpu))
    require_idle([gpu])
    output, rows, audit = prepare(root, gpu)
    state = {'status': 'running', 'expected_images': len(rows), 'generated_images': 0,
             'inference_git': audit['inference_git'], 'branch_steps': STEPS, 'physical_gpu': gpu,
             'fresh_training': False, 'quality_verified': False}
    generated = []
    _write(output / 'status.json', state)
    gallery(output, rows, generated, state)
    print(json.dumps({'output': str(output), 'expected_images': len(rows)}), flush=True)
    try:
        require_idle([gpu])
        run_group(output, 'scmt', rows, audit, gpu, render_gallery=gallery)
        generated = collect(output, rows)
        if (len(generated) != len(rows) or any(row['loaded_branches'] != list(STEPS)
                or row['loaded_source_group'] != 'scmt' for row in generated)
                or read_json(output / 'groups/scmt/status.json')['status'] != 'succeeded'
                or git_info() != audit['inference_git']):
            raise ValueError('inference incomplete or loaded sources differ')
        snapshots.check_locks(audit['source_locks'])
        state['status'] = 'succeeded'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
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
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    run(args.object_root, args.gpu)
