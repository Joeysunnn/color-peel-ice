"""Mailbox Subject1000: nine category-free prompts, seeds20-60, object-local assets."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd21_qwen_initializers_v1 import contract
from experiments.sd21_qwen_initializers_v1.inference import register_initial_input
from experiments.sd21_qwen_initializers_v1.run import ENVIRONMENT, require_idle
from experiments.sd21_qwen_cfg_compare_v1.run import validate_selection
from experiments.sd21_qwen_midpoint_transfer_v1.protocol import SOURCE, SOURCE_GIT
from experiments.sd21_qwen_midpoint_transfer_v1.run import _merge_locks, run_group, validate_output
from experiments.sd21_wood_branches_v1 import snapshots
from experiments.subject_multicolor_multimaterial_b.training import _write
from scripts.launch.colorpeel_run import git_info

ROOT = Path(__file__).resolve().parents[2]
STUDY = 'paper_subject_token_v1'
OBJECT = 'image_081__mask_130'
SEEDS = tuple(range(20, 61))
PROMPTS = {
    'plain': 'a photo of <S*>',
    'snow': 'a photo of <S*> in the snow',
    'street': 'a photo of <S*> in the street',
    'flowers': 'a photo of <S*> surrounded by flowers',
    'blue': 'a photo of <S*> in blue color',
    'white': 'a photo of <S*> in white color',
    'red': 'a photo of <S*> in red color',
    'metal': 'a photo of <S*> with a metal surface',
    'matte': 'a photo of <S*> with a matte surface',
}
SAMPLING = {'scheduler': 'PNDM', 'num_inference_steps': 100, 'guidance_scale': 3.5,
            'width': 512, 'height': 512, 'dtype': 'float16', 'negative_prompt': '',
            'safety_checker': True, 'generator': 'fresh_device_generator_per_image'}


def plan():
    return [{'id': f'{case}__seed{seed}', 'case': case, 'seed': seed, 'prompt': prompt,
             'mode': 'subject_only', 'branch_steps': {'subject': 1000},
             'negative_prompt': '', 'guidance_scale': 3.5,
             'image_path': f'images/{case}/seed{seed}.png'}
            for case, prompt in PROMPTS.items() for seed in SEEDS]


def shard(rows, gpus, gpu):
    return [row for row in rows if gpus[(row['seed'] - SEEDS[0]) % len(gpus)] == gpu]


def copy_checkpoint(item, destination):
    """Keep the producer receipt unchanged and record the byte-identical relocation."""
    original = Path(item['checkpoint'])
    snapshots.check_locks(item['hashes'])
    destination.mkdir(parents=True, exist_ok=False)
    copies = []
    for name in (item['weight'], item['token'] + '.bin', 'adaptation_config.json', 'receipt.json'):
        source, target = original / name, destination / name
        before = sha256(source)
        shutil.copyfile(source, target)
        if sha256(target) != before or sha256(source) != before:
            raise ValueError('checkpoint copy differs from its frozen source')
        copies.append({'source': str(source), 'copy': str(target), 'sha256': before})
    selected = {**item, 'original_checkpoint': str(original), 'checkpoint': str(destination),
                'path': str(destination), 'hashes': dict(item['hashes'])}
    _merge_locks(selected['hashes'], {row['copy']: row['sha256'] for row in copies})
    return selected, copies


def prepare(object_root, gpus):
    import torch
    from transformers import CLIPTokenizer
    from experiments.sd21_wood_branches_v1.prepare_model import verify_model
    from src.train.initializer_token_utils import load_initializer_manifest

    object_root = validate_output(object_root, SOURCE)
    if object_root.name != OBJECT:
        raise ValueError('this fixed study requires the image_081__mask_130 object folder')
    revision = git_info()
    paths = {name: SOURCE / name for name in ('source_audit.json', 'exports.json', 'status.json')}
    audit, exports, state = [read_json(paths[name]) for name in paths]
    if (audit['new_training_git'] != SOURCE_GIT or state.get('status') != 'succeeded'
            or state.get('new_training_git') != SOURCE_GIT):
        raise ValueError('source training must have succeeded at the pinned producer revision')
    model = verify_model(contract.MODEL_DIR)
    if audit['base_model'] != model['model']:
        raise ValueError('inference base differs from source training')
    tokenizer = CLIPTokenizer.from_pretrained(str(contract.MODEL_DIR), subfolder='tokenizer', local_files_only=True)
    manifest_path = Path(audit['initializer_manifest'])
    manifest = load_initializer_manifest(manifest_path, tokenizer, contract.MODEL_DIR)
    if manifest != audit['initializers'] or manifest['source']['image_sha256'] != contract.SOURCE_SHA:
        raise ValueError('initializer manifest or source image differs from the selected Mailbox')
    item = exports['subject']['1000']
    validate_selection(item, 'subject', manifest, audit, model['model'])
    if Path(item['checkpoint']).resolve() != (SOURCE / 'formal/subject/ckpts/step1000').resolve():
        raise ValueError('checkpoint differs from the exact user-selected export')
    locks = dict(audit['source_locks'])
    _merge_locks(locks, item['hashes'])
    _merge_locks(locks, {str(path): sha256(path) for path in (*paths.values(), manifest_path)})
    _merge_locks(locks, snapshots.model_locks(contract.MODEL_DIR, model))
    snapshots.check_locks(locks)
    shapes, empty = snapshots.adapter_schema('lora_r64')
    weights = torch.load(Path(item['checkpoint']) / item['weight'], map_location='cpu')
    snapshots.shared._check_state(torch, weights, shapes, empty, 'subject1000')
    if any(value.dtype != torch.float32 or not bool(torch.isfinite(value.half()).all())
           for value in weights.values()):
        raise ValueError('adapter must be FP32 and remain finite in FP16')
    vector = torch.load(Path(item['checkpoint']) / '<S*>.bin', map_location='cpu')
    if (not isinstance(vector, dict) or set(vector) != {'<S*>'}
            or not isinstance(vector['<S*>'], torch.Tensor) or vector['<S*>'].shape != (1024,)
            or vector['<S*>'].dtype != torch.float32 or not bool(torch.isfinite(vector['<S*>'].half()).all())):
        raise ValueError('require the paired finite FP32 Subject 1024-vector')
    del weights, vector
    if tokenizer.add_tokens(['<S*>']) != 1 or any(
            len(tokenizer.encode(row['prompt'], add_special_tokens=True)) > 77 for row in plan()):
        raise ValueError('unexpected token inventory or truncated prompt')
    if git_info() != revision:
        raise ValueError('checkout changed during preparation')
    selected, copies = copy_checkpoint(item, object_root / 'ckpts/subject/cb169c6/step1000')
    provenance = object_root / 'provenance/subject_cb169c6'
    provenance.mkdir(parents=True)
    for name, source in {**paths, 'initializers.json': manifest_path}.items():
        target = provenance / name
        shutil.copyfile(source, target)
        if sha256(target) != locks[str(source)]:
            raise ValueError('provenance copy differs')
        locks[str(target)] = sha256(target)
    output = object_root / 'infer' / f'20261107__subject1000_no_category__{revision["commit"][:7]}__seeds20-60'
    output.mkdir(parents=True)
    _merge_locks(locks, selected['hashes'])
    inference_audit = {'object_key': OBJECT, 'source_image_sha256': contract.SOURCE_SHA,
        'source_training_git': SOURCE_GIT, 'source_experiment': str(SOURCE), 'inference_git': revision,
        'selected_exports': {'subject': selected}, 'checkpoint_copies': copies,
        'source_locks': locks, 'sampling': SAMPLING, 'gpus': gpus, 'fresh_training': False}
    _write(output / 'source_audit.json', inference_audit)
    (output / 'generation_manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in plan()), encoding='utf8')
    _write(object_root / 'object.json', {'object_key': OBJECT, 'source_image_sha256': contract.SOURCE_SHA,
        'checkpoint': selected['checkpoint'], 'inference': str(output), 'source_training_git': SOURCE_GIT,
        'inference_git': revision, 'new_training': False, 'quality_verified': False})
    (object_root / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>' + OBJECT + '</h1>'
        + '<a href="' + output.relative_to(object_root).as_posix() + '/index.html">Subject1000 result gallery</a>', encoding='utf8')
    return output, inference_audit


def gallery(output, rows, generated, state, group=None):
    folder = output / 'groups' / group if group else output
    folder.mkdir(parents=True, exist_ok=True)
    prefix = '../../' if group else ''
    records = {r['id']: r for r in generated}
    body = [f'<h1>{OBJECT}: Subject1000, no category word</h1>',
            f'<p>{state["status"]}: {len(generated)}/{len(rows)}. Seeds20-60; CFG3.5; PNDM100; 512px.</p>',
            '<p>All seeds are retained, including safety-filtered outputs. Human review pending.</p>']
    for case, prompt in PROMPTS.items():
        body.append('<h2>' + html.escape(prompt) + '</h2><div class="grid">')
        for row in (r for r in rows if r['case'] == case):
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
        'body{font:16px sans-serif}.grid{display:flex;flex-wrap:wrap}figure{margin:8px;width:192px}'
        'small{display:block}</style>' + ''.join(body), encoding='utf8')


def collect(output, rows, gpus):
    planned, records, initial = {r['id']: r for r in rows}, {}, {}
    for gpu in gpus:
        allowed = {r['id'] for r in shard(rows, gpus, gpu)}
        ledger = output / f'groups/gpu{gpu}/generation_status.jsonl'
        if not ledger.exists():
            continue
        for line in ledger.read_text(encoding='utf8').splitlines():
            record = json.loads(line)
            if (record['id'] not in allowed or record['id'] in records
                    or any(record.get(k) != v for k, v in planned[record['id']].items())
                    or record['loaded_branches'] != ['subject'] or record['loaded_source_group'] != f'gpu{gpu}'
                    or sha256(output / record['image_path']) != record['image_sha256']
                    or record['status'] != ('safety_filtered' if record['nsfw_content_detected'] else 'ok')):
                raise ValueError('generated record differs from its plan, GPU assignment or saved bytes')
            register_initial_input(initial, record['seed'], record['initial_denoiser_input'])
            records[record['id']] = record
    generated = [records[r['id']] for r in rows if r['id'] in records]
    (output / 'generation_status.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in generated), encoding='utf8')
    _write(output / 'initial_input_audit.json', {'same_seed_equal_across_prompts': True, 'by_seed': initial})
    return generated


def worker(output, gpu):
    audit = read_json(output / 'source_audit.json')
    if gpu not in audit['gpus'] or os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise ValueError('worker GPU differs from assignment')
    require_idle([gpu])
    run_group(output, f'gpu{gpu}', shard(plan(), audit['gpus'], gpu), audit, gpu, render_gallery=gallery)


def run(object_root, gpus):
    if not gpus or len(set(gpus)) != len(gpus) or len(gpus) > len(SEEDS):
        raise ValueError('require unique GPU assignments and at least one seed per worker')
    require_idle(gpus)
    output, audit = prepare(object_root, gpus)
    rows, generated = plan(), []
    state = {'status': 'running', 'expected_images': len(rows), 'generated_images': 0,
             'inference_git': audit['inference_git'], 'gpus': gpus, 'fresh_training': False, 'quality_verified': False}
    _write(output / 'status.json', state)
    gallery(output, rows, [], state)
    print(json.dumps({'inference': str(output), 'expected_images': len(rows)}), flush=True)

    def launch(gpu):
        command = [sys.executable, '-u', '-B', '-m', 'experiments.' + STUDY + '.run',
                   '--output', str(output), '--worker-gpu', str(gpu)]
        _write(output / f'gpu{gpu}_command.json', {'command': command, 'physical_gpu': gpu})
        with (output / f'gpu{gpu}.log').open('x', encoding='utf8') as log:
            subprocess.run(command, cwd=ROOT, env={**os.environ, **ENVIRONMENT, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        require_idle(gpus)
        failures = {}
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = {pool.submit(launch, gpu): gpu for gpu in gpus}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    failures[str(futures[future])] = str(error)
        generated = collect(output, rows, gpus)
        state.update(generated_images=len(generated), safety_filtered_images=sum(r['nsfw_content_detected'] for r in generated))
        snapshots.check_locks(audit['source_locks'])
        if failures or len(generated) != len(rows) or git_info() != audit['inference_git'] or any(
                read_json(output / f'groups/gpu{gpu}/status.json')['status'] != 'succeeded' for gpu in gpus):
            raise RuntimeError(f'inference incomplete; outputs preserved: {failures}')
        state['status'] = 'succeeded'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        _write(output / 'status.json', state)
        gallery(output, rows, generated, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object-root', type=Path)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-gpu', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ.update(ENVIRONMENT)
    if args.worker_gpu is not None:
        if args.output is None:
            parser.error('worker requires --output')
        worker(args.output.resolve(), args.worker_gpu)
    else:
        if args.object_root is None:
            parser.error('require --object-root')
        run(args.object_root, args.gpus)
