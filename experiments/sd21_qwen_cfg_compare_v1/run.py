"""CFG-only comparison: Subject1000 and Material150 independently, seeds30-40."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd21_qwen_initializers_v1 import contract
from experiments.sd21_qwen_initializers_v1.run import ENVIRONMENT, require_idle
from experiments.sd21_qwen_midpoint_transfer_v1 import protocol as prior
from experiments.sd21_qwen_midpoint_transfer_v1.run import _merge_locks, collect, run_group
from experiments.sd21_wood_branches_v1 import snapshots
from experiments.subject_multicolor_multimaterial_b.training import _write
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
STUDY = 'sd21_qwen_cfg_compare_v1'
STEPS = {'subject': 1000, 'material': 150}
CFGS = (3.5, 5.0, 7.5)
GPUS = {'subject': 0, 'material': 1}
SOURCES = {'subject': prior.SOURCE, 'material': Path('/home/r12user5/Documents/Jiawei/colorpeel-runs/'
    'sd21_qwen_material150_200_v1/20261103-201500__mailbox__1a4ab03__42/experiment')}
SOURCE_GITS = {'subject': prior.SOURCE_GIT, 'material': {
    'branch': 'codex/sd21-qwen-material150-200-v1', 'commit': '1a4ab037e639f17051bc39495de41b36aa9006a4'}}


def plan(manifest):
    rows = []
    for old in prior.plan(manifest):
        branch = old['mode'].removesuffix('_only')
        if branch not in STEPS:
            continue
        for cfg in CFGS:
            tag = f'{cfg:g}'.replace('.', 'p')
            rows.append({**old, 'phase': f'step{STEPS[branch]}', 'branch_steps': {branch: STEPS[branch]},
                'guidance_scale': cfg, 'id': f'{branch}{STEPS[branch]}__{old["case"]}__seed{old["seed"]}__cfg{tag}',
                'image_path': f'images/{old["mode"]}/cfg{tag}/{old["case"]}/seed{old["seed"]}.png'})
    return rows


def validate_selection(item, branch, manifest, source_audit, model):
    expected = {'branch': branch, 'step': STEPS[branch], 'steps': STEPS[branch],
        'method': 'lora_r64', 'token': contract.TOKENS[branch], 'embedding_dim': 1024,
        'base_model': model, 'status': 'verified', 'paired_embedding_finite': True,
        'weight': 'pytorch_lora_kv_weights.bin'}
    if any(item.get(key) != value for key, value in expected.items()):
        raise ValueError('wrong branch, step, base model or paired checkpoint')
    directory = Path(item['checkpoint'])
    if (not directory.is_absolute() or not directory.resolve().is_relative_to(SOURCES[branch].resolve())
            or any(str(directory / name) not in item['hashes']
                   for name in (item['weight'], item['token'] + '.bin'))):
        raise ValueError('paired checkpoint must belong to its pinned source and be hash-locked')
    provenance = item['initializer_provenance']
    path = Path(source_audit['initializer_manifest'])
    entry = manifest['branches'][branch]
    expected = {'initializer_manifest': str(path), 'initializer_manifest_sha256': sha256(path),
                'initializer': entry['initializer'], 'initializer_token_id': entry['initializer_token_id']}
    if (any(provenance.get(key) != value for key, value in expected.items())
            or source_audit['source_locks'].get(provenance['initializer_audit']) != provenance['initializer_audit_sha256']):
        raise ValueError('initializer provenance differs from source training')


def gallery(output, rows, generated, state, mode):
    branch = mode.removesuffix('_only')
    folder = output / 'groups' / mode
    folder.mkdir(parents=True, exist_ok=True)
    records = {(r['case'], r['seed'], r['guidance_scale']): r for r in generated}
    body = [f'<h1>{branch}{STEPS[branch]}: CFG comparison</h1>',
        f'<p>{state["status"]}: {len(generated)}/{len(rows)}. Seeds30-40; PNDM100;512px;FP16; '
        'empty negative; safety checker enabled. User manual review; filtered images retained.</p>',
        '<p><a href="../../index.html">All branches</a></p><table><tr><th>case / seed</th>'
        + ''.join(f'<th>CFG{cfg:g}</th>' for cfg in CFGS) + '</tr>']
    for case, seed in dict.fromkeys((r['case'], r['seed']) for r in rows):
        body.append(f'<tr><th>{case} / {seed}</th>')
        for cfg in CFGS:
            row = records.get((case, seed, cfg))
            body.append('<td>pending</td>' if row is None else
                f'<td><a href="../../{row["image_path"]}"><img width="256" src="../../{row["image_path"]}"></a>'
                f'<p>{row["status"]}</p><small>{html.escape(row["prompt"])}</small></td>')
        body.append('</tr>')
    (folder / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<style>body{font:16px sans-serif}td{width:256px;vertical-align:top}</style>'
        + ''.join(body) + '</table>', encoding='utf8')


def prepare(output):
    from transformers import CLIPTokenizer
    from experiments.sd21_wood_branches_v1.prepare_model import verify_model
    from src.train.initializer_token_utils import load_initializer_manifest

    output = resolve_run_dir(Path(output))
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output)
            for p in (ROOT, contract.MODEL_DIR.resolve(), *(p.resolve() for p in SOURCES.values()))):
        raise ValueError('require fresh output outside protected source/code/model directories')
    revision = git_info()
    model = verify_model(contract.MODEL_DIR)
    tokenizer = CLIPTokenizer.from_pretrained(str(contract.MODEL_DIR), subfolder='tokenizer', local_files_only=True)
    locks, selected, manifest = snapshots.model_locks(contract.MODEL_DIR, model), {}, None
    for branch, source in SOURCES.items():
        audit, exports, status = [read_json(source / name) for name in ('source_audit.json', 'exports.json', 'status.json')]
        if (audit['new_training_git'] != SOURCE_GITS[branch] or status['status'] != 'succeeded'
                or audit['base_model'] != model['model']):
            raise ValueError('source training identity, completion or base differs')
        actual = load_initializer_manifest(Path(audit['initializer_manifest']), tokenizer, contract.MODEL_DIR)
        if actual != audit['initializers'] or (manifest is not None and actual != manifest):
            raise ValueError('the two sources must share the audited single-image initializer manifest')
        manifest = actual
        item = exports[branch][str(STEPS[branch])]
        validate_selection(item, branch, manifest, audit, model['model'])
        selected[branch] = item
        _merge_locks(locks, item['hashes'])
        _merge_locks(locks, {str(source / name): sha256(source / name)
                            for name in ('source_audit.json', 'exports.json', 'status.json')})
    snapshots.check_locks(locks)
    rows = plan(manifest)
    output.mkdir(parents=True)
    audit = {'study': STUDY, 'inference_git': revision, 'source_training_gits': SOURCE_GITS,
        'source_experiments': {b: str(p) for b, p in SOURCES.items()}, 'initializers': manifest,
        'selected_exports': selected, 'source_locks': locks, 'fresh_training': False,
        'sampling': {**prior.SAMPLING, 'guidance_scale_by_mode': {b + '_only': list(CFGS) for b in STEPS}}}
    _write(output / 'source_audit.json', audit)
    (output / 'generation_manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf8')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Subject1000 / Material150 CFG comparison</h1>'
        '<p>CFG3.5 /5 /7.5; seeds30-40;297 images. See status.json for execution state.</p>'
        + ''.join(f'<p><a href="groups/{b}_only/index.html">{b}{step}</a></p>' for b, step in STEPS.items()), encoding='utf8')
    for branch in STEPS:
        mode = branch + '_only'
        gallery(output, [r for r in rows if r['mode'] == mode], [], {'status': 'prepared'}, mode)
    return output, rows, audit


def worker(output, branch):
    gpu = GPUS[branch]
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise ValueError('worker GPU differs from assignment')
    audit = read_json(output / 'source_audit.json')
    rows = [r for r in plan(audit['initializers']) if r['mode'] == branch + '_only']
    run_group(output, branch + '_only', rows, audit, gpu, render_gallery=gallery)


def run(output):
    os.environ.update(ENVIRONMENT)
    require_idle(GPUS.values())
    output, rows, audit = prepare(output)
    state = {'status': 'running', 'expected_images': len(rows), 'generated_images': 0,
             'inference_git': audit['inference_git'], 'fresh_training': False, 'quality_verified': False, 'gpu_mapping': GPUS}
    _write(output / 'status.json', state)

    def launch(branch):
        command = [sys.executable, '-u', '-B', '-m', 'experiments.' + STUDY + '.run',
                   '--output', str(output), '--worker', branch]
        _write(output / f'{branch}_command.json', {'command': command, 'gpu': GPUS[branch]})
        with (output / f'{branch}.log').open('x', encoding='utf8') as log:
            subprocess.run(command, cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(GPUS[branch])},
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        require_idle(GPUS.values())
        failures = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(launch, b): b for b in STEPS}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    failures[futures[future]] = str(error)
        generated = collect(output, rows)
        state['generated_images'] = len(generated)
        snapshots.check_locks(audit['source_locks'])
        if failures or len(generated) != 297 or git_info() != audit['inference_git'] or any(
                read_json(output / f'groups/{b}_only/status.json')['status'] != 'succeeded' for b in STEPS):
            raise RuntimeError(f'CFG comparison incomplete; outputs preserved: {failures}')
        state.update(status='succeeded', safety_filtered_images=sum(r['nsfw_content_detected'] for r in generated))
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        _write(output / 'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=tuple(STEPS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    worker(args.output.resolve(), args.worker) if args.worker else run(args.output)
