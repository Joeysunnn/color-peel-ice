"""Read-only SD2.1 wood Subject400 + Texture600 composition, matching methods."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd15_token_local_kv_v1.evaluate import attention_masks
from experiments.sd21_wood_branches_v1 import contract, inference, snapshots
from experiments.sd21_wood_branches_v1.prepare_model import verify_model
from experiments.wood_spoon_composition_v1 import inference as review_helpers
from scripts.launch.colorpeel_run import git_info, read_config, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
SOURCE = contract.DEFAULT_OUTPUT_ROOT / '20261101-003759__sd21_wood_branches_v1__wood_S72_M8_T16__19049ce__42'
SOURCE_GIT = {'branch': 'codex/sd21-wood-branches-v1',
              'commit': '19049cee6194476c3c6e508e954bc28667e53352'}
PROMPT = 'a photo of <S*> spoon with <T*> texture'
BRANCH_STEPS = {'subject': 400, 'texture': 600}
SEEDS = tuple(range(30, 61))
write = review_helpers.write


def prompt_rows(method):
    if method not in contract.METHODS:
        raise ValueError('select original/fullrank or rank64')
    return [{'id': f'{method}__S400_T600__seed{seed}', 'method': method, 'branch': 'texture',
             'seed': seed, 'prompt': PROMPT, 'subject_step': 400, 'attribute_step': 600,
             'subject_images': 72, 'image_path': f'images/seed{seed}.png'} for seed in SEEDS]


def check_pair(item, branch, method):
    """Compare the existing export tensors with the exact selected training snapshot."""
    import torch
    from experiments.sd15_token_local_kv_v1 import snapshots as shared

    step, token = BRANCH_STEPS[branch], contract.TOKENS[branch]
    folder = Path(item['checkpoint'])
    snapshot = Path(item['run']) / 'checkpoints' / f'checkpoint-{step}'
    shapes, empty = snapshots.adapter_schema(method)
    adapter = torch.load(folder / item['weight'], map_location='cpu')
    saved = torch.load(snapshot / 'pytorch_model.bin', map_location='cpu')
    shared._check_state(torch, adapter, shapes, empty, 'composition export')
    shared._check_state(torch, saved, shapes, {}, 'composition snapshot')
    if any(not torch.equal(adapter[key], saved[key]) for key in shapes):
        raise ValueError('composition adapter differs from selected snapshot')
    token_id, audit = shared._embedding_audit(
        Path(item['run']) / 'checkpoints/embedding_update_audit.json', 1000, token)
    if audit['non_modifier_embedding_drift'].get('changed_rows') != 0:
        raise ValueError('ordinary embeddings changed in source training')
    vector = torch.load(folder / (token + '.bin'), map_location='cpu')
    encoder = torch.load(snapshot / 'pytorch_model_1.bin', map_location='cpu')
    table = encoder[shared.EMBEDDING_KEY]
    if (set(vector) != {token} or tuple(vector[token].shape) != (1024,)
            or table.ndim != 2 or table.shape[1] != 1024 or not 0 <= token_id < table.shape[0]
            or not bool(torch.isfinite(vector[token]).all())
            or not torch.equal(vector[token], table[token_id])):
        raise ValueError('composition token differs from selected1024-dimensional snapshot')


def prepare():
    receipt = verify_model()
    source_state = read_json(SOURCE / 'experiment.json')
    if source_state['status'] != 'succeeded' or source_state['git'] != SOURCE_GIT or source_state['smoke']:
        raise ValueError('expected the completed pinned SD2.1 experiment')
    locks = {str(SOURCE / 'experiment.json'): sha256(SOURCE / 'experiment.json'),
             **snapshots.model_locks(contract.MODEL_DIR, receipt)}
    sources, references = {}, {}
    for branch, step in BRANCH_STEPS.items():
        data = contract.verify_dataset(branch)
        references[branch] = data['references']
        locks.update(data['source_locks'])
        sources[branch] = {}
        for method in contract.METHODS:
            folder = SOURCE / branch / method / 'inference'
            receipt_path = folder / f'ckpts/step{step}/receipt.json'
            item, state = read_json(receipt_path), read_json(folder / 'status.json')
            index_path = folder / 'ckpts/index.json'
            index = read_json(index_path)
            run, checkpoint = Path(item['run']), Path(item['checkpoint'])
            expected = contract.build_config(branch, method)
            job = next(job for job in source_state['jobs']
                       if job['branch'] == branch and job['method'] == method)
            manifest = read_json(run / 'manifest.json')
            weight = 'pytorch_token_local_kv_weights.bin' if method == 'fullrank' else 'pytorch_lora_kv_weights.bin'
            if (item != index['steps'][str(step)] or index['git'] != SOURCE_GIT or index['smoke']
                    or state['status'] != 'succeeded' or state['git'] != SOURCE_GIT
                    or run != Path(job['run']) or checkpoint != folder / f'ckpts/step{step}'
                    or read_config(run / 'config.yaml') != expected
                    or manifest['status'] != 'succeeded' or manifest['returncode'] != 0
                    or manifest['git'] != SOURCE_GIT or item.get('base_model') != receipt['model']
                    or item.get('source_checkpoint') != str(run / 'checkpoints')
                    or any(item.get(key) != value for key, value in {
                        'branch': branch, 'method': method, 'step': step, 'steps': step,
                        'training_steps': 1000, 'token': contract.TOKENS[branch], 'weight': weight,
                        'embedding_dim': 1024, 'status': 'verified', 'paired_embedding_finite': True,
                        'concept_id': 'wood_spoon' if branch == 'subject' else 'wood_spoon_texture',
                        'final_snapshot_matches_export': False}.items())):
                raise ValueError('selected SD2.1 S400/T600 source or paired export differs')
            snapshot = run / 'checkpoints' / f'checkpoint-{step}'
            snapshot_paths = {str(snapshot / name) for name in ('pytorch_model.bin', 'pytorch_model_1.bin')}
            if set(item.get('source_snapshot_hashes', {})) != snapshot_paths:
                raise ValueError('selected snapshot source paths differ')
            paths = [run / 'config.yaml', run / 'manifest.json', checkpoint / weight,
                     checkpoint / (item['token'] + '.bin'), checkpoint / 'adaptation_config.json',
                     run / 'checkpoints/embedding_update_audit.json', run / 'checkpoints/training_metrics.jsonl']
            paths += [Path(path) for path in snapshot_paths]
            selected_locks = {str(path): item['hashes'][str(path)] for path in paths}
            if any(item['source_snapshot_hashes'][path] != selected_locks[path] for path in snapshot_paths):
                raise ValueError('selected source snapshot hashes disagree')
            selected_locks.update({str(path): sha256(path) for path in (receipt_path, index_path, folder / 'status.json')})
            snapshots.check_locks(selected_locks)
            check_pair(item, branch, method)
            locks.update(selected_locks)
            # Keep historical provenance intact, narrowing runtime locks to the selected pairs.
            sources[branch][method] = {**item, 'hashes': selected_locks}
    return {'inference_git': git_info(), 'training_gits': {'sd21': SOURCE_GIT},
            'source': str(SOURCE), 'sources': sources, 'references': references, 'source_locks': locks}


def load_pipeline(subject, texture, device):
    import torch
    pipe = inference.load_pipeline([], device)
    if subject['method'] != texture['method']:
        raise ValueError('match the two adapter methods')
    if subject['method'] == 'fullrank':
        pipe.unet.load_attn_procs(subject['checkpoint'], weight_name=subject['weight'], adaptation_mode='token_local_kv')
        from src.methods.colorpeel_ice.dual_token_local_kv import install_dual_token_local_kv
        state = torch.load(Path(texture['checkpoint']) / texture['weight'], map_location='cpu')
        install_dual_token_local_kv(pipe.unet, state, primary_label='subject')
    else:
        from experiments.lora_kv_material_v1.attention import install_dual_token_local_lora_kv
        states = [torch.load(Path(item['checkpoint']) / item['weight'], map_location='cpu')
                  for item in (subject, texture)]
        install_dual_token_local_lora_kv(pipe.unet, *states, rank=64, alpha=64)
    for item in (subject, texture):
        pipe.load_textual_inversion(item['checkpoint'], weight_name=item['token'] + '.bin')
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(audit_path, output, method, device='cuda:0', execute=False):
    output, audit_path = resolve_run_dir(Path(output)), Path(audit_path).resolve()
    audit, revision = read_json(audit_path), git_info()
    if (output.exists() or output.is_relative_to(ROOT) or audit['inference_git'] != revision
            or audit_path.is_relative_to(output)):
        raise ValueError('use a fresh external composition output at the audited revision')
    locks = {**audit['source_locks'], str(audit_path): sha256(audit_path)}
    snapshots.check_locks(locks)
    output.mkdir(parents=True)
    rows, generated, refs, copies = prompt_rows(method), [], {}, {}
    state = {'status': 'running' if execute else 'dry_run', 'method': method, 'branch': 'texture',
             'literal_subject': False, 'subject_step': 400, 'attribute_step': 600, 'subject_images': 72,
             'prompt': PROMPT, 'sampling': {**inference.SAMPLING, 'seeds': list(SEEDS)},
             'inference_git': revision, 'training_gits': audit['training_gits'],
             'expected_images': 31, 'generated_images': 0, 'retraining': False, 'monitoring': False,
             'internal_adapter_slots': {'subject': '<S*>', 'material': '<T*>'}}
    try:
        refs, copies = review_helpers.references(audit, 'texture', output)
        (output / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
        if execute:
            import torch
            pipe = load_pipeline(audit['sources']['subject'][method], audit['sources']['texture'][method], device)
            with (output / 'generation_status.jsonl').open('x', encoding='utf8') as ledger, torch.no_grad():
                for row in rows:
                    result = pipe(PROMPT, num_inference_steps=100, guidance_scale=3.5, width=512, height=512,
                                  generator=torch.Generator(device=device).manual_seed(row['seed']),
                                  cross_attention_kwargs=attention_masks(pipe, PROMPT, ['<S*>', '<T*>'], 3.5))
                    if (result.nsfw_content_detected is None or len(result.nsfw_content_detected) != 1
                            or len(result.images) != 1 or result.images[0].size != (512, 512)
                            or result.images[0].mode != 'RGB'):
                        raise ValueError('require one512px RGB image with a safety result')
                    path = output / row['image_path']; path.parent.mkdir(exist_ok=True)
                    with path.open('xb') as stream:
                        result.images[0].save(stream, format='PNG')
                    generated.append({**row, 'image_sha256': sha256(path),
                                      'status': 'safety_filtered' if result.nsfw_content_detected[0] else 'ok',
                                      'subject_adapter_loaded': True, 'texture_adapter_loaded': True})
                    ledger.write(json.dumps(generated[-1]) + '\n'); ledger.flush()
            del pipe
            torch.cuda.empty_cache()
        snapshots.check_locks(locks)
        snapshots.check_locks(copies)
        if git_info() != revision:
            raise ValueError('composition execution revision changed')
        state.update(status='succeeded' if execute else 'dry_run_verified', generated_images=len(generated))
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}', generated_images=len(generated))
        raise
    finally:
        state.update(source_locks=locks, reference_copy_hashes=copies)
        write(output / 'provenance.json', state)
        review_helpers.review(output, refs, generated or rows, state)
    return state


def run(output, gpus=(2, 3), execute=False):
    output = resolve_run_dir(Path(output))
    if output.exists() or output.is_relative_to(ROOT) or not gpus or len(set(gpus)) != len(gpus):
        raise ValueError('use a fresh external output and distinct GPUs')
    audit = prepare()
    if execute:
        stats = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
        used = {int(row.split(',')[0]): int(row.split(',')[1]) for row in stats.splitlines()}
        if any(used.get(gpu, 999999) > 2048 for gpu in gpus):
            raise ValueError('selected GPU is occupied; preserve existing jobs')
    output.mkdir(parents=True)
    write(output / 'source_audit.json', audit)
    state = {'status': 'running' if execute else 'dry_run_verified', 'git': git_info(), 'source': str(SOURCE),
             'subject_step': 400, 'texture_step': 600, 'seeds': list(SEEDS), 'expected_images': 62,
             'monitoring': False, 'training': False, 'jobs': []}
    for index, method in enumerate(contract.METHODS):
        state['jobs'].append({'method': method, 'gpu': gpus[index % len(gpus)], 'status': 'queued',
                             'output': str(output / 'methods' / method)})
    write(output / 'status.json', state)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>SD2.1 S400 + T600</h1>'
        + '<p>seed30–60; PNDM100 / CFG3.5 / 512px / fp16 / checker</p>'
        + ''.join(f'<p><a href="methods/{method}/index.html">{method}: results and training references</a></p>' for method in contract.METHODS)
        + '<p>Images appear as the finite jobs run. See status.json for completion.</p>'
        + '<table><tr><th>seed</th><th>original</th><th>rank64</th></tr>'
        + ''.join(f'<tr><th>{seed}</th>' + ''.join(
            f'<td><a href="methods/{method}/images/seed{seed}.png"><img loading="lazy" width="384" src="methods/{method}/images/seed{seed}.png"></a></td>'
            for method in contract.METHODS) + '</tr>' for seed in SEEDS) + '</table>', encoding='utf8')
    def queue(gpu):
        for job in state['jobs']:
            if job['gpu'] != gpu:
                continue
            command = [sys.executable, '-u', '-B', '-m', 'experiments.sd21_wood_branches_v1.composition',
                       '--audit', str(output / 'source_audit.json'), '--output', job['output'], '--method', job['method']]
            if execute:
                command.append('--execute')
            job.update(status='running', command=command)
            write(output / f'gpu_{gpu}.json', {'jobs': [item for item in state['jobs'] if item['gpu'] == gpu]})
            with (output / f'{job["method"]}.log').open('x') as log:
                result = subprocess.run(command, cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu),
                                        'PYTHONDONTWRITEBYTECODE': '1'}, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT)
            job['status'] = 'succeeded' if result.returncode == 0 else 'failed'
            write(output / f'gpu_{gpu}.json', {'jobs': [item for item in state['jobs'] if item['gpu'] == gpu]})
            if result.returncode:
                raise RuntimeError(f'{job["method"]} failed; see its log')
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(queue, gpu) for gpu in gpus]
            for future in futures:
                future.result()
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
    parser.add_argument('--gpus', nargs='+', type=int, default=[2, 3])
    parser.add_argument('--method', choices=contract.METHODS)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.method:
        if args.audit is None:
            parser.error('--method requires --audit')
        generate(args.audit, args.output, args.method, execute=args.execute)
    else:
        run(args.output, tuple(args.gpus), args.execute)
