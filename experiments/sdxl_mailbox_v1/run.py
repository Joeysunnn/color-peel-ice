"""Finite SDXL queues: train and test eight independent branch/method pairs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.sdxl_mailbox_v1 import contract, data
from experiments.sdxl_mailbox_v1.prepare_model import MODEL_DIR, sha256, verify_model

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')
    temporary.replace(path)


def git_revision():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    if git('status', '--porcelain'):
        raise ValueError('execute from a clean committed checkout')
    return {'commit': git('rev-parse', 'HEAD'), 'branch': git('branch', '--show-current'),
            'origin': git('remote', 'get-url', 'origin')}


def make_jobs(output, sources, model_dir, gpus, smoke=False):
    jobs = []
    for branch, recipe in contract.TRAINING.items():
        for method in contract.METHODS:
            group = Path(output) / branch / method
            config_path = group / 'config.json'
            config = {'model_dir': str(model_dir), 'dataset': sources[branch]['manifest'],
                      'branch': branch, 'method': method, 'token': recipe['token'],
                      'initializer': recipe['initializer'], 'steps': 2 if smoke else recipe['max_steps'],
                      'checkpoints': [2] if smoke else list(recipe['steps']),
                      'resolution': contract.RESOLUTION, 'seed': 42,
                      'inference_seeds': [30] if smoke else list(contract.SEEDS),
                      'smoke': smoke, 'cfg': recipe['cfg'], 'embedding_lr': contract.EMBEDDING_LR,
                      'kv_lr': contract.KV_LR[method], 'rank': 64, 'alpha': 64}
            train = [sys.executable, '-B', '-u', '-m', 'experiments.sdxl_mailbox_v1.train',
                     '--config', str(config_path), '--output', str(group / 'training')]
            infer = [sys.executable, '-B', '-u', '-m', 'experiments.sdxl_mailbox_v1.infer',
                     '--config', str(config_path), '--training', str(group / 'training'),
                     '--output', str(group / 'inference')]
            jobs.append({'branch': branch, 'method': method, 'group': str(group),
                         'gpu': gpus[len(jobs) % len(gpus)], 'config': config,
                         'config_path': str(config_path), 'training_command': train,
                         'inference_command': infer})
    return jobs


def report(job, state):
    group = Path(job['group'])
    folder = group / 'train_outputs'
    folder.mkdir(exist_ok=True)
    write(group / 'status.json', state)
    write(folder / 'status.json', state)
    (folder / 'SUMMARY.md').write_text(
        f"Status: {state['status']}. Training {state['completed_steps']}/{job['config']['steps']} updates.\n"
        f"Stop reason: {state.get('stop_reason')}. Startup does not establish transfer quality.\n", encoding='utf8')
    (folder / 'COMMANDS.md').write_text(
        '\n\n'.join('CUDA_VISIBLE_DEVICES=' + str(job['gpu']) + ' ' + shlex.join(job[key])
                    for key in ('training_command', 'inference_command')) + '\n', encoding='utf8')
    (folder / 'LOG.md').write_text(
        'Queue output: training.log and inference.log beside this file.\n'
        'Training metrics, audit and checkpoints: ../training/.\n'
        'Transfer images, ledger and review: ../inference/.\n', encoding='utf8')
    (folder / 'SCIENTIFIC_CHANGELOG.md').write_text(
        'SDXL base 1.0 replaces SD2.1; two learned token vectors, concatenated penultimate '
        'hidden states, projected EOS pool and size/crop conditioning. Frozen RGB is resized '
        '512 to 1024 with Lanczos; object masks use nearest sampling. V100 training uses '
        'mixed precision with FP32 trainable weights and VAE, not the old all-FP32 recipe. '
        'The new first-pass transfer seeds are 30-40 for every branch. No compositions.\n', encoding='utf8')
    (folder / 'COMPARABILITY_REPORT.md').write_text(
        'Both methods update only their target token vectors and token-local K/V residuals. '
        'Fullrank does not mean full UNet tuning. Rank64 has alpha64. Original images, '
        'captions, fixed order, mask-loss channel-sum normalization, learning rates and '
        'AdamW settings are retained. Mailbox Material initializer remains metal. '
        'PNDM100; CFG3.5 for S/M/T, CFG6 for Color. The reused common SD1.4 checker '
        'runs after SDXL decoding. Different base, resolution, dual encoding and precision '
        'mean this is a migration experiment rather than a strict base-only ablation.\n', encoding='utf8')


def run_queue(gpu, jobs, output, revision):
    queue = {'gpu': gpu, 'status': 'running', 'jobs': [], 'monitoring': False}
    path = Path(output) / f'gpu_{gpu}.json'
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONDONTWRITEBYTECODE': '1',
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'DIFFUSERS_OFFLINE': '1'}
    try:
        for job in jobs:
            if git_revision() != revision:
                raise ValueError('execution checkout changed during the finite queue')
            data.load_rows(job['config']['dataset'])
            state = {'status': 'training', 'branch': job['branch'], 'method': job['method'],
                     'git': revision, 'gpu': gpu, 'completed_steps': 0,
                     'checkpoint_steps': job['config']['checkpoints'], 'stop_reason': None}
            queue['jobs'].append(state)
            report(job, state)
            write(path, queue)
            try:
                for phase in ('training', 'inference'):
                    state['status'] = phase
                    report(job, state)
                    write(path, queue)
                    with (Path(job['group']) / 'train_outputs' / f'{phase}.log').open('x') as log:
                        subprocess.run(job[phase + '_command'], cwd=ROOT, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, check=True)
                    if phase == 'training':
                        state['completed_steps'] = job['config']['steps']
                state.update(status='succeeded', stop_reason='training and paired transfer completed')
            except BaseException as error:
                state.update(status='failed', stop_reason=f'{type(error).__name__}: {error}')
                raise
            finally:
                report(job, state)
                write(path, queue)
        queue['status'] = 'succeeded'
    except BaseException as error:
        queue.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(path, queue)
    return queue


def check_gpus(gpus):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    used = {int(row.split(',')[0]): int(row.split(',')[1]) for row in rows.splitlines()}
    if any(used.get(gpu, 999999) > 2048 for gpu in gpus):
        raise RuntimeError('selected GPU missing or occupied above 2GiB; preserve existing jobs')


def run(output, gpus=(1, 2, 3), model_dir=MODEL_DIR, execute=False, smoke=False):
    output, model_dir = Path(output).resolve(), Path(model_dir).resolve()
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError('use a fresh external experiment directory')
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError('require distinct nonnegative GPU indices')
    revision = git_revision()
    model = verify_model(model_dir)
    if execute:
        check_gpus(gpus)
    # Space was supplied by the user. Do not issue another capacity query.
    output.mkdir(parents=True)
    state = {'status': 'preparing', 'git': revision, 'model': model, 'gpus': list(gpus),
             'smoke': smoke, 'monitoring': False, 'composition': False,
             'command': shlex.join(sys.argv), 'new_output_only': True,
             'space_source': 'user: home 494.19GiB; var/tmp 30.92GiB; no recheck requested'}
    write(output / 'experiment.json', state)
    try:
        sources = data.prepare_data(output / 'dataset')
        jobs = make_jobs(output, sources, model_dir, gpus, smoke)
        for job in jobs:
            Path(job['group']).mkdir(parents=True)
            job['config'].update(git_commit=revision['commit'],
                                 model_source_sha256=sha256(model_dir / 'model_source.json'),
                                 dataset_manifest_sha256=sha256(job['config']['dataset']))
            write(job['config_path'], job['config'])
        state.update(status='running' if execute else 'dry_run_verified', dataset_sources=sources, jobs=jobs,
                     expected_transfer_images=sum(len(contract.transfer_cases(
                         job['branch'], job['config']['checkpoints'], smoke=smoke)) for job in jobs))
        write(output / 'experiment.json', state)
        links = ''.join(f'<li><a href="{job["branch"]}/{job["method"]}/inference/index.html">'
                        f'{html.escape(job["branch"])} / {job["method"]}</a></li>' for job in jobs)
        (output / 'index.html').write_text(
            '<!doctype html><meta charset="utf-8"><h1>SDXL Mailbox: four independent branches</h1>'
            '<p>Subject27 / Color9 / Material8 / Texture16. Fullrank and rank64. '
            'Single-branch transfer only. See experiment.json for live completion state.</p>'
            '<p><a href="dataset/index.html">Actual training images and captions</a> | '
            '<a href="dataset/receipt.json">Data provenance</a></p><ul>' + links + '</ul>', encoding='utf8')
        if not execute:
            return state
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(run_queue, gpu, [job for job in jobs if job['gpu'] == gpu],
                                   output, revision) for gpu in gpus]
            state['queues'] = [future.result() for future in futures]
        state['status'] = 'succeeded'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(output / 'experiment.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    run(args.output, args.gpus, args.model_dir, args.execute, args.smoke)
