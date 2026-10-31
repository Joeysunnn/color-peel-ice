"""Finite SD2.1 queues: six single-branch trainings, paired inference, stock spoon."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import html
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from experiments.sd21_wood_branches_v1.contract import (
    BRANCHES, METHODS, STEPS, STUDY, build_config, verify_dataset,
)
from experiments.sd21_wood_branches_v1.prepare_model import verify_model
from experiments.subject_multicolor_multimaterial_b.training import _write
from scripts.launch.colorpeel_run import git_info, read_config, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]


def make_jobs(output, revision, gpus, smoke=False, seeds=tuple(range(30, 61))):
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    jobs = []
    for branch in BRANCHES:
        for method in METHODS:
            group = output / branch / method
            config = build_config(branch, method, smoke=smoke)
            name = f"{stamp}__{STUDY}__{config['run']['variant']}__{revision['commit'][:7]}__42"
            training_run = group / 'training' / name
            config_path = group / 'config.json'
            train = [sys.executable, '-B', str(ROOT / 'scripts/launch/colorpeel_run.py'),
                     '--config', str(config_path), '--run-dir', str(training_run)]
            infer = [sys.executable, '-u', '-B', '-m',
                     'experiments.sd21_wood_branches_v1.inference', '--branch', branch,
                     '--method', method, '--run', str(training_run),
                     '--output', str(group / 'inference'), '--device', 'cuda:0',
                     '--seeds', *map(str, seeds), '--execute']
            if smoke:
                infer.append('--smoke')
            jobs.append({'branch': branch, 'method': method, 'gpu': gpus[len(jobs) % len(gpus)],
                         'group': str(group), 'run': str(training_run), 'config': config,
                         'config_path': str(config_path), 'training_command': train,
                         'inference_command': infer})
    return jobs


def report_training(group, state, job):
    folder = group / 'train_outputs'
    folder.mkdir(exist_ok=True)
    _write(group / 'status.json', state)
    _write(folder / 'status.json', state)
    (folder / 'SUMMARY.md').write_text(
        f"{state['status']}; {state['completed_steps']}/{state['max_steps']} training steps. "
        f"Stop reason: {state['stop_reason']}.\n", encoding='utf8')
    (folder / 'COMMANDS.md').write_text(
        f"CUDA_VISIBLE_DEVICES={job['gpu']} " + shlex.join(job['training_command'])
        + '\n\n' + shlex.join(job['inference_command']) + '\n', encoding='utf8')
    (folder / 'LOG.md').write_text(
        f"Training: [{job['run']}/logs/stdout.log]({job['run']}/logs/stdout.log).\n"
        'Queue subprocess output: training.log / inference.log in this directory.\n', encoding='utf8')
    (folder / 'SCIENTIFIC_CHANGELOG.md').write_text(
        'New base: pinned SD2.1-base 512, OpenCLIP 1024. All data, loss, optimizer, '
        'initializers, learning rates, augmentation and attention rules inherited. '
        'User requested 1000 steps and paired 400/600/800/1000 snapshots. '
        'Two-step smoke runs are runtime checks only. No compositions.\n', encoding='utf8')
    (folder / 'COMPARABILITY_REPORT.md').write_text(
        'Fullrank means fullrank token-local K/V residual plus target embedding; '
        'rank64 means rank64/alpha64 token-local K/V LoRA plus target embedding. '
        'Neither trains the full UNet. Frozen original data are read only. '
        'The SD2.1 text/UNet architecture necessarily differs from SD1.5; method and '
        '512px/PNDM100/CFG3.5 transfer settings are preserved. '
        'Startup verification does not establish output quality.\n', encoding='utf8')


def run_queue(gpu, jobs, output, revision, smoke=False, seeds=tuple(range(30, 61)), literal=False):
    queue = {'gpu': gpu, 'status': 'running', 'jobs': [], 'monitoring': False}
    path = output / f'gpu_{gpu}.json'
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu),
                   'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
                   'TRANSFORMERS_OFFLINE': '1', 'DIFFUSERS_OFFLINE': '1'}
    try:
        for job in jobs:
            if git_info() != revision:
                raise ValueError('execution checkout changed')
            verify_dataset(job['branch'])
            group = Path(job['group'])
            state = {'status': 'training', 'git': revision, 'gpu': gpu,
                     'branch': job['branch'], 'method': job['method'], 'run': job['run'],
                     'max_steps': job['config']['args']['max_train_steps'], 'completed_steps': 0,
                     'checkpoint_steps': job['config']['args']['checkpoint_steps'],
                     'best_metric': None, 'best_checkpoint': None, 'stop_reason': None}
            queue['jobs'].append(state)
            _write(path, queue)
            report_training(group, state, job)
            try:
                for phase, command in (('training', job['training_command']),
                                       ('inference', job['inference_command'])):
                    with (group / 'train_outputs' / f'{phase}.log').open('x') as log:
                        subprocess.run(command, cwd=ROOT, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, check=True)
                    if phase == 'training':
                        state.update(status='inference', completed_steps=state['max_steps'])
                        report_training(group, state, job)
                        _write(path, queue)
                verify_dataset(job['branch'])
                state.update(status='succeeded', stop_reason='max_steps reached; paired inference completed')
            except BaseException as error:
                state.update(status='failed', stop_reason=f'{type(error).__name__}: {error}')
                raise
            finally:
                report_training(group, state, job)
                _write(path, queue)
        if literal:
            command = [sys.executable, '-u', '-B', '-m',
                       'experiments.sd21_wood_branches_v1.inference', '--branch', 'literal',
                       '--output', str(output / 'literal_spoon'), '--device', 'cuda:0',
                       '--seeds', *map(str, seeds), '--execute']
            if smoke:
                command.append('--smoke')
            with (output / 'literal.log').open('x') as log:
                subprocess.run(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        queue['status'] = 'succeeded'
    except BaseException as error:
        queue.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        _write(path, queue)
    return queue


def run(output, gpus=(0, 2, 3), execute=False, smoke=False, seeds=tuple(range(30, 61))):
    output = resolve_run_dir(Path(output))
    revision = git_info()
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError('use a fresh external experiment directory')
    if (not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus)
            or not seeds or len(set(seeds)) != len(seeds)):
        raise ValueError('require distinct nonnegative GPUs and distinct seeds')
    model = verify_model()
    sources = {branch: verify_dataset(branch) for branch in BRANCHES}
    output.parent.mkdir(parents=True, exist_ok=True)
    required = (15 if smoke else 68) * 1024 ** 3
    if shutil.disk_usage(output.parent).free < required:
        raise OSError(f'require {required / 1024**3:g}GiB free for immutable training states')
    if execute:
        stats = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                         '--format=csv,noheader,nounits'], text=True)
        used = {int(row.split(',')[0]): int(row.split(',')[1]) for row in stats.splitlines()}
        if any(used.get(gpu, 999999) > 2048 for gpu in gpus):
            raise RuntimeError('selected GPU has more than 2GiB in use; preserve existing jobs')
    jobs = make_jobs(output, revision, gpus, smoke, seeds)
    output.mkdir()
    for job in jobs:
        Path(job['group']).mkdir(parents=True)
        _write(Path(job['config_path']), job['config'])
        if read_config(Path(job['config_path'])) != job['config']:
            raise ValueError('config YAML roundtrip changed the training arguments')
    state = {'status': 'running' if execute else 'dry_run_verified', 'git': revision,
             'model': model, 'dataset_sources': sources, 'jobs': jobs,
             'gpus': list(gpus), 'smoke': smoke, 'transfer_seeds': list(seeds),
             'checkpoint_steps': [2] if smoke else list(STEPS),
             'expected_transfer_images': 6 if smoke else 4 * 2 * len(seeds) * (9 + 2 * 4 * 3),
             'literal_images': 1 if smoke else 3 * len(seeds), 'monitoring': False,
             'command': shlex.join(sys.argv), 'new_output_only': True}
    _write(output / 'experiment.json', state)
    links = ''.join(f'<li><a href="{job["branch"]}/{job["method"]}/inference/index.html">'
                    f'{html.escape(job["branch"])} / {job["method"]}</a></li>' for job in jobs)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>SD2.1 wood spoon</h1>'
        '<p>Separate Subject / Material / Texture; original and rank64; no compositions.</p><ul>'
        + links + '<li><a href="literal_spoon/index.html">Stock literal spoon</a></li></ul>', encoding='utf8')
    if not execute:
        return state
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [executor.submit(run_queue, gpu, [job for job in jobs if job['gpu'] == gpu],
                                       output, revision, smoke, seeds, gpu == gpus[0]) for gpu in gpus]
            state['queues'] = [future.result() for future in futures]
        state['status'] = 'succeeded'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        _write(output / 'experiment.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 2, 3])
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(30, 61)))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    run(args.output, args.gpus, args.execute, args.smoke, args.seeds)
