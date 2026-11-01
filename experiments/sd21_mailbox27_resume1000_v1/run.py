"""Continue completed Mailbox27 original/rank64600 runs to1000, then transfer."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import shutil
import subprocess

from experiments.sd21_mailbox27_resume1000_v1 import contract, inference
from experiments.sd21_mailbox_subject27_v1.run import jobs
from experiments.sd21_wood_branches_v1 import run as shared
from experiments.sd21_wood_branches_v1.prepare_model import verify_model
from experiments.subject_multicolor_multimaterial_b.training import _write
from scripts.launch.colorpeel_run import git_info, read_config, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]


def report(group, state, job):
    state['resumed_from_step'] = 600
    state['completed_steps'] = max(600, state['completed_steps'])
    state['new_optimization_steps'] = state['completed_steps'] - 600
    shared.report_training(group, state, job)
    folder = group / 'train_outputs'
    (folder / 'SCIENTIFIC_CHANGELOG.md').write_text(
        'True full-state continuation from completed600, not a warm start. Restore adapter, text encoder, '
        'AdamW, constant scheduler, RNG and custom state. Reuse the same27 images in fixed order; '
        'first continued update601 uses row7. Train400 additional updates; save800/1000. '
        'Same base, losses, LRs, seed42 and token-local original/rank64 methods.\n', encoding='utf8')
    (folder / 'COMPARABILITY_REPORT.md').write_text(
        'Old600 state/results read only. Metrics1-600 and actual audit counters are inherited with hashes; '
        'new optimization metrics601-1000 are appended in this new run. Worker RNG was not checkpointed; '
        'this fixed-input recipe preserves sample order but no general bitwise-resume claim is made. '
        'Nine unchanged prompts including matte; seed30-40, PNDM100/CFG3.5/512/fp16/checker. '
        '198 images per method,396 total,44 matte. Startup is not quality acceptance.\n', encoding='utf8')


def run(output, gpus=(2, 3), execute=False):
    output = resolve_run_dir(Path(output))
    revision = git_info()
    if (output.exists() or not output.is_relative_to(contract.PARENT / 'continuations')
            or len(gpus) != 2 or len(set(gpus)) != 2 or any(gpu < 0 for gpu in gpus)):
        raise ValueError('use a fresh original-experiment continuation and two distinct GPUs')
    model = verify_model()
    data = contract.verify_dataset('subject')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exact parent state sizes, twice per method, plus2GiB for final/export pairs,
    # images, logs and free-space margin. Never remove historical snapshots.
    state_bytes = sum((contract.source_run(method) / 'checkpoints/checkpoint-600' / name).stat().st_size
        for method in contract.METHODS for name in contract.STATE_NAMES)
    required = 2 * state_bytes + 2 * 1024 ** 3
    free = shutil.disk_usage(output.parent).free
    if free < required:
        raise OSError(f'need{required}bytes for four new states and outputs; only{free}bytes free')
    if execute:
        stats = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
            '--format=csv,noheader,nounits'], text=True)
        used = {int(row.split(',')[0]): int(row.split(',')[1]) for row in stats.splitlines()}
        if any(used.get(gpu, 999999) > 2048 for gpu in gpus):
            raise RuntimeError('selected GPU occupied; preserve existing jobs')
    planned = jobs(output, revision, gpus, experiment_contract=contract, inference_module=inference)
    output.mkdir()
    _write(output / 'parent_sources.json', data)
    _write(output / 'authorization.json', {'request': '继续训练800/1000并transfer，包含matte',
        'parent': str(contract.PARENT), 'resume_step': 600, 'new_steps_per_method': 400,
        'checkpoint_steps': list(contract.STEPS), 'generation_seeds': list(inference.SEEDS),
        'methods': list(contract.METHODS), 'training_authorized': execute, 'old_results_read_only': True,
        'monitoring': False, 'git': revision})
    for job in planned:
        group = Path(job['group'])
        group.mkdir(parents=True)
        _write(Path(job['config_path']), job['config'])
        if read_config(Path(job['config_path'])) != job['config']:
            raise ValueError('launcher config roundtrip changed')
        preflight = [*job['training_command'][:-1], str(group / 'preflight' / Path(job['run']).name), '--dry-run']
        with (group / 'preflight.log').open('x') as log:
            subprocess.run(preflight, cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(job['gpu'])},
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
    state = {'status': 'running' if execute else 'dry_run_verified', 'git': revision, 'model': model,
        'study': contract.STUDY, 'jobs': planned, 'resume_step': 600, 'max_train_steps': 1000,
        'checkpoint_steps': list(contract.STEPS), 'generation_seeds': list(inference.SEEDS),
        'expected_images': 396, 'expected_matte_images': 44, 'monitoring': False,
        'disk_required_bytes': required, 'disk_free_bytes_at_preflight': free}
    _write(output / 'experiment.json', state)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Mailbox27 continuation600-1000</h1>'
        '<p>Original/rank64;800/1000 paired snapshots;9 prompts including matte;seeds30-40;396 images.</p>'
        + ''.join(f'<p><a href="methods/{m}/inference/index.html">{m}</a></p>' for m in contract.METHODS), encoding='utf8')
    if not execute:
        return state
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(shared.run_queue, gpu, [j for j in planned if j['gpu'] == gpu], output, revision,
                dataset_verifier=contract.verify_dataset, training_reporter=report) for gpu in gpus]
            state['queues'] = [future.result() for future in futures]
        contract.check_locks(data['source_locks'])
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
    parser.add_argument('--gpus', type=int, nargs=2, default=[2, 3])
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    run(args.output, tuple(args.gpus), args.execute)
