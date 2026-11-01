"""Stage Mailbox27, train two600-step methods and run paired200/400/600 transfer."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.sd21_mailbox_subject27_v1 import contract, inference
from experiments.sd21_wood_branches_v1 import run as shared
from experiments.sd21_wood_branches_v1.inference import reference_review
from experiments.sd21_wood_branches_v1.prepare_model import verify_model
from experiments.subject_multicolor_multimaterial_b.training import _write
from scripts.launch.colorpeel_run import git_info, read_config, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
REQUEST = '24张studio蓝绿M0/1、R.15/.85、三视角，加orange原图及blue/green原图换色；训练600步，保存200/400/600并transfer seed30–40。'


def jobs(output, revision, gpus, *, experiment_contract=contract, inference_module=inference):
    contract, inference = experiment_contract, inference_module
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    planned = []
    for number, method in enumerate(contract.METHODS):
        group = output / 'methods' / method
        config = contract.build_config('subject', method)
        name = f"{stamp}__{contract.STUDY}__{config['run']['variant']}__{revision['commit'][:7]}__42"
        train_run, config_path = group / 'training' / name, group / 'config.json'
        planned.append({'branch': 'subject', 'method': method, 'gpu': gpus[number % len(gpus)],
            'group': str(group), 'run': str(train_run), 'config': config, 'config_path': str(config_path),
            'training_command': [sys.executable, '-B', str(ROOT / 'scripts/launch/colorpeel_run.py'),
                '--config', str(config_path), '--run-dir', str(train_run)],
            'inference_command': [sys.executable, '-u', '-B', '-m', inference.__name__, '--run', str(train_run),
                '--output', str(group / 'inference'), '--method', method, '--execute']})
    return planned


def report(group, state, job):
    shared.report_training(group, state, job)
    folder = group / 'train_outputs'
    (folder / 'SCIENTIFIC_CHANGELOG.md').write_text(
        'SD2.1-base512; original/rank64 token-local K/V plus Subject embedding. '
        'Mailbox initializer, unchanged loss/optimizer/LRs/augmentation; training seed42. '
        'New source-cropped27 inputs,600 updates, paired200/400/600 exports. No old weights or resume.\n', encoding='utf8')
    (folder / 'COMPARABILITY_REPORT.md').write_text(
        '24 byte-identical source-framed studio endpoints plus original orange photo and two historical LCh recolors. '
        'Photos retain their original background and distinct historical paired mask. '
        '600 fixed-order updates: first6 rows23 exposures, remaining21 rows22 exposures. '
        'Nine historical Mailbox probes; seeds30-40 inclusive, PNDM100/CFG3.5/512/fp16/checker. '
        'Changed source framing/data/dose are not a single-factor training ablation. '
        'Fullrank means token-local residuals, not full UNet finetuning. Startup is not quality acceptance.\n', encoding='utf8')


def run(output, gpus=(2, 3), execute=False):
    from transformers import AutoTokenizer
    output = resolve_run_dir(Path(output))
    revision = git_info()
    if (output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(contract.SOURCE_ROOT)
            or not gpus or len(set(gpus)) != len(gpus) or any(type(gpu) is not int or gpu < 0 for gpu in gpus)):
        raise ValueError('require fresh external experiment and distinct nonnegative GPUs')
    model = verify_model()
    tokenizer = AutoTokenizer.from_pretrained(str(contract.MODEL_DIR), subfolder='tokenizer', use_fast=False, local_files_only=True)
    initial = tokenizer.encode('mailbox', add_special_tokens=False)
    if len(initial) != 1 or initial[0] in tokenizer.all_special_ids:
        raise ValueError('mailbox must be one ordinary SD2.1 token')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Measured SD2.1 snapshots: fullrank2.1GiB + rank641.8GiB per saved step.
    # Three paired steps ~12GiB; retain >8GiB margin for finals/exports/images.
    if shutil.disk_usage(output.parent).free < 20 * 1024 ** 3:
        raise OSError('require20GiB for six complete snapshots and outputs')
    if execute:
        stats = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
        used = {int(row.split(',')[0]): int(row.split(',')[1]) for row in stats.splitlines()}
        if any(used.get(gpu, 999999) > 2048 for gpu in gpus):
            raise RuntimeError('selected GPU occupied; preserve existing jobs')
    output.mkdir()
    os.environ[contract.DATASET_ENV] = str(output / 'dataset')
    data = contract.stage_dataset(output / 'dataset')
    (output / 'input_review').mkdir()
    reference_review(output / 'input_review', data['references'])
    planned = jobs(output, revision, gpus)
    _write(output / 'authorization.json', {'request': REQUEST, 'git': revision, 'training_authorized': execute,
        'methods': list(contract.METHODS), 'base_model': str(contract.MODEL_DIR), 'steps': 600,
        'checkpoint_steps': list(contract.STEPS), 'generation_seeds': list(inference.SEEDS),
        'source_metadata_unchanged': True, 'monitoring': False,
        'note': 'Current user request authorizes original orange anchor and these27 images; historical dataset-only/held-out flags remain untouched.'})
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
    state = {'status': 'running' if execute else 'dry_run_verified', 'git': revision, 'study': contract.STUDY,
        'jobs': planned, 'model': model, 'initializer_token_id': initial[0], 'training_images': 27,
        'max_train_steps': 600, 'checkpoint_steps': list(contract.STEPS), 'generation_seeds': list(inference.SEEDS),
        'expected_images': 594, 'monitoring': False}
    _write(output / 'experiment.json', state)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>SD2.1 Mailbox Subject27</h1>'
        '<p>Original/rank64;600updates;200/400/600;9prompts;seed30-40.</p>'
        '<a href="input_review/training_references.html">Actual27 RGB and captions</a>'
        + ''.join(f'<p><a href="methods/{method}/inference/index.html">{method} transfer</a></p>' for method in contract.METHODS), encoding='utf8')
    if not execute:
        return state
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(shared.run_queue, gpu, [j for j in planned if j['gpu'] == gpu], output, revision,
                dataset_verifier=contract.verify_dataset, training_reporter=report) for gpu in gpus]
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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[2, 3])
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    run(args.output, tuple(args.gpus), args.execute)
