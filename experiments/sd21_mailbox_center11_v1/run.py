"""Train two center11 methods on reserved GPU0, then paired single-branch transfer."""

import argparse
import os
from pathlib import Path
import subprocess

from experiments.sd21_mailbox_center11_v1 import contract, inference
from experiments.sd21_mailbox_subject27_v1.run import jobs
from experiments.sd21_wood_branches_v1 import run as shared
from experiments.sd21_wood_branches_v1.inference import reference_review
from experiments.sd21_wood_branches_v1.prepare_model import verify_model
from experiments.subject_multicolor_multimaterial_b.training import _write
from scripts.launch.colorpeel_run import git_info, read_config, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]


def report(group, state, job):
    shared.report_training(group, state, job)
    folder = group / 'train_outputs'
    (folder / 'SCIENTIFIC_CHANGELOG.md').write_text(
        'New from-base SD2.1 Subject experiment on the reviewed eleven inputs: eight center renders '
        'and three photographs/recolors.1500 updates, paired500/1000/1500 snapshots. '
        'Unchanged optimizer, LR, seed42,512px FP32 training, mask loss and fixed sample order. '
        'No old learned weights, resume or composition.\n', encoding='utf8')
    (folder / 'COMPARABILITY_REPORT.md').write_text(
        'Fullrank token-local K/V and rank64/alpha64 plus Subject embedding; initializer mailbox. '
        'First four rows137 exposures, other seven136: synthetic1092, photos408. '
        'Removing side views changes both coverage and synthetic/photo exposure ratio. '
        'Nine historical prompts, seeds30-40,PNDM100/CFG3.5/512/fp16/checker. '
        'Only GPU0 is used; SDXL GPUs1/2/3 and checkout are untouched. Startup is not quality acceptance.\n', encoding='utf8')


def require_idle_gpu0():
    stats = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                     '--format=csv,noheader,nounits'], text=True)
    rows = [row.strip().split(',') for row in stats.splitlines()]
    selected = next(row for row in rows if int(row[0]) == 0)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader,nounits'], text=True)
    if int(selected[2]) > 512 or any(row.split(',')[0].strip() == selected[1].strip()
                                    for row in processes.splitlines()):
        raise RuntimeError('GPU0 occupied; do not share it or use SDXL GPUs1/2/3')


def run(output, execute=False):
    from transformers import AutoTokenizer
    output = resolve_run_dir(Path(output))
    revision = git_info()
    if output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(contract.SOURCE_ROOT):
        raise ValueError('require a fresh external experiment directory')
    model = verify_model()
    tokenizer = AutoTokenizer.from_pretrained(str(contract.MODEL_DIR), subfolder='tokenizer',
                                              use_fast=False, local_files_only=True)
    initial = tokenizer.encode('mailbox', add_special_tokens=False)
    if len(initial) != 1 or initial[0] in tokenizer.all_special_ids:
        raise ValueError('mailbox must be one ordinary SD2.1 token')
    if execute:
        require_idle_gpu0()
    output.mkdir(parents=True)
    os.environ[contract.DATASET_ENV] = str(output / 'dataset')
    data = contract.stage_dataset(output / 'dataset')
    (output / 'input_review').mkdir()
    reference_review(output / 'input_review', data['references'])
    planned = jobs(output, revision, (0,), experiment_contract=contract, inference_module=inference)
    _write(output / 'authorization.json', {'request': '使用已确认11图，SD2.1 Subject fullrank和LoRA各1500步，保存500/1000/1500并单独transfer，不影响SDXL。',
        'git': revision, 'training_authorized': execute, 'source_metadata_unchanged': True,
        'checkpoint_steps': list(contract.STEPS), 'generation_seeds': list(inference.SEEDS),
        'gpu': 0, 'sdxl_reserved_gpus': [1, 2, 3], 'monitoring': False,
        'storage': 'Use home/Jiawei; user supplied sufficient capacity, no repeated disk-space query.'})
    for job in planned:
        group = Path(job['group'])
        group.mkdir(parents=True)
        _write(Path(job['config_path']), job['config'])
        if read_config(Path(job['config_path'])) != job['config']:
            raise ValueError('launcher config roundtrip changed')
        preflight = [*job['training_command'][:-1], str(group / 'preflight' / Path(job['run']).name), '--dry-run']
        with (group / 'preflight.log').open('x') as log:
            subprocess.run(preflight, cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'},
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        report(group, {'status': 'queued' if execute else 'dry_run_verified', 'completed_steps': 0,
                      'max_steps': 1500, 'stop_reason': None, 'best_metric': None, 'best_checkpoint': None}, job)
    state = {'status': 'running' if execute else 'dry_run_verified', 'git': revision,
        'jobs': planned, 'model': model, 'training_images': 11, 'max_train_steps': 1500,
        'checkpoint_steps': list(contract.STEPS), 'expected_images': 594,
        'initializer_token_id': initial[0], 'gpu': 0, 'monitoring': False}
    _write(output / 'experiment.json', state)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>SD2.1 Mailbox center11</h1>'
        '<p>Fullrank/rank64;1500 updates;500/1000/1500;9 prompts;seeds30-40;GPU0 serial queue.</p>'
        '<a href="input_review/training_references.html">Actual11 RGB and captions</a>'
        + ''.join(f'<p><a href="methods/{m}/inference/index.html">{m} transfer</a></p>' for m in contract.METHODS), encoding='utf8')
    if not execute:
        return state
    try:
        # Never acquire an SDXL queue's GPU, even between its subprocesses.
        require_idle_gpu0()
        state['queues'] = [shared.run_queue(0, planned, output, revision,
            dataset_verifier=contract.verify_dataset, training_reporter=report)]
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
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    run(args.output, args.execute)
