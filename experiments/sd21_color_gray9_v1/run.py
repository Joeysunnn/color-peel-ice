"""Two finite Color training/export/transfer queues; no background monitoring."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
import os
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.sd21_color_gray9_v1 import contract, inference
from experiments.sd21_wood_branches_v1 import run as shared
from experiments.sd21_wood_branches_v1.prepare_model import verify_model
from experiments.subject_multicolor_multimaterial_b.training import _write
from scripts.launch.colorpeel_run import git_info, read_config, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]


def jobs(output, revision, gpus, smoke=False, *, contract=contract, inference=inference):
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    result = []
    for number, method in enumerate(contract.METHODS):
        group = output / 'methods' / method
        config = contract.build_config('color', method, smoke=smoke)
        name = f"{stamp}__{contract.STUDY}__{config['run']['variant']}__{revision['commit'][:7]}__42"
        training_run, config_path = group / 'training' / name, group / 'config.json'
        infer = [sys.executable, '-u', '-B', '-m', inference.__name__,
                 '--run', str(training_run), '--output', str(group / 'inference'),
                 '--method', method, '--execute']
        if smoke:
            infer.append('--smoke')
        result.append({'branch': 'color', 'method': method, 'gpu': gpus[number % len(gpus)],
            'group': str(group), 'run': str(training_run), 'config': config, 'config_path': str(config_path),
            'training_command': [sys.executable, '-B', str(ROOT / 'scripts/launch/colorpeel_run.py'),
                                '--config', str(config_path), '--run-dir', str(training_run)],
            'inference_command': infer})
    return result


def report(group, state, job, *, contract=contract, inference=inference):
    shared.report_training(group, state, job)
    folder = group / 'train_outputs'
    (folder / 'SCIENTIFIC_CHANGELOG.md').write_text(
        'Current gray9 input explicitly authorized for SD2.1 Color training. '
        'Inherited Color method, orange initializer, token-local K/V, embedding/adapter learning rates, '
        'optimizer, valid-image-region loss without supplied object masks, fixed sample order. '
        f'No resume/old learned weights. Formal{contract.authorization()["max_train_steps"]} updates and{list(contract.STEPS)} paired snapshots. '
        'Separate smoke2 is a runtime check and never resumed into formal training.\n', encoding='utf8')
    (folder / 'COMPARABILITY_REPORT.md').write_text(
        'Fullrank is token-local K/V residual plus target embedding, not full UNet finetuning. '
        'rank64 uses alpha64 and the same target embedding. SD2.1-base512 OpenCLIP1024 differs from SD1.5. '
        f'Dataset provenance: {contract.SOURCE_NOTE} '
        f'Source captions/order retained;{contract.authorization()["max_train_steps"]} updates over nine images in fixed cycles. '
        f'Color PNDM100/CFG6/512px/fp16/checker and ten historical prompts retained, seeds{list(inference.SEEDS)}. '
        'No Subject composition, no additional loss, no effective hflip despite inherited hflip=true. '
        'Startup verification is not a transfer-quality verdict.\n', encoding='utf8')


def run(output, gpus=(2, 3), smoke=False, execute=False, *, contract=contract, inference=inference):
    from transformers import AutoTokenizer

    output = resolve_run_dir(Path(output))
    revision = git_info()
    if (output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(contract.SOURCE_ROOT)
            or contract.SOURCE_ROOT.is_relative_to(output) or not gpus or len(set(gpus)) != len(gpus)
            or any(type(gpu) is not int or gpu < 0 for gpu in gpus)):
        raise ValueError('require fresh external output and distinct nonnegative GPUs')
    model = verify_model()
    tokenizer = AutoTokenizer.from_pretrained(str(contract.MODEL_DIR), subfolder='tokenizer', local_files_only=True, use_fast=False)
    orange = tokenizer.encode('orange', add_special_tokens=False)
    if len(orange) != 1 or orange[0] in tokenizer.all_special_ids:
        raise ValueError('orange initializer must be one ordinary SD2.1 token')
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < (8 if smoke else getattr(contract, 'MIN_FREE_GIB', 24)) * 1024 ** 3:
        raise OSError('insufficient free space for immutable complete snapshots and outputs')
    if execute:
        stats = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                         '--format=csv,noheader,nounits'], text=True)
        used = {int(row.split(',')[0]): int(row.split(',')[1]) for row in stats.splitlines()}
        if any(used.get(gpu, 999999) > 2048 for gpu in gpus):
            raise RuntimeError('selected GPU is occupied; preserve current processes')
    output.mkdir()
    dataset = output / 'dataset'
    os.environ[contract.DATASET_ENV] = str(dataset)
    data = contract.stage_dataset(dataset)
    planned = jobs(output, revision, gpus, smoke, contract=contract, inference=inference)
    _write(output / 'authorization.json', {**contract.authorization(), 'source': str(contract.SOURCE_ROOT),
        'source_metadata_unchanged': True, 'runtime_smoke': smoke, 'monitoring': False})
    for job in planned:
        Path(job['group']).mkdir(parents=True)
        _write(Path(job['config_path']), job['config'])
        if read_config(Path(job['config_path'])) != job['config']:
            raise ValueError('launcher config roundtrip changed')
    state = {'status': 'running' if execute else 'dry_run_verified', 'git': revision,
             'study': contract.STUDY, 'data': data, 'model': model, 'jobs': planned, 'smoke': smoke,
             'orange_token_id': orange[0], 'training_images': 9, 'max_train_steps': 2 if smoke else contract.authorization()['max_train_steps'],
             'checkpoint_steps': [2] if smoke else list(contract.STEPS),
             'sampling': {**inference.SAMPLING, 'seeds': [30] if smoke else list(inference.SEEDS)},
             'generation_seeds': [30] if smoke else list(inference.SEEDS),
             'expected_images': 2 if smoke else len(contract.METHODS) * len(contract.STEPS) * len(inference.prompt_rows('fullrank', contract.STEPS[0])), 'monitoring': False}
    _write(output / 'experiment.json', state)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>SD2.1 Color gray9</h1>'
        f'<p>Original and rank64; paired{list(contract.STEPS)}; ten Color prompts; seeds{list(inference.SEEDS)}.</p>'
        + ''.join(f'<p><a href="methods/{method}/inference/index.html">{method}: all checkpoints and training references</a></p>'
                  for method in contract.METHODS), encoding='utf8')
    if not execute:
        return state
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(shared.run_queue, gpu, [j for j in planned if j['gpu'] == gpu],
                                   output, revision, smoke=smoke, dataset_verifier=contract.verify_dataset,
                                   training_reporter=partial(report, contract=contract, inference=inference)) for gpu in gpus]
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
    parser.add_argument('--gpus', nargs='+', type=int, default=[2, 3])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    run(args.output, tuple(args.gpus), args.smoke, args.execute)
