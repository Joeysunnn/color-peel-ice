"""Stage, preflight, then run two finite Color-training and transfer queues."""

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from experiments.mailbox_color_sd15_v1 import data, inference, prompts, training
from experiments.sd15_token_local_kv_v1.contract import read_json
from scripts.launch.colorpeel_run import git_info, read_config, resolve_run_dir


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def report(output, state, commands):
    folder = output / 'train_outputs'
    folder.mkdir(exist_ok=True)
    write(output / 'status.json', state)
    write(folder / 'status.json', state)
    write(folder / 'COMMANDS.json', commands)
    (folder / 'COMMANDS.md').write_text('\n\n'.join(shlex.join(row['command']) for row in commands) + '\n', encoding='utf8')
    (folder / 'SUMMARY.md').write_text(
        f"Status: {state['status']}. Two independent1000-update Color runs; checkpoints400/600/800/1000. "
        'Fixed existing Subject1000; no Subject retraining. Planned3224 images. No monitor or automatic retry.\n', encoding='utf8')
    notes = {
        'LOG.md': 'See fullrank.queue.log/lora_r64.queue.log, training/{method}/train_outputs, method status and step{N}/{method}/provenance.json.\n',
        'SCIENTIFIC_CHANGELOG.md': 'Copy the nine original-Mailbox orange Emission images and captions. Migrate SD1.4 to frozen SD1.5;100 to1000 updates; save400/600/800/1000; add rank64/alpha64. Set weight_decay0 instead of0.01 to prevent ordinary embedding drift. Keep embedding LR1e-5, fullrank KV1e-5, batch1,seed42,cos_weight0; LoRA KV5e-5. No new loss or mask. The historical hflip flag remains true but the inherited trainer does not execute flips.\n',
        'COMPARABILITY_REPORT.md': 'This is a base-model, dose and optimizer adaptation, not a base-model-only ablation. Compare fullrank/rank64 on identical data. Test ten original Color prompts plus three S+C prompts,seeds30..60 inclusive,PNDM100/CFG6/512/fp16/checker. Keep Subject1000 fixed and method matched. Historical Subject inference usedCFG3.5; all new probes use prior ColorCFG6. Execution does not prove disentanglement.\n'}
    for name, text in notes.items():
        (folder / name).write_text(text, encoding='utf8')


def worker(output, method, gpu):
    output = resolve_run_dir(Path(output))
    plan = read_json(output / 'plan.json')
    if plan['git'] != git_info() or method not in training.METHODS or gpu != plan['gpus'][training.METHODS.index(method)]:
        raise ValueError('worker method, GPU or checkout differs from the prepared plan')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', DIFFUSERS_OFFLINE='1')
    dataset = output / 'training_dataset'
    state = {'status': 'training', 'method': method, 'gpu': gpu, 'git': plan['git'],
             'max_steps': 1000, 'checkpoint_steps': list(training.STEPS), 'completed_inference_steps': [],
             'expected_images': 1612, 'monitoring': False}
    path = output / f'{method}.status.json'
    write(path, state)
    try:
        result = training.run(dataset, output / 'training' / method, method, gpu, True)
        if result['completed_steps'] != 1000 or result['status'] != 'training_verified':
            raise ValueError('Color training did not pass the completed1000-step audit')
        state.update(status='inference', completed_training_steps=1000)
        write(path, state)
        for step in training.STEPS:
            if git_info() != plan['git']:
                raise ValueError('checkout changed during scheduled inference')
            inference.run(dataset, output / f'step{step}' / method, method, step,
                training_index=output / 'training' / method / 'training_index.json',
                subject_index=output / 'subject_reference' / 'subject_index.json',
                execute_inference=True, device='cuda:0')
            state['completed_inference_steps'].append(step)
            write(path, state)
        state['status'] = 'execution_verified'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(path, state)
    return state


def run(output, gpus, execute=False):
    output = resolve_run_dir(Path(output))
    if (output.exists() or output.is_relative_to(training.ROOT) or len(gpus) != 2
            or any(type(gpu) is not int or gpu < 0 for gpu in gpus) or len(set(gpus)) != 2):
        raise ValueError('require a new external root and two distinct nonnegative GPUs')
    if shutil.disk_usage(output.parent).free < 30 * 1024**3:
        raise ValueError('require30GiB free for paired checkpoints, exports and transfer images')
    revision = git_info()
    output.mkdir(parents=True)
    state = {'status': 'preparing', 'git': revision, 'execute': execute, 'gpus': gpus,
             'training_images': 9, 'max_steps': 1000, 'checkpoint_steps': list(training.STEPS),
             'subject_step': 1000, 'expected_images': 3224, 'monitoring': False,
             'completed_methods': [], 'worker_pids': {}}
    commands = [{'method': method, 'command': [sys.executable, '-u', '-B', '-m',
        'experiments.mailbox_color_sd15_v1.pipeline', '--output', str(output),
        '--worker-method', method, '--gpu', str(gpu)]} for method, gpu in zip(training.METHODS, gpus)]
    report(output, state, commands)
    try:
        data.stage_dataset(output / 'training_dataset')
        bundle = training.build_configuration_bundle(output / 'training_dataset', output / 'training_plan')
        old = read_config(data.LEGACY_CONFIG)['args']
        comparison = {method: {key: {'old': old.get(key), 'new': config['args'].get(key)}
            for key in set(old) | set(config['args']) if old.get(key) != config['args'].get(key)}
            for method, config in bundle['configs'].items()}
        write(output / 'training_plan' / 'LEGACY_COLOR_COMPARISON.json', comparison)
        inference.prepare_subject(output / 'subject_reference')
        for method, gpu in zip(training.METHODS, gpus):
            training.run(output / 'training_dataset', output / 'preflight' / 'training' / method, method, gpu)
            for step in training.STEPS:
                inference.run(output / 'training_dataset', output / 'preflight' / f'step{step}' / method, method, step)
        write(output / 'plan.json', {'git': revision, 'gpus': gpus, 'commands': commands,
            'expected_images': 3224, 'protocol': read_json(Path(__file__).with_name('protocol.json'))})
        if git_info() != revision:
            raise ValueError('checkout changed during preflight')
        if not execute:
            state['status'] = 'preflight_verified'
            return state
        processes = []
        state['status'] = 'running'
        for row in commands:
            with (output / (row['method'] + '.queue.log')).open('xb') as log:
                child = subprocess.Popen(row['command'], cwd=training.ROOT,
                    env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
            processes.append((row['method'], child))
            state['worker_pids'][row['method']] = child.pid
        report(output, state, commands)
        failures = []
        for method, child in processes:
            code = child.wait()
            finished = read_json(output / f'{method}.status.json')
            if code or finished['status'] != 'execution_verified':
                failures.append(method)
            else:
                state['completed_methods'].append(method)
            report(output, state, commands)
        if failures:
            raise RuntimeError(f'queues failed without retry: {failures}')
        state['status'] = 'execution_verified'
        return state
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report(output, state, commands)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', nargs=2, type=int)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker-method', choices=training.METHODS)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.worker_method:
        if args.gpu is None:
            parser.error('--worker-method requires --gpu')
        worker(args.output, args.worker_method, args.gpu)
    else:
        if args.gpus is None:
            parser.error('provide two --gpus')
        print(json.dumps(run(args.output, args.gpus, args.execute)))
