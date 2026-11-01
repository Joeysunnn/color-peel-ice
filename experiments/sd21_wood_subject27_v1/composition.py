"""Compose Subject27 at400/600 with the previous Texture600, matching methods."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.sd21_wood_branches_v1 import composition as shared
from experiments.sd21_wood_subject27_v1 import contract, composition_source
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
STEPS = (400, 600)
STUDY = 'sd21_wood27_composition_v1'
PAUSE_RECEIPT = Path('/var/tmp/colorpeel-r12user5/runs/sd21_wood27_composition_v1/20261101-131914__transfer_pause.json')


def jobs(output, gpus):
    result = []
    for step in STEPS:
        for method in contract.METHODS:
            group = output / f'S{step}' / method
            command = [sys.executable, '-u', '-B', '-m',
                       'experiments.sd21_wood_subject27_v1.composition', '--audit',
                       str(output / 'source_audits' / f'S{step}.json'), '--output', str(group),
                       '--subject-step', str(step), '--method', method]
            result.append({'subject_step': step, 'texture_step': 600, 'method': method,
                           'gpu': gpus[len(result) % len(gpus)], 'output': str(group), 'command': command,
                           'status': 'queued'})
    return result


def review(output, planned):
    heads = ''.join(f'<th>S{job["subject_step"]} / {job["method"]}</th>' for job in planned)
    rows = ''.join(f'<tr><th>{seed}</th>' + ''.join(
        f'<td><a href="S{job["subject_step"]}/{job["method"]}/images/seed{seed}.png">'
        f'<img loading="lazy" width="320" src="S{job["subject_step"]}/{job["method"]}/images/seed{seed}.png"></a></td>'
        for job in planned) + '</tr>' for seed in shared.SEEDS)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>SD2.1 Subject27 + Texture600</h1>'
        f'<p>{shared.PROMPT}; seeds30–60; PNDM100 / CFG3.5 /512px/fp16/checker.</p>'
        '<p>Four finite31-image cases. See status.json and gpu_N.json for completion.</p>'
        + ''.join(f'<p><a href="S{job["subject_step"]}/{job["method"]}/index.html">'
                  f'S{job["subject_step"]}+T600 / {job["method"]}: results and actual training references</a></p>' for job in planned)
        + f'<table><tr><th>seed</th>{heads}</tr>{rows}</table>', encoding='utf8')


def run(output, gpus=(2, 3), execute=False):
    output = resolve_run_dir(Path(output))
    if (output.exists() or output.is_relative_to(ROOT) or not gpus or len(set(gpus)) != len(gpus)
            or any(type(gpu) is not int or gpu < 0 for gpu in gpus)):
        raise ValueError('use a fresh external output and distinct nonnegative GPUs')
    audits = composition_source.prepare()
    if execute:
        stats = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
        used = {int(row.split(',')[0]): int(row.split(',')[1]) for row in stats.splitlines()}
        if any(used.get(gpu, 999999) > 2048 for gpu in gpus):
            raise ValueError('selected GPU is occupied; preserve other tasks')
    output.mkdir(parents=True)
    (output / 'source_audits').mkdir()
    for step in STEPS:
        shared.write(output / 'source_audits' / f'S{step}.json', audits[step])
    if PAUSE_RECEIPT.is_file():
        shutil.copyfile(PAUSE_RECEIPT, output / 'TRANSFER_PAUSE.json')
    planned = jobs(output, gpus)
    state = {'status': 'running' if execute else 'dry_run_verified', 'git': git_info(),
             'subject_images': 27, 'subject_steps': list(STEPS), 'texture_step': 600,
             'seeds': list(shared.SEEDS), 'expected_images': 124, 'training': False,
             'monitoring': False, 'jobs': planned}
    shared.write(output / 'status.json', state)
    review(output, planned)
    def queue(gpu):
        selected = [job for job in planned if job['gpu'] == gpu]
        path = output / f'gpu_{gpu}.json'
        try:
            for job in selected:
                job['status'] = 'running'
                shared.write(path, {'gpu': gpu, 'jobs': selected})
                command = job['command'] + (['--execute'] if execute else [])
                job['execution_command'] = command
                with (output / f'S{job["subject_step"]}_{job["method"]}.log').open('x') as log:
                    result = subprocess.run(command, cwd=ROOT, env={**os.environ,
                        'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONDONTWRITEBYTECODE': '1'},
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                job['status'] = 'succeeded' if result.returncode == 0 else 'failed'
                shared.write(path, {'gpu': gpu, 'jobs': selected})
                if result.returncode:
                    raise RuntimeError(f'S{job["subject_step"]}/{job["method"]} failed; see log')
        except BaseException as error:
            shared.write(path, {'gpu': gpu, 'jobs': selected, 'error': f'{type(error).__name__}: {error}'})
            raise
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
        shared.write(output / 'status.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpus', nargs='+', type=int, default=[2, 3])
    parser.add_argument('--method', choices=contract.METHODS)
    parser.add_argument('--subject-step', type=int, choices=STEPS)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.method:
        if args.audit is None or args.subject_step is None:
            parser.error('--method requires --audit and --subject-step')
        shared.generate(args.audit, args.output, args.method, execute=args.execute,
                        subject_step=args.subject_step, subject_images=27)
    else:
        run(args.output, tuple(args.gpus), args.execute)
