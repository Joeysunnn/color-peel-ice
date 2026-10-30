"""Audit existing wood spoon tokens, then dispatch four finite composition jobs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys

from experiments.wood_spoon_composition_v1 import inference, source
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('fullrank', 'lora_r64')
BRANCHES = ('material', 'texture')


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def review(output, state, jobs):
    branches = state['branches']
    cells = []
    for job in jobs:
        branch, method = job['branch'], job['method']
        label = f'S{job["subject_step"]}+{branch[0].upper()}{state["attribute_steps"][branch]}: {method}'
        path = f'{job["folder"]}/index.html'
        cells.append(f'<li><a href="{path}">{label}</a></li>')
    comparisons = []
    if state['status'] == 'execution_verified':
        by_condition = {}
        for job in jobs:
            folder = output / job['folder']
            rows = [json.loads(line) for line in (folder / 'generation_status.jsonl').read_text().splitlines()]
            by_condition[job['folder']] = {row['seed']: row for row in rows}
        for seed in range(30, 61):
            comparisons.append(f'<tr><th>seed{seed}</th>')
            for job in jobs:
                row = by_condition[job['folder']][seed]
                image = f'{job["folder"]}/{row["image_path"]}'
                comparisons.append(f'<td><img loading="lazy" src="{html.escape(image, quote=True)}" width="256">'
                    + f'<p>{html.escape(row["status"])}</p></td>')
            comparisons.append('</tr>')
    heading = ' / '.join(f'{b[0].upper()}{state["attribute_steps"][b]}' for b in branches)
    subjects = '/'.join(f'S{s}' for s in state['subject_steps'])
    headers = ''.join(f'<th>S{j["subject_step"]}+{j["branch"][0].upper()} '
        + ('original' if j['method'] == 'fullrank' else 'rank64') + '</th>' for j in jobs)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Wood spoon composition</title>'
        + '<style>body{font:16px sans-serif}td,th{vertical-align:top;padding:8px}img{max-width:100%}</style>'
        + f'<h1>Wood spoon {subjects} + {heading}</h1><p>Status: {state["status"]};{state["expected_images"]} planned images;seeds30–60.</p>'
        + '<p>PNDM100/CFG3.5/512/fp16/checker. '
        + 'Both token embeddings and both matching token-local adapters are loaded. No retraining.</p>'
        + '<ul>' + ''.join(cells) + '</ul><table><tr><th>Seed</th>' + headers
        + '</tr>' + ''.join(comparisons) + '</table>', encoding='utf8')


def report(output, state, commands):
    write(output / 'status.json', state)
    folder = output / 'repro_outputs'
    folder.mkdir(exist_ok=True)
    write(folder / 'status.json', state)
    write(folder / 'COMMANDS.json', commands)
    attributes = ', '.join(f'{b}{s}' for b, s in state['attribute_steps'].items())
    subjects = '/'.join(str(s) for s in state['subject_steps'])
    notes = {
        'SUMMARY.md': f"Status: {state['status']}. Subject{subjects} + {attributes}; original/rank64, {state['expected_images']} images. No training.\n",
        'COMMANDS.md': '\n\n'.join(shlex.join(row['command']) for row in commands) + '\n',
        'LOG.md': 'See source_audit/audit.json, source audit log, queue_gpu*.json, logs/ and each method/branch provenance.json.\n',
        'SCIENTIFIC_CHANGELOG.md': f'Selected inference snapshots: Subject{subjects} + {attributes}. Subject uses spoon initialization/72 images and training total600; Texture uses texture initialization/16 images; selected Material uses material initialization/8 images. Attribute training total remains1000 even for snapshot500. Each branch loads its own same-step adapter and embedding. No training or source edits.\n',
        'COMPARABILITY_REPORT.md': 'Same seeds30–60 across selected compositions and both methods;PNDM100/CFG3.5/512/fp16/checker. S+M uses only S/M, S+T uses only S/T. Sources have different training revisions and durations, recorded separately. The historical second slot named material carries M or T explicitly. No additional controls.\n',
        'PATCHES.md': 'Add read-only source/paired-tensor audits, two-token composition runner and same-seed comparison HTML; existing models/results remain read-only.\n'}
    for name, text in notes.items():
        (folder / name).write_text(text, encoding='utf8')
    review(output, state, commands)


def run(output, gpus, execute=False, *, texture_step=1000, texture_only=False, subject_steps=(600,)):
    if not subject_steps or len(set(subject_steps)) != len(subject_steps):
        raise ValueError('select distinct Subject snapshot steps')
    for step in subject_steps:
        inference.prompt_rows('fullrank', 'texture', texture_step, subject_step=step)
    branches = ('texture',) if texture_only else BRANCHES
    attribute_steps = {b: texture_step if b == 'texture' else 1000 for b in branches}
    output = resolve_run_dir(Path(output))
    if (output.exists() or output.is_relative_to(ROOT) or not gpus
            or len(gpus) > 3 or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus)):
        raise ValueError('use a fresh external output and one to three distinct idle GPUs')
    output.mkdir(parents=True)
    state = {'status': 'auditing_sources', 'git': git_info(), 'gpus': gpus, 'execute': execute,
             'training': False, 'monitoring': False, 'expected_images': 62 * len(branches) * len(subject_steps), 'completed_jobs': [],
             'branches': list(branches), 'attribute_steps': attribute_steps, 'subject_steps': list(subject_steps)}
    jobs = []
    audits = {step: output / 'source_audit' / (f'step{step}' if len(subject_steps) > 1 else '') for step in subject_steps}
    for step in subject_steps:
        for branch in branches:
            for method in METHODS:
                folder = f'{branch}/{method}' if len(subject_steps) == 1 else f'step{step}_{branch}/{method}'
                jobs.append({'branch': branch, 'method': method, 'subject_step': step, 'folder': folder,
                    'command': [sys.executable, '-u', '-B', '-m',
                    'experiments.wood_spoon_composition_v1.inference', '--audit', str(audits[step] / 'audit.json'),
                    '--output', str(output / folder), '--method', method, '--branch', branch,
                    '--device', 'cuda:0', '--attribute-step', str(attribute_steps[branch]),
                    '--subject-step', str(step), '--execute-inference']})
    report(output, state, jobs)
    try:
        for step, audit in audits.items():
            source.prepare(audit, texture_step=texture_step, include_material=not texture_only, subject_step=step)
        for job in jobs:
            inference.run(audits[job['subject_step']] / 'audit.json', output / 'preflight' / job['folder'],
                          job['method'], job['branch'], attribute_step=attribute_steps[job['branch']],
                          subject_step=job['subject_step'])
        if git_info() != state['git']:
            raise ValueError('inference checkout changed during preparation')
        state['status'] = 'dry_run_verified'
        if not execute:
            return state
        (output / 'logs').mkdir()
        pending = queue.Queue()
        for job in jobs:
            pending.put(job)
        def gpu_queue(gpu):
            records = []
            while True:
                try:
                    job = pending.get_nowait()
                except queue.Empty:
                    return records
                record = {**job, 'gpu': gpu, 'status': 'running'}
                records.append(record)
                with (output / 'logs' / (job['folder'].replace('/', '__') + '.log')).open('xb') as log:
                    child = subprocess.Popen(job['command'], cwd=ROOT,
                        env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONDONTWRITEBYTECODE': '1'},
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    record['pid'] = child.pid
                    write(output / f'queue_gpu{gpu}.json', records)
                    code = child.wait()
                provenance = {}
                if code == 0:
                    try:
                        provenance = json.loads((output / job['folder'] / 'provenance.json').read_text())
                    except (OSError, ValueError) as error:
                        record['error'] = f'{type(error).__name__}: {error}'
                good = code == 0 and provenance.get('status') == 'succeeded' and provenance.get('generated_images') == 31
                record.update(returncode=code, status='succeeded' if good else 'failed')
                write(output / f'queue_gpu{gpu}.json', records)
                if not good:
                    raise RuntimeError('composition job failed; no automatic retry')
        state['status'] = 'inference'
        report(output, state, jobs)
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(gpu_queue, gpu) for gpu in gpus]
            completed = [row for future in futures for row in future.result()]
        state.update(status='execution_verified', generated_images=state['expected_images'],
                     completed_jobs=[{'method': row['method'], 'branch': row['branch'],
                                      'subject_step': row['subject_step']} for row in completed])
        return state
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report(output, state, jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--texture-step', type=int, choices=(500, 1000), default=1000)
    parser.add_argument('--texture-only', action='store_true')
    parser.add_argument('--subject-steps', nargs='+', type=int, choices=(200, 400, 600), default=[600])
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.gpus, args.execute,
                         texture_step=args.texture_step, texture_only=args.texture_only, subject_steps=args.subject_steps)))
