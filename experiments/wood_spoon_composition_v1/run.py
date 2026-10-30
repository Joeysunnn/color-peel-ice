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


def review(output, state):
    cells = []
    for branch in BRANCHES:
        for method in METHODS:
            label = f'S+{ "M" if branch == "material" else "T"}: {method}'
            path = f'{branch}/{method}/index.html'
            cells.append(f'<li><a href="{path}">{label}</a></li>')
    comparisons = []
    if state['status'] == 'execution_verified':
        by_condition = {}
        for branch in BRANCHES:
            for method in METHODS:
                folder = output / branch / method
                rows = [json.loads(line) for line in (folder / 'generation_status.jsonl').read_text().splitlines()]
                by_condition[branch, method] = {row['seed']: row for row in rows}
        for seed in range(30, 61):
            comparisons.append(f'<tr><th>seed{seed}</th>')
            for branch in BRANCHES:
                for method in METHODS:
                    row = by_condition[branch, method][seed]
                    image = f'{branch}/{method}/{row["image_path"]}'
                    comparisons.append(f'<td><img loading="lazy" src="{html.escape(image, quote=True)}" width="256">'
                        + f'<p>{html.escape(row["status"])}</p></td>')
            comparisons.append('</tr>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Wood spoon S+M / S+T</title>'
        + '<style>body{font:16px sans-serif}td,th{vertical-align:top;padding:8px}img{max-width:100%}</style>'
        + f'<h1>Wood spoon S600 + M/T1000</h1><p>Status: {state["status"]};124 planned images;seeds30–60.</p>'
        + '<p>Material initializer=material;Texture initializer=texture;PNDM100/CFG3.5/512/fp16/checker. '
        + 'Both token embeddings and both matching token-local adapters are loaded. No retraining.</p>'
        + '<ul>' + ''.join(cells) + '</ul><table><tr><th>Seed</th><th>S+M original</th><th>S+M rank64</th>'
        + '<th>S+T original</th><th>S+T rank64</th></tr>' + ''.join(comparisons) + '</table>', encoding='utf8')


def report(output, state, commands):
    write(output / 'status.json', state)
    folder = output / 'repro_outputs'
    folder.mkdir(exist_ok=True)
    write(folder / 'status.json', state)
    write(folder / 'COMMANDS.json', commands)
    notes = {
        'SUMMARY.md': f"Status: {state['status']}. Four31-image jobs compare Subject600+Material1000 and Subject600+Texture1000 with original/rank64. No training.\n",
        'COMMANDS.md': '\n\n'.join(shlex.join(row['command']) for row in commands) + '\n',
        'LOG.md': 'See source_audit/audit.json, source audit log, queue_gpu*.json, logs/ and each method/branch provenance.json.\n',
        'SCIENTIFIC_CHANGELOG.md': 'New inference composition only. Source wood Subject600 uses spoon initialization and72 training images; Material1000 uses material initialization and8 images; Texture1000 uses texture initialization and16 images. Each branch retains its independently trained adapter and same-step embedding. No new dataset, optimizer, loss or architecture.\n',
        'COMPARABILITY_REPORT.md': 'Use same seeds30–60 across two compositions and both methods;PNDM100/CFG3.5/512/fp16/checker. S+M uses only S/M, S+T uses only S/T. Sources have different training revisions and durations, recorded separately. No Color adapter or prompt. The historical second slot named material carries M or T explicitly. No native/literal/single-token controls added.\n',
        'PATCHES.md': 'Add read-only source/paired-tensor audits, two-token composition runner and same-seed comparison HTML; existing models/results remain read-only.\n'}
    for name, text in notes.items():
        (folder / name).write_text(text, encoding='utf8')
    review(output, state)


def run(output, gpus, execute=False):
    output = resolve_run_dir(Path(output))
    if (output.exists() or output.is_relative_to(ROOT) or not gpus
            or len(gpus) > 3 or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus)):
        raise ValueError('use a fresh external output and one to three distinct idle GPUs')
    output.mkdir(parents=True)
    state = {'status': 'auditing_sources', 'git': git_info(), 'gpus': gpus, 'execute': execute,
             'training': False, 'monitoring': False, 'expected_images': 124, 'completed_jobs': []}
    jobs = []
    for branch in BRANCHES:
        for method in METHODS:
            jobs.append({'branch': branch, 'method': method, 'command': [sys.executable, '-u', '-B', '-m',
                'experiments.wood_spoon_composition_v1.inference', '--audit', str(output / 'source_audit' / 'audit.json'),
                '--output', str(output / branch / method), '--method', method, '--branch', branch,
                '--device', 'cuda:0', '--execute-inference']})
    report(output, state, jobs)
    try:
        source.prepare(output / 'source_audit')
        for job in jobs:
            inference.run(output / 'source_audit' / 'audit.json', output / 'preflight' / job['branch'] / job['method'],
                          job['method'], job['branch'])
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
                with (output / 'logs' / f"{job['branch']}__{job['method']}.log").open('xb') as log:
                    child = subprocess.Popen(job['command'], cwd=ROOT,
                        env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONDONTWRITEBYTECODE': '1'},
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    record['pid'] = child.pid
                    write(output / f'queue_gpu{gpu}.json', records)
                    code = child.wait()
                provenance = {}
                if code == 0:
                    try:
                        provenance = json.loads((output / job['branch'] / job['method'] / 'provenance.json').read_text())
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
        state.update(status='execution_verified', generated_images=124,
                     completed_jobs=[{'method': row['method'], 'branch': row['branch']} for row in completed])
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
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.gpus, args.execute)))
