"""Eight paired seed42 controls, reusing four existing images read-only."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
import os
from pathlib import Path
import queue
import subprocess
import sys

from experiments.wood_spoon_composition_v1 import control_inference as inference, control_reuse, source
from experiments.wood_spoon_composition_v1.run import ROOT, write
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

GROUPS = ('mug', 'spoon', 'wood_spoon', 'subject')


def review(output, state):
    body = []
    references = ''
    reference_path = output / 'training_references.json'
    if reference_path.exists():
        for branch, rows in json.loads(reference_path.read_text())['references'].items():
            references += f'<details><summary>{branch}: {len(rows)} actual training images; applies only to arms loading this concept</summary>'
            references += ''.join(f'<figure><img loading="lazy" width="128" src="{html.escape(r["review_image"], quote=True)}">'
                + f'<figcaption>{html.escape(r["caption"])}</figcaption></figure>' for r in rows) + '</details>'
    for group in GROUPS:
        body.append(f'<tr><th>{group}</th>')
        for suffix in ('without_t', 'with_t'):
            case = group + '_' + suffix
            folder = output / 'cases' / case
            ledger = folder / 'generation_status.jsonl'
            rows = [json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
            origin = state['origins'].get(case, 'pending')
            prompt = html.escape(inference.CASES[case]['prompt'])
            image = f'<img width="384" src="cases/{case}/{rows[0]["image_path"]}">' if rows else '<p>Pending</p>'
            body.append(f'<td>{image}<p>{prompt}</p><p>{origin}</p><a href="cases/{case}/index.html">Provenance</a></td>')
        body.append('</tr>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Texture500 seed42 controls</title>'
        + '<style>body{font:16px sans-serif}td,th{vertical-align:top;padding:12px}img{max-width:100%}</style>'
        + f'<h1>Texture500 / rank64 / seed42</h1><p>{state["status"]};8 cases;Subject72 step600.</p>'
        + '<p>PNDM100 / CFG3.5 /512px/fp16/checker. No-T removes the T embedding and adapter as well as the token. '
        + 'The Subject row retains S600 in both columns. Existing outputs are copied byte-for-byte.</p>'
        + '<table><tr><th>Subject</th><th>Without T</th><th>With T500</th></tr>' + ''.join(body) + '</table>'
        + references, encoding='utf8')


def report(output, state, jobs):
    write(output / 'status.json', state)
    folder = output / 'repro_outputs'
    folder.mkdir(exist_ok=True)
    write(folder / 'status.json', state)
    write(folder / 'COMMANDS.json', jobs)
    (folder / 'SUMMARY.md').write_text(f"Status: {state['status']};8 matched cases,4 reused and4 generated. "
        'Only rank64,seed42,S72 step600,T500. No training, export or monitoring.\n', encoding='utf8')
    (folder / 'COMPARABILITY_REPORT.md').write_text(
        'Same frozen SD1.5,PNDM100,CFG3.5,512px,fp16,checker and fresh device generator per image. '
        'No-T ordinary nouns use the same unadapted common loader as the historical mug baseline; '
        'the learned Subject no-T arm keeps only S600. Copied images retain their historical provenance. '
        'This evaluates complete T concept removal, not a token-only or K/V-only ablation. '
        'One seed is diagnostic and does not establish general reliability.\n', encoding='utf8')
    review(output, state)


def run(output, gpus, execute=False):
    output = resolve_run_dir(Path(output))
    if (output.exists() or output.is_relative_to(ROOT) or not gpus or len(gpus) > 2
            or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus)):
        raise ValueError('use a fresh external experiment root and one or two available GPUs')
    revision = git_info()
    output.mkdir(parents=True)
    state = {'status': 'preparing', 'git': revision, 'gpus': gpus, 'seed': 42, 'method': 'lora_r64',
             'subject_images': 72, 'subject_step': 600, 'texture_step': 500, 'expected_images': 8,
             'origins': {}, 'training': False, 'monitoring': False}
    jobs = []
    report(output, state, jobs)
    try:
        audit_path = output / 'source_audit/audit.json'
        audit = source.prepare(audit_path.parent, texture_step=500, include_material=False)
        _, reference_locks = inference.references(audit, output, inference.prompt_row('subject_with_t'))
        for case in inference.CASES:
            receipt = control_reuse.copy_existing(output / 'cases' / case, case, audit)
            if receipt is not None:
                state['origins'][case] = 'copied'
                continue
            state['origins'][case] = 'new inference'
            inference.run(audit_path, output / 'preflight' / case, case)
            jobs.append({'case': case, 'command': [sys.executable, '-u', '-B', '-m',
                'experiments.wood_spoon_composition_v1.control_inference', '--audit', str(audit_path),
                '--output', str(output / 'cases' / case), '--case', case, '--execute-inference']})
        if len(jobs) != 4 or list(state['origins'].values()).count('copied') != 4:
            raise ValueError('expected four exact historical matches and four missing controls')
        if git_info() != revision:
            raise ValueError('execution revision changed')
        state['status'] = 'dry_run_verified'
        if not execute:
            return state
        pending = queue.Queue()
        for job in jobs:
            pending.put(job)
        (output / 'logs').mkdir()
        state['status'] = 'inference'
        report(output, state, jobs)
        def worker(gpu):
            records = []
            while True:
                try:
                    job = pending.get_nowait()
                except queue.Empty:
                    return
                record = {**job, 'gpu': gpu, 'status': 'running'}
                records.append(record)
                with (output / 'logs' / (job['case'] + '.log')).open('xb') as log:
                    child = subprocess.Popen(job['command'], cwd=ROOT, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT,
                        env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONDONTWRITEBYTECODE': '1'})
                    record['pid'] = child.pid
                    write(output / f'queue_gpu{gpu}.json', records)
                    code = child.wait()
                proof_path = output / 'cases' / job['case'] / 'provenance.json'
                proof = json.loads(proof_path.read_text()) if code == 0 else {}
                good = code == 0 and proof.get('status') == 'succeeded' and proof.get('generated_images') == 1
                record.update(returncode=code, status='succeeded' if good else 'failed')
                write(output / f'queue_gpu{gpu}.json', records)
                if not good:
                    raise RuntimeError('control inference failed; no automatic retry')
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(worker, gpu) for gpu in gpus]
            for future in futures:
                future.result()
        if reference_locks:
            source.verify_locks(reference_locks)
        state.update(status='execution_verified', copied_images=4, generated_images=4, total_images=8)
        return state
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report(output, state, jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpus', required=True, nargs='+', type=int)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.gpus, args.execute)))
