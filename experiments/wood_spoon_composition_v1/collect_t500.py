"""Copy completed72-image Subject compositions and generate missing180 cases."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
import os
from pathlib import Path
import queue
import subprocess
import sys

from experiments.wood_spoon_composition_v1 import inference, reuse_composition, source180
from experiments.wood_spoon_composition_v1.run import ROOT, write
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

BASE = Path('/var/tmp/colorpeel-r12user5/runs/wood_spoon_composition_v1')


def previous(step, method):
    if step == 600:
        return BASE / '20261030-203708__S600_T500__03583f4__30/experiment/texture' / method
    return BASE / '20261030-221534__S200_S400_T500__6ce851f__30/experiment' / f'step{step}_texture' / method


def jobs(output):
    rows = []
    for count, steps in ((72, (200, 400, 600)), (180, (750, 1000))):
        for step in steps:
            for method in inference.METHODS:
                folder = f's{count}_step{step}/{method}'
                row = {'subject_images': count, 'subject_step': step, 'method': method,
                       'folder': folder, 'kind': 'copied' if count == 72 else 'generated'}
                if count == 180:
                    row['command'] = [sys.executable, '-u', '-B', '-m',
                        'experiments.wood_spoon_composition_v1.inference', '--audit',
                        str(output / 'source_audit' / f'step{step}/audit.json'), '--output', str(output / folder),
                        '--branch', 'texture', '--method', method, '--attribute-step', '500',
                        '--subject-step', str(step), '--subject-images', '180', '--execute-inference']
                rows.append(row)
    return rows


def report(output, state, cases):
    write(output / 'status.json', state)
    repro = output / 'repro_outputs'
    repro.mkdir(exist_ok=True)
    write(repro / 'status.json', state)
    write(repro / 'COMMANDS.json', cases)
    (repro / 'SUMMARY.md').write_text(
        f"Status: {state['status']}. Copy186 existing S72+T500 images; generate124 S180+T500 images. "
        'Both methods matched, seeds30–60. No training or monitoring.\n', encoding='utf8')
    (repro / 'COMPARABILITY_REPORT.md').write_text(
        'S72 snapshots200/400/600 come from a600-update run; S180 snapshots750/1000 come from a1000-update run. '
        'This compares different dataset sizes and training durations. T500, prompt, seeds and sampler stay fixed. '
        'Copied files retain historical provenance unchanged; reuse receipts record byte hashes. '
        'Each new adapter and embedding uses its own same-step snapshot; old fullrank training is reused.\n', encoding='utf8')
    links, headers, images = [], [], []
    for case in cases:
        label = f'{case["subject_images"]} images S{case["subject_step"]} {case["method"]}'
        links.append(f'<li><a href="{case["folder"]}/index.html">{label}</a> ({case["kind"]})</li>')
        headers.append(f'<th>{label}</th>')
    if state['status'] == 'execution_verified':
        ledgers = {c['folder']: {r['seed']: r for r in map(json.loads,
            (output / c['folder'] / 'generation_status.jsonl').read_text().splitlines())} for c in cases}
        for seed in range(30, 61):
            images.append(f'<tr><th>seed{seed}</th>')
            for case in cases:
                row = ledgers[case['folder']][seed]
                path = html.escape(f'{case["folder"]}/{row["image_path"]}', quote=True)
                images.append(f'<td><img loading="lazy" width="256" src="{path}"><p>{row["status"]}</p></td>')
            images.append('</tr>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        + '<title>Wood spoon T500:72/180 Subject comparison</title>'
        + '<style>body{font:16px sans-serif}th,td{padding:8px;vertical-align:top}</style>'
        + f'<h1>Wood spoon Subject + Texture500</h1><p>{state["status"]};310 planned images;'
        + '186 copied +124 new;seeds30–60;PNDM100/CFG3.5/512/fp16/checker.</p><ul>'
        + ''.join(links) + '</ul><table><tr><th>Seed</th>' + ''.join(headers) + '</tr>'
        + ''.join(images) + '</table>', encoding='utf8')


def run(output, gpus, execute=False):
    output = resolve_run_dir(Path(output))
    if (output.exists() or output.is_relative_to(ROOT) or not gpus or len(gpus) > 3
            or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus)):
        raise ValueError('use a fresh external output and one to three distinct available GPUs')
    revision = git_info()
    output.mkdir(parents=True)
    cases = jobs(output)
    state = {'status': 'preparing', 'git': revision, 'gpus': gpus, 'training': False,
             'monitoring': False, 'expected_images': 310, 'copied_images': 0, 'new_images': 124}
    report(output, state, cases)
    try:
        for case in cases:
            if case['kind'] == 'copied':
                reuse_composition.copy_completed(previous(case['subject_step'], case['method']),
                    output / case['folder'], case['subject_step'], case['method'])
                state['copied_images'] += 31
        for step in (750, 1000):
            source180.prepare(output / 'source_audit' / f'step{step}', step)
        pending = queue.Queue()
        for case in cases:
            if case['kind'] != 'generated':
                continue
            inference.run(output / 'source_audit' / f'step{case["subject_step"]}/audit.json',
                output / 'preflight' / case['folder'], case['method'], 'texture',
                attribute_step=500, subject_step=case['subject_step'], subject_images=180)
            pending.put(case)
        if git_info() != revision:
            raise ValueError('execution revision changed')
        state['status'] = 'dry_run_verified'
        if not execute:
            return state
        (output / 'logs').mkdir()
        state['status'] = 'inference'
        report(output, state, cases)
        def worker(gpu):
            records = []
            while True:
                try:
                    case = pending.get_nowait()
                except queue.Empty:
                    return
                record = {**case, 'gpu': gpu, 'status': 'running'}
                records.append(record)
                with (output / 'logs' / (case['folder'].replace('/', '__') + '.log')).open('xb') as log:
                    child = subprocess.Popen(case['command'], cwd=ROOT, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT,
                        env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONDONTWRITEBYTECODE': '1'})
                    record['pid'] = child.pid
                    write(output / f'queue_gpu{gpu}.json', records)
                    code = child.wait()
                proof = json.loads((output / case['folder'] / 'provenance.json').read_text()) if code == 0 else {}
                good = code == 0 and proof.get('status') == 'succeeded' and proof.get('generated_images') == 31
                record.update(returncode=code, status='succeeded' if good else 'failed')
                write(output / f'queue_gpu{gpu}.json', records)
                if not good:
                    raise RuntimeError('new composition failed; no automatic retry')
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(worker, gpu) for gpu in gpus]
            for future in futures:
                future.result()
        state.update(status='execution_verified', generated_images=124, total_images=310)
        return state
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report(output, state, cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpus', required=True, nargs='+', type=int)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.gpus, args.execute)))
