"""Hand off completed Color training's remaining inference to idle GPUs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import shlex
import signal
import subprocess
import sys
import time

from experiments.mailbox_color_sd15_v1 import inference, parallel_job, prompts, training
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

SOURCE = Path('/var/tmp/colorpeel-r12user5/runs/mailbox_color_sd15_v1/'
    '20261030-144030__original_mailbox_color1000__16f9fcc__42/experiment')
SOURCE_CHECKOUT = Path('/var/tmp/colorpeel-r12user5/checkouts/mailbox-color-sd15-v1')
TRAINING_GIT = {'branch': 'codex/mailbox-color-sd15-v1', 'commit': '16f9fcc9544fbdbac562a12c7e1e7c3e7ade7821'}
AUDIT_SCRIPT = '''
import contextlib,json,sys
from pathlib import Path
with contextlib.redirect_stdout(sys.stderr):
    from experiments.mailbox_color_sd15_v1 import inference,training
    from scripts.launch.colorpeel_run import git_info
    root=Path.cwd().resolve()
    if any(not Path(m.__file__).resolve().is_relative_to(root) for m in (inference,training)):
        raise ValueError('source audit imports escaped the frozen checkout')
    source=Path(sys.argv[1]);dataset=source/'training_dataset'
    colors=[];subjects=[];locks={}
    for method in training.METHODS:
        state=inference.read_json(source/'training'/method/'status.json')
        if state.get('status')!='training_verified' or state.get('completed_steps')!=1000:
            raise ValueError('both Color methods must have completed training')
        index=source/'training'/method/'training_index.json'
        colors.append(inference.color_selection(index,dataset,method))
        subject,checked=inference.subject_selection(source/'subject_reference'/'subject_index.json',method)
        subjects.append(subject);locks.update(checked)
        locks[str(index)]=inference.sha256(index)
        locks.update(colors[-1]['hashes'])
    for name in ('concepts.json','training_assets_manifest.jsonl','source_locks.json','dataset_audit.json','approval.json'):
        p=dataset/name;locks[str(p)]=inference.sha256(p)
    locks.update(inference.read_json(dataset/'source_locks.json'))
    locks[str(source/'plan.json')]=inference.sha256(source/'plan.json')
    revision=git_info()
print(json.dumps({'training_git':revision,'color_runs':colors,'subject_runs':subjects,'source_locks':locks},allow_nan=False))
'''


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def source_revision():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(SOURCE_CHECKOUT), *args], text=True).strip()
    if git('status', '--porcelain'):
        raise ValueError('original checkout must remain clean')
    if {'branch': git('rev-parse', '--abbrev-ref', 'HEAD'), 'commit': git('rev-parse', 'HEAD')} != TRAINING_GIT:
        raise ValueError('original training checkout changed')
    allowed = {'experiments/mailbox_color_sd15_v1/parallel.py',
               'experiments/mailbox_color_sd15_v1/parallel_job.py',
               'experiments/mailbox_color_sd15_v1/README.md',
               'tests/test_mailbox_color_parallel.py', 'tests/test_mailbox_color_parallel_job.py'}
    changed = subprocess.check_output(['git', '-C', str(training.ROOT), 'diff',
        TRAINING_GIT['commit'], 'HEAD', '--name-only'], text=True).splitlines()
    if not set(changed) <= allowed:
        raise ValueError('parallel checkout changes existing execution code beyond scheduling')
    for name in ('data.py', 'training.py', 'inference.py', 'prompts.py', 'protocol.json'):
        relative = Path('experiments/mailbox_color_sd15_v1') / name
        if sha256(SOURCE_CHECKOUT / relative) != sha256(training.ROOT / relative):
            raise ValueError('parallel continuation must preserve original training and sampling code')


def source_audit(output):
    source_revision()
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
    for key in ('PYTHONPATH', 'PYTHONHOME', 'GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_COMMON_DIR'):
        env.pop(key, None)
    result = subprocess.run([sys.executable, '-B', '-c', AUDIT_SCRIPT, str(SOURCE)], cwd=SOURCE_CHECKOUT,
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output / 'source_audit.log').write_text(result.stderr, encoding='utf8')
    if result.returncode:
        raise RuntimeError('strict original-revision Color/Subject audit failed; see source_audit.log')
    audit = json.loads(result.stdout)
    if audit['training_git'] != TRAINING_GIT:
        raise ValueError('source audit revision differs')
    for selection in audit['color_runs']:
        for step in training.STEPS:
            directory = Path(selection['checkpoint']) / f'checkpoint-{step}'
            for name in ('pytorch_model.bin', 'pytorch_model_1.bin'):
                p = directory / name
                audit['source_locks'][str(p)] = sha256(p)
    audit.update(source=str(SOURCE), source_code=str(SOURCE_CHECKOUT), inference_git=git_info())
    inference.check_locks(audit['source_locks'])
    return audit


def process_info(pid):
    root = Path('/proc') / str(pid)
    try:
        status = dict(line.split(':', 1) for line in (root / 'status').read_text().splitlines() if ':' in line)
        if status['State'].strip().startswith('Z'):
            return None
        return {'pid': pid, 'uid': int(status['Uid'].split()[0]),
            'ignores_sigint': bool(int(status['SigIgn'].strip(), 16) & (1 << (signal.SIGINT - 1))),
            'args': (root / 'cmdline').read_bytes().decode().strip('\0').split('\0'),
            'starttime': (root / 'stat').read_text().split(') ', 1)[1].split()[19]}
    except FileNotFoundError:
        return None


def stop_original(output):
    """Interrupt only this experiment's inference workers; never training/other users."""
    state = read_json(SOURCE / 'status.json')
    launch = read_json(SOURCE.parent / 'LAUNCH_RECEIPT.json')
    identities = []
    for method in training.METHODS:
        status = read_json(SOURCE / 'training' / method / 'status.json')
        if status.get('status') != 'training_verified' or status.get('completed_steps') != 1000:
            raise ValueError('refuse to interrupt unfinished training')
        pid = state['worker_pids'][method]
        info = process_info(pid)
        if info is None:
            continue
        expected = [sys.executable, '-u', '-B', '-m', 'experiments.mailbox_color_sd15_v1.pipeline',
            '--output', str(SOURCE), '--worker-method', method, '--gpu', str(state['gpus'][training.METHODS.index(method)])]
        if info['uid'] != os.getuid() or info['args'] != expected or info['ignores_sigint']:
            raise ValueError('worker identity, ownership or SIGINT handler differs; no signals sent')
        identities.append(info)
    parent = process_info(launch['coordinator_pid'])
    if parent and (parent['uid'] != os.getuid() or '--output' not in parent['args']
                   or parent['args'][parent['args'].index('--output') + 1] != str(SOURCE)):
        raise ValueError('original coordinator identity differs')
    receipt = {'reason': 'user_requested_idle_GPU_parallel_inference', 'training_restarted': False,
               'signal': 'SIGINT', 'workers': identities, 'parent': parent, 'status': 'handoff_requested'}
    write(output / 'handoff.json', receipt)
    for info in identities:
        if process_info(info['pid']) != info:
            raise ValueError('worker identity changed before signal')
        os.kill(info['pid'], signal.SIGINT)
    deadline = time.monotonic() + 45
    pids = [info['pid'] for info in identities] + ([parent['pid']] if parent else [])
    while any(process_info(pid) is not None for pid in pids):
        if time.monotonic() > deadline:
            raise RuntimeError('handoff still draining; no new inference workers were started')
        time.sleep(1)
    receipt['status'] = 'original_workers_and_coordinator_exited'
    write(output / 'handoff.json', receipt)
    return receipt


def report(output, state):
    write(output / 'status.json', state)
    folder = output / 'repro_outputs'
    folder.mkdir(exist_ok=True)
    write(folder / 'status.json', state)
    notes = {
        'SUMMARY.md': f"Status: {state['status']}. Three-GPU continuation; expected3224 distinct images including reused verified PNGs. No retraining.\n",
        'COMMANDS.md': shlex.join([sys.executable, '-u', '-B', '-m', 'experiments.mailbox_color_sd15_v1.parallel', '--output', str(output), '--gpus', *map(str, state['gpus'])] + (['--execute', '--handoff'] if state['execute'] else [])) + '\n',
        'LOG.md': 'See source_audit.log, source_audit.json, handoff.json, queue_gpuN.json, logs/ and step*/method/provenance.json.\n',
        'SCIENTIFIC_CHANGELOG.md': 'Scheduling only: reuse completed1000-step training, same per-step paired adapters/embeddings, Subject1000,13 prompts,seeds30..60,PNDM100,CFG6,512,fp16,checker. Verify/copy completed ledger-prefix PNGs; generate only missing rows. No retry or source overwrite.\n',
        'COMPARABILITY_REPORT.md': 'Training/inference sources retain original16f9fcc revision, new scheduling code records its own revision. Three separate GPU processes change scheduling only. Interrupted old queues remain preserved; their failed orchestration status does not indicate failed training.\n',
        'PATCHES.md': 'Add audited prefix reuse, explicit SIGINT handoff and independent GPU queues in new checkout; original running checkout unchanged.\n'}
    for name, text in notes.items():
        (folder / name).write_text(text, encoding='utf8')
    links = ''.join(f'<li><a href="step{s}/{m}/index.html">{m}: step{s}</a></li>'
                    for s in training.STEPS for m in training.METHODS)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Mailbox Color parallel continuation</h1>'
        + f'<p>Status: {state["status"]}; same training, prompts and seeds; verified old PNGs reused.</p><ul>{links}</ul>', encoding='utf8')


def run(output, gpus, execute=False, handoff=False):
    output = resolve_run_dir(Path(output))
    if (output.exists() or not output.is_relative_to(SOURCE.parent / 'continuations')
            or len(gpus) != 3 or len(set(gpus)) != 3 or any(g < 0 for g in gpus)
            or (handoff and not execute)):
        raise ValueError('use a fresh continuation below the original experiment container and three distinct GPUs')
    output.mkdir(parents=True)
    state = {'status': 'auditing_sources', 'git': git_info(), 'training_git': TRAINING_GIT,
             'source': str(SOURCE), 'gpus': gpus, 'execute': execute,
             'monitoring': False, 'training_restarted': False, 'expected_images': 3224}
    report(output, state)
    try:
        audit = source_audit(output)
        if handoff:
            stop_original(output)
        else:
            old = read_json(SOURCE / 'status.json')
            if any(process_info(pid) is not None for pid in old.get('worker_pids', {}).values()):
                raise ValueError('source inference workers must exit before snapshotting their ledgers')
        plans = []
        for step in training.STEPS:
            for method in training.METHODS:
                checked = parallel_job.inspect_source(SOURCE, audit, method, step)
                audit['source_locks'].update(checked['source_locks'])
                plans.append({'method': method, 'step': step, 'reused_images': len(checked['rows']),
                              'missing_images': 403 - len(checked['rows'])})
        inference.check_locks(audit['source_locks'])
        write(output / 'source_audit.json', audit)
        write(output / 'job_plan.json', plans)
        state.update(status='dry_run_verified', reused_images=sum(p['reused_images'] for p in plans),
                     pending_images=sum(p['missing_images'] for p in plans))
        report(output, state)
        if not execute:
            return state
        jobs = queue.Queue()
        for plan in plans:
            jobs.put(plan)
        (output / 'logs').mkdir()
        def gpu_queue(gpu):
            records = []
            while True:
                try:
                    job = jobs.get_nowait()
                except queue.Empty:
                    return records
                name = f"{job['method']}_step{job['step']}"
                command = [sys.executable, '-u', '-B', '-m', 'experiments.mailbox_color_sd15_v1.parallel',
                    '--output', str(output), '--job-method', job['method'], '--job-step', str(job['step']), '--gpu', str(gpu)]
                record = {**job, 'gpu': gpu, 'status': 'running', 'command': command}
                records.append(record)
                with (output / 'logs' / (name + '.log')).open('xb') as log:
                    child = subprocess.Popen(command, cwd=training.ROOT,
                        env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONDONTWRITEBYTECODE': '1'},
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    record['pid'] = child.pid
                    write(output / f'queue_gpu{gpu}.json', records)
                    code = child.wait()
                record.update(returncode=code, status='succeeded' if code == 0 else 'failed')
                write(output / f'queue_gpu{gpu}.json', records)
                if code:
                    raise RuntimeError(f'{name} failed without retry; see its log')
        state['status'] = 'inference'
        report(output, state)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(gpu_queue, gpu) for gpu in gpus]
            for future in futures:
                future.result()
        state['status'] = 'execution_verified'
        return state
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report(output, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', nargs=3, type=int)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--handoff', action='store_true')
    parser.add_argument('--job-method', choices=training.METHODS)
    parser.add_argument('--job-step', type=int, choices=training.STEPS)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.job_method:
        if args.gpu is None or args.job_step is None:
            parser.error('job requires --job-step and --gpu')
        root = resolve_run_dir(args.output)
        audit = read_json(root / 'source_audit.json')
        if audit['inference_git'] != git_info():
            raise ValueError('parallel checkout changed')
        parallel_job.run(SOURCE, audit, root / f'step{args.job_step}' / args.job_method,
                         args.job_method, args.job_step, device='cuda:0', execute=True)
    else:
        if args.gpus is None:
            parser.error('provide three --gpus')
        print(json.dumps(run(args.output, args.gpus, args.execute, args.handoff)))
