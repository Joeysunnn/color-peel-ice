"""A bounded, GPU0-only watchdog for the already authorized center11 experiment."""

import argparse
from datetime import datetime
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import time

PARENT = Path('/home/r12user5/Documents/Jiawei/colorpeel-runs/sd21_mailbox_center11_v1/'
              '20261102-172717__center11_both1500__87d6e18__42')
CHECKOUT = Path('/home/r12user5/Documents/Jiawei/colorpeel-checkouts/sd21-mailbox-center11-v1')
PYTHON = '/home/r12user5/miniforge3/envs/colorpeel017/bin/python'
COMMIT = '87d6e18ad340d2c930010c8e79db4e163a08a05f'
BRANCH = 'codex/sd21-mailbox-center11-1500'
METHODS, STEPS = ('fullrank', 'lora_r64'), (500, 1000, 1500)
STATE_FILES = ('pytorch_model.bin', 'pytorch_model_1.bin', 'optimizer.bin',
               'scheduler.bin', 'random_states_0.pkl', 'custom_checkpoint_0.pkl')


def read(path, default=None):
    return json.loads(path.read_text(encoding='utf8')) if path.exists() else default


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def jsonlines(path):
    if not path.exists():
        return []
    # Writers append a newline with each complete record; defer a partial final write.
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines(keepends=True)
            if line.endswith('\n') and line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_info(run):
    metrics = jsonlines(run / 'checkpoints/training_metrics.jsonl')
    finite = all(row.get('losses_finite', {}).get('all') is True
                 and isinstance(row.get('total_loss'), (int, float))
                 and math.isfinite(row['total_loss']) for row in metrics)
    saved = [step for step in STEPS if all((run / f'checkpoints/checkpoint-{step}' / name).is_file()
             and (run / f'checkpoints/checkpoint-{step}' / name).stat().st_size for name in STATE_FILES)]
    manifest = read(run / 'manifest.json', {})
    complete = (manifest.get('status') == 'succeeded' and manifest.get('returncode') == 0
                and manifest.get('git') == {'commit': COMMIT, 'branch': BRANCH}
                and [row.get('step') for row in metrics] == list(range(1, 1501))
                and finite and saved == list(STEPS))
    return {'steps': len(metrics), 'finite': finite, 'checkpoints': saved, 'complete': complete}


def inference_info(output, train_run):
    ledgers = {step: jsonlines(output / f'step{step}/generation_status.jsonl') for step in STEPS}
    state = read(output / 'status.json', {})
    counts = {str(step): len(rows) for step, rows in ledgers.items()}
    complete = state.get('status') == 'succeeded'
    if complete:
        if (counts != {str(step): 99 for step in STEPS} or state.get('generated_images') != 297
                or state.get('completed_steps') != list(STEPS)):
            raise ValueError('successful inference has an incomplete transfer matrix')
        exported = read(output / 'ckpts/index.json', {})
        if (exported.get('training', {}).get('run') != str(train_run)
                or set(exported.get('steps', {})) != {str(step) for step in STEPS}):
            raise ValueError('transfer export does not match selected training and checkpoints')
        for step, rows in ledgers.items():
            planned = jsonlines(output / f'step{step}/generation_manifest.jsonl')
            if len(planned) != 99 or len({row['id'] for row in rows}) != 99:
                raise ValueError('duplicate or missing transfer cases')
            for row, expected in zip(rows, planned):
                path = (output / f'step{step}' / row['image_path']).resolve()
                if (any(row.get(key) != value for key, value in expected.items())
                        or row.get('status') not in ('ok', 'safety_filtered')
                        or not path.is_relative_to((output / f'step{step}/images').resolve())
                        or not path.is_file() or digest(path) != row.get('image_sha256')):
                    raise ValueError('transfer case or PNG checksum mismatch')
            if exported['steps'][str(step)].get('paired_embedding_finite') is not True:
                raise ValueError('paired checkpoint export not verified')
    return {'images': sum(counts.values()), 'per_step': counts, 'complete': complete}


def active_processes(parent, proc_root=Path('/proc')):
    result = []
    for proc in proc_root.iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            if (any(arg.startswith(str(parent) + os.sep) for arg in args)
                    and any('experiments.sd21_mailbox_center11_v1.run' == arg or
                            'experiments.sd21_mailbox_center11_v1.inference' == arg or
                            arg.endswith(('/scripts/launch/colorpeel_run.py', '/src/train/train_colorpeel.py'))
                            for arg in args)):
                result.append(int(proc.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return result


def gpu0_idle():
    stats = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                     '--format=csv,noheader,nounits'], text=True)
    row = next(line.split(',') for line in stats.splitlines() if line.split(',')[0].strip() == '0')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader,nounits'], text=True)
    return int(row[2]) < 512 and not any(line.split(',')[0].strip() == row[1].strip()
                                         for line in processes.splitlines())


def verify_checkout():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(CHECKOUT), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != COMMIT or git('branch', '--show-current') != BRANCH or git('status', '--porcelain'):
        raise ValueError('original execution checkout changed; automatic recovery refused')


def next_action(observed, active, dead_checks, attempts):
    if all(row['training']['complete'] and row['inference']['complete'] for row in observed.values()):
        return 'completed', None
    if any(not row['training']['finite'] for row in observed.values()):
        return 'needs_attention', 'nonfinite training loss; no automatic science changes'
    if active:
        return 'watching', None
    if dead_checks < 2:
        return 'confirming_exit', None
    for method in METHODS:
        for phase in ('training', 'inference'):
            if not observed[method][phase]['complete']:
                if sum(item['method'] == method and item['phase'] == phase for item in attempts) >= 2:
                    return 'needs_attention', f'{method}/{phase}: two recovery attempts exhausted'
                return 'recover', (method, phase)


def fatal_evidence(paths):
    markers = ('ValueError:', 'AssertionError', 'TypeError:', 'KeyError:', 'NameError:', 'AttributeError:', 'ImportError:',
               'ModuleNotFoundError:', 'SyntaxError:', 'FileNotFoundError:', 'No space left on device')
    for path in paths:
        if path.exists():
            tail = path.read_text(encoding='utf8', errors='replace')[-16000:]
            if any(marker in tail for marker in markers):
                return {'log': str(path), 'tail': tail[-4000:]}


def launch_recovery(parent, job, method, phase, record):
    verify_checkout()
    if active_processes(parent) or not gpu0_idle():
        return None
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'PYTHONDONTWRITEBYTECODE': '1',
                   'COLORPEEL_OUTPUT_ROOT': str(PARENT.parents[1]),
                   'COLORPEEL_MAILBOX_CENTER11_DATASET': str(parent / 'experiment/dataset'),
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'DIFFUSERS_OFFLINE': '1'}
    preflight = ('import sys; from pathlib import Path; '
                 'from experiments.sd21_mailbox_center11_v1 import contract; '
                 'from scripts.launch.colorpeel_run import read_config; '
                 'contract.verify_dataset("subject"); '
                 'assert read_config(Path(sys.argv[1])) == contract.build_config("subject", sys.argv[2])')
    subprocess.run([PYTHON, '-B', '-c', preflight, job['config_path'], method], cwd=CHECKOUT,
                   env=environment, stdin=subprocess.DEVNULL, check=True, capture_output=True)
    number = 1 + sum(a['method'] == method and a['phase'] == phase for a in record['attempts'])
    folder = parent / 'recovery' / method / f'{phase}-{number}'
    location = record['locations'][method]
    if phase == 'training':
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        target = folder / (stamp + '__' + Path(job['run']).name.split('__', 1)[1])
        command = [PYTHON, '-B', str(CHECKOUT / 'scripts/launch/colorpeel_run.py'),
                   '--config', job['config_path'], '--run-dir', str(target)]
        location['training'] = str(target)
    else:
        target = folder / 'inference'
        command = [PYTHON, '-B', '-u', '-m', 'experiments.sd21_mailbox_center11_v1.inference',
                   '--run', location['training'], '--output', str(target), '--method', method, '--execute']
        location['inference'] = str(target)
    attempt = {'method': method, 'phase': phase, 'attempt': number, 'output': str(target),
               'command': command, 'log': str(folder / 'subprocess.log'), 'started_at': time.time()}
    record['attempts'].append(attempt)
    write(parent / 'monitor/state.json', record)  # Persist intent before spawning, including crash windows.
    folder.mkdir(parents=True)
    with Path(attempt['log']).open('xb') as log:
        child = subprocess.Popen(command, cwd=CHECKOUT, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    attempt['pid'] = child.pid
    write(parent / 'monitor/state.json', record)
    return child


def run(parent=PARENT):
    import fcntl
    parent = Path(parent).resolve()
    if parent != PARENT:
        raise ValueError('this watchdog is scoped only to the authorized center11 run')
    output = parent / 'monitor'
    output.mkdir(exist_ok=True)
    lock = (output / 'lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(parent / 'experiment/experiment.json')
    if (plan['git'] != {'commit': COMMIT, 'branch': BRANCH} or plan['checkpoint_steps'] != list(STEPS)
            or plan['training_images'] != 11 or any(j['gpu'] != 0 for j in plan['jobs'])):
        raise ValueError('experiment plan differs from the authorized run')
    jobs = {job['method']: job for job in plan['jobs']}
    record = read(output / 'state.json', {'attempts': [], 'locations': {
        m: {'training': jobs[m]['run'], 'inference': str(Path(jobs[m]['group']) / 'inference')} for m in METHODS}})
    dead, previous, last_progress, last_event, child = 0, None, time.time(), None, None
    while True:
        try:
            if child is not None:
                child.poll()  # Reap completed children; /proc handles monitor restarts too.
            observed = {}
            for m, location in record['locations'].items():
                observed[m] = {'training': training_info(Path(location['training'])),
                               'inference': inference_info(Path(location['inference']), Path(location['training']))}
            active = active_processes(parent)
            dead = 0 if active else dead + 1
            status, reason = next_action(observed, active, dead, record['attempts'])
            logs = [Path(j['group']) / 'train_outputs' / f'{phase}.log'
                    for j in jobs.values() for phase in ('training', 'inference')]
            logs += [Path(a['log']) for a in record['attempts']]
            progress = (json.dumps(observed, sort_keys=True), tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size)
                       for p in logs if p.exists()))
            if progress != previous:
                previous, last_progress = progress, time.time()
            if active and time.time() - last_progress > 1800:
                status, reason = 'needs_attention', 'no recorded progress for30 minutes; processes left intact'
            if status == 'recover':
                method, phase = reason
                relevant = [Path(jobs[method]['group']) / 'train_outputs' / f'{phase}.log']
                relevant += [Path(a['log']) for a in record['attempts'] if a['method'] == method and a['phase'] == phase]
                if phase == 'training':
                    relevant.append(Path(record['locations'][method]['training']) / 'logs/stdout.log')
                evidence = fatal_evidence(relevant)
                if evidence:
                    status, reason = 'needs_attention', evidence
                else:
                    child = launch_recovery(parent, jobs[method], method, phase, record)
                    status, reason = ('recovering', {'method': method, 'phase': phase}) if child else ('waiting_gpu0', None)
                    dead = 0
            state = {'status': status, 'reason': reason, 'checked_at': datetime.now().astimezone().isoformat(),
                     'methods': observed, 'active_pids': active, 'locations': record['locations'],
                     'attempts': record['attempts'], 'gpu': 0, 'expected_images': 594, 'interval_seconds': 60}
        except Exception as error:
            state = {'status': 'needs_attention', 'reason': f'{type(error).__name__}: {error}',
                     'checked_at': datetime.now().astimezone().isoformat(), 'locations': record['locations'],
                     'attempts': record['attempts']}
            if isinstance(error, subprocess.CalledProcessError) and error.stderr:
                state['stderr'] = error.stderr.decode(errors='replace')[-4000:] if isinstance(error.stderr, bytes) else error.stderr[-4000:]
        write(output / 'status.json', state)
        event = (state['status'], json.dumps(state.get('reason'), sort_keys=True),
                 tuple((m, info['training']['complete'], info['inference']['complete'], info['training']['checkpoints'])
                       for m, info in state.get('methods', {}).items()))
        if event != last_event:
            with (output / 'events.jsonl').open('a', encoding='utf8') as handle:
                handle.write(json.dumps(state) + '\n')
            last_event = event
        links = ''.join(f'<p>{m}: <a href="{html.escape(os.path.relpath(v["inference"], output))}/index.html">transfer</a></p>'
                        for m, v in record['locations'].items())
        (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>center11 monitor</h1>'
            + '<pre>' + html.escape(json.dumps(state, indent=2)) + '</pre>' + links, encoding='utf8')
        if state['status'] == 'completed':
            return state
        time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=PARENT)
    run(parser.parse_args().parent)
