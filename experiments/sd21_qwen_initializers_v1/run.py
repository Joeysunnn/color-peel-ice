"""Prepare Mailbox data, verify four short runs, train four branches and infer."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd21_wood_branches_v1 import snapshots
from experiments.subject_multicolor_multimaterial_b.training import _write
from scripts.launch.colorpeel_run import git_info, read_config, resolve_run_dir
from . import contract, data

ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT = {'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
               'TRANSFORMERS_OFFLINE': '1', 'DIFFUSERS_OFFLINE': '1'}


def require_idle(gpus):
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    indices = {int(row.split(',')[0]): row.split(',')[1].strip() for row in query.splitlines() if row.strip()}
    if not set(gpus).issubset(indices):
        raise ValueError('requested GPUs unavailable')
    selected = {indices[index] for index in gpus}
    query = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    if any(row.split(',')[0].strip() in selected for row in query.splitlines()):
        raise RuntimeError('requested GPU occupied; no existing process is stopped')


def prepare(output, initializer_manifest):
    from transformers import CLIPTokenizer
    from src.train.initializer_token_utils import load_initializer_manifest
    from experiments.sd21_wood_branches_v1.prepare_model import verify_model
    from .inference import plan

    output, initializer_manifest = resolve_run_dir(Path(output)), Path(initializer_manifest).resolve()
    if output.exists() or any(output.is_relative_to(root) or root.is_relative_to(output)
                              for root in (ROOT, contract.MODEL_DIR, initializer_manifest.parent)):
        raise ValueError('require a fresh external output separate from code, model and VLM receipts')
    revision = git_info()
    model = verify_model(contract.MODEL_DIR)
    tokenizer = CLIPTokenizer.from_pretrained(str(contract.MODEL_DIR), subfolder='tokenizer', local_files_only=True)
    initializers = load_initializer_manifest(initializer_manifest, tokenizer, contract.MODEL_DIR)
    if initializers['source']['image_sha256'] != contract.SOURCE_SHA:
        raise ValueError('no audited derived dataset for this new input image')
    output.mkdir(parents=True)
    datasets = data.prepare(output / 'dataset', initializers, tokenizer)
    locks = {**snapshots.model_locks(contract.MODEL_DIR, model), **initializers['source_locks'],
             str(initializer_manifest): sha256(initializer_manifest)}
    for item in datasets.values():
        locks.update(item['source_locks'])
    snapshots.check_locks(locks)
    audit = {'new_training_git': revision, 'initializer_manifest': str(initializer_manifest),
             'initializers': initializers, 'datasets': datasets, 'source_locks': locks,
             'base_model': model['model'], 'auxiliary_objectives': 'all disabled',
             'source_scope': 'single original Mailbox; reused verified derived data; fresh weights'}
    _write(output / 'source_audit.json', audit)
    _write(output / 'generation_plan.json', plan(initializers))
    state = {'status': 'prepared', 'branches': {}, 'generated_images': 0, 'expected_images': 112,
             'fresh_training': True, 'gpu_mapping': contract.GPUS, 'new_training_git': revision}
    _write(output / 'status.json', state)
    return state


def execute(command, log, gpu):
    with Path(log).open('x', encoding='utf8') as stream:
        subprocess.run(command, cwd=ROOT, env={**os.environ, **ENVIRONMENT, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                       stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, check=True)


def export(experiment, branch, run, output, smoke):
    result = snapshots.audit_export(run, branch, 'lora_r64', output, smoke=smoke,
                                    experiment_contract=contract.recipe(experiment, branch))
    audit = read_json(experiment / 'source_audit.json')
    selected = audit['initializers']['branches'][branch]
    path = Path(run) / 'checkpoints/initializer_audit.json'
    initial = read_json(path)
    manifest = Path(audit['initializer_manifest'])
    expected = {'initializer_manifest': str(manifest), 'initializer_manifest_sha256': sha256(manifest),
                'initializer': selected['initializer'], 'initializer_token_id': selected['initializer_token_id']}
    adaptation = read_json(Path(run) / 'checkpoints/adaptation_config.json')
    if (any(initial.get(key) != value or adaptation.get(key) != value for key, value in expected.items())
            or initial.get('modifier_token') != contract.TOKENS[branch]
            or initial.get('initialized_equal_base') is not True or initial.get('initializer_unchanged') is not True
            or initial['base_initializer_sha256'] != initial['modifier_after_copy_sha256']):
        raise ValueError('actual initialized embedding or exported initializer provenance differs')
    provenance = {**expected, 'initializer_audit': str(path), 'initializer_audit_sha256': sha256(path)}
    result['source_locks'][str(path)] = sha256(path)
    result['source_locks'][str(manifest)] = sha256(manifest)
    for step, item in result['steps'].items():
        item['initializer_provenance'] = provenance
        item['hashes'].update({str(path): sha256(path), str(manifest): sha256(manifest)})
        _write(output / f'step{step}/receipt.json', item)
    _write(output / 'index.json', result)
    return result


def train(experiment):
    experiment = Path(experiment).resolve()
    audit, state = [read_json(experiment / name) for name in ('source_audit.json', 'status.json')]
    if state['status'] != 'prepared' or git_info() != audit['new_training_git']:
        raise ValueError('training requires a fresh prepared experiment and unchanged checkout')
    snapshots.check_locks(audit['source_locks'])
    require_idle(tuple(contract.GPUS.values()))
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    report = experiment / 'train_outputs'
    report.mkdir()

    def launch(branch, smoke):
        phase = 'smoke' if smoke else 'formal'
        folder = experiment / phase / branch
        folder.mkdir(parents=True)
        config = contract.build_config(experiment, branch, smoke=smoke)
        config_path = folder / 'config.json'
        _write(config_path, config)
        if read_config(config_path) != config:
            raise ValueError('serialized config changed numeric types')
        name = f"{stamp}__{contract.STUDY}__{config['run']['variant']}__{audit['new_training_git']['commit'][:7]}__42"
        run = folder / 'training' / name
        command = [sys.executable, '-u', '-B', str(ROOT / 'scripts/launch/colorpeel_run.py'),
                   '--config', str(config_path), '--run-dir', str(run)]
        _write(folder / 'command.json', {'command': command, 'gpu': contract.GPUS[branch]})
        execute(command, folder / 'training.log', contract.GPUS[branch])
        return run

    try:
        exports = {}
        for smoke in (True, False):
            state['status'] = 'smoke_training' if smoke else 'formal_training'
            _write(experiment / 'status.json', state)
            runs, failures = {}, {}
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(launch, branch, smoke): branch for branch in contract.BRANCHES}
                for future in as_completed(futures):
                    branch = futures[future]
                    try:
                        runs[branch] = future.result()
                    except Exception as error:
                        failures[branch] = f'{type(error).__name__}: {error}'
            if failures:
                raise RuntimeError(f'branch failures preserved: {failures}')
            state['status'] = 'smoke_auditing' if smoke else 'formal_auditing'
            _write(experiment / 'status.json', state)
            for branch, run in runs.items():
                result = export(experiment, branch, run, run.parent.parent / 'ckpts', smoke)
                audit['source_locks'].update(result['source_locks'])
                for item in result['steps'].values():
                    audit['source_locks'].update(item['hashes'])
                state['branches'][branch] = {'status': 'smoke_verified' if smoke else 'trained_and_exported',
                                             'run': str(run), 'steps': 2 if smoke else contract.STEPS[branch][-1]}
                if not smoke:
                    exports[branch] = {str(step): item for step, item in result['steps'].items()}
                _write(experiment / 'status.json', state)
            _write(experiment / 'source_audit.json', audit)
            require_idle(tuple(contract.GPUS.values()))
        _write(experiment / 'exports.json', exports)
        state['status'] = 'inference'
        _write(experiment / 'status.json', state)
        execute([sys.executable, '-u', '-B', '-m', 'experiments.sd21_qwen_initializers_v1.inference',
                 '--experiment', str(experiment)], experiment / 'inference.log', 0)
        completion = read_json(experiment / 'inference/status.json')
        if completion['status'] != 'succeeded' or completion['generated_images'] != 112:
            raise ValueError('fixed 112-image inference incomplete')
        snapshots.check_locks(audit['source_locks'])
        if git_info() != audit['new_training_git']:
            raise ValueError('checkout changed during experiment')
        state.update(status='succeeded', generated_images=112)
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        _write(experiment / 'status.json', state)
        _write(report / 'status.json', {**state, 'best_checkpoint': None, 'best_metric': None})
        (report / 'SUMMARY.md').write_text(
            f"Status: {state['status']}. Four 2-step startup checks, then fresh S/M/T1000 and C150. "
            "Paired midpoint/final exports. No auxiliary preservation or TokenVerse loss. "
            "Technical completion does not establish visual quality or background improvement.\n", encoding='utf8')
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'train'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initializer-manifest', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        if args.initializer_manifest is None:
            parser.error('prepare requires --initializer-manifest')
        prepare(args.output, args.initializer_manifest)
    else:
        train(args.output)
