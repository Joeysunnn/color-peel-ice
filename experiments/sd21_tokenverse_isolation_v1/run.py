"""Prepare a reviewed bank, run the real probe, train six arms, then evaluate112 images."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.sd21_background_diagnosis_v1 import run as diagnosis
from experiments.sd21_tokenverse_isolation_v1 import contract
from experiments.subject_multicolor_multimaterial_b.training import _write

sct = diagnosis.sct
ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT = {'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
               'TRANSFORMERS_OFFLINE': '1', 'DIFFUSERS_OFFLINE': '1'}


def require_idle(gpus):
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    indices = {int(row.split(',')[0]): row.split(',')[1].strip() for row in query.splitlines() if row.strip()}
    if not set(gpus).issubset(indices):
        raise ValueError('the fixed experiment GPUs are not available')
    selected = {indices[index] for index in gpus}
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    if any(row.split(',')[0].strip() in selected for row in processes.splitlines()):
        raise RuntimeError('a requested GPU is occupied; no existing process is stopped')


def prepare(output, previous_path):
    from experiments.sd21_tokenverse_isolation_v1 import assets, inference

    output = sct.st.resolve_run_dir(Path(output))
    roots = (ROOT, sct.st.SUBJECT_PARENT, sct.st.source.TEXTURE_ROOT, sct.COLOR_ROOT,
             contract.MODEL_DIR, contract.texture.NATURAL_DATASET, Path(previous_path).resolve().parent)
    if output.exists() or any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots):
        raise ValueError('require a fresh output separate from all existing code, data and model sources')
    require_idle((0,))
    audit = diagnosis.prepare(previous_path)
    contract.configure_data()
    audit['datasets'] = {branch: contract.verify_dataset(branch) for branch in contract.BRANCHES}
    for data in audit['datasets'].values():
        sct.st.source._merge(audit['source_locks'], data['source_locks'])
    audit['new_training_git'] = sct.st.git_info()
    output.mkdir(parents=True)
    state = {'status': 'preparing_bank', 'new_training_git': audit['new_training_git'],
             'branches': {}, 'expected_images': 112, 'generated_images': 0,
             'gpu_mapping': contract.GPUS, 'fresh_training': True}
    sct.write(output / 'status.json', state)
    sct.write(output / 'source_audit.json', audit)
    (output / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in inference.plan()), encoding='utf8')
    try:
        os.environ.update(ENVIRONMENT, CUDA_VISIBLE_DEVICES='0')
        prompts = [row['caption'] for data in audit['datasets'].values() for row in data['references']]
        bank = assets.generate(output / 'bank', concept_prompts=prompts)
        audit['bank'] = bank
        sct.st.source._merge(audit['source_locks'], bank['source_locks'])
        sct.write(output / 'source_audit.json', audit)
        # Copy only visible references for the review; original data stay immutable.
        sct.references(audit, output)
        sct.st.shared.snapshots.check_locks(audit['source_locks'])
        state['status'] = 'awaiting_bank_visual_review'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        sct.write(output / 'status.json', state)
    return state


def checked_experiment(experiment):
    from experiments.sd21_tokenverse_isolation_v1 import assets

    experiment = Path(experiment).resolve()
    audit, state = [sct.st.read_json(experiment / name) for name in ('source_audit.json', 'status.json')]
    if sct.st.git_info() != audit['new_training_git']:
        raise ValueError('the experiment checkout changed')
    assets.verify_bank(audit['bank']['manifest_path'])
    sct.st.shared.snapshots.check_locks(audit['source_locks'])
    return experiment, audit, state


def execute(command, log, gpu):
    with Path(log).open('x', encoding='utf8') as stream:
        subprocess.run(command, cwd=ROOT, env={**os.environ, **ENVIRONMENT, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                       stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, check=True)


def train(experiment):
    experiment, audit, state = checked_experiment(experiment)
    if state['status'] != 'awaiting_bank_visual_review':
        raise ValueError('training requires the fresh prepared bank, not a partial or completed run')
    review_path = experiment / 'bank_review.json'
    review = sct.st.read_json(review_path)
    if (review.get('status') != 'accepted' or review.get('reviewed_count') != 25
            or review.get('manifest_sha256') != sct.sha256(audit['bank']['manifest_path'])
            or review.get('reviewer') != 'assistant_visual_review'):
        raise ValueError('all25 bank images need a caption/content visual review before training')
    sct.st.source._merge(audit['source_locks'], {str(review_path): sct.sha256(review_path)})
    sct.write(experiment / 'source_audit.json', audit)
    require_idle(tuple(contract.GPUS.values()))
    state['status'] = 'gradient_probe'
    sct.write(experiment / 'status.json', state)
    try:
        execute([sys.executable, '-u', '-B', '-m', 'experiments.sd21_tokenverse_isolation_v1.probe',
                 '--experiment', str(experiment)], experiment / 'gradient_probe.log', 0)
        if sct.st.read_json(experiment / 'gradient_probe.json').get('passed') is not True:
            raise ValueError('real joint-canvas gradient probe failed')
        require_idle(tuple(contract.GPUS.values()))
        state['status'] = 'training'
        sct.write(experiment / 'status.json', state)
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')

        def train_branch(branch):
            group = experiment / branch
            group.mkdir()
            result = {}
            for arm in contract.ARMS:
                folder = group / arm
                folder.mkdir()
                config = contract.build_config(branch, arm, audit['bank']['manifest_path'])
                config_path = folder / 'config.json'
                _write(config_path, config)
                from scripts.launch.colorpeel_run import read_config
                if read_config(config_path) != config:
                    raise ValueError('serialized configuration changed numeric types')
                name = f"{stamp}__{contract.STUDY}__{config['run']['variant']}__{audit['new_training_git']['commit'][:7]}__42"
                run = folder / 'training' / name
                branch_state = {'branch': branch, 'arm': arm, 'status': 'training', 'gpu': contract.GPUS[branch], 'run': str(run)}
                sct.write(group / 'status.json', branch_state)
                execute([sys.executable, '-u', '-B', str(ROOT / 'scripts/launch/colorpeel_run.py'),
                         '--config', str(config_path), '--run-dir', str(run)], folder / 'training.log', contract.GPUS[branch])
                # Export happens after every training process has released its GPU.
                result[arm] = str(run)
                branch_state['status'] = 'trained'
                sct.write(group / 'status.json', branch_state)
            return result

        failures, runs = {}, {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(train_branch, branch): branch for branch in contract.BRANCHES}
            for future in as_completed(futures):
                branch = futures[future]
                try:
                    runs[branch] = future.result()
                    state['branches'][branch] = {'status': 'trained', 'runs': runs[branch]}
                except Exception as error:
                    failures[branch] = f'{type(error).__name__}: {error}'
                    state['branches'][branch] = {'status': 'failed', 'error': failures[branch]}
                sct.write(experiment / 'status.json', state)
        if failures:
            raise RuntimeError(f'training failures preserved; other independent branches completed: {failures}')
        state['status'] = 'auditing_exports'
        sct.write(experiment / 'status.json', state)
        sct.st.shared.snapshots.check_locks(audit['source_locks'])
        exports = {}
        for branch in contract.BRANCHES:
            exports[branch] = {}
            for arm, run in runs[branch].items():
                result = sct.st.shared.snapshots.audit_export(run, branch, 'lora_r64',
                    experiment / branch / arm / 'ckpts',
                    experiment_contract=contract.recipe(branch, arm, audit['bank']['manifest_path']))
                exports[branch][arm] = {str(step): item for step, item in result['steps'].items()}
                sct.st.source._merge(audit['source_locks'], result['source_locks'])
                for item in result['steps'].values():
                    sct.st.source._merge(audit['source_locks'], item['hashes'])
            state['branches'][branch]['status'] = 'trained_and_exported'
            sct.write(experiment / 'status.json', state)
        sct.write(experiment / 'exports.json', exports)
        sct.write(experiment / 'source_audit.json', audit)
        state['status'] = 'inference'
        sct.write(experiment / 'status.json', state)
        execute([sys.executable, '-u', '-B', '-m', 'experiments.sd21_tokenverse_isolation_v1.inference',
                 '--experiment', str(experiment)], experiment / 'inference.log', 0)
        completion = sct.st.read_json(experiment / 'inference/status.json')
        if completion['status'] != 'succeeded' or completion['generated_images'] != 112:
            raise ValueError('the fixed112 evaluation did not finish')
        sct.st.shared.snapshots.check_locks(audit['source_locks'])
        state.update(status='succeeded', generated_images=112)
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        sct.write(experiment / 'status.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'train'), required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-audit', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        if args.source_audit is None:
            parser.error('prepare requires --source-audit')
        prepare(args.output, args.source_audit)
    else:
        train(args.output)
