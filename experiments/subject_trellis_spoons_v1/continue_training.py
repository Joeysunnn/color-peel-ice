"""Continue only missing spoon jobs; reuse completed180 fullrank read-only."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.subject_multicolor_multimaterial_b.inference import _locks, export_subject_step
from experiments.subject_trellis_dataset_v1 import train_pipeline as pipeline
from experiments.subject_trellis_spoons_v1 import inference, reuse, stage_training, training
from experiments.subject_trellis_spoons_v1.train_pipeline import commands
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

REQUEST = ('修复 spoon 初始化限制，复用已完成的 fullrank，不要重训，继续两把勺子的180张 rank64、'
    '72张 original和rank64训练，各1000步，并完成500/750/1000步的infer。每个实验仍放同一文件夹，'
    '严格本地修改提交推送、服务器拉取运行，旧结果不要动，启动后不用监控。')


def remaining_jobs(dataset, output, gpu):
    jobs = [job for job in commands(dataset, output, gpu) if job['name'] != 'fullrank']
    for job in jobs:
        if job['phase'] == 'inference':
            job['command'].append('--reuse-fullrank')
    return jobs


def run(subject, output, gpu, execute=False):
    output = resolve_run_dir(Path(output))
    source = reuse.SOURCE_ROOT / subject
    if (output.exists() or not output.is_relative_to(source / 'continuations')
            or type(gpu) is not int or gpu < 0):
        raise ValueError('use a fresh continuation directory inside the same subject experiment')
    dataset = source / 'all180/training_dataset'
    git = git_info()
    fullrank, receipt = reuse.completed_fullrank(dataset, source / 'all180/training/fullrank')
    output.mkdir(parents=True)
    report = output / 'train_outputs'
    report.mkdir()
    work = output / 'all180'
    work.mkdir()
    jobs = remaining_jobs(dataset, work, gpu)
    subset = output / 'blue_green72'
    planned = jobs + commands(subset / 'training_dataset', subset, gpu)
    training._write(report / 'COMMANDS.json', planned)
    (report / 'COMMANDS.md').write_text('\n\n'.join(
        f"{job['phase']} {job['name']}:\n" + shlex.join(job['command'])
        + '\nEnvironment: ' + json.dumps(job['environment']) for job in planned) + '\n', encoding='utf8')
    training._write(output / 'approval.json', {'request': REQUEST, 'retrain_completed_fullrank': False,
        'new_steps_per_method': 1000, 'checkpoint_steps': list(inference.STEPS), 'monitoring': False})
    training._write(output / 'fullrank_reuse.json', receipt)
    for name, text in {
        'LOG.md': 'New180 rank64 and inference logs are here. Blue/green72 has its own train_outputs. Old failed logs/status stay historical. Stop on failure; no retry or monitoring.\n',
        'SCIENTIFIC_CHANGELOG.md': 'Permit spoon in the existing Subject LoRA initializer whitelist. No optimizer, loss, precision, seed, rank/alpha, data or sampling changes. Completed180 fullrank and its same-step embeddings are reused without training or editing source files.\n',
        'COMPARABILITY_REPORT.md': 'Mixed-revision comparison: fullrank retains the recorded original commit; newly trained rank64 uses the initializer-validation fix. Both exact source revisions and immutable hashes are recorded in the combined index.72 trains both methods afresh with the inherited recipes. No completion or disentanglement claim follows from startup.\n',
    }.items():
        (report / name).write_text(text, encoding='utf8')
    state = {'status': 'planned', 'git': git, 'subject_id': subject, 'gpu': gpu,
        'mode': 'continue missing jobs' if execute else 'preflight', 'execute': execute,
        'fullrank_reused': True, 'source_fullrank': str(source / 'all180/training/fullrank'),
        'completed_steps': {'all180/fullrank': 1000, 'all180/lora_r64': 0,
                            'blue_green72/fullrank': 0, 'blue_green72/lora_r64': 0},
        'inference_steps': {'all180': [], 'blue_green72': []}, 'active_job': None,
        'max_steps_per_new_method': 1000, 'checkpoint_steps': list(inference.STEPS),
        'monitoring': False, 'best_metric': None, 'best_checkpoint': None, 'stop_reason': None}

    def save():
        training._write(output / 'status.json', state)
        training._write(report / 'status.json', state)
        (report / 'SUMMARY.md').write_text(
            f"Status: {state['status']}. Completed180 fullrank reused, never retrained.\n"
            f"New training steps: {state['completed_steps']}; inference: {state['inference_steps']}.\n"
            f"Active: {state['active_job']}; stop: {state['stop_reason']}.\n", encoding='utf8')

    save()
    try:
        if not execute:
            training.run(dataset, work / 'training/lora_r64', 'lora_r64', gpu)
            for step in inference.STEPS:
                export_subject_step(fullrank, step, work / 'fullrank_export_preflight' / f'step{step}')
                inference.run(dataset, work / f'step{step}', step)
        else:
            for job in jobs:
                if git_info() != git:
                    raise ValueError('continuation checkout changed')
                _locks(receipt['locks'])
                state.update(status=job['phase'], active_job='all180/' + job['name'])
                save()
                with (report / f"{job['name']}.log").open('x', encoding='utf8') as log:
                    result = subprocess.run(job['command'], cwd=training.ROOT,
                        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', **job['environment']},
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"{job['name']} failed; see continuation train_outputs")
                if job['phase'] == 'training':
                    folder = work / 'training/lora_r64'
                    status = training.read_json(folder / 'status.json')
                    index = training.read_json(folder / 'training_index.json')
                    if (status.get('status') != 'training_verified' or status.get('completed_steps') != 1000
                            or index.get('git') != git or len(index.get('runs', [])) != 1):
                        raise ValueError('new rank64 did not complete its audited1000 steps')
                    combined = {'git': git, 'source_gits': {'fullrank': reuse.OLD_GIT, 'lora_r64': git},
                        'runs': [fullrank, index['runs'][0]], 'reuse_receipt': receipt}
                    path = work / 'training_index.json'
                    training._write(path, combined)
                    reuse.selection_loader(path, dataset)
                    state['completed_steps']['all180/lora_r64'] = 1000
                else:
                    step = int(job['name'][4:])
                    proof = training.read_json(work / job['name'] / 'provenance.json')
                    if (proof.get('status') != 'succeeded' or proof.get('generated_images') != 20
                            or proof.get('step') != step or proof.get('sampling') != inference.SAMPLING
                            or proof.get('training_images') != 180 or len(proof.get('contact_sheets', [])) != 3):
                        raise ValueError('incomplete180 paired-checkpoint inference')
                    state['inference_steps']['all180'].append(step)
                save()
        state.update(status='training' if execute else 'preflight', active_job='blue_green72')
        save()
        result = pipeline.run(stage_training.SOURCES[subject]['candidates'], subset, gpu, execute, True,
            stage_module=stage_training, training_module=training, inference_module=inference,
            command_builder=commands, subject_noun='spoon')
        if result['status'] != ('execution_verified' if execute else 'dry_run_verified'):
            raise ValueError('incomplete72 comparison')
        if execute:
            state['completed_steps'].update({'blue_green72/fullrank': 1000, 'blue_green72/lora_r64': 1000})
            state['inference_steps']['blue_green72'] = list(inference.STEPS)
        _locks(receipt['locks'])
        if git_info() != git:
            raise ValueError('continuation checkout changed')
        state.update(status='execution_verified' if execute else 'dry_run_verified', active_job=None,
            stop_reason='all requested missing jobs completed' if execute else 'preflight passed; no training executed')
        return state
    except BaseException as error:
        state.update(status='failed', stop_reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', required=True, choices=tuple(stage_training.SOURCES))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.subject, args.output, args.gpu, args.execute)))
