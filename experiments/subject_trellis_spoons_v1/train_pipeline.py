"""Run one spoon's all180 and blue/green72 comparisons in one experiment folder."""

import argparse
import json
from pathlib import Path
import shlex
import sys

from experiments.subject_trellis_dataset_v1 import train_pipeline as shared
from experiments.subject_trellis_spoons_v1 import inference, stage_training, training
from scripts.launch.colorpeel_run import git_info, resolve_run_dir


def commands(dataset, output, gpu):
    jobs = shared.commands(dataset, output, gpu)
    for job in jobs:
        job['command'][3] = job['command'][3].replace(
            'subject_trellis_dataset_v1', 'subject_trellis_spoons_v1')
    return jobs


def run(subject, output, gpu, execute=False):
    output = resolve_run_dir(Path(output))
    candidates = stage_training.SOURCES[subject]['candidates']
    if (output.exists() or output.is_relative_to(training.ROOT)
            or output.is_relative_to(stage_training.SOURCE_ROOT.parent)):
        raise ValueError('require a fresh external experiment outside historical sources')
    git = git_info()
    output.mkdir(parents=True)
    report = output / 'train_outputs'
    report.mkdir()
    state = {'status': 'planned', 'git': git, 'subject_id': subject, 'gpu': gpu,
        'mode': 'full kickoff' if execute else 'dry run', 'resume': False,
        'max_steps_per_method': 1000, 'checkpoint_steps': [500, 750, 1000],
        'datasets': {'all180': 180, 'blue_green72': 72}, 'completed_experiments': [],
        'active_experiment': None, 'monitoring': False, 'best_metric': None,
        'best_checkpoint': None, 'stop_reason': None}

    def report_state():
        training._write(report / 'status.json', state)
        training._write(output / 'status.json', state)

    report_state()
    (report / 'SUMMARY.md').write_text(
        'One independent spoon; all180 then blue_green72. Each comparison runs original/fullrank, '
        'rank64/alpha64, then paired500/750/1000 transfer tests. No resume or monitoring. '
        'Only the spoon initializer/class word changes from the prior Subject hyperparameters.\n', encoding='utf8')
    invocation = [sys.executable, '-B', '-m', 'experiments.subject_trellis_spoons_v1.train_pipeline',
        '--subject', subject, '--output', str(output), '--gpu', str(gpu)] + (['--execute'] if execute else [])
    documents = {
        'COMMANDS.md': shlex.join(invocation) + '\nExact child commands are in each comparison/train_outputs/COMMANDS.json.\n',
        'LOG.md': 'See all180 and blue_green72 train_outputs logs and training/{fullrank,lora_r64}. Queue stops on failure; no retry or monitoring.\n',
        'SCIENTIFIC_CHANGELOG.md': 'Independent spoon data; initialization noun and transfer class word change from mailbox to spoon. Original/fullrank and rank64/alpha64 hyperparameters, seed42,1000 steps and paired500/750/1000 snapshots stay fixed.\n',
        'COMPARABILITY_REPORT.md': 'Two data doses share each source geometry:180 images versus its blue/green72 subset. Sequential exposure dose differs. Five material prompts at both doses;72 additionally retains the four prior context/color probes. Seeds42/43,PNDM100,CFG3.5,512px,fp16,safety checker. Startup evidence does not imply completion or disentanglement.\n',
    }
    for name, text in documents.items():
        (report / name).write_text(text, encoding='utf8')
    try:
        for name, subset in (('all180', False), ('blue_green72', True)):
            if git_info() != git:
                raise ValueError('checkout changed during the spoon experiment queue')
            state.update(status='running' if execute else 'planning', active_experiment=name)
            report_state()
            result = shared.run(candidates, output / name, gpu, execute, subset,
                stage_module=stage_training, training_module=training, inference_module=inference,
                command_builder=commands, subject_noun='spoon')
            expected = 'execution_verified' if execute else 'dry_run_verified'
            if result['status'] != expected:
                raise ValueError('spoon comparison did not reach the requested execution state')
            state['completed_experiments'].append(name)
            report_state()
        state.update(status='execution_verified' if execute else 'dry_run_verified',
            active_experiment=None, stop_reason='all requested comparisons completed')
        return state
    except BaseException as error:
        state.update(status='failed', stop_reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        report_state()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', choices=tuple(stage_training.SOURCES), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.subject, args.output, args.gpu, args.execute)))
