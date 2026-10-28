"""Train both inherited Subject methods on one spoon's72 images for600 steps."""

import argparse
import json
from pathlib import Path
import sys

from experiments.subject_trellis_dataset_v1 import train_pipeline as shared
from experiments.subject_trellis_spoons600_v1 import inference, stage_training, training


def commands(dataset, output, gpu):
    prefix = [sys.executable, '-B', '-m']
    jobs = []
    for method in training.METHODS:
        jobs.append({'name': method, 'phase': 'training', 'environment': {}, 'command': [
            *prefix, 'experiments.subject_trellis_spoons600_v1.training', '--dataset', str(dataset),
            '--output', str(output / 'training' / method), '--method', method, '--gpu', str(gpu), '--execute-training']})
    for step in inference.STEPS:
        jobs.append({'name': f'step{step}', 'phase': 'inference', 'environment': {'CUDA_VISIBLE_DEVICES': str(gpu)},
            'command': [*prefix, 'experiments.subject_trellis_spoons600_v1.inference', '--dataset', str(dataset),
                '--output', str(output / f'step{step}'), '--step', str(step),
                '--training-index', str(output / 'training_index.json'), '--device', 'cuda:0', '--execute-inference']})
    return jobs


def run(subject, output, gpu, execute=False):
    return shared.run(stage_training.SOURCES[subject]['candidates'], output, gpu, execute, True,
        stage_module=stage_training, training_module=training, inference_module=inference,
        command_builder=commands, subject_noun='spoon', max_steps=600)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', choices=tuple(stage_training.SOURCES), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.subject, args.output, args.gpu, args.execute)))
