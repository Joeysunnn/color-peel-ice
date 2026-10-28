"""Reuse the frozen Subject recipes, with spoon as the initialization noun."""

import argparse
from pathlib import Path

from experiments.subject_multicolor_multimaterial_b import training as baseline
from experiments.subject_trellis_spoons_v1.stage_training import verify_dataset

ROOT, METHODS = baseline.ROOT, baseline.METHODS
STUDY = 'subject_trellis_spoons_training_v1'
ARG_CHANGES = baseline.ARG_CHANGES | {'initializer_token'}
_write, read_json = baseline._write, baseline.read_json


def build_config(method, dataset):
    audit = read_json(Path(dataset) / 'dataset_audit.json')
    subject, count = audit['subject_id'], audit['rows']
    if subject not in ('metal_spoon', 'wood_spoon') or count not in (72, 180):
        raise ValueError('require a declared spoon subject and 180 or 72 images')
    config = baseline.build_config(method, dataset)
    size = 'blue_green72' if count == 72 else 'all180'
    config.update(status='subject_trellis_spoons_prepared', subject_id=subject)
    config['run'].update(study=STUDY, variant=f'{subject}_{size}_{method}_steps1000')
    config['args']['initializer_token'] = 'spoon'
    return config


def build_configuration_bundle(dataset, output):
    return baseline.build_configuration_bundle(dataset, output, verify_dataset=verify_dataset,
        config_builder=build_config, allowed_arg_changes=ARG_CHANGES)


def run(dataset, output, method, gpu, execute_training=False):
    return baseline.run(dataset, output, method, gpu, execute_training, verify_dataset=verify_dataset,
        config_builder=build_config, allowed_arg_changes=ARG_CHANGES)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--method', required=True, choices=METHODS)
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--execute-training', action='store_true')
    args = parser.parse_args()
    run(args.dataset, args.output, args.method, args.gpu, args.execute_training)
