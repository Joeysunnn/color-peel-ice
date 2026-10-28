"""Train the inherited spoon Subject recipes for exactly600 updates."""

import argparse
from pathlib import Path

from experiments.subject_multicolor_multimaterial_b import training as baseline
from experiments.subject_trellis_spoons_v1 import training as previous
from experiments.subject_trellis_spoons600_v1.stage_training import verify_dataset

ROOT, METHODS = previous.ROOT, previous.METHODS
STUDY = 'subject_trellis_spoons600_v1'
STEPS = (200, 400, 600)
ARG_CHANGES = previous.ARG_CHANGES
_write, read_json = baseline._write, baseline.read_json


def build_config(method, dataset):
    audit = read_json(Path(dataset) / 'dataset_audit.json')
    if audit['rows'] != 72:
        raise ValueError('the600-step spoon experiment requires blue/green72 images')
    config = previous.build_config(method, dataset)
    config['status'] = 'subject_trellis_spoons600_prepared'
    config['run'].update(study=STUDY, variant=f"{audit['subject_id']}_blue_green72_{method}_steps600")
    config['args'].update(max_train_steps=600, checkpoint_steps=list(STEPS))
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
