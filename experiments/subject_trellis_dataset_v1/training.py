"""Use unchanged Experiment B recipes for the approved TRELLIS180 Subject dataset."""

import argparse
from pathlib import Path

from experiments.subject_trellis_dataset_v1.stage_training import verify_dataset
from experiments.subject_multicolor_multimaterial_b import training as baseline

ROOT, METHODS = baseline.ROOT, baseline.METHODS
STUDY = "subject_trellis180_training_v1"
_write, read_json = baseline._write, baseline.read_json


def build_config(method, dataset):
    config = baseline.build_config(method, dataset)
    config["status"] = "subject_trellis180_prepared"
    config["run"]["study"] = STUDY
    return config


def build_configuration_bundle(dataset, output):
    return baseline.build_configuration_bundle(dataset, output,
        verify_dataset=verify_dataset, config_builder=build_config)


def run(dataset, output, method, gpu, execute_training=False):
    return baseline.run(dataset, output, method, gpu, execute_training,
        verify_dataset=verify_dataset, config_builder=build_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--execute-training", action="store_true")
    args = parser.parse_args()
    run(args.dataset, args.output, args.method, args.gpu, args.execute_training)
