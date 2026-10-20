"""Reuse Experiment B Subject recipes and launcher for the accepted C dataset."""

import argparse
from pathlib import Path

from experiments.subject_albedo_pbr_c import data
from experiments.subject_multicolor_multimaterial_b import training as baseline
from experiments.sd15_token_local_kv_v1.contract import read_json

STUDY = "subject_albedo_pbr_c"
METHODS = baseline.METHODS
ROOT = baseline.ROOT
_write = baseline._write


def build_config(method, dataset):
    config = baseline.build_config(method, dataset)
    config["status"] = "subject_experiment_c_prepared"
    config["run"]["study"] = STUDY
    return config


def build_configuration_bundle(dataset, output):
    return baseline.build_configuration_bundle(dataset, output,
        verify_dataset=data.verify_dataset, config_builder=build_config)


def run(dataset, output, method, gpu, execute_training=False):
    return baseline.run(dataset, output, method, gpu, execute_training,
        verify_dataset=data.verify_dataset, config_builder=build_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--execute-training", action="store_true")
    args = parser.parse_args()
    run(args.dataset, args.output, args.method, args.gpu, args.execute_training)


if __name__ == "__main__":
    main()
