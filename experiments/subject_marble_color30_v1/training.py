"""Use unchanged Experiment B recipes for approved MARBLE Subject datasets."""

import argparse
from pathlib import Path

from experiments.subject_marble_color30_v1.stage_training import verify_dataset
from experiments.subject_multicolor_multimaterial_b import training as baseline

ROOT, METHODS = baseline.ROOT, baseline.METHODS
STUDY = "subject_marble_color30_training_v1"
_write, read_json = baseline._write, baseline.read_json


def build_config(method, dataset):
    config = baseline.build_config(method, dataset)
    config["status"] = "subject_marble35_prepared"
    config["run"]["study"] = STUDY
    audit_path = Path(dataset) / "dataset_audit.json"
    if audit_path.is_file() and read_json(audit_path).get("source_mode") == "foreground_only":
        config["status"] = "subject_marble_foreground35_prepared"
        config["run"]["study"] = "subject_marble_foreground_training_v1"
        if read_json(audit_path).get("schema") == "marble_subject14_training/v1":
            config["status"] = "subject_marble_foreground_pink_cyan14_prepared"
            config["run"]["study"] = "subject_marble_foreground_pink_cyan_training_v1"
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
