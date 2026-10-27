"""Reuse the fixed material-transfer probes for paired TRELLIS180 Subject snapshots."""

import argparse
from pathlib import Path

from experiments.subject_marble_color30_v1 import inference as baseline
from experiments.subject_trellis_dataset_v1 import training

MATERIALS, METHODS, SAMPLING, STEPS = baseline.MATERIALS, baseline.METHODS, baseline.SAMPLING, baseline.STEPS
prompt_rows = baseline.prompt_rows


def run(dataset, output, step, training_index=None, execute_inference=False, device="cuda:0"):
    return baseline.run(dataset, output, step, training_index, execute_inference, device,
        training_module=training, reference_count=180, experiment=training.STUDY,
        reference_label="TRELLIS.2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, choices=STEPS, required=True)
    parser.add_argument("--training-index", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-inference", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.execute_inference:
        parser.error("--dry-run and --execute-inference are mutually exclusive")
    run(args.dataset, args.output, args.step, args.training_index, args.execute_inference, args.device)


if __name__ == "__main__":
    main()
