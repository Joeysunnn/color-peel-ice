"""Reuse the fixed material and blue/green transfer probes for spoon snapshots."""

import argparse
from pathlib import Path

from experiments.subject_marble_color30_v1 import inference as baseline
from experiments.subject_trellis_dataset_v1 import inference as previous
from experiments.subject_trellis_spoons_v1 import training

MATERIALS, METHODS, SAMPLING, STEPS = baseline.MATERIALS, baseline.METHODS, baseline.SAMPLING, baseline.STEPS
TRANSFERS, PROBE_LABELS = previous.TRANSFERS, previous.PROBE_LABELS


def prompt_rows(step):
    return [{**row, 'prompt': row['prompt'].replace('<S*> mailbox', '<S*> spoon')}
            for row in previous.prompt_rows(step)]


def blue_green_prompt_rows(step):
    return [{**row, 'prompt': row['prompt'].replace('<S*> mailbox', '<S*> spoon')}
            for row in previous.blue_green_prompt_rows(step)]


def expected_images(dataset):
    count = training.read_json(Path(dataset) / 'dataset_audit.json')['rows']
    if count not in (72, 180):
        raise ValueError('require 180 or blue/green72 spoon images')
    return 36 if count == 72 else 20


def run(dataset, output, step, training_index=None, execute_inference=False, device='cuda:0'):
    audit = training.read_json(Path(dataset) / 'dataset_audit.json')
    subset = audit['rows'] == 72
    expected_images(dataset)
    return baseline.run(dataset, output, step, training_index, execute_inference, device,
        training_module=training, reference_count=audit['rows'], experiment=training.STUDY,
        reference_label=f"TRELLIS.2 {audit['subject_id']} " + ('blue/green72' if subset else 'all180'),
        probe_rows=blue_green_prompt_rows if subset else prompt_rows,
        probe_labels=PROBE_LABELS if subset else MATERIALS,
        baseline_note='Same material probes and sampling; subject noun changed from mailbox to spoon. '
            + ('Blue/green72 retains the four independent context/color probes.' if subset else 'No added probes.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--step', type=int, choices=STEPS, required=True)
    parser.add_argument('--training-index', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--execute-inference', action='store_true')
    args = parser.parse_args()
    if args.dry_run and args.execute_inference:
        parser.error('--dry-run and --execute-inference are mutually exclusive')
    run(args.dataset, args.output, args.step, args.training_index, args.execute_inference, args.device)
