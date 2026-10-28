"""Run the unchanged spoon72 probes at paired200/400/600 Subject snapshots."""

import argparse
from functools import partial
from pathlib import Path

from experiments.subject_marble_color30_v1 import inference as baseline
from experiments.subject_multicolor_multimaterial_b.inference import export_subject_step
from experiments.subject_trellis_spoons_v1 import inference as previous
from experiments.subject_trellis_spoons600_v1 import training

MATERIALS, METHODS, SAMPLING = previous.MATERIALS, previous.METHODS, previous.SAMPLING
PROBE_LABELS, TRANSFERS = previous.PROBE_LABELS, previous.TRANSFERS
STEPS = (200, 400, 600)
snapshot_exporter = partial(export_subject_step, steps=STEPS, total_steps=600)


def prompt_rows(step):
    if type(step) is not int or step not in STEPS:
        raise ValueError('expected a declared200/400/600 Subject checkpoint')
    return [{**row, 'step': step} for row in previous.blue_green_prompt_rows(500)]


def expected_images(dataset):
    if training.read_json(Path(dataset) / 'dataset_audit.json')['rows'] != 72:
        raise ValueError('require the complete blue/green72 spoon dataset')
    return 36


def run(dataset, output, step, training_index=None, execute_inference=False, device='cuda:0'):
    expected_images(dataset)
    audit = training.read_json(Path(dataset) / 'dataset_audit.json')
    return baseline.run(dataset, output, step, training_index, execute_inference, device,
        training_module=training, reference_count=72, experiment=training.STUDY,
        reference_label=f"TRELLIS.2 {audit['subject_id']} blue/green72;600-step training",
        probe_rows=prompt_rows, probe_labels=PROBE_LABELS, snapshot_exporter=snapshot_exporter,
        baseline_note='Same five material and four independent context/color probes, seeds42/43 and sampling. '
            'Only training dose/checkpoint plan changes to600 updates and200/400/600 paired snapshots.')


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
