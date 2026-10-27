"""Reuse the fixed material-transfer probes for paired TRELLIS180 Subject snapshots."""

import argparse
from pathlib import Path

from experiments.subject_marble_color30_v1 import inference as baseline
from experiments.subject_trellis_dataset_v1 import training

MATERIALS, METHODS, SAMPLING, STEPS = baseline.MATERIALS, baseline.METHODS, baseline.SAMPLING, baseline.STEPS
prompt_rows = baseline.prompt_rows
TRANSFERS = (
    ("in the snow", "context_transfer"),
    ("surrounded by flowers", "context_transfer"),
    ("in white color", "color_transfer"),
    ("in orange color", "color_transfer"),
)
PROBE_LABELS = (*MATERIALS, *(condition for condition, _ in TRANSFERS))


def blue_green_prompt_rows(step):
    rows = prompt_rows(step)
    for method in METHODS:
        for condition, test_type in TRANSFERS:
            slug = condition.replace(" ", "_")
            for seed in SAMPLING["seeds"]:
                rows.append({"id": f"{method}__{slug}__seed{seed}", "method": method,
                    "condition": condition, "test_type": test_type, "seed": seed, "step": step,
                    "prompt": f"a photo of <S*> mailbox {condition}",
                    "image_path": f"images/{method}/transfers/{slug}__seed{seed}.png"})
    return rows


def expected_images(dataset):
    return 36 if training.read_json(Path(dataset) / "dataset_audit.json")["rows"] == 72 else 20


def run(dataset, output, step, training_index=None, execute_inference=False, device="cuda:0"):
    count = training.read_json(Path(dataset) / "dataset_audit.json")["rows"]
    subset = count == 72
    return baseline.run(dataset, output, step, training_index, execute_inference, device,
        training_module=training, reference_count=72 if subset else 180,
        experiment=training.build_config(METHODS[0], dataset)["run"]["study"],
        reference_label="TRELLIS.2 blue/green72" if subset else "TRELLIS.2",
        probe_rows=blue_green_prompt_rows if subset else None,
        probe_labels=PROBE_LABELS if subset else None)


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
