"""Validate nine Fold runs and derive immutable held-out generation configs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


FOLDS = ("A", "B", "C")
TRAINING_SEEDS = (42, 43, 44)
FINAL_ARTIFACTS = (
    "pytorch_custom_diffusion_weights.bin",
    "<s1*>.bin",
    "<s2*>.bin",
    "<s3*>.bin",
    "<c1*>.bin",
    "<c2*>.bin",
    "<c3*>.bin",
)
EXPECTED_MODIFIER_TOKENS = {"<s1*>", "<s2*>", "<s3*>", "<c1*>", "<c2*>", "<c3*>"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
    return rows


def expected_variant(fold_id: str, seed: int) -> str:
    return f"multiview_v2_fold_{fold_id.lower()}_seed{seed}"


def validate_training_run(run_dir: Path, commit: str) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "succeeded" or manifest.get("returncode") != 0:
        raise ValueError(f"training run is not succeeded: {run_dir}")
    if manifest.get("git", {}).get("commit") != commit:
        raise ValueError(f"training commit mismatch: {run_dir}")
    variant = manifest.get("run", {}).get("variant")
    seed = int(manifest.get("run", {}).get("seed"))
    matches = [
        fold_id
        for fold_id in FOLDS
        if variant == expected_variant(fold_id, seed)
    ]
    if seed not in TRAINING_SEEDS or len(matches) != 1:
        raise ValueError(f"unexpected Fold variant/seed: {variant!r}/{seed}")
    fold_id = matches[0]

    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    args = config["args"]
    locked = {
        "max_train_steps": 1500,
        "cos_weight": 0.2,
        "adam_weight_decay": 0.01,
        "initializer_token": "cube+sphere+cylinder+red+turquoise+gray",
        "mixed_precision": "no",
    }
    for key, expected in locked.items():
        if args.get(key) != expected:
            raise ValueError(f"{run_dir}: config {key} is not locked value {expected!r}")
    if config.get("protocol", {}).get("fold_id") != fold_id:
        raise ValueError(f"{run_dir}: config fold does not match variant")
    if config.get("protocol", {}).get("gt_masks_in_training") is not False:
        raise ValueError(f"{run_dir}: GT-mask training exclusion is not recorded")

    metrics = read_jsonl(run_dir / "checkpoints" / "training_metrics.jsonl")
    if len(metrics) != 1500 or [row.get("step") for row in metrics] != list(range(1, 1501)):
        raise ValueError(f"{run_dir}: training metrics are not steps 1..1500")
    metric_fields = (
        "reconstruction_loss",
        "caa_loss",
        "caa_weighted_loss",
        "total_loss",
        "learning_rate",
    )
    if not all(
        isinstance(row.get(field), (int, float)) and math.isfinite(row[field])
        for row in metrics
        for field in metric_fields
    ):
        raise ValueError(f"{run_dir}: non-finite training metric")

    audit = json.loads(
        (run_dir / "checkpoints" / "embedding_update_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if audit.get("observed_optimization_steps") != 1500:
        raise ValueError(f"{run_dir}: embedding audit step mismatch")
    token_rows = audit.get("modifier_tokens", [])
    token_names = [row.get("token") for row in token_rows]
    if (
        len(token_rows) != 6
        or set(token_names) != EXPECTED_MODIFIER_TOKENS
        or len(token_names) != len(set(token_names))
        or not all(
        row.get("exposure_steps", 0) > 0
        and row.get("nonzero_gradient_steps") == row.get("exposure_steps")
        and isinstance(row.get("initial_final_l2_delta"), (int, float))
        and math.isfinite(row["initial_final_l2_delta"])
        and row["initial_final_l2_delta"] > 0
        for row in token_rows
        )
    ):
        raise ValueError(f"{run_dir}: modifier-token audit failed")
    if audit.get("non_modifier_embedding_drift", {}).get("enforced") is not False:
        raise ValueError(f"{run_dir}: official AdamW drift policy changed")

    checkpoint_dir = run_dir / "checkpoints"
    if not (checkpoint_dir / "checkpoint-1000").is_dir():
        raise ValueError(f"{run_dir}: checkpoint-1000 missing")
    for name in FINAL_ARTIFACTS:
        path = checkpoint_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"{run_dir}: final artifact missing or empty: {name}")
    stdout = (run_dir / "logs" / "stdout.log").read_text(encoding="utf-8")
    for token in ("<s1*>", "<s2*>", "<s3*>", "<c1*>", "<c2*>", "<c3*>"):
        if f"Loaded textual inversion embedding for {token}." not in stdout:
            raise ValueError(f"{run_dir}: post-save reload evidence missing for {token}")

    return {
        "fold_id": fold_id,
        "training_seed": seed,
        "variant": variant,
        "run_dir": str(run_dir.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "training_commit": commit,
        "metrics_rows": 1500,
        "audit_steps": 1500,
        "checkpoint_1000": True,
        "final_artifacts": list(FINAL_ARTIFACTS),
        "reload_succeeded": True,
    }


def discover_training_runs(training_root: Path, commit: str) -> list[dict[str, Any]]:
    inventory = []
    for manifest_path in sorted(training_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        variant = str(manifest.get("run", {}).get("variant", ""))
        if not variant.startswith("multiview_v2_fold_"):
            continue
        if manifest.get("status") != "succeeded":
            continue
        inventory.append(validate_training_run(manifest_path.parent, commit))
    expected = {(fold, seed) for fold in FOLDS for seed in TRAINING_SEEDS}
    actual = {(row["fold_id"], row["training_seed"]) for row in inventory}
    if actual != expected or len(inventory) != len(expected):
        raise ValueError(f"expected Fold/seed matrix {sorted(expected)}, got {sorted(actual)}")
    return sorted(inventory, key=lambda row: (row["fold_id"], row["training_seed"]))


def generation_config(
    row: dict[str, Any], protocol_path: Path
) -> dict[str, Any]:
    fold = row["fold_id"]
    seed = row["training_seed"]
    return {
        "schema_version": 1,
        "status": "ready_after_human_review",
        "stage": "generate_multiview",
        "run": {
            "study": "clevr_subject_color_3x3",
            "variant": f"multiview_v2_heldout_generate_fold_{fold.lower()}_train{seed}",
            "seed": seed,
        },
        "environment": {"CUDA_VISIBLE_DEVICES": "3"},
        "data_manifest": str(protocol_path.resolve()),
        "args": {
            "model-dir": row["checkpoint_dir"],
            "parent-training-run": row["run_dir"],
            "evaluation-protocol": str(protocol_path.resolve()),
            "fold-id": fold,
            "training-seed": seed,
            "pretrained-model-name-or-path": "CompVis/stable-diffusion-v1-4",
            "device": "cuda:0",
            "dtype": "float16",
            "disable-safety-checker": True,
            "acknowledge-safety-risk": True,
            "skip-existing": True,
        },
        "protocol": {
            "evaluation_protocol_id": "clevr_subject_color_3x3_multiview_heldout_bundle_v1",
            "complete_bundle_only": True,
            "generation_seed_start": 42,
            "generation_seed_end_inclusive": 61,
            "prompts": 9,
            "seen_images": 120,
            "held_out_images": 60,
            "expected_images": 180,
            "num_inference_steps": 100,
            "guidance_scale": 6.0,
            "safety_checker_policy": "disabled_after_confirmed_false_positive_with_explicit_acknowledgement",
        },
    }


def bundle_config(training_root: Path, protocol_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "ready_after_generation",
        "stage": "bundle_multiview",
        "run": {
            "study": "clevr_subject_color_3x3",
            "variant": "multiview_v2_heldout_bundle",
            "seed": 42,
        },
        "data_manifest": str(protocol_path.resolve()),
        "args": {
            "generation-root": str(training_root.resolve()),
            "evaluation-protocol": str(protocol_path.resolve()),
        },
        "protocol": {
            "evaluation_protocol_id": "clevr_subject_color_3x3_multiview_heldout_bundle_v1",
            "expected_generation_runs": 9,
            "expected_images": 1620,
            "human_review_is_primary": True,
        },
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def plan(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = discover_training_runs(args.training_root.resolve(), args.training_commit)
    configs_dir = output_dir / "generation_configs"
    configs_dir.mkdir()
    config_rows = []
    for row in inventory:
        config = generation_config(row, args.evaluation_protocol)
        filename = (
            f"fold_{row['fold_id'].lower()}_train{row['training_seed']}.json"
        )
        config_path = configs_dir / filename
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        config_rows.append(
            {
                **row,
                "generation_config": str(config_path.resolve()),
                "generation_variant": config["run"]["variant"],
            }
        )
    write_jsonl(config_rows, output_dir / "training_inventory.jsonl")
    bundle_path = output_dir / "bundle_config.json"
    bundle_path.write_text(
        json.dumps(
            bundle_config(args.training_root, args.evaluation_protocol),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result = {
        "status": "planned",
        "training_commit": args.training_commit,
        "training_runs": 9,
        "generation_runs": 9,
        "images_per_generation_run": 180,
        "expected_images": 1620,
        "seen_images": 1080,
        "held_out_images": 540,
        "generation_configs": [row["generation_config"] for row in config_rows],
        "bundle_config": str(bundle_path.resolve()),
        "training_inventory": str((output_dir / "training_inventory.jsonl").resolve()),
    }
    (output_dir / "evaluation_plan.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(plan(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
