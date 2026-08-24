"""Locked complete-bundle evaluation protocol for CLEVR multiview folds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


EVALUATION_PROTOCOL_ID = "clevr_subject_color_3x3_multiview_heldout_bundle_v1"
EXPECTED_RENDER_PROTOCOL_ID = "clevr_subject_color_3x3_multiview_v2"
DEFAULT_GENERATION_SEEDS = tuple(range(42, 62))
DEFAULT_INFERENCE_STEPS = 100
DEFAULT_GUIDANCE_SCALE = 6.0
EXPECTED_DTYPE = "float16"
EXPECTED_SAFETY_POLICY = (
    "disabled_after_confirmed_false_positive_with_explicit_acknowledgement"
)

SUBJECTS = (
    ("s1", "<s1*>", "cube"),
    ("s2", "<s2*>", "sphere"),
    ("s3", "<s3*>", "cylinder"),
)
COLORS = (
    ("c1", "<c1*>", "red"),
    ("c2", "<c2*>", "cyan"),
    ("c3", "<c3*>", "gray"),
)


def read_evaluation_protocol(path: Path) -> dict[str, Any]:
    path = Path(path)
    raw = path.read_bytes()
    protocol = json.loads(raw.decode("utf-8"))
    if protocol.get("schema_version") != 1:
        raise ValueError("evaluation protocol schema version must be 1")
    if protocol.get("protocol_id") != EVALUATION_PROTOCOL_ID:
        raise ValueError(
            "expected multiview evaluation protocol "
            f"{EVALUATION_PROTOCOL_ID!r}, got {protocol.get('protocol_id')!r}"
        )
    if protocol.get("source_render_protocol_id") != EXPECTED_RENDER_PROTOCOL_ID:
        raise ValueError("evaluation protocol does not reference multiview render v2")
    if protocol.get("training_seeds") != [42, 43, 44]:
        raise ValueError("evaluation protocol training seeds must be 42, 43, 44")
    if protocol.get("generation_seeds") != list(DEFAULT_GENERATION_SEEDS):
        raise ValueError("evaluation protocol generation seeds must be 42 through 61")
    expected_subjects = [
        {"id": item[0], "token": item[1], "label": item[2]} for item in SUBJECTS
    ]
    expected_colors = [
        {"id": item[0], "token": item[1], "label": item[2]} for item in COLORS
    ]
    if protocol.get("subjects") != expected_subjects:
        raise ValueError("evaluation protocol subject definitions changed")
    if protocol.get("colors") != expected_colors:
        raise ValueError("evaluation protocol color definitions changed")
    if protocol.get("prompt_template") != (
        "a photo of {subject_token} shape in {color_token} color"
    ):
        raise ValueError("evaluation prompt template changed")
    sampling = protocol.get("sampling", {})
    if sampling.get("num_inference_steps") != DEFAULT_INFERENCE_STEPS:
        raise ValueError("evaluation protocol must use 100 inference steps")
    if sampling.get("guidance_scale") != DEFAULT_GUIDANCE_SCALE:
        raise ValueError("evaluation protocol must use guidance scale 6.0")
    if sampling.get("dtype") != EXPECTED_DTYPE:
        raise ValueError("evaluation protocol must use float16 sampling")
    if sampling.get("safety_checker_policy") != EXPECTED_SAFETY_POLICY:
        raise ValueError("evaluation protocol safety-checker policy changed")
    expected = protocol.get("expected", {})
    locked_expected = {
        "checkpoints": 9,
        "prompts_per_checkpoint": 9,
        "images_per_checkpoint": 180,
        "seen_images_per_checkpoint": 120,
        "held_out_images_per_checkpoint": 60,
        "total_images": 1620,
        "total_seen_images": 1080,
        "total_held_out_images": 540,
    }
    if expected != locked_expected:
        raise ValueError("evaluation protocol expected-count contract changed")
    folds = protocol.get("folds")
    if not isinstance(folds, list) or len(folds) != 3:
        raise ValueError("multiview protocol must define exactly three folds")
    if {fold.get("id") for fold in folds} != {"A", "B", "C"}:
        raise ValueError("multiview protocol must define folds A, B, and C")
    all_held_out = [
        cell
        for fold_id in ("A", "B", "C")
        for cell in held_out_cells(protocol, fold_id)
    ]
    expected_cells = {
        (subject[2], color[2]) for subject in SUBJECTS for color in COLORS
    }
    if len(all_held_out) != 9 or set(all_held_out) != expected_cells:
        raise ValueError("the three folds must hold out every cell exactly once")
    protocol["_source_sha256"] = hashlib.sha256(raw).hexdigest()
    return protocol


def held_out_cells(protocol: dict[str, Any], fold_id: str) -> set[tuple[str, str]]:
    fold_id = str(fold_id).upper()
    matches = [fold for fold in protocol["folds"] if fold.get("id") == fold_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate fold id: {fold_id!r}")
    cells = {tuple(cell) for cell in matches[0].get("held_out", [])}
    valid_cells = {
        (subject[2], color[2]) for subject in SUBJECTS for color in COLORS
    }
    if len(cells) != 3 or not cells <= valid_cells:
        raise ValueError(f"fold {fold_id} must contain three valid held-out cells")
    if {shape for shape, _ in cells} != {subject[2] for subject in SUBJECTS}:
        raise ValueError(f"fold {fold_id} must hold out every subject exactly once")
    if {color for _, color in cells} != {color[2] for color in COLORS}:
        raise ValueError(f"fold {fold_id} must hold out every color exactly once")
    return cells


def build_manifest(
    protocol: dict[str, Any],
    *,
    fold_id: str,
    training_seed: int,
    generation_seeds: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """Build nine complete prompts × twenty paired generation seeds."""

    fold_id = str(fold_id).upper()
    training_seed = int(training_seed)
    if training_seed not in protocol["training_seeds"]:
        raise ValueError(f"unsupported training seed: {training_seed}")
    if generation_seeds is None:
        generation_seeds = protocol["generation_seeds"]
    generation_seeds = tuple(int(seed) for seed in generation_seeds)
    if len(generation_seeds) != len(set(generation_seeds)):
        raise ValueError("generation seeds must be unique")
    held_out = held_out_cells(protocol, fold_id)
    rows: list[dict[str, Any]] = []
    cell_index = 0
    for subject_id, subject_token, subject_label in SUBJECTS:
        for color_id, color_token, color_label in COLORS:
            split = (
                "held_out"
                if (subject_label, color_label) in held_out
                else "seen"
            )
            prompt = f"a photo of {subject_token} shape in {color_token} color"
            for generation_seed in generation_seeds:
                item_id = (
                    f"mv2-fold-{fold_id.lower()}-train-{training_seed}-"
                    f"cell-{cell_index:02d}-gen-{generation_seed}"
                )
                rows.append(
                    {
                        "id": item_id,
                        "category": "multiview_grid",
                        "evaluation_protocol_id": protocol["protocol_id"],
                        "render_protocol_id": protocol["source_render_protocol_id"],
                        "fold_id": fold_id,
                        "training_seed": training_seed,
                        "combination_status": split,
                        "held_out": split == "held_out",
                        "cell_index": cell_index,
                        "prompt_index": cell_index,
                        "prompt": prompt,
                        "generation_seed": generation_seed,
                        "seed": generation_seed,
                        "num_inference_steps": DEFAULT_INFERENCE_STEPS,
                        "guidance_scale": DEFAULT_GUIDANCE_SCALE,
                        "subject_id": subject_id,
                        "subject_token": subject_token,
                        "subject_label": subject_label,
                        "expected_shape": subject_label,
                        "color_id": color_id,
                        "color_token": color_token,
                        "color_label": color_label,
                        "expected_color": color_label,
                        "fixed_subject_group": (
                            f"fold-{fold_id.lower()}-train-{training_seed}-"
                            f"subject-{subject_label}-gen-{generation_seed}"
                        ),
                        "fixed_color_group": (
                            f"fold-{fold_id.lower()}-train-{training_seed}-"
                            f"color-{color_label}-gen-{generation_seed}"
                        ),
                        "image_path": (
                            "images/multiview_grid/"
                            f"{item_id}.png"
                        ),
                    }
                )
            cell_index += 1

    expected = 9 * len(generation_seeds)
    if len(rows) != expected or len({row["id"] for row in rows}) != expected:
        raise AssertionError("multiview evaluation manifest is incomplete")
    return rows


def validate_campaign_manifest(
    rows: Iterable[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    """Validate the exact nine-checkpoint, 1620-image campaign contract."""

    items = list(rows)
    expected = protocol["expected"]
    if len(items) != expected["total_images"]:
        raise ValueError(
            f"expected {expected['total_images']} campaign rows, found {len(items)}"
        )
    ids = [row.get("id") for row in items]
    paths = [row.get("image_path") for row in items]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("every campaign row must have a non-empty string id")
    if len(ids) != len(set(ids)):
        raise ValueError("campaign manifest contains duplicate ids")
    if any(not isinstance(value, str) or not value for value in paths):
        raise ValueError("every campaign row must have a non-empty image path")
    if len(paths) != len(set(paths)):
        raise ValueError("campaign manifest contains duplicate image paths")

    expected_checkpoints = {
        (fold_id, seed)
        for fold_id in ("A", "B", "C")
        for seed in protocol["training_seeds"]
    }
    checkpoints = {(row.get("fold_id"), row.get("training_seed")) for row in items}
    if checkpoints != expected_checkpoints:
        raise ValueError("campaign manifest does not contain the 3x3 checkpoint matrix")

    expected_cells = {
        (subject[2], color[2]) for subject in SUBJECTS for color in COLORS
    }
    expected_generation_seeds = set(protocol["generation_seeds"])
    sampling = protocol["sampling"]
    for fold_id, training_seed in sorted(expected_checkpoints):
        checkpoint_rows = [
            row
            for row in items
            if row.get("fold_id") == fold_id
            and row.get("training_seed") == training_seed
        ]
        if len(checkpoint_rows) != expected["images_per_checkpoint"]:
            raise ValueError("each checkpoint must contribute exactly 180 rows")
        split_counts = {
            split: sum(row.get("combination_status") == split for row in checkpoint_rows)
            for split in ("seen", "held_out")
        }
        if split_counts != {
            "seen": expected["seen_images_per_checkpoint"],
            "held_out": expected["held_out_images_per_checkpoint"],
        }:
            raise ValueError("each checkpoint must contain 120 seen and 60 held-out rows")
        expected_held_out = held_out_cells(protocol, fold_id)
        locked_rows = {
            row["id"]: row
            for row in build_manifest(
                protocol, fold_id=fold_id, training_seed=training_seed
            )
        }
        locked_fields = (
            "prompt",
            "cell_index",
            "prompt_index",
            "generation_seed",
            "seed",
            "subject_id",
            "subject_token",
            "subject_label",
            "expected_shape",
            "color_id",
            "color_token",
            "color_label",
            "expected_color",
            "fixed_subject_group",
            "fixed_color_group",
        )
        cell_seeds: dict[tuple[str, str], set[int]] = {}
        for row in checkpoint_rows:
            locked = locked_rows.get(row.get("id"))
            if locked is None or any(row.get(field) != locked[field] for field in locked_fields):
                raise ValueError("campaign row differs from the locked prompt/seed schema")
            cell = (row.get("subject_label"), row.get("color_label"))
            cell_seeds.setdefault(cell, set()).add(row.get("generation_seed"))
            is_held_out = cell in expected_held_out
            if row.get("held_out") is not is_held_out:
                raise ValueError("held_out boolean does not match the locked fold mapping")
            if row.get("combination_status") != ("held_out" if is_held_out else "seen"):
                raise ValueError("combination_status does not match the locked fold mapping")
            if row.get("category") != "multiview_grid":
                raise ValueError("campaign contains a non-multiview category")
            if row.get("evaluation_protocol_id") != protocol["protocol_id"]:
                raise ValueError("campaign protocol id mismatch")
            if row.get("render_protocol_id") != protocol["source_render_protocol_id"]:
                raise ValueError("campaign render protocol id mismatch")
            if row.get("num_inference_steps") != sampling["num_inference_steps"]:
                raise ValueError("campaign inference-step setting changed")
            if row.get("guidance_scale") != sampling["guidance_scale"]:
                raise ValueError("campaign guidance-scale setting changed")
            if row.get("dtype") != sampling["dtype"]:
                raise ValueError("campaign dtype setting changed")
            if row.get("safety_checker_disabled") is not True:
                raise ValueError("campaign safety checker must be explicitly disabled")
            if row.get("safety_risk_acknowledged") is not True:
                raise ValueError("campaign safety risk must be explicitly acknowledged")
            if row.get("protocol_fingerprint_sha256") != protocol["_source_sha256"]:
                raise ValueError("campaign protocol fingerprint mismatch")
        if set(cell_seeds) != expected_cells or any(
            seeds != expected_generation_seeds for seeds in cell_seeds.values()
        ):
            raise ValueError("each checkpoint must contain all nine cells with seeds 42..61")
        fingerprints = {row.get("model_fingerprint_sha256") for row in checkpoint_rows}
        if len(fingerprints) != 1 or not next(iter(fingerprints), None):
            raise ValueError("each checkpoint must bind one non-empty model fingerprint")
        expected_variant = f"multiview_v2_fold_{fold_id.lower()}_seed{training_seed}"
        variants = {row.get("parent_training_variant") for row in checkpoint_rows}
        if variants != {expected_variant}:
            raise ValueError("campaign parent training variant does not match checkpoint")
    return items
