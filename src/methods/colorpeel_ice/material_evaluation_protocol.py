"""Locked three-axis complete-bundle evaluation protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from src.methods.colorpeel_ice.prepare_clevr_multiview_material import EXPECTED_FOLDS


PROTOCOL_ID = "clevr_subject_color_material_3x3x2_heldout_bundle_v1"
RENDER_PROTOCOL_ID = "clevr_subject_color_material_3x3x2_multiview_v3"
SUBJECTS = (("s1", "<s1*>", "cube"), ("s2", "<s2*>", "sphere"), ("s3", "<s3*>", "cylinder"))
COLORS = (("c1", "<c1*>", "red"), ("c2", "<c2*>", "cyan"), ("c3", "<c3*>", "gray"))
MATERIALS = (("m1", "<m1*>", "metal"), ("m2", "<m2*>", "rubber"))
GENERATION_SEEDS = tuple(range(42, 62))
TRAINING_SEEDS = (42, 43, 44)
EXPECTED = {"checkpoints": 9, "prompts_per_checkpoint": 18, "images_per_checkpoint": 360,
            "seen_images_per_checkpoint": 240, "held_out_images_per_checkpoint": 120,
            "total_images": 3240, "total_seen_images": 2160, "total_held_out_images": 1080}


def read_protocol(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if value.get("schema_version") != 1 or value.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("material evaluation protocol identity changed")
    if value.get("source_render_protocol_id") != RENDER_PROTOCOL_ID:
        raise ValueError("material evaluation render protocol changed")
    if value.get("training_seeds") != list(TRAINING_SEEDS) or value.get("generation_seeds") != list(GENERATION_SEEDS):
        raise ValueError("material evaluation seeds changed")
    for name, expected in (("subjects", SUBJECTS), ("colors", COLORS), ("materials", MATERIALS)):
        locked = [{"id": item[0], "token": item[1], "label": item[2]} for item in expected]
        if value.get(name) != locked:
            raise ValueError(f"material evaluation {name} changed")
    if value.get("prompt_template") != (
        "a photo of {subject_token} shape in {color_token} color with {material_token} material"
    ):
        raise ValueError("material evaluation prompt changed")
    if value.get("sampling") != {"num_inference_steps": 100, "guidance_scale": 6.0, "dtype": "float16",
                                 "safety_checker_policy": "disabled_after_confirmed_false_positive_with_explicit_acknowledgement"}:
        raise ValueError("material evaluation sampling changed")
    if value.get("expected") != EXPECTED:
        raise ValueError("material evaluation counts changed")
    folds = {fold.get("id"): {tuple(cell) for cell in fold.get("held_out", [])} for fold in value.get("folds", [])}
    if folds != EXPECTED_FOLDS:
        raise ValueError("material evaluation folds changed")
    value["_source_sha256"] = hashlib.sha256(raw).hexdigest()
    return value


def held_out_cells(protocol: dict[str, Any], fold_id: str) -> set[tuple[str, str, str]]:
    fold_id = str(fold_id).upper()
    matches = [fold for fold in protocol["folds"] if fold["id"] == fold_id]
    if len(matches) != 1:
        raise ValueError(f"unknown fold: {fold_id}")
    return {tuple(cell) for cell in matches[0]["held_out"]}


def build_manifest(protocol: dict[str, Any], *, fold_id: str, training_seed: int,
                   generation_seeds: Iterable[int] | None = None) -> list[dict[str, Any]]:
    fold_id = str(fold_id).upper(); training_seed = int(training_seed)
    if training_seed not in TRAINING_SEEDS:
        raise ValueError("unsupported training seed")
    seeds = tuple(GENERATION_SEEDS if generation_seeds is None else map(int, generation_seeds))
    if len(seeds) != len(set(seeds)):
        raise ValueError("generation seeds must be unique")
    held_out = held_out_cells(protocol, fold_id)
    rows = []
    for cell_index, (subject, color, material) in enumerate(
        (s, c, m) for s in SUBJECTS for c in COLORS for m in MATERIALS
    ):
        triple = (subject[2], color[2], material[2]); is_held_out = triple in held_out
        prompt = protocol["prompt_template"].format(
            subject_token=subject[1], color_token=color[1], material_token=material[1]
        )
        for seed in seeds:
            item_id = (f"mat-fold-{fold_id.lower()}-train-{training_seed}-"
                       f"cell-{cell_index:02d}-gen-{seed}")
            group_prefix = f"fold-{fold_id.lower()}-train-{training_seed}-gen-{seed}"
            rows.append({
                "id": item_id, "category": "material_grid", "evaluation_protocol_id": PROTOCOL_ID,
                "render_protocol_id": RENDER_PROTOCOL_ID, "fold_id": fold_id, "training_seed": training_seed,
                "combination_status": "held_out" if is_held_out else "seen", "held_out": is_held_out,
                "cell_index": cell_index, "prompt": prompt, "generation_seed": seed, "seed": seed,
                "num_inference_steps": 100, "guidance_scale": 6.0,
                "subject_id": subject[0], "subject_token": subject[1], "subject_label": subject[2],
                "expected_shape": subject[2], "color_id": color[0], "color_token": color[1],
                "color_label": color[2], "expected_color": color[2], "material_id": material[0],
                "material_token": material[1], "material_label": material[2], "expected_material": material[2],
                "fixed_shape_color_group": f"{group_prefix}-shape-{subject[2]}-color-{color[2]}",
                "fixed_shape_material_group": f"{group_prefix}-shape-{subject[2]}-material-{material[2]}",
                "fixed_color_material_group": f"{group_prefix}-color-{color[2]}-material-{material[2]}",
                "image_path": f"images/material_grid/{item_id}.png",
            })
    expected = 18 * len(seeds)
    if len(rows) != expected or len({row["id"] for row in rows}) != expected:
        raise AssertionError("material manifest is incomplete")
    return rows


def build_full_grid_manifest(protocol: dict[str, Any], generation_seeds: Iterable[int] | None = None) -> list[dict[str, Any]]:
    """Build the seed-42 full-grid gate without assigning seen/held-out labels."""
    rows = build_manifest(
        protocol, fold_id="A", training_seed=42, generation_seeds=generation_seeds
    )
    output = []
    for row in rows:
        item_id = row["id"].replace("mat-fold-a-train-42", "mat-full-grid-train-42")
        output.append({**row, "id": item_id, "evaluation_scope": "full_grid_baseline",
                       "combination_status": "full_grid", "held_out": False,
                       "image_path": f"images/material_grid/{item_id}.png"})
    return output


def validate_campaign(rows: Iterable[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    items = list(rows)
    if len(items) != EXPECTED["total_images"] or len({row.get("id") for row in items}) != len(items):
        raise ValueError("campaign must contain 3240 unique rows")
    if len({row.get("image_path") for row in items}) != len(items):
        raise ValueError("campaign image paths must be unique")
    checkpoints = {(row.get("fold_id"), row.get("training_seed")) for row in items}
    if checkpoints != {(fold, seed) for fold in "ABC" for seed in TRAINING_SEEDS}:
        raise ValueError("campaign checkpoint matrix changed")
    for fold, seed in checkpoints:
        subset = [row for row in items if row.get("fold_id") == fold and row.get("training_seed") == seed]
        locked = {row["id"]: row for row in build_manifest(protocol, fold_id=fold, training_seed=seed)}
        if len(subset) != 360 or {row.get("id") for row in subset} != set(locked):
            raise ValueError("checkpoint slice changed")
        if sum(row.get("held_out") is True for row in subset) != 120:
            raise ValueError("checkpoint split counts changed")
        for row in subset:
            expected = locked[row["id"]]
            for field in expected:
                if field != "image_path" and row.get(field) != expected[field]:
                    raise ValueError(f"campaign row changed locked field {field}")
            if row.get("dtype") != "float16" or row.get("safety_checker_disabled") is not True:
                raise ValueError("campaign sampling record changed")
            if row.get("safety_risk_acknowledged") is not True:
                raise ValueError("campaign safety acknowledgement missing")
            if row.get("protocol_fingerprint_sha256") != protocol["_source_sha256"]:
                raise ValueError("campaign protocol fingerprint mismatch")
            if not row.get("model_fingerprint_sha256"):
                raise ValueError("campaign model fingerprint missing")
    return items
