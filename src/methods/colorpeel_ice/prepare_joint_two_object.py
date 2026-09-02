#!/usr/bin/env python3
"""Build the immutable joint two-object binding training package."""

from __future__ import annotations

import argparse
import filecmp
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.colorpeel_ice.prepare_clevr_multiview import (  # noqa: E402
    _require,
    _require_empty_output_dir,
    _write_json,
    _write_jsonl,
)
from src.methods.colorpeel_ice.prepare_clevr_two_object import _read_json, _read_jsonl, _stage  # noqa: E402


EXPERIMENT_DIR = REPO_ROOT / "experiments" / "clevr_two_object_joint_binding_v2"
BASE_CONFIG = EXPERIMENT_DIR / "configs" / "train_seed42.json"
SMOKE2_CONFIG = EXPERIMENT_DIR / "configs" / "smoke_2step.json"
SMOKE18_CONFIG = EXPERIMENT_DIR / "configs" / "smoke_18step.json"
MODIFIER_TOKENS = ("<s1*>", "<s2*>", "<s3*>", "<c1*>", "<c2*>", "<c3*>", "<m1*>", "<m2*>")
PROMPT_TEMPLATE = (
    "a photo of {left_subject} shape in {left_color} color with {left_material} material on the left "
    "and {right_subject} shape in {right_color} color with {right_material} material on the right"
)


def _prompt_and_groups(row: dict[str, Any]) -> tuple[str, list[list[str]]]:
    _require([obj.get("side") for obj in row.get("objects", [])] == ["left", "right"],
             f"Scene object order changed: {row.get('scene_id')}")
    left, right = row["objects"]
    groups = [
        [left["subject_token"], left["color_token"], left["material_token"]],
        [right["subject_token"], right["color_token"], right["material_token"]],
    ]
    _require(set(groups[0]).isdisjoint(groups[1]), f"Scene token groups overlap: {row['scene_id']}")
    prompt = PROMPT_TEMPLATE.format(
        left_subject=groups[0][0], left_color=groups[0][1], left_material=groups[0][2],
        right_subject=groups[1][0], right_color=groups[1][1], right_material=groups[1][2],
    )
    return prompt, groups


def _prepared_state_sources(prepared_root: Path) -> dict[str, dict[str, Path]]:
    concepts = _read_json(prepared_root / "training" / "concepts.json")
    _require(isinstance(concepts, list) and len(concepts) == 18, "Expected 18 prepared semantic concepts")
    sources = {}
    for concept in concepts:
        image_dir = Path(concept["instance_data_dir"]).resolve()
        mask_dir = Path(concept["instance_mask_dir"]).resolve()
        state_id = image_dir.name
        _require(state_id == mask_dir.name and state_id not in sources, "Prepared state directories changed")
        sources[state_id] = {"images": image_dir, "masks": mask_dir}
    return sources


def _source_paths(row: dict[str, Any], sources: dict[str, dict[str, Path]]) -> tuple[Path, Path, Path]:
    name = f"{row['scene_id']}__view_{row['view_index']:02d}"
    left, right = row["objects"]
    left_image = sources[left["state_id"]]["images"] / f"{name}.jpg"
    right_image = sources[right["state_id"]]["images"] / f"{name}.jpg"
    left_mask = sources[left["state_id"]]["masks"] / f"{name}.png"
    right_mask = sources[right["state_id"]]["masks"] / f"{name}.png"
    for path in (left_image, right_image, left_mask, right_mask):
        _require(path.is_file(), f"Prepared training artifact missing: {path}")
    _require(filecmp.cmp(left_image, right_image, shallow=False), f"Duplicated RGB differs for {name}")
    return left_image, left_mask, right_mask


def _stage_concept(
    rows: list[dict[str, Any]], sources: dict[str, dict[str, Path]], root: Path
) -> dict[str, Any]:
    prompt, groups = _prompt_and_groups(rows[0])
    image_dir = root / "images"
    left_dir, right_dir = root / "masks_left", root / "masks_right"
    for directory in (image_dir, left_dir, right_dir):
        directory.mkdir(parents=True)
    for row in rows:
        current_prompt, current_groups = _prompt_and_groups(row)
        _require((current_prompt, current_groups) == (prompt, groups), "Concept rows changed prompt or groups")
        image, left_mask, right_mask = _source_paths(row, sources)
        name = f"{row['scene_id']}__view_{row['view_index']:02d}"
        _stage(image, image_dir / f"{name}.jpg")
        _stage(left_mask, left_dir / f"{name}.png")
        _stage(right_mask, right_dir / f"{name}.png")
    stems = {path.stem for path in image_dir.iterdir()}
    _require(stems == {path.stem for path in left_dir.iterdir()} == {path.stem for path in right_dir.iterdir()},
             f"Joint staging stems differ: {root}")
    return {
        "instance_prompt": [prompt],
        "instance_data_dir": str(image_dir.resolve()),
        "instance_mask_dirs": {"left": str(left_dir.resolve()), "right": str(right_dir.resolve())},
        "modifier_token_groups": groups,
    }


def _expected_present(groups: list[list[str]]) -> list[str]:
    present = {token for group in groups for token in group}
    return [token for token in MODIFIER_TOKENS if token in present]


def _write_training_config(template_path: Path, concepts_path: Path, manifest_path: Path, output_path: Path,
                           expected_rows: list[dict[str, Any]] | None = None) -> None:
    config = _read_json(template_path)
    config["args"]["concepts_list"] = str(concepts_path.resolve())
    config["data_manifest"] = str(manifest_path.resolve())
    if expected_rows is not None:
        groups = [_prompt_and_groups(row)[1] for row in expected_rows]
        expected_sequence = [_expected_present(item) for item in groups]
        counts = {token: sum(token in row for row in expected_sequence) for token in MODIFIER_TOKENS}
        config["protocol"]["expected_modifier_token_pairs"] = expected_sequence
        config["protocol"]["expected_exposure_counts"] = counts
    _write_json(output_path, config)


def build_package(prepared_root: Path, output_dir: Path) -> dict[str, Any]:
    prepared_root = prepared_root.resolve()
    status = _read_json(prepared_root / "protocol_status.json")
    _require(status.get("status") == "validated_pending_human_review", "Prepared renderer status changed")
    realized_path = prepared_root / "realized_scenes.jsonl"
    realized = _read_jsonl(realized_path)
    _require(len(realized) == 360, "Expected 360 realized two-object views")
    train_rows = [row for row in realized if row.get("split") == "train"]
    _require(len(train_rows) == 288, "Expected 288 joint training images")
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in train_rows:
        by_scene.setdefault(row["scene_id"], []).append(row)
    _require(len(by_scene) == 18 and set(map(len, by_scene.values())) == {16}, "Expected 18 scenes x 16 views")
    sources = _prepared_state_sources(prepared_root)
    _require_empty_output_dir(output_dir)

    full_root = output_dir / "full_training"
    full_concepts = []
    for scene_id in sorted(by_scene):
        rows = sorted(by_scene[scene_id], key=lambda row: row["view_index"])
        full_concepts.append(_stage_concept(rows, sources, full_root / "train_assets" / scene_id))
    full_concepts_path = full_root / "concepts.json"
    _write_json(full_concepts_path, full_concepts)
    _write_training_config(BASE_CONFIG, full_concepts_path, realized_path, full_root / "train_config_seed42.json")

    first_rows = [min(rows, key=lambda row: row["view_index"]) for _, rows in sorted(by_scene.items())]
    smoke_specs = (
        ("smoke_2step", first_rows[:2], SMOKE2_CONFIG),
        ("smoke_18step", first_rows, SMOKE18_CONFIG),
    )
    smoke_configs = {}
    for name, selected, template in smoke_specs:
        smoke_root = output_dir / "smokes" / name
        concepts = [
            _stage_concept([row], sources, smoke_root / "train_assets" / row["scene_id"])
            for row in selected
        ]
        concepts_path = smoke_root / "concepts.json"
        _write_json(concepts_path, concepts)
        config_path = smoke_root / "train_config.json"
        _write_training_config(template, concepts_path, realized_path, config_path, selected)
        smoke_configs[name] = str(config_path)

    package_manifest = [
        {
            "scene_id": row["scene_id"],
            "view_index": row["view_index"],
            "objects": row["objects"],
            "prompt": _prompt_and_groups(row)[0],
            "modifier_token_groups": _prompt_and_groups(row)[1],
        }
        for row in sorted(train_rows, key=lambda item: (item["scene_id"], item["view_index"]))
    ]
    _write_jsonl(output_dir / "joint_training_manifest.jsonl", package_manifest)
    result = {
        "schema_version": 1,
        "status": "ready_for_joint_binding_smokes",
        "protocol_id": "clevr_two_object_joint_binding_v2",
        "prepared_root": str(prepared_root),
        "joint_training_image_count": 288,
        "scene_count": 18,
        "views_per_scene": 16,
        "shared_modifier_token_count": 8,
        "smoke_configs": smoke_configs,
        "full_config": str((full_root / "train_config_seed42.json").resolve()),
    }
    _write_json(output_dir / "package_status.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_package(args.prepared_root, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
