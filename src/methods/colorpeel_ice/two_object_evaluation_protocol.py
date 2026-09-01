"""Locked seen/unseen two-object generation protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.methods.colorpeel_ice.prepare_clevr_two_object import (
    COLORS,
    DEFAULT_MANIFEST,
    MATERIALS,
    SHAPES,
    build_states,
)


EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "experiments" / "clevr_two_object_subject_color_material"
DEFAULT_EVALUATION_PROTOCOL = EXPERIMENT_DIR / "manifests" / "two_object_evaluation.json"
PROTOCOL_ID = "clevr_two_object_subject_color_material_eval_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_protocol(path: Path = DEFAULT_EVALUATION_PROTOCOL) -> dict[str, Any]:
    path = Path(path).resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "training_manifest": "clevr_two_object_manifest.json",
        "prompt_template": "a photo of {left_bundle} on the left and {right_bundle} on the right",
        "pair_groups": {
            "seen": "metal(s,c) with rubber((s+1)%3,(c+1)%3)",
            "unseen": "metal(s,c) with rubber((s+2)%3,(c+1)%3)",
        },
        "orientations": ["forward", "swapped"],
        "generation_seeds": {"start": 42, "stop_inclusive": 61},
        "num_inference_steps": 100,
        "guidance_scale": 6.0,
        "safety_checker_disabled": True,
        "expected_pair_count_per_group": 9,
        "expected_scene_count": 36,
        "expected_image_count": 720,
    }
    if protocol != expected:
        raise ValueError("two-object evaluation protocol differs from the locked v1 protocol")
    protocol["_source_path"] = str(path)
    protocol["_source_sha256"] = file_sha256(path)
    return protocol


def _bundle(state: dict[str, Any]) -> str:
    return (
        f"{state['subject_token']} shape in {state['color_token']} color "
        f"with {state['material_token']} material"
    )


def build_manifest(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    states = build_states(manifest)
    by_key = {(row["shape"], row["color"], row["material"]): row for row in states}
    seeds = range(
        protocol["generation_seeds"]["start"],
        protocol["generation_seeds"]["stop_inclusive"] + 1,
    )
    rows = []
    pair_sets: dict[str, set[frozenset[str]]] = {"seen": set(), "unseen": set()}
    appearances = {(group, state["state_id"], side): 0
                   for group in ("seen", "unseen") for state in states for side in ("left", "right")}
    for group, shape_offset in (("seen", 1), ("unseen", 2)):
        for shape_index, shape in enumerate(SHAPES):
            for color_index, color in enumerate(COLORS):
                pair_index = shape_index * len(COLORS) + color_index
                metal = by_key[(shape, color, MATERIALS[0])]
                rubber = by_key[(SHAPES[(shape_index + shape_offset) % 3], COLORS[(color_index + 1) % 3], MATERIALS[1])]
                pair_sets[group].add(frozenset((metal["state_id"], rubber["state_id"])))
                for orientation, (left, right) in (
                    ("forward", (metal, rubber)),
                    ("swapped", (rubber, metal)),
                ):
                    appearances[(group, left["state_id"], "left")] += 1
                    appearances[(group, right["state_id"], "right")] += 1
                    scene_id = f"{group}_pair_{pair_index:02d}_{orientation}"
                    prompt = protocol["prompt_template"].format(
                        left_bundle=_bundle(left), right_bundle=_bundle(right)
                    )
                    for seed in seeds:
                        rows.append({
                            "id": f"{scene_id}_seed_{seed:02d}",
                            "pair_group": group,
                            "pair_index": pair_index,
                            "orientation": orientation,
                            "scene_id": scene_id,
                            "generation_seed": seed,
                            "prompt": prompt,
                            "left": {key: left[key] for key in (
                                "state_id", "shape", "color", "material", "subject_token", "color_token", "material_token"
                            )},
                            "right": {key: right[key] for key in (
                                "state_id", "shape", "color", "material", "subject_token", "color_token", "material_token"
                            )},
                            "num_inference_steps": protocol["num_inference_steps"],
                            "guidance_scale": protocol["guidance_scale"],
                            "image_path": f"images/{group}/{scene_id}/seed_{seed:02d}.png",
                        })
    if pair_sets["seen"] & pair_sets["unseen"]:
        raise ValueError("seen and unseen pair sets overlap")
    if set(appearances.values()) != {1}:
        raise ValueError("each state must appear once on each side in each pair group")
    if len(rows) != protocol["expected_image_count"]:
        raise ValueError("two-object evaluation row count changed")
    return rows
