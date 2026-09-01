#!/usr/bin/env python3
"""Plan and realize the locked controlled two-object CLEVR protocol."""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

from PIL import Image, ImageChops, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.colorpeel_ice.multiview_render_contract import (  # noqa: E402
    EXPECTED_PROFILE_V4,
    canonical_sha256,
    validate_two_object_render_requests,
)
from src.methods.colorpeel_ice.prepare_clevr_multiview import (  # noqa: E402
    ProtocolError,
    _file_sha256,
    _require,
    _require_empty_output_dir,
    _resolved_under,
    _validate_view_metadata,
    _vector_matches,
    _write_json,
    _write_jsonl,
)

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


EXPERIMENT_DIR = REPO_ROOT / "experiments" / "clevr_two_object_subject_color_material"
DEFAULT_MANIFEST = EXPERIMENT_DIR / "manifests" / "clevr_two_object_manifest.json"
DEFAULT_PROTOCOL = EXPERIMENT_DIR / "manifests" / "clevr_two_object_protocol.json"
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "two_object_base.yaml"
SMOKE_2STEP_CONFIG = EXPERIMENT_DIR / "configs" / "smoke_2step.yaml"
SMOKE_18STEP_CONFIG = EXPERIMENT_DIR / "configs" / "smoke_18step.yaml"
SHAPES = ("cube", "sphere", "cylinder")
COLORS = ("red", "cyan", "gray")
MATERIALS = ("metal", "rubber")
PROMPT_TEMPLATE = "a photo of {subject_token} shape in {color_token} color with {material_token} material"
MODIFIER_TOKEN = "<s1*>+<s2*>+<s3*>+<c1*>+<c2*>+<c3*>+<m1*>+<m2*>"
INITIALIZER_TOKEN = "cube+sphere+cylinder+red+turquoise+gray+metal+rubber"
RENDERER_FIELDS = ("camera", "light", "background", "scene_json", "image", "masks", "background_mask")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read valid JSON from {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    value = json.loads(line)
                    _require(isinstance(value, dict), f"Expected object at {path}:{line_number}")
                    records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read valid JSONL from {path}: {exc}") from exc
    return records


def build_states(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    _require(manifest.get("version") == 1, "Manifest version changed")
    _require(manifest.get("study") == "clevr_two_object_subject_color_material", "Study changed")
    _require(manifest.get("prompt_template") == PROMPT_TEMPLATE, "Prompt template changed")
    dimensions = {}
    for field, expected in (("shapes", SHAPES), ("colors", COLORS), ("materials", MATERIALS)):
        values = manifest.get(field, [])
        _require([item.get("name") for item in values] == list(expected), f"Locked {field} order changed")
        dimensions[field] = {item["name"]: item for item in values}
    _require([dimensions["colors"][name]["initializer"] for name in COLORS] ==
             ["red", "turquoise", "gray"], "Color initializers changed")
    states = []
    for shape_index, shape in enumerate(SHAPES):
        for color_index, color in enumerate(COLORS):
            for material_index, material in enumerate(MATERIALS):
                shape_item = dimensions["shapes"][shape]
                color_item = dimensions["colors"][color]
                material_item = dimensions["materials"][material]
                values = {
                    "subject_token": shape_item["token"],
                    "color_token": color_item["token"],
                    "material_token": material_item["token"],
                }
                states.append({
                    "state_id": f"{shape}_{color}_{material}",
                    "state_index": (shape_index * 3 + color_index) * 2 + material_index,
                    "shape": shape,
                    "color": color,
                    "material": material,
                    "nominal_rgb": color_item["rgb"],
                    **values,
                    "instance_prompt": [PROMPT_TEMPLATE.format(**values)],
                })
    _require(len(states) == 18, "Expected 18 semantic states")
    return states


def build_scenes(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["shape"], row["color"], row["material"]): row for row in states}
    scenes = []
    for shape_index, shape in enumerate(SHAPES):
        for color_index, color in enumerate(COLORS):
            pair_index = shape_index * 3 + color_index
            first = by_key[(shape, color, "metal")]
            second = by_key[(SHAPES[(shape_index + 1) % 3], COLORS[(color_index + 1) % 3], "rubber")]
            for orientation, (left, right) in enumerate(((first, second), (second, first))):
                scene_id = f"pair_{pair_index:02d}_{'forward' if orientation == 0 else 'swapped'}"
                objects = []
                for side, state in (("left", left), ("right", right)):
                    objects.append({key: state[key] for key in (
                        "state_id", "state_index", "shape", "color", "material", "nominal_rgb",
                        "subject_token", "color_token", "material_token", "instance_prompt",
                    )} | {"side": side})
                scenes.append({
                    "scene_id": scene_id,
                    "scene_index": pair_index * 2 + orientation,
                    "pair_index": pair_index,
                    "orientation": "forward" if orientation == 0 else "swapped",
                    "objects": objects,
                })
    _require(len(scenes) == 18, "Expected 18 oriented scene cells")
    appearances = {(state["state_id"], side): 0 for state in states for side in ("left", "right")}
    for scene in scenes:
        left, right = scene["objects"]
        _require(left["shape"] != right["shape"], f"Same shape in {scene['scene_id']}")
        _require(left["color"] != right["color"], f"Same color in {scene['scene_id']}")
        _require(left["material"] != right["material"], f"Same material in {scene['scene_id']}")
        for obj in scene["objects"]:
            appearances[(obj["state_id"], obj["side"])] += 1
    _require(set(appearances.values()) == {1}, "Every state must appear once on each side")
    return scenes


def validate_protocol(manifest: dict[str, Any], protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    states = build_states(manifest)
    scenes = build_scenes(states)
    _require(protocol.get("version") == 1, "Protocol version changed")
    _require(protocol.get("protocol_id") == "clevr_two_object_subject_color_material_v1", "Protocol ID changed")
    _require(protocol.get("base_manifest") == "clevr_two_object_manifest.json", "Base manifest changed")
    _require(protocol.get("views_per_scene") == 20, "views_per_scene must be 20")
    _require(protocol.get("view_splits") == {
        "train": {"start": 0, "stop": 16}, "audit": {"start": 16, "stop": 20},
    }, "View split changed")
    _require(protocol.get("render_seed") == {
        "base": 520000, "pair_stride": 100,
        "formula": "base + pair_index * pair_stride + view_index",
        "paired_orientations": ["forward", "swapped"],
    }, "Render seed rule changed")
    expected_renderer = {
        "id": EXPECTED_PROFILE_V4["profile_id"],
        "config": "../configs/multiview_render_v4_two_object.json",
        "background": EXPECTED_PROFILE_V4["background"],
    }
    _require(protocol.get("renderer_profile") == expected_renderer, "Renderer profile changed")
    _require(protocol.get("training") == {
        "one_object_bundle_per_sample": True,
        "shared_model": True,
        "shared_modifier_tokens": 8,
        "instance_mask_reconstruction_loss": True,
        "cross_object_caa": False,
        "train_object_records": 576,
    }, "Training contract changed")
    return states, scenes


def load_inputs(manifest_path: Path, protocol_path: Path):
    manifest, protocol = _read_json(manifest_path), _read_json(protocol_path)
    _require(isinstance(manifest, dict) and isinstance(protocol, dict), "Inputs must be JSON objects")
    states, scenes = validate_protocol(manifest, protocol)
    return manifest, protocol, states, scenes


def build_render_requests(manifest: dict[str, Any], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    _, scenes = validate_protocol(manifest, protocol)
    requests = []
    by_pair = {
        pair_index: [scene for scene in scenes if scene["pair_index"] == pair_index]
        for pair_index in range(9)
    }
    for pair_index in range(9):
        for view_index in range(20):
            for scene in by_pair[pair_index]:
                requests.append({
                    **scene,
                    "view_index": view_index,
                    "split": "train" if view_index < 16 else "audit",
                    "render_seed": 520000 + pair_index * 100 + view_index,
                    "renderer_profile_id": EXPECTED_PROFILE_V4["profile_id"],
                    "renderer_profile_sha256": canonical_sha256(EXPECTED_PROFILE_V4),
                    **{field: None for field in RENDERER_FIELDS},
                })
    validate_render_requests(requests)
    return requests


def validate_render_requests(requests: list[dict[str, Any]]) -> None:
    try:
        validate_two_object_render_requests(requests)
    except ValueError as exc:
        raise ProtocolError(str(exc)) from exc


def plan_protocol(manifest: dict[str, Any], protocol: dict[str, Any], output_dir: Path,
                  renderer: Path | None) -> dict[str, Any]:
    requests = build_render_requests(manifest, protocol)
    _require_empty_output_dir(output_dir)
    request_path = output_dir / "render_requests.jsonl"
    _write_jsonl(request_path, requests)
    available = renderer is not None and renderer.resolve().is_file()
    status = {
        "status": "planned" if available else "blocked",
        "blocked_reason": None if available else "two_object_renderer_not_provided_or_missing",
        "renderer": str(renderer.resolve()) if renderer else None,
        "renderer_available": available,
        "renderer_profile_id": EXPECTED_PROFILE_V4["profile_id"],
        "renderer_profile_sha256": canonical_sha256(EXPECTED_PROFILE_V4),
        "render_request_manifest": str(request_path.resolve()),
        "request_count": 360,
        "paired_smoke_prefix_count": 2,
    }
    _write_json(output_dir / "protocol_status.json", status)
    return status


def _binary_mask(path: Path) -> Image.Image:
    with Image.open(path) as source:
        _require(source.mode == "L", f"Mask must use L mode: {path}")
        image = source.copy()
    _require(image.size == (512, 512), f"Mask must be 512x512: {path}")
    values = {value for value, count in enumerate(image.histogram()) if count}
    _require(values <= {0, 255}, f"Mask is not binary: {path}")
    return image


def _validate_artifacts(image_path: Path, mask_paths: dict[str, Path], background_path: Path) -> dict[str, int]:
    with Image.open(image_path) as image:
        _require(image.mode == "RGB" and image.size == (512, 512), f"RGB must be 512x512: {image_path}")
    left, right = _binary_mask(mask_paths["left"]), _binary_mask(mask_paths["right"])
    background = _binary_mask(background_path)
    _require(ImageChops.multiply(left, right).getbbox() is None, "Object masks overlap")
    union = ImageChops.lighter(left, right)
    complement = ImageChops.invert(union)
    _require(ImageChops.difference(complement, background).getbbox() is None,
             "Background mask is not the complement of the object union")
    counts = {"left": left.histogram()[255], "right": right.histogram()[255]}
    for side, count in counts.items():
        _require(0.005 <= count / (512 * 512) <= 0.45, f"{side} mask ratio outside 0.005-0.45")
        bbox = {"left": left, "right": right}[side].getbbox()
        _require(bbox is not None and bbox[0] > 0 and bbox[1] > 0 and bbox[2] < 512 and bbox[3] < 512,
                 f"{side} object mask touches an image edge")
    return counts


def validate_realization(render_root: Path, records: list[dict[str, Any]], manifest: dict[str, Any],
                         protocol: dict[str, Any]) -> list[dict[str, Any]]:
    expected = build_render_requests(manifest, protocol)
    expected_by_key = {(row["scene_id"], row["view_index"]): row for row in expected}
    actual_by_key = {(row.get("scene_id"), row.get("view_index")): row for row in records}
    _require(len(records) == len(actual_by_key) == 360 and set(actual_by_key) == set(expected_by_key),
             "Realization must contain all 360 scene/views exactly once")
    contract = _read_json(render_root / "render_contract.json")
    _require(contract.get("schema_version") == 1, "Render contract version changed")
    _require(contract.get("profile_id") == EXPECTED_PROFILE_V4["profile_id"], "Render profile changed")
    _require(contract.get("profile_sha256") == canonical_sha256(EXPECTED_PROFILE_V4), "Profile hash changed")
    _require(contract.get("requests_sha256") == canonical_sha256(expected), "Request hash changed")
    _require(contract.get("request_count") == 360, "Render request count changed")
    _require(set(contract.get("asset_sha256", {})) >= {"material_metal", "material_rubber"},
             "Both material asset hashes are required")
    contract_hash = canonical_sha256(contract)
    hashes_by_scene = {row["scene_id"]: set() for row in expected}
    realized = []
    for key, expected_row in expected_by_key.items():
        row = actual_by_key[key]
        for field in ("scene_id", "scene_index", "pair_index", "orientation", "objects", "view_index",
                      "split", "render_seed", "renderer_profile_id", "renderer_profile_sha256"):
            _require(row.get(field) == expected_row[field], f"Realization changed {field} for {key}")
        _require(row.get("background") == EXPECTED_PROFILE_V4["background"], f"Background changed for {key}")
        _validate_view_metadata(row, key, EXPECTED_PROFILE_V4)
        paths = {
            "image": _resolved_under(render_root, row.get("image"), "image"),
            "scene_json": _resolved_under(render_root, row.get("scene_json"), "scene_json"),
            "background_mask": _resolved_under(render_root, row.get("background_mask"), "background_mask"),
        }
        masks = row.get("masks")
        _require(isinstance(masks, dict) and set(masks) == {"left", "right"}, f"Bad masks for {key}")
        mask_paths = {side: _resolved_under(render_root, value, f"mask_{side}") for side, value in masks.items()}
        hashes = row.get("artifact_sha256", {})
        for field, path in paths.items():
            _require(hashes.get(field) == _file_sha256(path), f"Artifact hash changed: {key} {field}")
        for side, path in mask_paths.items():
            _require(hashes.get(f"mask_{side}") == _file_sha256(path), f"Mask hash changed: {key} {side}")
        counts = _validate_artifacts(paths["image"], mask_paths, paths["background_mask"])
        _require(row.get("foreground_pixels") == counts, f"Foreground counts changed for {key}")
        _require(row.get("render_contract_sha256") == contract_hash, f"Contract hash changed for {key}")
        scene = _read_json(paths["scene_json"])
        _require(scene.get("renderer_profile_id") == EXPECTED_PROFILE_V4["profile_id"], f"Scene profile changed for {key}")
        _require(scene.get("render_seed") == row["render_seed"] == scene.get("cycles_seed"), f"Scene seed changed for {key}")
        _require(scene.get("camera") == row.get("camera") and scene.get("light") == row.get("light"),
                 f"Scene view metadata changed for {key}")
        _require(scene.get("background") == EXPECTED_PROFILE_V4["background"], f"Scene background changed for {key}")
        _require(scene.get("asset_sha256") == contract.get("asset_sha256"), f"Scene assets changed for {key}")
        renderer = scene.get("renderer", {})
        _require(renderer.get("blender_version") == "4.2.11" and renderer.get("engine") == "CYCLES",
                 f"Renderer runtime changed for {key}")
        _require(renderer.get("cycles_device") == "CUDA" and renderer.get("cycles_samples") == 512,
                 f"Cycles runtime changed for {key}")
        _require(len(renderer.get("cuda_devices", [])) == 1 and
                 "V100" in renderer["cuda_devices"][0].get("name", ""), f"GPU changed for {key}")
        _require(scene.get("objects") == row.get("rendered_objects"), f"Scene objects changed for {key}")
        _require([obj.get("side") for obj in scene.get("objects", [])] == ["left", "right"],
                 f"Scene must contain ordered left/right objects for {key}")
        for requested, rendered in zip(row["objects"], scene["objects"]):
            for field in ("side", "state_id", "shape", "color", "material"):
                _require(rendered.get(field) == requested[field], f"Rendered {field} changed for {key}")
            expected_xy = EXPECTED_PROFILE_V4["objects"]["positions_xy"][requested["side"]]
            _require(_vector_matches(rendered.get("3d_coords", [])[:2], expected_xy),
                     f"Rendered position changed for {key} {requested['side']}")
            material_key = f"material_{requested['material']}"
            _require(rendered.get("material_asset_sha256") == contract["asset_sha256"].get(material_key),
                     f"Rendered material asset changed for {key} {requested['side']}")
        midpoint = [
            sum(float(obj["3d_coords"][axis]) for obj in scene["objects"]) / 2.0
            for axis in range(3)
        ]
        _require(_vector_matches(row["camera"].get("look_at_target"), midpoint),
                 f"Camera does not target the scene midpoint for {key}")
        hashes_by_scene[row["scene_id"]].add(_file_sha256(paths["image"]))
        realized.append(row)
    for scene_id, hashes in hashes_by_scene.items():
        _require(len(hashes) == 20, f"Scene {scene_id} must have 20 unique RGB images")
    return realized


def _stage(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        same = destination.is_symlink() and destination.resolve() == source.resolve()
        same = same or (destination.is_file() and filecmp.cmp(destination, source, shallow=False))
        _require(same, f"Existing staged file differs: {destination}")
        return
    try:
        os.symlink(source.resolve(), destination)
    except OSError:
        shutil.copy2(source, destination)


def build_training_outputs(render_root: Path, realized: list[dict[str, Any]], states: list[dict[str, Any]],
                           output_dir: Path, base_config_path: Path) -> dict[str, Any]:
    _require(yaml is not None, "PyYAML is required")
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    _require(base_config["args"]["modifier_token"] == MODIFIER_TOKEN, "Eight-token modifier order changed")
    _require(base_config["args"]["initializer_token"] == INITIALIZER_TOKEN, "Initializer order changed")
    asset_root = output_dir / "training" / "train_assets"
    mask_root = output_dir / "training" / "train_masks"
    concepts = []
    staged_count = 0
    for state in states:
        image_dir, mask_dir = asset_root / state["state_id"], mask_root / state["state_id"]
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        selected = []
        for row in realized:
            if row["split"] != "train":
                continue
            for obj in row["objects"]:
                if obj["state_id"] == state["state_id"]:
                    selected.append((row, obj["side"]))
        _require(len(selected) == 32, f"State {state['state_id']} must have 32 object records")
        for row, side in sorted(selected, key=lambda item: (item[0]["scene_id"], item[0]["view_index"])):
            name = f"{row['scene_id']}__view_{row['view_index']:02d}"
            _stage(_resolved_under(render_root, row["image"], "image"), image_dir / f"{name}.jpg")
            _stage(_resolved_under(render_root, row["masks"][side], f"mask_{side}"), mask_dir / f"{name}.png")
            staged_count += 1
        _require(len(list(image_dir.iterdir())) == len(list(mask_dir.iterdir())) == 32,
                 f"Staging count changed for {state['state_id']}")
        _require(all(path.suffix == ".jpg" for path in image_dir.iterdir()), f"Image staging contaminated: {image_dir}")
        _require(all(path.suffix == ".png" for path in mask_dir.iterdir()), f"Mask staging contaminated: {mask_dir}")
        concepts.append({
            "instance_prompt": state["instance_prompt"],
            "instance_data_dir": str(image_dir.resolve()),
            "instance_mask_dir": str(mask_dir.resolve()),
        })
    _require(staged_count == 576, "Expected 576 staged object records")
    concepts_path = output_dir / "training" / "concepts.json"
    _write_json(concepts_path, concepts)
    config = json.loads(json.dumps(base_config))
    config["args"]["concepts_list"] = str(concepts_path.resolve())
    config["data_manifest"] = str((output_dir / "realized_scenes.jsonl").resolve())
    config_path = output_dir / "training" / "train_config_seed42.json"
    _write_json(config_path, config)
    return {"concepts": str(concepts_path), "config": str(config_path), "object_record_count": staged_count}


def build_authorized_training_package(
    prepared_root: Path,
    output_dir: Path,
    states: list[dict[str, Any]],
    base_config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Create immutable smoke/full configs after the renderer human gate passes."""
    _require(yaml is not None, "PyYAML is required")
    prepared_root = prepared_root.resolve()
    status = _read_json(prepared_root / "protocol_status.json")
    _require(status.get("status") == "validated_pending_human_review", "Prepared renderer status changed")
    _require(status.get("training_object_record_count") == 576, "Prepared object record count changed")
    realized_manifest = prepared_root / "realized_scenes.jsonl"
    _require(len(_read_jsonl(realized_manifest)) == 360, "Prepared scene manifest must contain 360 rows")
    concepts_path = prepared_root / "training" / "concepts.json"
    concepts = _read_json(concepts_path)
    _require(isinstance(concepts, list) and len(concepts) == len(states), "Expected 18 prepared concepts")

    verified = []
    for state, concept in zip(states, concepts):
        _require(concept.get("instance_prompt") == state["instance_prompt"],
                 f"Prepared prompt changed for {state['state_id']}")
        image_dir = Path(concept.get("instance_data_dir", "")).resolve()
        mask_dir = Path(concept.get("instance_mask_dir", "")).resolve()
        images = sorted(image_dir.glob("*.jpg"))
        masks = sorted(mask_dir.glob("*.png"))
        _require(len(images) == len(masks) == 32, f"Prepared count changed for {state['state_id']}")
        _require([path.stem for path in images] == [path.stem for path in masks],
                 f"Prepared image/mask stems differ for {state['state_id']}")
        verified.append((state, images[0], masks[0]))

    _require_empty_output_dir(output_dir)
    generated_configs = {}
    for smoke_name, selected, template_path in (
        ("smoke_2step", verified[:2], SMOKE_2STEP_CONFIG),
        ("smoke_18step", verified, SMOKE_18STEP_CONFIG),
    ):
        smoke_root = output_dir / "smokes" / smoke_name
        smoke_concepts = []
        for state, image, mask in selected:
            image_dir = smoke_root / "train_assets" / state["state_id"]
            mask_dir = smoke_root / "train_masks" / state["state_id"]
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            _stage(image, image_dir / image.name)
            _stage(mask, mask_dir / mask.name)
            smoke_concepts.append({
                "instance_prompt": state["instance_prompt"],
                "instance_data_dir": str(image_dir.resolve()),
                "instance_mask_dir": str(mask_dir.resolve()),
            })
        smoke_concepts_path = smoke_root / "concepts.json"
        _write_json(smoke_concepts_path, smoke_concepts)
        config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        config["args"]["concepts_list"] = str(smoke_concepts_path.resolve())
        config["data_manifest"] = str(realized_manifest.resolve())
        config_path = smoke_root / "train_config.json"
        _write_json(config_path, config)
        generated_configs[smoke_name] = str(config_path)

    full_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    full_config["args"]["concepts_list"] = str(concepts_path.resolve())
    full_config["data_manifest"] = str(realized_manifest.resolve())
    full_config_path = output_dir / "full_training" / "train_config_seed42.json"
    full_config_path.parent.mkdir(parents=True)
    _write_json(full_config_path, full_config)
    decision = {
        "status": "passed",
        "training_authorized": True,
        "authorized_at": "2026-09-01",
        "human_observation": "A small minority of medium-to-heavy occlusions was accepted.",
        "scope": "two_object_smokes_then_seed42_1500_steps",
    }
    _write_json(output_dir / "human_gate_decision.json", decision)
    result = {
        "status": "ready_for_training_smokes",
        "prepared_root": str(prepared_root),
        "realized_scene_count": 360,
        "full_training_object_record_count": 576,
        "shared_modifier_token_count": 8,
        "training_uses_gt_instance_masks": True,
        "smoke_configs": generated_configs,
        "full_config": str(full_config_path),
        "human_gate": decision,
    }
    _write_json(output_dir / "training_package_status.json", result)
    return result


def write_review_outputs(render_root: Path, realized: list[dict[str, Any]], output_dir: Path):
    review_path = output_dir / "two_object_render_human_review.csv"
    fields = ["generation_id", "scene_id", "pair_index", "orientation", "view_index", "split", "image",
              "left_state", "right_state", "left_correct", "right_correct", "masks_aligned", "objects_separate",
              "objects_complete", "background_neutral", "lighting_ok", "artifact_or_invalid", "confidence", "comment"]
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in realized:
            writer.writerow({
                "generation_id": f"{row['scene_id']}:v{row['view_index']:02d}",
                "scene_id": row["scene_id"], "pair_index": row["pair_index"],
                "orientation": row["orientation"], "view_index": row["view_index"],
                "split": row["split"], "image": row["image"],
                "left_state": row["objects"][0]["state_id"], "right_state": row["objects"][1]["state_id"],
            })
    selected = (0, 4, 8, 12, 16)
    by_key = {(row["scene_id"], row["view_index"]): row for row in realized}
    scene_ids = sorted({row["scene_id"] for row in realized})
    sheet = Image.new("RGB", (1000, len(scene_ids) * 224), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, scene_id in enumerate(scene_ids):
        for column, view in enumerate(selected):
            row = by_key[(scene_id, view)]
            with Image.open(_resolved_under(render_root, row["image"], "image")) as source:
                tile = source.convert("RGB"); tile.thumbnail((192, 192), Image.Resampling.LANCZOS)
                sheet.paste(tile, (column * 200 + (200 - tile.width) // 2, row_index * 224 + 24))
            draw.text((column * 200 + 4, row_index * 224 + 4), f"{scene_id} v{view:02d}", fill="black")
    sheet_path = output_dir / "two_object_18scene_contact_sheet.png"
    sheet.save(sheet_path)
    pair_paths = []
    for pair_index in range(9):
        pair_sheet = Image.new("RGB", (1000, 2 * 224), "white")
        pair_draw = ImageDraw.Draw(pair_sheet)
        for row_index, orientation in enumerate(("forward", "swapped")):
            scene_id = f"pair_{pair_index:02d}_{orientation}"
            for column, view in enumerate(selected):
                row = by_key[(scene_id, view)]
                with Image.open(_resolved_under(render_root, row["image"], "image")) as source:
                    tile = source.convert("RGB"); tile.thumbnail((192, 192), Image.Resampling.LANCZOS)
                    pair_sheet.paste(tile, (column * 200 + (200 - tile.width) // 2, row_index * 224 + 24))
                pair_draw.text((column * 200 + 4, row_index * 224 + 4), f"{orientation} v{view:02d}", fill="black")
        path = output_dir / "orientation_pairs" / f"pair_{pair_index:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True); pair_sheet.save(path); pair_paths.append(path)
    return review_path, sheet_path, pair_paths


def realize_protocol(manifest: dict[str, Any], protocol: dict[str, Any], render_root: Path,
                     render_manifest: Path, output_dir: Path, base_config: Path) -> dict[str, Any]:
    _require_empty_output_dir(output_dir)
    states, _ = validate_protocol(manifest, protocol)
    realized = validate_realization(render_root.resolve(), _read_jsonl(render_manifest), manifest, protocol)
    realized_path = output_dir / "realized_scenes.jsonl"
    _write_jsonl(realized_path, realized)
    training = build_training_outputs(render_root.resolve(), realized, states, output_dir, base_config)
    review, sheet, pair_sheets = write_review_outputs(render_root.resolve(), realized, output_dir)
    gate = {
        "status": "pending_human_review", "training_authorized": False,
        "required_checks": ["both_objects_correct", "instances_separate", "masks_aligned",
                            "objects_not_clipped", "left_right_balance", "fixed_neutral_background"],
    }
    _write_json(output_dir / "human_gate_decision.json", gate)
    result = {
        "status": "validated_pending_human_review", "realized_scene_count": 360,
        "training_object_record_count": 576, "training_uses_gt_instance_masks": True,
        "shared_modifier_token_count": 8, "cross_object_caa": False,
        "realized_manifest": str(realized_path), "training": training,
        "human_review_csv": str(review), "contact_sheet": str(sheet),
        "orientation_pair_sheets": [str(path) for path in pair_sheets],
    }
    _write_json(output_dir / "protocol_status.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "realize", "authorize-training"))
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--renderer", type=Path)
    parser.add_argument("--render-root", type=Path)
    parser.add_argument("--render-manifest", type=Path)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Iterable[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    manifest, protocol, _, _ = load_inputs(args.base_manifest, args.protocol)
    if args.command == "plan":
        result = plan_protocol(manifest, protocol, args.output_dir.resolve(), args.renderer)
    elif args.command == "realize":
        _require(args.render_root is not None and args.render_manifest is not None,
                 "--render-root and --render-manifest are required for realize")
        result = realize_protocol(manifest, protocol, args.render_root, args.render_manifest,
                                  args.output_dir.resolve(), args.base_config)
    else:
        _require(args.prepared_root is not None, "--prepared-root is required for authorize-training")
        result = build_authorized_training_package(
            args.prepared_root, args.output_dir.resolve(), build_states(manifest), args.base_config
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
