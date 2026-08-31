#!/usr/bin/env python3
"""Plan and validate the locked CLEVR 3x3 multiview held-out protocol.

The ``plan`` command emits render requests only.  Camera, lighting, background,
scene and file fields remain null until a real renderer supplies a realization
manifest.  The ``realize`` command validates all 180 rendered views and builds
separate, image-only ColorPeel assets for folds A/B/C.  It never modifies the
single-view baseline staging and never places GT masks in training directories.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

from PIL import Image, ImageChops, ImageDraw, ImageOps, ImageStat

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.colorpeel_ice.multiview_render_contract import (
    EXPECTED_PROFILE,
    EXPECTED_PROFILE_V2,
    canonical_sha256,
    look_at_alignment,
    look_at_y_up_alignment,
    orbit_jitter_metadata,
    orbit_location,
    official_jitter_metadata,
    spherical_pose,
)

try:
    import yaml
except ImportError:  # pragma: no cover - only needed while realizing train configs
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_DIR = REPO_ROOT / "experiments" / "clevr_subject_color_3x3" / "manifests"
DEFAULT_BASE_MANIFEST = MANIFEST_DIR / "clevr_3x3_manifest.json"
DEFAULT_PROTOCOL = MANIFEST_DIR / "clevr_multiview_protocol.json"
DEFAULT_BASE_CONFIG = (
    REPO_ROOT
    / "experiments"
    / "clevr_subject_color_3x3"
    / "configs"
    / "multiview_base_turquoise.yaml"
)
RENDERER_FIELDS = (
    "camera", "light", "background", "scene_json", "image", "mask", "background_mask",
)
EXPECTED_MODIFIER_TOKEN = "<s1*>+<s2*>+<s3*>+<c1*>+<c2*>+<c3*>"
EXPECTED_INITIALIZER_TOKEN = "cube+sphere+cylinder+red+turquoise+gray"
EXPECTED_RENDERER_PROFILE = {
    "id": "multiview_render_v1",
    "config": "../configs/multiview_render.json",
    "background": {
        "profile_id": "clevr_neutral_fixed_v1",
        "varied": False,
        "world_rgba": [0.05, 0.05, 0.05, 1.0],
        "ground_rgba": [0.5, 0.5, 0.5, 1.0],
    },
}
EXPECTED_RENDERER_PROFILE_V2 = {
    "id": "multiview_render_v2",
    "config": "../configs/multiview_render_v2.json",
    "background": EXPECTED_PROFILE_V2["background"],
}
EXPECTED_PROTOCOLS = {
    "clevr_subject_color_3x3_multiview_v1": {
        "schema": "clevr_multiview_protocol.schema.json",
        "version": 1,
        "profile": EXPECTED_PROFILE,
        "renderer_profile": EXPECTED_RENDERER_PROFILE,
    },
    "clevr_subject_color_3x3_multiview_v2": {
        "schema": "clevr_multiview_protocol_v2.schema.json",
        "version": 2,
        "profile": EXPECTED_PROFILE_V2,
        "renderer_profile": EXPECTED_RENDERER_PROFILE_V2,
    },
}
EXPECTED_FOLDS = {
    "A": {("cube", "red"), ("sphere", "cyan"), ("cylinder", "gray")},
    "B": {("cube", "cyan"), ("sphere", "gray"), ("cylinder", "red")},
    "C": {("cube", "gray"), ("sphere", "red"), ("cylinder", "cyan")},
}


def _protocol_spec(protocol: dict[str, Any]) -> dict[str, Any]:
    protocol_id = protocol.get("protocol_id")
    _require(protocol_id in EXPECTED_PROTOCOLS, "Protocol ID changed")
    return EXPECTED_PROTOCOLS[protocol_id]


TRAINING_SEEDS = (42, 43, 44)


class ProtocolError(RuntimeError):
    """Raised when the locked protocol or a renderer realization is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read valid JSON from {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                _require(isinstance(value, dict), f"Expected an object at {path}:{line_number}")
                records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read valid JSONL from {path}: {exc}") from exc
    return records


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def load_inputs(base_manifest_path: Path, protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    base_manifest = _read_json(base_manifest_path)
    protocol = _read_json(protocol_path)
    _require(isinstance(base_manifest, dict), "Base manifest must be an object")
    _require(isinstance(protocol, dict), "Protocol manifest must be an object")
    validate_protocol(base_manifest, protocol)
    return base_manifest, protocol


def validate_protocol(base_manifest: dict[str, Any], protocol: dict[str, Any]) -> None:
    spec = _protocol_spec(protocol)
    _require(protocol.get("$schema") == spec["schema"], "Protocol schema changed")
    _require(protocol.get("version") == spec["version"], "Protocol version changed")
    _require(protocol.get("base_manifest") == "clevr_3x3_manifest.json", "Protocol base manifest changed")
    samples = base_manifest.get("samples", [])
    _require(len(samples) == 9, "Base manifest must contain exactly nine cells")
    _require(len({sample.get("id") for sample in samples}) == 9, "Cell IDs must be unique")
    expected_grid = {
        (shape, color)
        for shape in ("cube", "sphere", "cylinder")
        for color in ("red", "cyan", "gray")
    }
    actual_grid = {(sample.get("shape"), sample.get("color")) for sample in samples}
    _require(actual_grid == expected_grid, "Base manifest must be the locked 3x3 Cartesian grid")
    _require(all(sample.get("material") == "metal" for sample in samples), "All cells must use metal")

    _require(protocol.get("views_per_cell") == 20, "views_per_cell must be 20")
    _require(protocol.get("view_splits") == {
        "train": {"start": 0, "stop": 16},
        "audit": {"start": 16, "stop": 20},
    }, "View split must be train 0:16 and audit 16:20")
    _require(protocol.get("render_seed") == {
        "base": 420000,
        "cell_stride": 100,
        "formula": "base + cell_index * cell_stride + view_index",
    }, "Render seed rule differs from the locked protocol")
    _require(protocol.get("renderer_profile") == spec["renderer_profile"], "Renderer profile changed")

    folds = protocol.get("folds", [])
    actual_folds = {
        fold.get("id"): {tuple(pair) for pair in fold.get("held_out", [])}
        for fold in folds
    }
    _require(actual_folds == EXPECTED_FOLDS, "Held-out folds A/B/C differ from the locked matchings")
    _require(len(folds) == 3, "Protocol must define exactly three folds")
    for fold_id, held_out in actual_folds.items():
        train_cells = expected_grid - held_out
        for shape in ("cube", "sphere", "cylinder"):
            partners = {color for candidate_shape, color in train_cells if candidate_shape == shape}
            _require(len(partners) == 2, f"Fold {fold_id} leaks subject axis: {shape} has {len(partners)} partners")
        for color in ("red", "cyan", "gray"):
            partners = {shape for shape, candidate_color in train_cells if candidate_color == color}
            _require(len(partners) == 2, f"Fold {fold_id} leaks color axis: {color} has {len(partners)} partners")

    contract = protocol.get("realization_contract", {})
    _require(contract.get("renderer_owned_fields") == list(RENDERER_FIELDS), "Renderer-owned fields changed")
    _require(contract.get("resolution") == [512, 512], "Realized views must be 512x512")
    _require(contract.get("image_mode") == "RGB", "Realized image mode must be RGB")
    _require(contract.get("mask_mode") == "L", "Realized mask mode must be L")
    _require(contract.get("mask_values") == [0, 255], "Realized mask values must be [0, 255]")
    _require(contract.get("empirical_rgb") == {
        "space": "srgb_u8",
        "statistic": "masked_mean",
        "source": "realized_view_gt_mask",
    }, "Empirical RGB contract changed")


def build_render_requests(base_manifest: dict[str, Any], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    profile = _protocol_spec(protocol)["profile"]
    requests: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(base_manifest["samples"]):
        for view_index in range(20):
            split = "train" if view_index < 16 else "audit"
            request = {
                "cell_id": cell["id"],
                "cell_index": cell_index,
                "shape": cell["shape"],
                "color": cell["color"],
                "material": "metal",
                "subject_token": cell["subject_token"],
                "color_token": cell["color_token"],
                "nominal_rgb": cell["rgb"],
                "view_index": view_index,
                "split": split,
                "render_seed": 420000 + cell_index * 100 + view_index,
                "renderer_profile_id": profile["profile_id"],
                "renderer_profile_sha256": canonical_sha256(profile),
                "camera": None,
                "light": None,
                "background": None,
                "scene_json": None,
                "image": None,
                "mask": None,
                "background_mask": None,
                "empirical_rgb": None,
            }
            requests.append(request)
    validate_render_requests(requests, base_manifest, protocol)
    return requests


def validate_render_requests(
    requests: list[dict[str, Any]],
    base_manifest: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    profile = _protocol_spec(protocol)["profile"]
    _require(len(requests) == 180, f"Expected 180 render requests, got {len(requests)}")
    keys = {(record.get("cell_id"), record.get("view_index")) for record in requests}
    _require(len(keys) == 180, "Each cell/view pair must be unique")
    cells = {sample["id"]: (index, sample) for index, sample in enumerate(base_manifest["samples"])}
    for record in requests:
        cell_id = record.get("cell_id")
        _require(cell_id in cells, f"Unknown cell_id: {cell_id}")
        cell_index, cell = cells[cell_id]
        view_index = record.get("view_index")
        _require(isinstance(view_index, int) and 0 <= view_index < 20, f"Invalid view_index for {cell_id}")
        expected_split = "train" if view_index < 16 else "audit"
        _require(record.get("split") == expected_split, f"Wrong split for {cell_id} view {view_index}")
        _require(record.get("cell_index") == cell_index, f"Wrong cell_index for {cell_id}")
        _require(record.get("render_seed") == 420000 + cell_index * 100 + view_index,
                 f"Wrong render_seed for {cell_id} view {view_index}")
        _require(record.get("renderer_profile_id") == profile["profile_id"],
                 f"Wrong renderer profile for {cell_id}")
        _require(record.get("renderer_profile_sha256") == canonical_sha256(profile),
                 f"Wrong renderer profile hash for {cell_id}")
        for field in ("shape", "color", "subject_token", "color_token"):
            _require(record.get(field) == cell[field], f"Wrong {field} for {cell_id}")
        _require(record.get("nominal_rgb") == cell["rgb"], f"Wrong nominal_rgb for {cell_id}")
        _require(record.get("material") == "metal", f"Wrong material for {cell_id}")
        for field in RENDERER_FIELDS:
            _require(record.get(field) is None, f"Protocol generator must not fabricate {field}")
        _require(record.get("empirical_rgb") is None, "Protocol generator must not fabricate empirical_rgb")

    for cell_id in cells:
        records = [record for record in requests if record["cell_id"] == cell_id]
        _require(sum(record["split"] == "train" for record in records) == 16, f"{cell_id} needs 16 train views")
        _require(sum(record["split"] == "audit" for record in records) == 4, f"{cell_id} needs 4 audit views")


def plan_protocol(
    base_manifest: dict[str, Any],
    protocol: dict[str, Any],
    output_dir: Path,
    renderer: Path | None,
) -> dict[str, Any]:
    requests = build_render_requests(base_manifest, protocol)
    profile = _protocol_spec(protocol)["profile"]
    output_dir = output_dir.resolve()
    _require_empty_output_dir(output_dir)
    requests_path = output_dir / "render_requests.jsonl"
    _write_jsonl(requests_path, requests)
    renderer_path = renderer.resolve() if renderer is not None else None
    renderer_available = renderer_path is not None and renderer_path.is_file()
    status = {
        "status": "planned" if renderer_available else "blocked",
        "blocked_reason": None if renderer_available else "multiview_renderer_not_provided_or_missing",
        "renderer": str(renderer_path) if renderer_path is not None else None,
        "renderer_available": renderer_available,
        "renderer_profile_id": profile["profile_id"],
        "renderer_profile_sha256": canonical_sha256(profile),
        "render_request_manifest": str(requests_path),
        "request_count": 180,
        "images_created": 0,
        "renderer_owned_fields_populated": False,
    }
    _write_json(output_dir / "protocol_status.json", status)
    return status


def _resolved_under(root: Path, relative: Any, field: str) -> Path:
    _require(isinstance(relative, str) and relative, f"Realized {field} must be a nonempty relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ProtocolError(f"Realized {field} escapes render root: {relative}") from exc
    _require(candidate.is_file(), f"Missing realized {field}: {candidate}")
    return candidate


def _require_empty_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        _require(output_dir.is_dir(), f"Output path is not a directory: {output_dir}")
        _require(not any(output_dir.iterdir()), f"Output directory must be empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_key(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _validate_vector_sum(base: Any, offset: Any, final: Any, label: str) -> None:
    _require(
        all(isinstance(value, (int, float)) for vector in (base, offset, final) for value in vector)
        if all(isinstance(vector, list) and len(vector) == 3 for vector in (base, offset, final))
        else False,
        f"{label} vectors must contain three numeric values",
    )
    _require(
        all(abs((float(base[index]) + float(offset[index])) - float(final[index])) <= 1e-6 for index in range(3)),
        f"{label} final location does not equal base plus jitter",
    )


def _vector_matches(actual: Any, expected: list[float]) -> bool:
    return (
        isinstance(actual, list)
        and len(actual) == len(expected)
        and all(
            isinstance(value, (int, float)) and abs(float(value) - target) <= 1e-6
            for value, target in zip(actual, expected)
        )
    )


def _numeric_vector(actual: Any, length: int) -> bool:
    return (
        isinstance(actual, list)
        and len(actual) == length
        and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in actual)
    )


def _numeric_mapping_matches(actual: Any, expected: dict[str, float]) -> bool:
    return (
        isinstance(actual, dict)
        and set(actual) == set(expected)
        and all(
            isinstance(actual[name], (int, float))
            and abs(float(actual[name]) - float(expected[name])) <= 1e-6
            for name in expected
        )
    )


def _validate_v1_camera(camera: dict[str, Any], expected: dict[str, Any], key: tuple[str, int]) -> None:
    _require(camera.get("jitter_offset") == expected["camera_offset"], f"Camera jitter disagrees for {key}")
    _require(camera.get("rotation_policy") == "preserve_base_scene", f"Camera rotation policy disagrees for {key}")
    _validate_vector_sum(
        camera.get("base_location"), camera.get("jitter_offset"), camera.get("final_location"), f"Camera {key}"
    )
    _require(isinstance(camera.get("rotation_euler"), list) and len(camera["rotation_euler"]) == 3,
             f"Camera rotation is missing for {key}")


def _validate_v2_camera(
    camera: dict[str, Any], expected: dict[str, Any], profile: dict[str, Any], key: tuple[str, int]
) -> None:
    _require(camera.get("sampling_model") == profile["camera"]["sampling_model"],
             f"Camera sampling model disagrees for {key}")
    expected_target_policy = (
        "scene_midpoint" if profile["profile_id"] == "multiview_render_v4_two_object"
        else "object_location"
    )
    _require(camera.get("target_policy") == expected_target_policy, f"Camera target policy disagrees for {key}")
    _require(camera.get("rotation_policy") == profile["camera"]["rotation_policy"],
             f"Camera rotation policy disagrees for {key}")
    _require(camera.get("base_constraint_policy") == profile["camera"]["base_constraint_policy"],
             f"Camera base constraint policy disagrees for {key}")
    constraints = camera.get("base_constraints")
    _require(isinstance(constraints, list), f"Camera base constraints are missing for {key}")
    for constraint in constraints:
        _require(
            isinstance(constraint, dict)
            and set(constraint) == {"name", "type", "mute", "influence", "target"}
            and isinstance(constraint["name"], str)
            and isinstance(constraint["type"], str)
            and isinstance(constraint["mute"], bool)
            and isinstance(constraint["influence"], (int, float)),
            f"Camera base constraint metadata is invalid for {key}",
        )
    _require(camera.get("final_constraints_muted") is True,
             f"Camera base constraints were not muted for {key}")
    target = camera.get("look_at_target")
    _require(isinstance(target, list) and len(target) == 3 and
             all(isinstance(value, (int, float)) for value in target),
             f"Camera look-at target is invalid for {key}")
    jitter = camera.get("orbit_jitter")
    expected_jitter = expected["camera_orbit_jitter"]
    _require(_numeric_mapping_matches(jitter, expected_jitter), f"Camera orbit jitter disagrees for {key}")
    _require(-profile["camera"]["azimuth_jitter_degrees"] <= float(jitter["azimuth_degrees"]) <
             profile["camera"]["azimuth_jitter_degrees"], f"Camera azimuth jitter is out of range for {key}")
    _require(-profile["camera"]["elevation_jitter_degrees"] <= float(jitter["elevation_degrees"]) <
             profile["camera"]["elevation_jitter_degrees"], f"Camera elevation jitter is out of range for {key}")
    _require(-profile["camera"]["distance_jitter_fraction"] <= float(jitter["distance_fraction"]) <
             profile["camera"]["distance_jitter_fraction"], f"Camera distance jitter is out of range for {key}")

    base_pose = camera.get("base_pose", {})
    final_pose = camera.get("final_pose", {})
    base_location = base_pose.get("location")
    final_location = final_pose.get("location")
    _require(_numeric_vector(base_location, 3), f"Camera base location is invalid for {key}")
    _require(_numeric_vector(final_location, 3), f"Camera final location is invalid for {key}")
    recomputed_base = spherical_pose(base_location, target)
    _require(_numeric_mapping_matches(base_pose.get("spherical"), recomputed_base),
             f"Camera base spherical pose disagrees for {key}")
    requested = {
        "radius": recomputed_base["radius"] * (1.0 + float(jitter["distance_fraction"])),
        "azimuth_degrees": (
            recomputed_base["azimuth_degrees"] + float(jitter["azimuth_degrees"]) + 180.0
        ) % 360.0 - 180.0,
        "elevation_degrees": recomputed_base["elevation_degrees"] + float(jitter["elevation_degrees"]),
    }
    _require(requested["radius"] > 0.0 and -89.0 < requested["elevation_degrees"] < 89.0,
             f"Camera requested orbit pose is invalid for {key}")
    _require(_numeric_mapping_matches(camera.get("requested_final_spherical"), requested),
             f"Camera requested final spherical pose disagrees for {key}")
    expected_location = orbit_location(
        target, requested["radius"], requested["azimuth_degrees"], requested["elevation_degrees"]
    )
    _require(_vector_matches(final_location, expected_location), f"Camera final orbit location disagrees for {key}")
    _require(_numeric_mapping_matches(final_pose.get("spherical"), spherical_pose(final_location, target)),
             f"Camera final spherical pose disagrees for {key}")
    quaternion = final_pose.get("rotation_quaternion_wxyz")
    _require(isinstance(quaternion, list) and len(quaternion) == 4 and
             all(isinstance(value, (int, float)) for value in quaternion),
             f"Camera final quaternion is invalid for {key}")
    quaternion_norm = math.sqrt(sum(float(value) ** 2 for value in quaternion))
    _require(abs(quaternion_norm - 1.0) <= 1e-6, f"Camera final quaternion is not normalized for {key}")
    alignment = look_at_alignment(final_location, target, quaternion)
    y_up_alignment = look_at_y_up_alignment(final_location, target, quaternion)
    _require(alignment >= 1.0 - 1e-6, f"Camera does not look at object center for {key}")
    _require(y_up_alignment >= 1.0 - 1e-6, f"Camera roll does not preserve Y-up for {key}")
    look_at = camera.get("look_at", {})
    _require(look_at.get("track_axis") == "-Z" and look_at.get("up_axis") == "Y",
             f"Camera look-at axes disagree for {key}")
    _require(isinstance(look_at.get("alignment_cosine"), (int, float)) and
             abs(float(look_at["alignment_cosine"]) - alignment) <= 1e-6,
             f"Camera look-at alignment metadata disagrees for {key}")
    _require(isinstance(look_at.get("y_up_alignment_cosine"), (int, float)) and
             abs(float(look_at["y_up_alignment_cosine"]) - y_up_alignment) <= 1e-6,
             f"Camera Y-up alignment metadata disagrees for {key}")
    projected = look_at.get("target_projected_xy")
    _require(_vector_matches(projected, [0.5, 0.5]), f"Camera target is not at optical center for {key}")
    _require(_vector_matches(camera.get("shift_xy"), [0.0, 0.0]), f"Camera lens shift changed for {key}")
    for pose_name, pose in (("base", base_pose), ("final", final_pose)):
        _require(isinstance(pose.get("rotation_euler_xyz"), list) and len(pose["rotation_euler_xyz"]) == 3,
                 f"Camera {pose_name} Euler rotation is missing for {key}")
        _require(isinstance(pose.get("rotation_quaternion_wxyz"), list) and
                 len(pose["rotation_quaternion_wxyz"]) == 4,
                 f"Camera {pose_name} quaternion is missing for {key}")


def _validate_view_metadata(
    record: dict[str, Any], key: tuple[str, int], profile: dict[str, Any] = EXPECTED_PROFILE
) -> None:
    expected = (
        official_jitter_metadata(record["render_seed"], profile)
        if profile["profile_id"] == "multiview_render_v1"
        else orbit_jitter_metadata(record["render_seed"], profile)
    )
    camera = record["camera"]
    _require(camera.get("name") == "Camera", f"Camera name disagrees for {key}")
    if profile["profile_id"] == "multiview_render_v1":
        _validate_v1_camera(camera, expected, key)
    else:
        _validate_v2_camera(camera, expected, profile, key)
    _require(isinstance(camera.get("lens"), (int, float)) and camera["lens"] > 0, f"Camera lens is invalid for {key}")
    _require(isinstance(camera.get("sensor_width"), (int, float)) and camera["sensor_width"] > 0,
             f"Camera sensor is invalid for {key}")

    light = record["light"]
    _require(light.get("order") == profile["lights"]["order"], f"Light order disagrees for {key}")
    lights = light.get("lights", {})
    _require(set(lights) == set(profile["lights"]["order"]), f"Light set disagrees for {key}")
    for name in profile["lights"]["order"]:
        metadata = lights[name]
        expected_offset = expected["light_offsets"][name]
        _require(metadata.get("jitter_offset") == expected_offset, f"{name} jitter disagrees for {key}")
        _validate_vector_sum(
            metadata.get("base_location"), metadata.get("jitter_offset"), metadata.get("final_location"),
            f"{name} {key}",
        )
        _require(metadata.get("rgb") == [1.0, 1.0, 1.0], f"{name} color disagrees for {key}")
        _require(isinstance(metadata.get("type"), str) and metadata["type"], f"{name} type is missing for {key}")
        _require(isinstance(metadata.get("energy"), (int, float)), f"{name} energy is missing for {key}")
    fixed_lights = light.get("fixed_lights", {})
    _require(set(fixed_lights) == set(profile["lights"]["fixed_order"]),
             f"Fixed light set disagrees for {key}")
    for name in profile["lights"]["fixed_order"]:
        metadata = fixed_lights[name]
        _require(metadata.get("base_location") == metadata.get("final_location"),
                 f"Fixed light {name} moved for {key}")
        _require(metadata.get("rgb") == [1.0, 1.0, 1.0], f"Fixed light {name} color disagrees for {key}")
        _require(isinstance(metadata.get("type"), str) and metadata["type"],
                 f"Fixed light {name} type is missing for {key}")
        _require(isinstance(metadata.get("energy"), (int, float)),
                 f"Fixed light {name} energy is missing for {key}")


def _validate_realized_image(
    image_path: Path,
    mask_path: Path,
    background_mask_path: Path,
) -> tuple[list[float], int]:
    try:
        with (
            Image.open(image_path) as image,
            Image.open(mask_path) as mask,
            Image.open(background_mask_path) as background_mask,
        ):
            image.load()
            mask.load()
            background_mask.load()
            _require(image.size == (512, 512) and image.mode == "RGB", f"Invalid RGB image: {image_path}")
            _require(mask.size == (512, 512) and mask.mode == "L", f"Invalid L mask: {mask_path}")
            _require(
                background_mask.size == (512, 512) and background_mask.mode == "L",
                f"Invalid L background mask: {background_mask_path}",
            )
            histogram = mask.histogram()
            values = [value for value, count in enumerate(histogram) if count]
            _require(values == [0, 255], f"Mask must contain exactly [0, 255]: {mask_path}")
            background_histogram = background_mask.histogram()
            background_values = [value for value, count in enumerate(background_histogram) if count]
            _require(
                background_values == [0, 255],
                f"Background mask must contain exactly [0, 255]: {background_mask_path}",
            )
            difference = ImageChops.difference(background_mask, ImageOps.invert(mask))
            _require(difference.getbbox() is None, f"Object/background masks are not complements: {mask_path}")
            foreground = histogram[255]
            _require(foreground > 0, f"Mask is empty: {mask_path}")
            ratio = foreground / (512 * 512)
            _require(0.005 <= ratio <= 0.90, f"Mask ratio is outside 0.005-0.90: {mask_path}")
            bbox = mask.getbbox()
            _require(bbox is not None, f"Mask bounding box is empty: {mask_path}")
            _require(
                bbox[0] > 0 and bbox[1] > 0 and bbox[2] < 512 and bbox[3] < 512,
                f"Object mask touches an image edge: {mask_path}",
            )
            mean = [round(value, 6) for value in ImageStat.Stat(image, mask=mask).mean]
            return mean, foreground
    except OSError as exc:
        raise ProtocolError(f"Cannot decode realized image or mask: {exc}") from exc


def validate_realization(
    render_root: Path,
    realized_records: list[dict[str, Any]],
    base_manifest: dict[str, Any],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    render_root = render_root.resolve()
    _require(render_root.is_dir(), f"Render root is not a directory: {render_root}")
    spec = _protocol_spec(protocol)
    profile = spec["profile"]
    renderer_profile = spec["renderer_profile"]
    expected = build_render_requests(base_manifest, protocol)
    contract = _read_json(render_root / "render_contract.json")
    _require(isinstance(contract, dict), "Render contract must be an object")
    _require(contract.get("schema_version") == 1, "Render contract version changed")
    _require(contract.get("profile_id") == profile["profile_id"], "Render contract profile changed")
    _require(contract.get("profile_sha256") == canonical_sha256(profile), "Render profile hash changed")
    _require(contract.get("requests_sha256") == canonical_sha256(expected), "Render request hash changed")
    _require(contract.get("request_count") == 180, "Render contract request count changed")
    _require(isinstance(contract.get("asset_sha256"), dict) and contract["asset_sha256"],
             "Render contract asset hashes are missing")
    contract_sha256 = canonical_sha256(contract)
    expected_by_key = {(record["cell_id"], record["view_index"]): record for record in expected}
    _require(len(realized_records) == 180, f"Expected 180 realized views, got {len(realized_records)}")
    actual_by_key = {(record.get("cell_id"), record.get("view_index")): record for record in realized_records}
    _require(len(actual_by_key) == 180 and set(actual_by_key) == set(expected_by_key),
             "Realization must contain every cell/view exactly once")

    realized: list[dict[str, Any]] = []
    used_paths: dict[str, set[Path]] = {
        field: set() for field in ("scene_json", "image", "mask", "background_mask")
    }
    image_hashes: dict[str, set[str]] = {sample["id"]: set() for sample in base_manifest["samples"]}
    metadata_values: dict[str, dict[str, set[str]]] = {
        sample["id"]: {field: set() for field in ("camera", "light")}
        for sample in base_manifest["samples"]
    }
    base_camera_values: dict[str, set[str]] = {sample["id"]: set() for sample in base_manifest["samples"]}
    for expected_record in expected:
        key = (expected_record["cell_id"], expected_record["view_index"])
        supplied = actual_by_key[key]
        for field in (
            "cell_index", "shape", "color", "material", "subject_token", "color_token",
            "nominal_rgb", "split", "render_seed", "renderer_profile_id", "renderer_profile_sha256",
        ):
            _require(supplied.get(field) == expected_record[field], f"Realization changed {field} for {key}")
        for field in ("camera", "light", "background"):
            _require(isinstance(supplied.get(field), dict) and supplied[field],
                     f"Renderer must populate nonempty {field} metadata for {key}")
        _require(
            supplied["background"] == renderer_profile["background"],
            f"Renderer changed the fixed background for {key}",
        )
        for field in ("camera", "light"):
            metadata_values[expected_record["cell_id"]][field].add(_metadata_key(supplied[field]))
        _validate_view_metadata(supplied, key, profile)
        if profile["profile_id"] == "multiview_render_v2":
            base_camera_values[expected_record["cell_id"]].add(_metadata_key(supplied["camera"]["base_pose"]))

        paths = {
            field: _resolved_under(render_root, supplied.get(field), field)
            for field in ("scene_json", "image", "mask", "background_mask")
        }
        for field, path in paths.items():
            _require(path not in used_paths[field], f"Realized {field} is reused: {path}")
            used_paths[field].add(path)
        scene = _read_json(paths["scene_json"])
        _require(isinstance(scene, dict), f"Scene JSON must be an object: {paths['scene_json']}")
        for field in ("render_seed", "camera", "light", "background"):
            _require(scene.get(field) == supplied[field], f"Scene {field} disagrees with realization for {key}")
        _require(scene.get("renderer_profile_id") == renderer_profile["id"],
                 f"Scene renderer profile disagrees for {key}")
        _require(scene.get("cycles_seed") == expected_record["render_seed"], f"Scene cycles seed disagrees for {key}")
        renderer = scene.get("renderer", {})
        _require(renderer.get("blender_version") == "4.2.11", f"Scene Blender version disagrees for {key}")
        _require(renderer.get("engine") == "CYCLES", f"Scene render engine disagrees for {key}")
        _require(renderer.get("cycles_samples") == 512, f"Scene Cycles samples disagree for {key}")
        _require(renderer.get("cycles_device") == "CUDA", f"Scene Cycles device disagrees for {key}")
        devices = renderer.get("cuda_devices", [])
        _require(len(devices) == 1 and "V100" in devices[0].get("name", ""), f"Scene CUDA device disagrees for {key}")
        _require(isinstance(scene.get("asset_sha256"), dict) and scene["asset_sha256"],
                 f"Scene asset hashes are missing for {key}")
        _require(scene["asset_sha256"] == contract["asset_sha256"], f"Scene asset hashes disagree for {key}")
        objects = scene.get("objects", [])
        _require(len(objects) == 1, f"Scene must contain one object for {key}")
        for field in ("shape", "color", "material"):
            _require(objects[0].get(field) == expected_record[field], f"Scene {field} disagrees for {key}")
        _require(objects[0].get("material_backend") == "clevr_asset_node_group",
                 f"Scene material backend disagrees for {key}")
        _require(objects[0].get("rotation") == 0.0, f"Scene object rotation disagrees for {key}")
        _require(objects[0].get("nominal_scale") == 1.3, f"Scene nominal scale disagrees for {key}")
        expected_scale = 1.3 / math.sqrt(2.0) if expected_record["shape"] == "cube" else 1.3
        _require(
            _vector_matches(
                objects[0].get("applied_scale"),
                [expected_scale, expected_scale, expected_scale],
            ),
            f"Scene applied scale disagrees for {key}",
        )
        _require(
            _vector_matches(objects[0].get("3d_coords"), [0.0, 0.0, expected_scale]),
            f"Scene object coordinates disagree for {key}",
        )
        if profile["profile_id"] == "multiview_render_v2":
            _require(
                _vector_matches(supplied["camera"].get("look_at_target"), objects[0]["3d_coords"]),
                f"Camera look-at target is not the object center for {key}",
            )

        hashes = supplied.get("artifact_sha256", {})
        _require(isinstance(hashes, dict), f"Artifact hashes are missing for {key}")
        for field, path in paths.items():
            _require(hashes.get(field) == _file_sha256(path), f"Artifact hash disagrees for {key}: {field}")
        _require(supplied.get("renderer_profile_id") == renderer_profile["id"],
                 f"Realization renderer profile disagrees for {key}")
        _require(supplied.get("render_contract_sha256") == contract_sha256,
                 f"Render contract hash disagrees for {key}")

        mean, foreground = _validate_realized_image(
            paths["image"], paths["mask"], paths["background_mask"]
        )
        _require(supplied.get("foreground_pixels") == foreground,
                 f"Foreground pixel count disagrees for {key}")
        image_hashes[expected_record["cell_id"]].add(_file_sha256(paths["image"]))
        record = {**expected_record, **supplied}
        record["empirical_rgb"] = {
            "value": mean,
            "space": "srgb_u8",
            "statistic": "masked_mean",
            "source": "realized_view_gt_mask",
            "source_image": supplied["image"],
            "source_mask": supplied["mask"],
            "foreground_pixels": foreground,
        }
        realized.append(record)
    for cell_id, hashes in image_hashes.items():
        _require(len(hashes) == 20, f"Cell {cell_id} must contain 20 distinct rendered images")
        for field, values in metadata_values[cell_id].items():
            _require(len(values) > 1, f"Cell {cell_id} has no realized {field} variation")
        if profile["profile_id"] == "multiview_render_v2":
            _require(len(base_camera_values[cell_id]) == 1,
                     f"Cell {cell_id} changed the base camera pose across views")
    return realized


def write_human_review_outputs(
    render_root: Path,
    realized: list[dict[str, Any]],
    base_manifest: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    review_path = output_dir / "multiview_human_review.csv"
    fieldnames = [
        "generation_id", "cell_id", "shape", "color", "material", "render_seed",
        "view_index", "split", "image", "mask", "background_mask", "observed_shape",
        "observed_color", "object_complete", "object_clipped", "mask_aligned",
        "lighting_ok", "background_neutral", "artifact_or_invalid", "confidence",
        "reviewer_id", "comment",
    ]
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in realized:
            writer.writerow({
                "generation_id": f"{record['cell_id']}:v{record['view_index']:02d}",
                "cell_id": record["cell_id"],
                "shape": record["shape"],
                "color": record["color"],
                "material": record["material"],
                "render_seed": record["render_seed"],
                "view_index": record["view_index"],
                "split": record["split"],
                "image": record["image"],
                "mask": record["mask"],
                "background_mask": record["background_mask"],
            })

    selected_views = (0, 4, 8, 12, 16)
    tile_width, tile_height, label_height = 200, 200, 24
    sheet = Image.new("RGB", (tile_width * 5, (tile_height + label_height) * 9), "white")
    draw = ImageDraw.Draw(sheet)
    records_by_key = {(record["cell_id"], record["view_index"]): record for record in realized}
    for row_index, cell in enumerate(base_manifest["samples"]):
        for column_index, view_index in enumerate(selected_views):
            record = records_by_key[(cell["id"], view_index)]
            with Image.open(_resolved_under(render_root, record["image"], "image")) as source:
                thumbnail = source.convert("RGB")
                thumbnail.thumbnail((tile_width - 8, tile_height - 8), Image.Resampling.LANCZOS)
                x = column_index * tile_width + (tile_width - thumbnail.width) // 2
                y = row_index * (tile_height + label_height) + label_height + (tile_height - thumbnail.height) // 2
                sheet.paste(thumbnail, (x, y))
            draw.text(
                (column_index * tile_width + 4, row_index * (tile_height + label_height) + 4),
                f"{cell['id']} v{view_index:02d} {record['split']}",
                fill="black",
            )
    contact_sheet_path = output_dir / "multiview_contact_sheet.png"
    sheet.save(contact_sheet_path)
    return review_path, contact_sheet_path


def _stage_image(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        same_target = destination.is_symlink() and destination.resolve() == source.resolve()
        same_content = destination.is_file() and filecmp.cmp(destination, source, shallow=False)
        _require(same_target or same_content, f"Existing staged file differs: {destination}")
        return
    try:
        os.symlink(source.resolve(), destination)
    except OSError:
        shutil.copy2(source, destination)


def _load_training_config(path: Path) -> dict[str, Any]:
    _require(yaml is not None, "PyYAML is required to derive executable fold training configs")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProtocolError(f"Cannot read baseline training config: {path}") from exc
    _require(isinstance(value, dict) and value.get("stage") == "train", "Baseline config must be a train mapping")
    args = value.get("args", {})
    _require(args.get("modifier_token") == EXPECTED_MODIFIER_TOKEN, "Multiview modifier token mapping changed")
    _require(
        args.get("initializer_token") == EXPECTED_INITIALIZER_TOKEN,
        "Multiview base config must use the selected single-token turquoise initializer",
    )
    return value


def build_fold_outputs(
    render_root: Path,
    realized: list[dict[str, Any]],
    base_manifest: dict[str, Any],
    protocol: dict[str, Any],
    output_dir: Path,
    base_config_path: Path,
) -> list[dict[str, Any]]:
    render_root = render_root.resolve()
    base_config = _load_training_config(base_config_path)
    cells_by_pair = {(cell["shape"], cell["color"]): cell for cell in base_manifest["samples"]}
    fold_summaries: list[dict[str, Any]] = []
    realized_manifest_path = output_dir.resolve() / "realized_views.jsonl"
    for fold in protocol["folds"]:
        fold_id = fold["id"]
        held_out = {tuple(pair) for pair in fold["held_out"]}
        train_cells = set(cells_by_pair) - held_out
        train_records = [
            record for record in realized
            if (record["shape"], record["color"]) in train_cells and record["split"] == "train"
        ]
        _require(len(train_records) == 96, f"Fold {fold_id} must contain 96 training views")
        _require(not any((record["shape"], record["color"]) in held_out for record in train_records),
                 f"Fold {fold_id} contains a held-out cell")
        _require(not any(record["split"] == "audit" for record in train_records),
                 f"Fold {fold_id} contains audit views")

        fold_dir = output_dir.resolve() / "folds" / f"fold_{fold_id.lower()}"
        assets_dir = fold_dir / "train_assets"
        concepts: list[dict[str, Any]] = []
        for pair in sorted(train_cells, key=lambda pair: next(
            index for index, cell in enumerate(base_manifest["samples"])
            if (cell["shape"], cell["color"]) == pair
        )):
            cell = cells_by_pair[pair]
            cell_dir = assets_dir / cell["id"]
            _require(not cell_dir.exists(), f"Training asset directory already exists: {cell_dir}")
            cell_dir.mkdir(parents=True, exist_ok=True)
            records = sorted(
                (record for record in train_records if record["cell_id"] == cell["id"]),
                key=lambda record: record["view_index"],
            )
            _require(len(records) == 16, f"Fold {fold_id} cell {cell['id']} must have 16 views")
            for record in records:
                source = _resolved_under(render_root, record["image"], "image")
                _stage_image(source, cell_dir / f"view_{record['view_index']:02d}.jpg")
            expected_names = {f"view_{index:02d}.jpg" for index in range(16)}
            actual_names = {path.name for path in cell_dir.iterdir()}
            _require(actual_names == expected_names, f"Fold {fold_id} cell {cell['id']} staging is contaminated")
            concepts.append({"instance_prompt": cell["instance_prompt"], "instance_data_dir": str(cell_dir)})

        concepts_path = fold_dir / "concepts.json"
        _write_json(concepts_path, concepts)
        seen_audit = [
            record for record in realized
            if (record["shape"], record["color"]) in train_cells and record["split"] == "audit"
        ]
        held_out_records = [
            record for record in realized if (record["shape"], record["color"]) in held_out
        ]
        held_out_train_views = [record for record in held_out_records if record["split"] == "train"]
        held_out_audit_views = [record for record in held_out_records if record["split"] == "audit"]
        fold_protocol = {
            "fold_id": fold_id,
            "training_seeds": list(TRAINING_SEEDS),
            "held_out_cells": [list(pair) for pair in fold["held_out"]],
            "train_cells": [list(pair) for pair in sorted(train_cells)],
            "train_view_count": len(train_records),
            "seen_audit_view_count": len(seen_audit),
            "held_out_view_count": len(held_out_records),
            "held_out_train_view_count": len(held_out_train_views),
            "held_out_audit_view_count": len(held_out_audit_views),
            "training_uses_gt_masks": False,
            "train_record_ids": [f"{record['cell_id']}:v{record['view_index']:02d}" for record in train_records],
            "seen_audit_record_ids": [f"{record['cell_id']}:v{record['view_index']:02d}" for record in seen_audit],
            "held_out_record_ids": [f"{record['cell_id']}:v{record['view_index']:02d}" for record in held_out_records],
            "held_out_train_record_ids": [
                f"{record['cell_id']}:v{record['view_index']:02d}" for record in held_out_train_views
            ],
            "held_out_audit_record_ids": [
                f"{record['cell_id']}:v{record['view_index']:02d}" for record in held_out_audit_views
            ],
        }
        _write_json(fold_dir / "fold_protocol.json", fold_protocol)

        for seed in TRAINING_SEEDS:
            train_config = json.loads(json.dumps(base_config))
            train_config["status"] = "pending_human_review"
            variant_prefix = "multiview" if protocol["version"] == 1 else "multiview_v2"
            train_config["run"]["variant"] = f"{variant_prefix}_fold_{fold_id.lower()}_seed{seed}"
            train_config["run"]["seed"] = seed
            train_config["data_manifest"] = str(realized_manifest_path)
            train_config["args"]["concepts_list"] = str(concepts_path)
            train_config["args"]["seed"] = seed
            train_config.setdefault("protocol", {})["multiview_protocol"] = protocol["protocol_id"]
            train_config["protocol"]["fold_id"] = fold_id
            train_config["protocol"]["held_out_cells"] = fold_protocol["held_out_cells"]
            train_config["protocol"]["views_per_training_cell"] = 16
            train_config["protocol"]["training_seed"] = seed
            _write_json(fold_dir / f"train_config_seed{seed}.json", train_config)
        fold_summaries.append(fold_protocol)
    return fold_summaries


def realize_protocol(
    base_manifest: dict[str, Any],
    protocol: dict[str, Any],
    render_root: Path,
    render_manifest: Path,
    output_dir: Path,
    base_config: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    _require_empty_output_dir(output_dir)
    records = _read_jsonl(render_manifest)
    realized = validate_realization(render_root, records, base_manifest, protocol)
    realized_path = output_dir / "realized_views.jsonl"
    _write_jsonl(realized_path, realized)
    folds = build_fold_outputs(render_root, realized, base_manifest, protocol, output_dir, base_config)
    review_path, contact_sheet_path = write_human_review_outputs(
        render_root, realized, base_manifest, output_dir
    )
    result = {
        "status": "validated",
        "realized_view_count": len(realized),
        "train_views_per_cell": 16,
        "audit_views_per_cell": 4,
        "fold_train_view_count": 96,
        "training_seeds": list(TRAINING_SEEDS),
        "training_uses_gt_masks": False,
        "realized_manifest": str(realized_path),
        "human_review_csv": str(review_path),
        "contact_sheet": str(contact_sheet_path),
        "folds": folds,
    }
    _write_json(output_dir / "protocol_status.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="Write deterministic render requests without fabricating views")
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--renderer", type=Path, help="Real multiview renderer entrypoint; absence is recorded as blocked")
    realize = subparsers.add_parser("realize", help="Validate renderer outputs and build image-only fold assets")
    realize.add_argument("--render-root", type=Path, required=True)
    realize.add_argument("--render-manifest", type=Path, required=True)
    realize.add_argument("--output-dir", type=Path, required=True)
    realize.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    return parser


def main(argv: Iterable[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    base_manifest, protocol = load_inputs(args.base_manifest, args.protocol)
    if args.command == "plan":
        result = plan_protocol(base_manifest, protocol, args.output_dir, args.renderer)
    else:
        result = realize_protocol(
            base_manifest, protocol, args.render_root, args.render_manifest, args.output_dir, args.base_config
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
