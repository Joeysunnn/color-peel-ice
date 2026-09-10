"""Frozen, Blender-independent renderer color calibration protocol helpers."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from src.methods.colorpeel_ice import natural_image_target_selection as selection
from src.methods.colorpeel_ice.natural_image_masks import canonical_sha256


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_RELPATH = "experiments/natural_image_subject_color_pilot/configs/d1_renderer_color_calibration_protocol_v1.json"
FROZEN_PROTOCOL_CANONICAL_SHA256 = "476d01c5a383ead7d1ceec5c276bb26d809cf2fabf51cd81e0c49695cc4aa48e"
FROZEN_REQUEST_HASHES = {
    "preflight": "c2b5a4078f9407c2e92d555924a89ad5ba5640eeee1ef5c38e41402253ea08ff",
    "reduced_direct": "0cbbd38a7411fe02faaa4573130a40c3e6da53adc86fdeda6174af6f70254585",
    "full_invariance": "36cd0f957f8049e84d08e73cfa278155539c7aefa4951cf07dba1e01444a2aff",
    "full_expansion": "4ad0ad54e01634f0916c746372d089922a397b0e0b61a2c657d63cbf776c2b3a",
}

SCHEMA = "renderer_color_calibration_protocol/v1"
PROTOCOL_ID = "d1_renderer_color_calibration_protocol_v1"
L_INPUT = 50.0
IMAGE_AREA = 512 * 512
PREFLIGHT_BASE_SEED = 810000
REQUEST_BASE_SEED = 820000
SHAPE_ORDINAL = {"cube": 0, "sphere": 1, "cylinder": 2}
METADATA_REQUIRED_FIELDS = [
    "git_commit",
    "protocol_canonical_sha256",
    "selection_canonical_sha256",
    "source_audit_manifest_sha256",
    "blender_version",
    "blender_build_identifier",
    "base_scene_relative_path",
    "base_scene_sha256",
    "material_asset_relative_path",
    "material_asset_sha256",
    "shape_asset_relative_path",
    "shape_asset_sha256",
    "stable_id",
    "target_a",
    "target_b",
    "target_C",
    "target_h_degrees",
    "L_input",
    "linear_rgb",
    "socket_rgba",
    "preview_srgb_uint8",
    "gamut",
    "stage",
    "shape",
    "view_index",
    "render_seed",
    "camera",
    "lights",
    "world",
    "ground",
    "color_management",
    "image_relative_path",
    "image_sha256",
    "mask_relative_path",
    "mask_sha256",
    "measurement_stats",
    "decision",
]

PROTOCOL_KEYS = {
    "schema",
    "protocol_id",
    "study",
    "frozen_inputs",
    "target_definition",
    "conversion",
    "renderer_contract",
    "scene_contract",
    "planning",
    "request_hashes",
    "measurement",
    "fallback",
    "metadata_contract",
    "approval_state",
}
FROZEN_INPUT_KEYS = {"selection_config"}
SELECTION_INPUT_KEYS = {
    "repo_relative_path",
    "canonical_sha256",
    "source_audit_manifest_sha256",
    "selected_stable_id_order_sha256",
    "pipeline_assignment_sha256",
}
TARGET_DEFINITION_KEYS = {
    "space",
    "frozen_token_components",
    "observed_median_L_is_metadata_only",
    "lightness_is_fixed_target_token",
    "renderer_L_input",
}
CONVERSION_KEYS = {
    "path",
    "white_point",
    "xyz_to_linear_srgb_matrix",
    "epsilon",
    "kappa",
    "gamut",
    "preview",
    "separate_fields",
}
RENDERER_CONTRACT_KEYS = {
    "renderer_profile_id",
    "engine",
    "resolution",
    "samples",
    "calibration_image",
    "mask_image",
    "material",
    "color_management",
    "legacy_renderer_note",
}
SCENE_CONTRACT_KEYS = {"blender", "object", "camera", "lights", "background", "runtime_assets"}
SCENE_BLENDER_KEYS = {"version", "render_engine", "cycles_device", "samples", "resolution"}
SCENE_OBJECT_KEYS = {"scale", "position_xy", "rotation_z_degrees", "photometric_material"}
SCENE_CAMERA_KEYS = {
    "view_count",
    "azimuth_degrees",
    "radius",
    "elevation_degrees",
    "look_at",
    "rotation_policy",
    "lens_sensor_shift_policy",
    "jitter",
    "record_runtime_fields",
}
SCENE_LIGHTS_KEYS = {
    "named_lights",
    "position_policy",
    "energy_policy",
    "rgb",
    "jitter",
    "record_runtime_fields",
}
SCENE_BACKGROUND_KEYS = {
    "world_rgba",
    "ground_rgba",
    "ground_metallic",
    "ground_roughness",
}
SCENE_RUNTIME_ASSET_KEYS = {"base_scene_sha256", "material_asset_sha256", "shape_asset_sha256", "path_policy"}
COLOR_MANAGEMENT_KEYS = {
    "display_device",
    "view_transform",
    "look",
    "exposure",
    "gamma",
    "unsupported_setting_policy",
}
REQUEST_HASH_KEYS = {"preflight", "reduced_direct", "full_invariance", "full_expansion"}
PLANNING_KEYS = {"preflight", "reduced_direct", "full_invariance", "full_expansion", "excluded_roles"}
PREFLIGHT_KEYS = {
    "description",
    "linear_rgb_probes",
    "shape",
    "view_indices",
    "expected_byte_tolerance",
    "expected_lab_delta_max",
}
DIRECT_PLAN_KEYS = {"roles", "target_count", "shapes", "view_indices", "request_count"}
EXPANSION_PLAN_KEYS = {"source", "request_count"}
MEASUREMENT_KEYS = {
    "mask",
    "object_ratio_min_inclusive",
    "object_ratio_max_exclusive",
    "interior_radius",
    "interior_count_min",
    "eligible_lightness",
    "eligible_count_min",
    "render_stats",
    "target_p90",
    "direct_target_pass",
    "overall_pass",
}
FALLBACK_KEYS = {
    "trigger",
    "roles",
    "candidate",
    "coarse",
    "refine",
    "bounds",
    "candidate_filter",
    "candidate_views",
    "objective",
    "tie_break",
    "acceptance",
    "full_invariance_confirmation",
    "max_render_budget_per_failing_color",
    "refine_anchor",
    "evaluation_policy",
    "max_unique_renderable_evaluations",
}
INTERIOR_RADIUS_KEYS = {"r_raw", "r_min", "r_max", "selection", "interior"}
DIRECT_PASS_KEYS = {"median_e_ch_max", "p90_e_ch_max"}
OVERALL_PASS_KEYS = {"all_targets_pass", "overall_render_p90_e_ch_max"}
COARSE_KEYS = {"L_values", "chroma_scales", "candidate_count"}
REFINE_KEYS = {"L_delta", "scale_delta", "max_candidate_count"}
BOUNDS_KEYS = {"L_min", "L_max", "scale_min", "scale_max"}
CANDIDATE_VIEWS_KEYS = {"shape", "view_indices"}


class RendererColorCalibrationError(ValueError):
    """Raised when the frozen calibration protocol or supplied measurements drift."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RendererColorCalibrationError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RendererColorCalibrationError(f"Cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON must be an object: {path}")
    return value


def _strict_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    actual = set(value)
    require(actual == expected, f"{label} fields differ; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    return value


def _looks_absolute_path(value: str) -> bool:
    return value.startswith("/") or value.startswith("\\") or re.match(r"^[A-Za-z]:[\\/]", value) is not None


def _reject_absolute_strings(value: Any, label: str = "protocol") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _reject_absolute_strings(nested, f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_absolute_strings(nested, f"{label}[{index}]")
    elif isinstance(value, str):
        require(not _looks_absolute_path(value), f"{label} must not contain an absolute path")


def _validate_protocol_structure(protocol: Any) -> dict[str, Any]:
    value = _strict_keys(protocol, PROTOCOL_KEYS, "protocol")
    _reject_absolute_strings(value)
    require(value["schema"] == SCHEMA, "Unexpected protocol schema")
    require(value["protocol_id"] == PROTOCOL_ID, "Unexpected protocol_id")
    require(value["study"] == "natural_image_subject_color_pilot", "Unexpected study")

    frozen = _strict_keys(value["frozen_inputs"], FROZEN_INPUT_KEYS, "frozen_inputs")
    selected = _strict_keys(frozen["selection_config"], SELECTION_INPUT_KEYS, "frozen_inputs.selection_config")
    require(selected["repo_relative_path"] == selection.CONFIG_RELPATH, "selection config path drifted")
    require(selected["canonical_sha256"] == selection.FROZEN_CONFIG_CANONICAL_SHA256, "selection canonical hash drifted")
    require(selected["source_audit_manifest_sha256"] == selection.SOURCE_AUDIT_MANIFEST_SHA256, "source audit manifest hash drifted")
    require(selected["selected_stable_id_order_sha256"] == selection.canonical_sha256(selection.SELECTED_STABLE_IDS),
            "selected stable-id order hash drifted")
    require(selected["pipeline_assignment_sha256"] == selection.canonical_sha256(selection._assignment_rows()),
            "pipeline assignment hash drifted")

    target = _strict_keys(value["target_definition"], TARGET_DEFINITION_KEYS, "target_definition")
    require(target["space"] == "CIELAB D65", "target_definition.space drifted")
    require(target["frozen_token_components"] == ["a", "b", "C", "h_degrees"], "target components drifted")
    require(target["observed_median_L_is_metadata_only"] is True, "observed L must remain metadata only")
    require(target["lightness_is_fixed_target_token"] is False, "target lightness flag must remain false")
    require(target["renderer_L_input"] == L_INPUT, "renderer L_input drifted")

    conversion = _strict_keys(value["conversion"], CONVERSION_KEYS, "conversion")
    require(conversion == {
        "path": "Lab D65 -> XYZ D65 -> linear sRGB float64",
        "white_point": [0.95047, 1.0, 1.08883],
        "xyz_to_linear_srgb_matrix": [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        "epsilon": "216/24389",
        "kappa": "24389/27",
        "gamut": "validate linear channels against [0,1]; no clipping",
        "preview": "linear sRGB -> nonlinear sRGB uint8 with round-half-up",
        "separate_fields": ["preview_srgb_uint8", "linear_rgb", "socket_rgba", "decoded_png_rgb_uint8"],
    }, "conversion contract drifted")

    renderer = _strict_keys(value["renderer_contract"], RENDERER_CONTRACT_KEYS, "renderer_contract")
    calibration_image = _strict_keys(renderer["calibration_image"], {"format", "mode", "bits_per_channel"}, "renderer_contract.calibration_image")
    mask_image = _strict_keys(renderer["mask_image"], {"format", "mode", "values"}, "renderer_contract.mask_image")
    require(renderer["engine"] == "Cycles", "renderer engine drifted")
    require(renderer["resolution"] == [512, 512], "renderer resolution drifted")
    require(renderer["samples"] == 512, "renderer samples drifted")
    require(renderer["material"] == "Rubber", "renderer material drifted")
    require(renderer["renderer_profile_id"] == "d1_renderer_color_calibration_v1", "renderer profile drifted")
    require(renderer["legacy_renderer_note"] == "Existing CLEVR renderer writes nominal_rgb/255 directly to the Color socket and does not explicitly fix color management; this protocol defines a future adapter contract only.",
            "legacy renderer note drifted")
    require(calibration_image == {"format": "PNG", "mode": "RGB", "bits_per_channel": 8}, "calibration image contract drifted")
    require(mask_image == {"format": "PNG", "mode": "L", "values": [0, 255]}, "mask image contract drifted")
    color_management = _strict_keys(renderer["color_management"], COLOR_MANAGEMENT_KEYS, "color_management")
    require(color_management == {
        "display_device": "sRGB",
        "view_transform": "Standard",
        "look": None,
        "exposure": 0.0,
        "gamma": 1.0,
        "unsupported_setting_policy": "abort_and_record_available_settings",
    }, "color-management contract drifted")

    scene = _strict_keys(value["scene_contract"], SCENE_CONTRACT_KEYS, "scene_contract")
    scene_blender = _strict_keys(scene["blender"], SCENE_BLENDER_KEYS, "scene_contract.blender")
    scene_object = _strict_keys(scene["object"], SCENE_OBJECT_KEYS, "scene_contract.object")
    scene_camera = _strict_keys(scene["camera"], SCENE_CAMERA_KEYS, "scene_contract.camera")
    scene_lights = _strict_keys(scene["lights"], SCENE_LIGHTS_KEYS, "scene_contract.lights")
    scene_background = _strict_keys(scene["background"], SCENE_BACKGROUND_KEYS, "scene_contract.background")
    scene_assets = _strict_keys(scene["runtime_assets"], SCENE_RUNTIME_ASSET_KEYS, "scene_contract.runtime_assets")
    require(scene_blender == {"version": "4.2.11", "render_engine": "CYCLES", "cycles_device": "CUDA",
                              "samples": 512, "resolution": [512, 512]}, "scene blender contract drifted")
    require(scene_object == {"scale": 1.3, "position_xy": [0.0, 0.0], "rotation_z_degrees": 0.0,
                             "photometric_material": "Rubber"}, "scene object contract drifted")
    require(scene_camera == {
        "view_count": 20,
        "azimuth_degrees": "base_scene_camera_azimuth_degrees + 18*view_index",
        "radius": "preserve_base_scene_camera_radius",
        "elevation_degrees": "preserve_base_scene_camera_elevation_degrees",
        "look_at": "object_center",
        "rotation_policy": "look_at_object_center_negative_z_y_up",
        "lens_sensor_shift_policy": "require_base_scene_lens_sensor_zero_shift_and_record",
        "jitter": "none",
        "record_runtime_fields": [
            "base_scene_camera_azimuth_degrees",
            "base_scene_camera_radius",
            "base_scene_camera_elevation_degrees",
            "lens_mm",
            "sensor_width_mm",
            "sensor_height_mm",
            "shift_x",
            "shift_y",
        ],
    }, "scene camera contract drifted")
    require(scene_lights == {
        "named_lights": ["Lamp_Key", "Lamp_Back", "Lamp_Fill", "Area"],
        "position_policy": "preserve_base_scene_positions_and_record",
        "energy_policy": "preserve_base_scene_energies_and_record",
        "rgb": [1.0, 1.0, 1.0],
        "jitter": "none",
        "record_runtime_fields": ["name", "position", "energy", "rgb"],
    }, "scene lights contract drifted")
    require(scene_background == {"world_rgba": [0.05, 0.05, 0.05, 1.0],
                                 "ground_rgba": [0.5, 0.5, 0.5, 1.0],
                                 "ground_metallic": 0.0,
                                 "ground_roughness": 1.0}, "scene background contract drifted")
    require(scene_assets == {"base_scene_sha256": "runtime_recorded",
                             "material_asset_sha256": "runtime_recorded",
                             "shape_asset_sha256": "runtime_recorded",
                             "path_policy": "runtime relative inputs and SHA256 recorded for base scene, material and shape assets; no absolute paths"},
            "scene runtime asset contract drifted")

    planning = _strict_keys(value["planning"], PLANNING_KEYS, "planning")
    preflight = _strict_keys(planning["preflight"], PREFLIGHT_KEYS, "planning.preflight")
    reduced = _strict_keys(planning["reduced_direct"], DIRECT_PLAN_KEYS, "planning.reduced_direct")
    full = _strict_keys(planning["full_invariance"], DIRECT_PLAN_KEYS, "planning.full_invariance")
    expansion = _strict_keys(planning["full_expansion"], EXPANSION_PLAN_KEYS, "planning.full_expansion")
    require(preflight["linear_rgb_probes"] == [
        [0.18, 0.18, 0.18],
        [0.50, 0.50, 0.50],
        [0.75, 0.10, 0.05],
        [0.05, 0.55, 0.90],
    ], "preflight probes drifted")
    require(preflight["shape"] == "sphere" and preflight["view_indices"] == [0], "preflight shape/view drifted")
    require(preflight["description"] == "Encoding validation only; not Rubber photometry.", "preflight description drifted")
    require(preflight["expected_byte_tolerance"] == 1 and preflight["expected_lab_delta_max"] == 0.5,
            "preflight thresholds drifted")
    require(reduced == {"roles": ["pipeline_development"], "target_count": 6, "shapes": ["sphere"], "view_indices": [0, 8, 16], "request_count": 18},
            "reduced_direct plan drifted")
    require(full == {"roles": ["pipeline_development"], "target_count": 6, "shapes": ["cube", "sphere", "cylinder"], "view_indices": [0, 4, 8, 12, 16], "request_count": 90},
            "full_invariance plan drifted")
    require(expansion == {"source": "full_invariance minus reusable reduced_direct keys", "request_count": 72},
            "full_expansion plan drifted")
    require(planning["excluded_roles"] == ["protocol_confirmation", "stress_test"], "excluded roles drifted")

    request_hashes = _strict_keys(value["request_hashes"], REQUEST_HASH_KEYS, "request_hashes")
    require(request_hashes == FROZEN_REQUEST_HASHES, "request hash contract drifted")

    measurement = _strict_keys(value["measurement"], MEASUREMENT_KEYS, "measurement")
    _strict_keys(measurement["interior_radius"], INTERIOR_RADIUS_KEYS, "measurement.interior_radius")
    direct_pass = _strict_keys(measurement["direct_target_pass"], DIRECT_PASS_KEYS, "measurement.direct_target_pass")
    overall_pass = _strict_keys(measurement["overall_pass"], OVERALL_PASS_KEYS, "measurement.overall_pass")
    require(measurement == {
        "mask": "binary object/background complement-validated mask",
        "object_ratio_min_inclusive": 0.005,
        "object_ratio_max_exclusive": 0.90,
        "interior_radius": {
            "r_raw": "round-half-up(0.025*min(s,sqrt(A)))",
            "r_min": "max(1,floor(0.01*s))",
            "r_max": "max(r_min,floor(0.08*s))",
            "selection": "clamp(r_raw,r_min,r_max)",
            "interior": "distance_transform_edt(object_mask) >= r",
        },
        "interior_count_min": "max(0.15*A,0.002*image_area)",
        "eligible_lightness": "5 < L < 95",
        "eligible_count_min": "count >= 0.80*I and count >= max(0.12*A,0.0015*image_area)",
        "render_stats": ["median_L", "median_a", "median_b", "iqr_L", "iqr_a", "iqr_b", "e_ch"],
        "target_p90": "nearest-rank index ceil(0.90*n)-1",
        "direct_target_pass": {"median_e_ch_max": 3.0, "p90_e_ch_max": 5.0},
        "overall_pass": {"all_targets_pass": True, "overall_render_p90_e_ch_max": 5.0},
    }, "measurement contract drifted")
    require(measurement["object_ratio_min_inclusive"] == 0.005, "object ratio lower gate drifted")
    require(measurement["object_ratio_max_exclusive"] == 0.90, "object ratio upper gate drifted")
    require(direct_pass == {"median_e_ch_max": 3.0, "p90_e_ch_max": 5.0}, "direct target thresholds drifted")
    require(overall_pass == {"all_targets_pass": True, "overall_render_p90_e_ch_max": 5.0}, "overall threshold drifted")

    fallback = _strict_keys(value["fallback"], FALLBACK_KEYS, "fallback")
    coarse = _strict_keys(fallback["coarse"], COARSE_KEYS, "fallback.coarse")
    refine = _strict_keys(fallback["refine"], REFINE_KEYS, "fallback.refine")
    bounds = _strict_keys(fallback["bounds"], BOUNDS_KEYS, "fallback.bounds")
    candidate_views = _strict_keys(fallback["candidate_views"], CANDIDATE_VIEWS_KEYS, "fallback.candidate_views")
    require(coarse == {"L_values": [35, 40, 45, 50, 55, 60, 65, 70, 75],
                       "chroma_scales": [0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20],
                       "candidate_count": 63}, "coarse fallback plan drifted")
    require(refine == {"L_delta": [-2, -1, 0, 1, 2], "scale_delta": [-0.04, -0.02, 0.0, 0.02, 0.04],
                       "max_candidate_count": 25}, "refine fallback plan drifted")
    require(bounds == {"L_min": 35.0, "L_max": 75.0, "scale_min": 0.60, "scale_max": 1.20},
            "fallback bounds drifted")
    require(candidate_views == {"shape": "sphere", "view_indices": [0, 8, 16]}, "fallback candidate views drifted")
    require(fallback["max_render_budget_per_failing_color"] == 279, "fallback render budget drifted")
    require(fallback["roles"] == ["pipeline_development"], "fallback roles drifted")
    require(fallback["candidate"] == "Lab=(L_input, scale*a, scale*b), fixed target hue", "fallback candidate drifted")
    require(fallback["refine_anchor"] == "exact best in-gamut coarse member for the same target", "fallback refine anchor drifted")
    require(fallback["evaluation_policy"] == "evaluate all in-gamut coarse candidates, then all unique in-gamut refine candidates; no early stop", "fallback evaluation policy drifted")
    require(fallback["max_unique_renderable_evaluations"] == 88, "fallback evaluation cap drifted")
    require(fallback["trigger"] == "after encoding preflight passes and a pipeline_development direct target fails",
            "fallback trigger drifted")
    require(fallback["candidate_filter"] == "discard out-of-gamut before rendering", "fallback filter drifted")
    require(fallback["objective"] == "median(e_ch)+0.25*p90(e_ch)+0.02*abs(median_render_L-50)",
            "fallback objective drifted")
    require(fallback["tie_break"] == ["lower_p90", "closer_to_L50_scale1", "lower_L", "lower_scale"],
            "fallback tie-break drifted")
    require(fallback["acceptance"] == "choose minimum objective among passing candidates; otherwise expose diagnostic only",
            "fallback acceptance drifted")
    require(fallback["full_invariance_confirmation"] == "selected passing fallback must pass full 15-render target matrix unchanged",
            "fallback invariance confirmation drifted")

    metadata = _strict_keys(value["metadata_contract"], {"required_fields"}, "metadata_contract")
    require(metadata["required_fields"] == METADATA_REQUIRED_FIELDS, "metadata required fields drifted")
    approval = _strict_keys(value["approval_state"], {
        "selection_layer_approved_target_count",
        "downstream_eligible",
        "recoloring_approved",
        "training_approved",
    }, "approval_state")
    require(approval == {
        "selection_layer_approved_target_count": 11,
        "downstream_eligible": True,
        "recoloring_approved": False,
        "training_approved": False,
    }, "approval state drifted")
    require(canonical_sha256(value) == FROZEN_PROTOCOL_CANONICAL_SHA256, "protocol canonical hash drift")
    return value


def load_protocol(
    path: str | Path = REPO_ROOT / PROTOCOL_RELPATH,
    selection_path: str | Path = REPO_ROOT / selection.CONFIG_RELPATH,
) -> dict[str, Any]:
    """Load only the frozen repository protocol and validate linked selection provenance."""
    protocol_path = Path(path)
    require(protocol_path.resolve() == (REPO_ROOT / PROTOCOL_RELPATH).resolve(),
            "protocol path must be the frozen repository protocol")
    value = read_json(protocol_path)
    require(canonical_sha256(value) == FROZEN_PROTOCOL_CANONICAL_SHA256, "protocol canonical hash drift")
    protocol = _validate_protocol_structure(value)
    selected_config = selection.load_selection_config(selection_path)
    linked = protocol["frozen_inputs"]["selection_config"]
    require(canonical_sha256(selected_config) == linked["canonical_sha256"], "linked selection canonical hash drift")
    require(selected_config["source_audit"]["manifest_sha256"] == linked["source_audit_manifest_sha256"],
            "linked source audit manifest hash drift")
    require(selected_config["canonical_hashes"]["selected_stable_id_order_sha256"] == linked["selected_stable_id_order_sha256"],
            "linked selected-order hash drift")
    require(selected_config["canonical_hashes"]["pipeline_assignment_sha256"] == linked["pipeline_assignment_sha256"],
            "linked assignment hash drift")
    return protocol


def round_half_up(value: float) -> int:
    require(math.isfinite(value) and value >= 0.0, "round_half_up only accepts finite nonnegative values")
    return int(math.floor(value + 0.5))


def _lab_f(value: float) -> float:
    epsilon = 216 / 24389
    kappa = 24389 / 27
    cube = value * value * value
    return cube if cube > epsilon else (116 * value - 16) / kappa


def lab_d65_to_xyz_d65(L: float, a: float, b: float) -> list[float]:
    require(all(math.isfinite(float(value)) for value in [L, a, b]), "Lab values must be finite")
    fy = (float(L) + 16.0) / 116.0
    fx = fy + float(a) / 500.0
    fz = fy - float(b) / 200.0
    white = [0.95047, 1.0, 1.08883]
    return [_lab_f(fx) * white[0], _lab_f(fy) * white[1], _lab_f(fz) * white[2]]


def lab_d65_to_linear_srgb(L: float, a: float, b: float) -> list[float]:
    x, y, z = lab_d65_to_xyz_d65(L, a, b)
    matrix = [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]
    return [sum(row[index] * [x, y, z][index] for index in range(3)) for row in matrix]


def assert_linear_rgb_in_gamut(linear_rgb: Sequence[float]) -> list[float]:
    require(len(linear_rgb) == 3, "linear RGB must contain exactly 3 channels")
    result = [float(channel) for channel in linear_rgb]
    for index, channel in enumerate(result):
        require(math.isfinite(channel), f"linear RGB channel {index} must be finite")
        require(0.0 <= channel <= 1.0, f"linear RGB channel {index} is out of gamut: {channel}")
    return result


def srgb_to_linear_channel(channel: float) -> float:
    value = float(channel)
    require(0.0 <= value <= 1.0, "sRGB channel must be in [0,1]")
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def linear_to_srgb_channel(channel: float) -> float:
    value = assert_linear_rgb_in_gamut([channel, 0.0, 0.0])[0]
    return 12.92 * value if value <= 0.0031308 else 1.055 * (value ** (1 / 2.4)) - 0.055


def linear_rgb_to_srgb_uint8(linear_rgb: Sequence[float]) -> list[int]:
    linear = assert_linear_rgb_in_gamut(linear_rgb)
    return [round_half_up(255.0 * linear_to_srgb_channel(channel)) for channel in linear]


def color_payload_for_target(record: Mapping[str, Any], L_input: float = L_INPUT) -> dict[str, Any]:
    target = record["target"]
    linear_rgb = lab_d65_to_linear_srgb(L_input, target["a"], target["b"])
    linear_rgb = assert_linear_rgb_in_gamut(linear_rgb)
    return {
        "L_input": float(L_input),
        "target_a": target["a"],
        "target_b": target["b"],
        "target_C": target["C"],
        "target_h_degrees": target["h_degrees"],
        "linear_rgb": linear_rgb,
        "socket_rgba": linear_rgb + [1.0],
        "preview_srgb_uint8": linear_rgb_to_srgb_uint8(linear_rgb),
        "gamut": "in",
    }


def _stable_slug(stable_id: str) -> str:
    return stable_id.replace("D1GT:", "D1GT_").replace("/", "_").replace(".png", "")


def _development_records(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    return selection.pipeline_development_records(config)


def _request_seed(target_index: int, shape_index: int, view_index: int) -> int:
    return REQUEST_BASE_SEED + target_index * 100 + shape_index * 20 + view_index


def build_preflight_requests(protocol: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    value = _validate_protocol_structure(dict(protocol)) if protocol is not None else load_protocol()
    probes = value["planning"]["preflight"]["linear_rgb_probes"]
    rows = []
    for index, linear_rgb in enumerate(probes):
        linear = assert_linear_rgb_in_gamut(linear_rgb)
        rows.append({
            "stage": "preflight",
            "request_id": f"preflight_probe_{index:03d}_sphere_view_00",
            "probe_index": index,
            "shape": value["planning"]["preflight"]["shape"],
            "material": "emission_probe",
            "view_index": 0,
            "render_seed": PREFLIGHT_BASE_SEED + index,
            "linear_rgb": linear,
            "socket_rgba": linear + [1.0],
            "expected_decoded_srgb_uint8": linear_rgb_to_srgb_uint8(linear),
            "expected_byte_tolerance": value["planning"]["preflight"]["expected_byte_tolerance"],
            "expected_lab_delta_max": value["planning"]["preflight"]["expected_lab_delta_max"],
        })
    require(len(rows) == 4, "preflight request count drifted")
    require(canonical_sha256(rows) == FROZEN_REQUEST_HASHES["preflight"], "preflight request hash drifted")
    return rows


def _validate_request_measurements(
    rows: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]],
) -> None:
    require(len(rows) == len(expected), "measurement request coverage count differs")
    requests = {row["request_id"]: row for row in expected}
    seen = set()
    for row in rows:
        identity = row.get("request_id")
        require(isinstance(identity, str) and identity in requests and identity not in seen,
                "missing, extra or duplicate measurement request identity")
        seen.add(identity)
        request = requests[identity]
        require(all(key in row for key in request), "measurement request payload fields are missing")
        require(canonical_sha256({key: row[key] for key in request}) == canonical_sha256(request),
                "measurement request payload differs")
    require(seen == set(requests), "measurement request coverage differs")


def summarize_preflight_measurements(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate all four frozen emission results before permitting photometry."""
    expected = build_preflight_requests()
    _validate_request_measurements(rows, expected)
    by_id = {row["request_id"]: row for row in rows}
    probes = []
    for request in expected:
        row = by_id[request["request_id"]]
        decoded = row.get("decoded_png_rgb_uint8")
        require(isinstance(decoded, list) and len(decoded) == 3
                and all(type(channel) is int and 0 <= channel <= 255 for channel in decoded),
                "decoded_png_rgb_uint8 must contain three integers in [0,255]")
        delta = row.get("decoded_lab_delta")
        require(type(delta) in {int, float} and math.isfinite(delta), "decoded_lab_delta must be finite")
        byte_errors = [abs(actual - predicted) for actual, predicted in zip(decoded, request["expected_decoded_srgb_uint8"])]
        passed = all(error <= 1 for error in byte_errors) and 0 <= delta <= 0.5
        probes.append({"request_id": request["request_id"], "probe_index": request["probe_index"],
                       "byte_errors": byte_errors, "decoded_lab_delta": delta, "passed": passed})
    return {"stage": "preflight", "probe_count": len(probes), "probes": probes,
            "overall_pass": all(probe["passed"] for probe in probes)}


def _build_target_requests(
    stage: str,
    shapes: Sequence[str],
    views: Sequence[int],
    config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    records = _development_records(config)
    rows = []
    for target_index, record in enumerate(records):
        stable_id = record["source"]["stable_id"]
        payload = color_payload_for_target(record)
        for shape in shapes:
            for view_index in views:
                rows.append({
                    "stage": stage,
                    "request_id": f"{_stable_slug(stable_id)}_{shape}_view_{view_index:02d}",
                    "target_index": target_index,
                    "stable_id": stable_id,
                    "pipeline_role": "pipeline_development",
                    "shape": shape,
                    "material": "Rubber",
                    "view_index": int(view_index),
                    "render_seed": _request_seed(target_index, SHAPE_ORDINAL[shape], int(view_index)),
                    **deepcopy(payload),
                })
    return rows


def render_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(row["stable_id"]), str(row["shape"]), int(row["view_index"]))


def build_reduced_direct_requests(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    rows = _build_target_requests("reduced_direct", ["sphere"], [0, 8, 16], config)
    require(len(rows) == 18, "reduced_direct request count drifted")
    require(canonical_sha256(rows) == FROZEN_REQUEST_HASHES["reduced_direct"], "reduced_direct request hash drifted")
    return rows


def build_full_invariance_requests(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    rows = _build_target_requests("full_invariance", ["cube", "sphere", "cylinder"], [0, 4, 8, 12, 16], config)
    require(len(rows) == 90, "full_invariance request count drifted")
    require(canonical_sha256(rows) == FROZEN_REQUEST_HASHES["full_invariance"], "full_invariance request hash drifted")
    return rows


def build_full_expansion_requests(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    reduced = {render_key(row) for row in build_reduced_direct_requests(config)}
    rows = [dict(row, stage="full_expansion") for row in build_full_invariance_requests(config) if render_key(row) not in reduced]
    require(len(rows) == 72, "full_expansion request count drifted")
    require(canonical_sha256(rows) == FROZEN_REQUEST_HASHES["full_expansion"], "full_expansion request hash drifted")
    return rows


def request_plan_summary(protocol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = _validate_protocol_structure(dict(protocol)) if protocol is not None else load_protocol()
    preflight = build_preflight_requests(value)
    reduced = build_reduced_direct_requests()
    full = build_full_invariance_requests()
    expansion = build_full_expansion_requests()
    return {
        "protocol_canonical_sha256": canonical_sha256(value),
        "counts": {
            "preflight": len(preflight),
            "reduced_direct": len(reduced),
            "full_invariance": len(full),
            "full_expansion": len(expansion),
        },
        "hashes": {
            "preflight": canonical_sha256(preflight),
            "reduced_direct": canonical_sha256(reduced),
            "full_invariance": canonical_sha256(full),
            "full_expansion": canonical_sha256(expansion),
        },
        "development_stable_ids": [row["source"]["stable_id"] for row in _development_records()],
    }


def derive_interior_contract(
    object_area: int,
    bbox_w: int,
    bbox_h: int,
    image_area: int = IMAGE_AREA,
    background_area: int | None = None,
    interior_area: int | None = None,
    eligible_count: int | None = None,
) -> dict[str, Any]:
    require(type(object_area) is int and object_area > 0, "object_area must be positive integer")
    require(type(image_area) is int and image_area > 0, "image_area must be positive integer")
    require(type(bbox_w) is int and bbox_w > 0 and type(bbox_h) is int and bbox_h > 0, "bbox dimensions must be positive integers")
    require(eligible_count is None or interior_area is not None, "eligible_count requires interior_area")
    if background_area is not None:
        require(type(background_area) is int and background_area >= 0, "background_area must be nonnegative integer")
        require(object_area + background_area == image_area, "object/background areas must complement image_area")
    ratio = object_area / image_area
    object_ratio_ok = 0.005 <= ratio < 0.90
    s = min(bbox_w, bbox_h)
    r_raw = round_half_up(0.025 * min(s, math.sqrt(object_area)))
    r_min = max(1, math.floor(0.01 * s))
    r_max = max(r_min, math.floor(0.08 * s))
    radius = max(r_min, min(r_max, r_raw))
    interior_min = max(0.15 * object_area, 0.002 * image_area)
    eligible_min = max(0.12 * object_area, 0.0015 * image_area)
    result = {
        "object_area": object_area,
        "background_area": background_area,
        "image_area": image_area,
        "object_ratio": ratio,
        "object_ratio_ok": object_ratio_ok,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "s": s,
        "r_raw": r_raw,
        "r_min": r_min,
        "r_max": r_max,
        "radius": radius,
        "interior_area_min": interior_min,
        "eligible_count_min": eligible_min,
    }
    if interior_area is not None:
        require(type(interior_area) is int and 0 <= interior_area <= object_area, "interior_area must be within object area")
        interior_ok = interior_area >= interior_min
        result.update({"interior_area": interior_area, "interior_ok": interior_ok})
        if eligible_count is not None:
            require(type(eligible_count) is int and 0 <= eligible_count <= interior_area, "eligible_count must be within interior area")
            eligible_ok = eligible_count >= 0.80 * interior_area and eligible_count >= eligible_min
            result.update({"eligible_count": eligible_count, "eligible_ok": eligible_ok})
    return result


def nearest_rank_p90(values: Sequence[float]) -> float:
    require(len(values) > 0, "nearest-rank p90 requires at least one value")
    ordered = sorted(float(value) for value in values)
    require(all(math.isfinite(value) for value in ordered), "p90 values must be finite")
    index = math.ceil(0.90 * len(ordered)) - 1
    return ordered[index]


def median(values: Sequence[float]) -> float:
    require(len(values) > 0, "median requires at least one value")
    ordered = sorted(float(value) for value in values)
    require(all(math.isfinite(value) for value in ordered), "median values must be finite")
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * ordered[mid - 1] + 0.5 * ordered[mid]


def summarize_render_measurement(row: Mapping[str, Any]) -> dict[str, Any]:
    for key in ["median_L", "median_a", "median_b", "iqr_L", "iqr_a", "iqr_b", "target_a", "target_b"]:
        require(key in row and math.isfinite(float(row[key])), f"{key} must be finite")
    for key in ["iqr_L", "iqr_a", "iqr_b"]:
        require(float(row[key]) >= 0, f"{key} must be nonnegative")
    if "e_ch" in row:
        require(math.isfinite(float(row["e_ch"])), "e_ch must be finite")
    for key in ["object_ratio_ok", "interior_ok", "eligible_ok"]:
        require(row.get(key) is True, f"{key} must be explicitly true")
    e_ch = math.hypot(float(row["median_a"]) - float(row["target_a"]), float(row["median_b"]) - float(row["target_b"]))
    require(math.isfinite(e_ch), "e_ch must be finite")
    result = dict(row)
    result["e_ch"] = e_ch
    return result


def _summarize_chroma_measurements(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(rows) > 0, "at least one render measurement is required")
    rendered = [summarize_render_measurement(row) for row in rows]
    grouped: dict[str, list[dict[str, Any]]] = {}
    identities = set()
    target_ab = {}
    for row in rendered:
        require(all(key in row for key in ["stable_id", "shape", "view_index"]), "render identity is required")
        require(row["shape"] in SHAPE_ORDINAL and type(row["view_index"]) is int and 0 <= row["view_index"] < 20,
                "invalid shape/view render identity")
        identity = render_key(row)
        require(identity not in identities, "duplicate render identity")
        identities.add(identity)
        stable_id = str(row["stable_id"])
        ab = (float(row["target_a"]), float(row["target_b"]))
        require(stable_id not in target_ab or target_ab[stable_id] == ab, "inconsistent target a,b for stable_id")
        target_ab[stable_id] = ab
        grouped.setdefault(str(row["stable_id"]), []).append(row)
    targets = {}
    for stable_id, target_rows in grouped.items():
        errors = [row["e_ch"] for row in target_rows]
        target_summary = {
            "render_count": len(target_rows),
            "median_e_ch": median(errors),
            "p90_e_ch": nearest_rank_p90(errors),
        }
        target_summary["direct_pass"] = target_summary["median_e_ch"] <= 3.0 and target_summary["p90_e_ch"] <= 5.0
        targets[stable_id] = target_summary
    all_errors = [row["e_ch"] for row in rendered]
    overall_p90 = nearest_rank_p90(all_errors)
    return {
        "render_count": len(rendered),
        "targets": targets,
        "overall_render_p90_e_ch": overall_p90,
        "overall_pass": all(summary["direct_pass"] for summary in targets.values()) and overall_p90 <= 5.0,
    }


def summarize_chroma_measurements(rows: Sequence[Mapping[str, Any]], stage: str) -> dict[str, Any]:
    """Gate a complete direct matrix, including reusable reduced/expansion results."""
    require(stage in {"reduced_direct", "full_invariance"}, "direct gate requires reduced_direct or full_invariance")
    expected_rows = build_reduced_direct_requests() if stage == "reduced_direct" else build_full_invariance_requests()
    require(len(rows) == len(expected_rows), "direct matrix coverage count differs")
    require(all(all(key in row for key in ["stable_id", "shape", "view_index", "stage"])
                for row in rows), "direct request identity and stage are required")
    stages = {row["stage"] for row in rows}
    merged = stage == "full_invariance" and stages == {"reduced_direct", "full_expansion"}
    require(stages == {stage} or merged, "wrong direct measurement stage")
    expected = {render_key(row): row for row in expected_rows}
    reduced_keys = {render_key(row) for row in build_reduced_direct_requests()}
    seen = set()
    for row in rows:
        key = render_key(row)
        require(key in expected and key not in seen, "direct matrix has extra or duplicate render identity")
        seen.add(key)
        request = expected[key]
        expected_stage = ("reduced_direct" if key in reduced_keys else "full_expansion") if merged else stage
        require(row["stage"] == expected_stage, "wrong direct measurement stage for render identity")
        require(all(field in row and row[field] == value for field, value in request.items() if field != "stage"),
                "direct request payload differs from frozen request")
    require(seen == set(expected), "direct matrix coverage differs")
    return dict(_summarize_chroma_measurements(rows), stage=stage)


def _scale_values(start: int, stop: int, step: int) -> list[float]:
    return [value / 100.0 for value in range(start, stop + 1, step)]


def _fallback_candidate(target_a: float, target_b: float, L: float, scale: float, source: str) -> dict[str, Any]:
    linear_rgb = lab_d65_to_linear_srgb(L, scale * target_a, scale * target_b)
    in_gamut = all(math.isfinite(channel) and 0.0 <= channel <= 1.0 for channel in linear_rgb)
    candidate = {
        "source": source,
        "L_input": float(L),
        "chroma_scale": float(scale),
        "candidate_a": scale * target_a,
        "candidate_b": scale * target_b,
        "linear_rgb": linear_rgb,
        "gamut": "in" if in_gamut else "out",
    }
    if in_gamut:
        candidate["socket_rgba"] = linear_rgb + [1.0]
        candidate["preview_srgb_uint8"] = linear_rgb_to_srgb_uint8(linear_rgb)
    return candidate


def coarse_fallback_candidates(target_a: float, target_b: float) -> list[dict[str, Any]]:
    require(math.isfinite(float(target_a)) and math.isfinite(float(target_b)), "fallback target must be finite")
    rows = [
        _fallback_candidate(target_a, target_b, L, scale, "coarse")
        for L in range(35, 76, 5)
        for scale in _scale_values(60, 120, 10)
    ]
    require(len(rows) == 63, "coarse fallback candidate count drifted")
    return rows


def _refine_fallback_candidates(target_a: float, target_b: float, best_coarse: Mapping[str, Any]) -> list[dict[str, Any]]:
    require(any(dict(best_coarse) == row for row in coarse_fallback_candidates(target_a, target_b) if row["gamut"] == "in"),
            "refine anchor must be an exact in-gamut coarse member for the same target")
    base_L = float(best_coarse["L_input"])
    base_scale = float(best_coarse["chroma_scale"])
    rows = []
    seen: set[tuple[float, float]] = set()
    for delta_L in [-2, -1, 0, 1, 2]:
        L = min(75.0, max(35.0, base_L + delta_L))
        for delta_scale in [-0.04, -0.02, 0.0, 0.02, 0.04]:
            scale = min(1.20, max(0.60, base_scale + delta_scale))
            key = (round(L, 8), round(scale, 8))
            if key in seen:
                continue
            seen.add(key)
            rows.append(_fallback_candidate(target_a, target_b, key[0], key[1], "refine"))
    require(len(rows) <= 25, "refine fallback candidate count exceeds cap")
    return rows


def _development_target(target_a: float, target_b: float) -> dict[str, Any]:
    matches = [record for record in _development_records()
               if (target_a, target_b) == (record["target"]["a"], record["target"]["b"])]
    require(len(matches) == 1, "fallback planning requires a pipeline_development target")
    return matches[0]


def _validated_coarse_summaries(
    target_a: float, target_b: float, summaries: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    record = _development_target(target_a, target_b)
    expected = {(row["L_input"], row["chroma_scale"]): row
                for row in coarse_fallback_candidates(target_a, target_b) if row["gamut"] == "in"}
    require(isinstance(summaries, (list, tuple)) and len(summaries) == len(expected),
            "complete in-gamut coarse summaries are required")
    seen = set()
    validated = []
    for summary in summaries:
        require(isinstance(summary, dict) and all(key in summary for key in ["L_input", "chroma_scale"]),
                "coarse summary candidate identity is required")
        key = (summary["L_input"], summary["chroma_scale"])
        require(key in expected and key not in seen, "extra or duplicate coarse summary")
        seen.add(key)
        _validate_candidate_summary(summary, expected[key], record)
        validated.append(deepcopy(summary))
    require(seen == set(expected), "complete in-gamut coarse summaries are required")
    selected = _rank_fallback_candidates(validated)["diagnostic_best"]
    return validated, selected


def refine_fallback_candidates(
    target_a: float, target_b: float, coarse_summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Refine the best anchor established by the complete measured coarse grid."""
    _, selected = _validated_coarse_summaries(target_a, target_b, coarse_summaries)
    core = _fallback_candidate(target_a, target_b, selected["L_input"], selected["chroma_scale"], "coarse")
    return _refine_fallback_candidates(target_a, target_b, core)


def renderable_candidates(candidates: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [deepcopy(dict(row)) for row in candidates if row.get("gamut") == "in"]


def fallback_candidate_search_plan(
    target_a: float,
    target_b: float,
    coarse_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    evaluated, selected = _validated_coarse_summaries(target_a, target_b, coarse_summaries)
    coarse = coarse_fallback_candidates(target_a, target_b)
    core = _fallback_candidate(target_a, target_b, selected["L_input"], selected["chroma_scale"], "coarse")
    refine = _refine_fallback_candidates(target_a, target_b, core)
    rows = []
    seen: set[tuple[float, float]] = set()
    for row in renderable_candidates(coarse + refine):
        key = (round(row["L_input"], 8), round(row["chroma_scale"], 8))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    require(len(rows) <= 88, "fallback candidate evaluation cap exceeded")
    return {"coarse_count": len(coarse), "evaluated_coarse_count": len(evaluated), "selected_anchor": selected,
            "refine_count": len(refine), "evaluation_count": len(rows), "candidates": rows}


def build_fallback_candidate_requests(
    target_a: float, target_b: float, candidate: Mapping[str, Any], *,
    coarse_summaries: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Bind each measured color candidate to its exact three-view render requests."""
    record = _development_target(target_a, target_b)
    require(candidate.get("source") in {"coarse", "refine"}, "fallback candidate source must be coarse or refine")
    if candidate["source"] == "coarse":
        grid = coarse_fallback_candidates(target_a, target_b)
    else:
        require(coarse_summaries is not None, "refine measurements require complete coarse summaries")
        grid = refine_fallback_candidates(target_a, target_b, coarse_summaries)
    require(any(canonical_sha256(dict(candidate)) == canonical_sha256(expected) for expected in grid if expected["gamut"] == "in"),
            "fallback candidate core must match the in-gamut grid for the same target")
    return _fallback_requests(record, candidate, f"fallback_{candidate['source']}", ["sphere"], [0, 8, 16])


def _fallback_requests(
    record: Mapping[str, Any], candidate: Mapping[str, Any], stage: str,
    shapes: Sequence[str], views: Sequence[int],
) -> list[dict[str, Any]]:
    stable_id = record["source"]["stable_id"]
    target_index = selection.PIPELINE_ROLES["pipeline_development"].index(stable_id)
    L = int(candidate["L_input"])
    scale_percent = round_half_up(candidate["chroma_scale"] * 100)
    source_index = {"coarse": 0, "refine": 1}[candidate["source"]]
    rows = []
    for shape in shapes:
        for view in views:
            rows.append({
                "stage": stage,
                "request_id": f"{stage}_{_stable_slug(stable_id)}_L{L}_scale{scale_percent}_{shape}_view_{view:02d}",
                "stable_id": stable_id,
                "target_index": target_index,
                "pipeline_role": "pipeline_development",
                "target_a": record["target"]["a"],
                "target_b": record["target"]["b"],
                "target_C": record["target"]["C"],
                "target_h_degrees": record["target"]["h_degrees"],
                "shape": shape,
                "material": "Rubber",
                "view_index": view,
                "render_seed": (900000 + target_index * 1000000 + source_index * 200000
                                + (L - 35) * 3660 + (scale_percent - 60) * 60
                                + SHAPE_ORDINAL[shape] * 20 + view),
                **deepcopy(dict(candidate)),
            })
    return rows


def _validate_candidate_summary(
    summary: Mapping[str, Any], core: Mapping[str, Any], record: Mapping[str, Any],
) -> None:
    summary_fields = {"stable_id", "target_a", "target_b", "render_count", "median_e_ch", "p90_e_ch",
                      "median_render_L", "passes", "objective", "request_payloads_sha256"}
    require(set(summary) == set(core) | summary_fields, "candidate summary fields differ")
    require(canonical_sha256({key: summary[key] for key in core}) == canonical_sha256(core), "candidate core differs")
    require((summary["stable_id"], summary["target_a"], summary["target_b"]) ==
            (record["source"]["stable_id"], record["target"]["a"], record["target"]["b"]), "candidate summary target differs")
    requests = _fallback_requests(record, core, f"fallback_{core['source']}", ["sphere"], [0, 8, 16])
    require(summary["request_payloads_sha256"] == canonical_sha256(requests), "candidate summary request identities differ")
    require(type(summary["render_count"]) is int and summary["render_count"] == 3, "candidate summary needs three renders")
    for field in ["median_e_ch", "p90_e_ch", "median_render_L", "objective"]:
        require(type(summary[field]) in {int, float} and math.isfinite(summary[field]), f"candidate {field} must be finite")
    require(0 <= summary["median_e_ch"] <= summary["p90_e_ch"], "candidate error summary is inconsistent")
    objective = summary["median_e_ch"] + 0.25 * summary["p90_e_ch"] + 0.02 * abs(summary["median_render_L"] - 50.0)
    require(summary["objective"] == objective, "candidate objective differs from frozen formula")
    require(summary["passes"] is (summary["median_e_ch"] <= 3.0 and summary["p90_e_ch"] <= 5.0),
            "candidate pass decision differs from thresholds")


def summarize_fallback_candidate(
    candidate: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *,
    coarse_summaries: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    require(len(rows) == 3 and all(key in rows[0] for key in ["target_a", "target_b"]),
            "fallback measurements require three candidate-bound views")
    target_a, target_b = rows[0]["target_a"], rows[0]["target_b"]
    requests = build_fallback_candidate_requests(target_a, target_b, candidate, coarse_summaries=coarse_summaries)
    _validate_request_measurements(rows, requests)
    summary = _summarize_chroma_measurements(rows)
    errors = [summarize_render_measurement(row)["e_ch"] for row in rows]
    median_render_L = median([float(row["median_L"]) for row in rows])
    result = dict(candidate)
    result.update({
        "stable_id": requests[0]["stable_id"],
        "target_a": target_a,
        "target_b": target_b,
        "render_count": len(rows),
        "median_e_ch": median(errors),
        "p90_e_ch": nearest_rank_p90(errors),
        "median_render_L": median_render_L,
        "passes": summary["overall_pass"],
        "request_payloads_sha256": canonical_sha256(requests),
    })
    result["objective"] = result["median_e_ch"] + 0.25 * result["p90_e_ch"] + 0.02 * abs(median_render_L - 50.0)
    return result


def _rank_fallback_candidates(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(candidates) > 0, "at least one fallback candidate summary is required")
    for row in candidates:
        for key in ["L_input", "chroma_scale", "objective", "p90_e_ch"]:
            require(key in row and math.isfinite(float(row[key])), f"candidate {key} must be finite")

    def objective_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
        closeness = math.hypot(float(row["L_input"]) - 50.0, float(row["chroma_scale"]) - 1.0)
        return (float(row["objective"]), float(row["p90_e_ch"]), closeness, float(row["L_input"]), float(row["chroma_scale"]))

    passing = [row for row in candidates if row.get("passes") is True]
    diagnostic = min(candidates, key=objective_key)
    if not passing:
        return {"accepted": None, "diagnostic_best": deepcopy(dict(diagnostic)), "accepted_fallback": False}
    accepted = min(passing, key=objective_key)
    return {"accepted": deepcopy(dict(accepted)), "diagnostic_best": deepcopy(dict(diagnostic)), "accepted_fallback": True}


def choose_fallback_candidate(
    target_a: float, target_b: float,
    coarse_summaries: Sequence[Mapping[str, Any]], refine_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select only after every unique renderable coarse/refine candidate was evaluated."""
    plan = fallback_candidate_search_plan(target_a, target_b, coarse_summaries)
    record = _development_target(target_a, target_b)
    expected = {(row["L_input"], row["chroma_scale"]): row
                for row in plan["candidates"] if row["source"] == "refine"}
    require(isinstance(refine_summaries, (list, tuple)) and len(refine_summaries) == len(expected),
            "complete unique in-gamut refine summaries are required")
    seen = set()
    for summary in refine_summaries:
        require(isinstance(summary, dict) and all(key in summary for key in ["L_input", "chroma_scale"]),
                "refine summary candidate identity is required")
        key = (summary["L_input"], summary["chroma_scale"])
        require(key in expected and key not in seen, "extra or duplicate refine summary")
        seen.add(key)
        _validate_candidate_summary(summary, expected[key], record)
    require(seen == set(expected), "complete unique in-gamut refine summaries are required")
    result = _rank_fallback_candidates(list(coarse_summaries) + list(refine_summaries))
    result.update(evaluated_coarse_count=plan["evaluated_coarse_count"], evaluated_refine_count=len(refine_summaries),
                  evaluated_candidate_count=plan["evaluation_count"], selected_coarse_anchor=plan["selected_anchor"])
    return result


def build_fallback_confirmation_requests(
    target_a: float, target_b: float,
    coarse_summaries: Sequence[Mapping[str, Any]], refine_summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Construct the unchanged 15-render matrix for the fully evaluated search winner."""
    selected = choose_fallback_candidate(target_a, target_b, coarse_summaries, refine_summaries)
    require(selected["accepted_fallback"], "fallback confirmation requires an accepted complete-search candidate")
    winner = selected["accepted"]
    core = _fallback_candidate(target_a, target_b, winner["L_input"], winner["chroma_scale"], winner["source"])
    return _fallback_requests(_development_target(target_a, target_b), core, "fallback_confirmation",
                              ["cube", "sphere", "cylinder"], [0, 4, 8, 12, 16])


def summarize_fallback_confirmation(
    rows: Sequence[Mapping[str, Any]], target_a: float, target_b: float,
    coarse_summaries: Sequence[Mapping[str, Any]], refine_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    requests = build_fallback_confirmation_requests(target_a, target_b, coarse_summaries, refine_summaries)
    _validate_request_measurements(rows, requests)
    result = _summarize_chroma_measurements(rows)
    result.update(stage="fallback_confirmation", confirmed=result["overall_pass"],
                  candidate_request_payloads_sha256=canonical_sha256(requests))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-config", default=REPO_ROOT / PROTOCOL_RELPATH, type=Path)
    args = parser.parse_args(argv)
    try:
        protocol = load_protocol(args.protocol_config)
        summary = request_plan_summary(protocol)
    except (RendererColorCalibrationError, selection.NaturalImageTargetSelectionError) as exc:
        parser.exit(2, f"renderer color calibration validation aborted: {exc}\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
