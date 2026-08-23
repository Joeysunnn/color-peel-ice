"""Pure, Blender-independent constants for the locked multiview renderer."""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any


EXPECTED_PROFILE_V1 = {
    "schema_version": 1,
    "profile_id": "multiview_render_v1",
    "blender": {
        "version": "4.2.11",
        "render_engine": "CYCLES",
        "cycles_device": "CUDA",
        "cycles_samples": 512,
        "cycles_seed": "render_seed",
        "resolution": [512, 512],
        "jpeg_quality": 95,
    },
    "object": {
        "scale": 1.3,
        "rotation_z_degrees": 0.0,
        "position_xy": [0.0, 0.0],
        "material": "metal",
    },
    "camera": {
        "name": "Camera",
        "jitter_distribution": "official_clevr_uniform_additive_xyz",
        "jitter_magnitude": 0.5,
        "rotation_policy": "preserve_base_scene",
    },
    "lights": {
        "order": ["Lamp_Key", "Lamp_Back", "Lamp_Fill"],
        "fixed_order": ["Area"],
        "jitter_distribution": "official_clevr_uniform_additive_xyz",
        "jitter_magnitude": 1.0,
        "rgb": [1.0, 1.0, 1.0],
    },
    "background": {
        "profile_id": "clevr_neutral_fixed_v1",
        "varied": False,
        "world_rgba": [0.05, 0.05, 0.05, 1.0],
        "ground_rgba": [0.5, 0.5, 0.5, 1.0],
    },
    "rng": {
        "implementation": "python_random_Random",
        "seed": "render_seed",
        "draw_order": ["camera_xyz", "Lamp_Key_xyz", "Lamp_Back_xyz", "Lamp_Fill_xyz"],
    },
}

EXPECTED_PROFILE_V2 = {
    "schema_version": 1,
    "profile_id": "multiview_render_v2",
    "blender": {
        "version": "4.2.11",
        "render_engine": "CYCLES",
        "cycles_device": "CUDA",
        "cycles_samples": 512,
        "cycles_seed": "render_seed",
        "resolution": [512, 512],
        "jpeg_quality": 95,
    },
    "object": {
        "scale": 1.3,
        "rotation_z_degrees": 0.0,
        "position_xy": [0.0, 0.0],
        "material": "metal",
    },
    "camera": {
        "name": "Camera",
        "sampling_model": "orbit_look_at_object_center",
        "azimuth_jitter_degrees": 18.0,
        "elevation_jitter_degrees": 10.0,
        "distance_jitter_fraction": 0.05,
        "rotation_policy": "look_at_object_center_negative_z_y_up",
        "base_constraint_policy": "mute_before_explicit_look_at",
    },
    "lights": {
        "order": ["Lamp_Key", "Lamp_Back", "Lamp_Fill"],
        "fixed_order": ["Area"],
        "jitter_distribution": "official_clevr_uniform_additive_xyz",
        "jitter_magnitude": 1.0,
        "rgb": [1.0, 1.0, 1.0],
    },
    "background": {
        "profile_id": "clevr_neutral_fixed_v2",
        "varied": False,
        "world_rgba": [0.05, 0.05, 0.05, 1.0],
        "ground_rgba": [0.5, 0.5, 0.5, 1.0],
    },
    "rng": {
        "implementation": "python_random_Random",
        "seed": "render_seed",
        "draw_order": [
            "camera_azimuth", "camera_elevation", "camera_distance",
            "Lamp_Key_xyz", "Lamp_Back_xyz", "Lamp_Fill_xyz",
        ],
    },
}

# Backwards-compatible alias: v1 callers and its canonical fingerprint remain unchanged.
EXPECTED_PROFILE = EXPECTED_PROFILE_V1
EXPECTED_PROFILES = {
    EXPECTED_PROFILE_V1["profile_id"]: EXPECTED_PROFILE_V1,
    EXPECTED_PROFILE_V2["profile_id"]: EXPECTED_PROFILE_V2,
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_profile(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, dict):
        raise ValueError("Renderer profile must be an object")
    expected = EXPECTED_PROFILES.get(profile.get("profile_id"))
    if expected is None or profile != expected:
        raise ValueError("Renderer profile differs from locked multiview_render_v1/v2")
    return profile


def official_jitter_metadata(render_seed: int, profile: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(render_seed)

    def draw(magnitude: float) -> list[float]:
        return [2.0 * magnitude * (rng.random() - 0.5) for _ in range(3)]

    return {
        "camera_offset": draw(profile["camera"]["jitter_magnitude"]),
        "light_offsets": {
            name: draw(profile["lights"]["jitter_magnitude"])
            for name in profile["lights"]["order"]
        },
    }


def orbit_jitter_metadata(render_seed: int, profile: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(render_seed)

    def scalar(magnitude: float) -> float:
        return 2.0 * magnitude * (rng.random() - 0.5)

    def xyz(magnitude: float) -> list[float]:
        return [scalar(magnitude) for _ in range(3)]

    return {
        "camera_orbit_jitter": {
            "azimuth_degrees": scalar(profile["camera"]["azimuth_jitter_degrees"]),
            "elevation_degrees": scalar(profile["camera"]["elevation_jitter_degrees"]),
            "distance_fraction": scalar(profile["camera"]["distance_jitter_fraction"]),
        },
        "light_offsets": {
            name: xyz(profile["lights"]["jitter_magnitude"])
            for name in profile["lights"]["order"]
        },
    }


def spherical_pose(location: list[float], target: list[float]) -> dict[str, float]:
    delta = [float(location[index]) - float(target[index]) for index in range(3)]
    radius = sum(value * value for value in delta) ** 0.5
    if radius <= 0.0:
        raise ValueError("Camera radius must be positive")
    horizontal = (delta[0] * delta[0] + delta[1] * delta[1]) ** 0.5
    return {
        "radius": radius,
        "azimuth_degrees": math.degrees(math.atan2(delta[1], delta[0])),
        "elevation_degrees": math.degrees(math.atan2(delta[2], horizontal)),
    }


def orbit_location(target: list[float], radius: float, azimuth_degrees: float,
                   elevation_degrees: float) -> list[float]:
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    horizontal = radius * math.cos(elevation)
    return [
        float(target[0]) + horizontal * math.cos(azimuth),
        float(target[1]) + horizontal * math.sin(azimuth),
        float(target[2]) + radius * math.sin(elevation),
    ]


def quaternion_rotate_vector(quaternion: list[float], vector: list[float]) -> list[float]:
    if len(quaternion) != 4 or len(vector) != 3:
        raise ValueError("Quaternion/vector dimensions are invalid")
    w, x, y, z = (float(value) for value in quaternion)
    vx, vy, vz = (float(value) for value in vector)
    dot_uv = x * vx + y * vy + z * vz
    dot_uu = x * x + y * y + z * z
    cross = [y * vz - z * vy, z * vx - x * vz, x * vy - y * vx]
    return [
        2.0 * dot_uv * x + (w * w - dot_uu) * vx + 2.0 * w * cross[0],
        2.0 * dot_uv * y + (w * w - dot_uu) * vy + 2.0 * w * cross[1],
        2.0 * dot_uv * z + (w * w - dot_uu) * vz + 2.0 * w * cross[2],
    ]


def look_at_alignment(final_location: list[float], target: list[float],
                      rotation_quaternion: list[float]) -> float:
    direction = [float(target[index]) - float(final_location[index]) for index in range(3)]
    norm = sum(value * value for value in direction) ** 0.5
    if norm <= 0.0:
        raise ValueError("Camera and look-at target must differ")
    direction = [value / norm for value in direction]
    forward = quaternion_rotate_vector(rotation_quaternion, [0.0, 0.0, -1.0])
    forward_norm = sum(value * value for value in forward) ** 0.5
    return sum(direction[index] * forward[index] / forward_norm for index in range(3))


def look_at_y_up_alignment(final_location: list[float], target: list[float],
                           rotation_quaternion: list[float]) -> float:
    """Return local +Y alignment with world +Z projected onto the image plane."""
    direction = [float(target[index]) - float(final_location[index]) for index in range(3)]
    direction_norm = sum(value * value for value in direction) ** 0.5
    if direction_norm <= 0.0:
        raise ValueError("Camera and look-at target must differ")
    direction = [value / direction_norm for value in direction]
    world_up = [0.0, 0.0, 1.0]
    projection = sum(world_up[index] * direction[index] for index in range(3))
    expected_up = [world_up[index] - projection * direction[index] for index in range(3)]
    expected_norm = sum(value * value for value in expected_up) ** 0.5
    if expected_norm <= 1e-12:
        raise ValueError("Y-up is undefined at an orbit pole")
    expected_up = [value / expected_norm for value in expected_up]
    actual_up = quaternion_rotate_vector(rotation_quaternion, [0.0, 1.0, 0.0])
    actual_norm = sum(value * value for value in actual_up) ** 0.5
    if actual_norm <= 0.0:
        raise ValueError("Camera quaternion has zero norm")
    return sum(expected_up[index] * actual_up[index] / actual_norm for index in range(3))
