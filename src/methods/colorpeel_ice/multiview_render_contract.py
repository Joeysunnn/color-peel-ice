"""Pure, Blender-independent constants for the locked multiview renderer."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any


EXPECTED_PROFILE = {
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


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_profile(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, dict):
        raise ValueError("Renderer profile must be an object")
    if profile != EXPECTED_PROFILE:
        raise ValueError("Renderer profile differs from locked multiview_render_v1")
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
