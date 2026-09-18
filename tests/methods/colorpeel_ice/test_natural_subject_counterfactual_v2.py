from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.methods.colorpeel_ice import natural_subject_counterfactual_v2 as v2


ROOT = Path(__file__).parents[3]


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = np.full((9, 9, 3), [20, 180, 40], dtype=np.uint8)
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[3:6, 2:5] = 255
    alpha = mask.astype(np.uint16) * 257
    foreground = np.full((9, 9, 3), [220, 30, 80], dtype=np.uint8)
    recolored = source.copy()
    recolored[mask == 255] = foreground[mask == 255]
    return recolored, source, mask, alpha


def _variant(**overrides: object) -> dict[str, object]:
    return {
        "background_rgb": [100, 110, 120],
        "scale": 1.0,
        "translate_x": 0,
        "translate_y": 0,
        "mirror": False,
    } | overrides


def test_same_fixed_variant_is_byte_identical_and_background_is_exact_outside_mask():
    first = v2.apply_variant(*_inputs(), _variant())
    second = v2.apply_variant(*_inputs(), _variant())
    image, mask, metrics = first
    assert hashlib.sha256(image.tobytes()).hexdigest() == hashlib.sha256(second[0].tobytes()).hexdigest()
    assert hashlib.sha256(mask.tobytes()).hexdigest() == hashlib.sha256(second[1].tobytes()).hexdigest()
    assert set(np.unique(mask).tolist()) <= {0, 255}
    assert np.array_equal(image[mask == 0], np.array([100, 110, 120], dtype=np.uint8))
    assert metrics["bbox_xyxy_inclusive"] == [2, 3, 4, 5]


def test_translation_scale_and_mirror_apply_to_the_same_binary_mask_geometry():
    _, _, mask, _ = _inputs()
    translated = v2.apply_variant(*_inputs(), _variant(translate_x=2, translate_y=-1))[1]
    mirrored = v2.apply_variant(*_inputs(), _variant(mirror=True))[1]
    scaled = v2.apply_variant(*_inputs(), _variant(scale=0.5))[1]
    assert np.array_equal(translated, np.roll(np.roll(mask, -1, axis=0), 2, axis=1))
    assert np.array_equal(mirrored, np.fliplr(mask))
    assert 0 < int((scaled == 255).sum()) < int((mask == 255).sum())


def test_soft_edge_is_recomposited_over_the_new_background_without_old_background_leakage():
    source = np.full((5, 5, 3), [10, 200, 20], dtype=np.uint8)
    mask = np.zeros((5, 5), dtype=np.uint8); mask[2, 2] = 255
    alpha = np.zeros((5, 5), dtype=np.uint16); alpha[2, 2] = 32768
    foreground = np.array([210, 30, 80], dtype=np.uint8)
    recolored = source.copy()
    recolored[2, 2] = np.floor(0.5 * foreground + 0.5 * source[2, 2] + 0.5).astype(np.uint8)
    image, transformed_mask, _ = v2.apply_variant(recolored, source, mask, alpha, _variant(background_rgb=[90, 90, 90]))
    expected = np.floor(0.5 * foreground + 0.5 * np.array([90, 90, 90]) + 0.5).astype(np.uint8)
    assert np.array_equal(image[2, 2], expected)
    assert np.array_equal(image[transformed_mask == 0], np.array([90, 90, 90], dtype=np.uint8))


def test_protocol_has_a_finite_held_out_five_by_five_subject_grid():
    protocol = json.loads((ROOT / "experiments/natural_image_subject_color_pilot/configs/d1_subject_counterfactual_v2_protocol_v1.json").read_text(encoding="utf-8"))
    assert protocol["source_repaired_subject"]["stable_id"] == "D1GT:81/130.png"
    assert len(protocol["auxiliary_colors"]) == len(protocol["variants"]) == 5
    assert protocol["expected_record_count"] == 25
    assert protocol["approval_state"] == {
        "subject_counterfactual_v2_asset_generation_approved": True,
        "subject_training_approved": False,
        "joint_training_approved": False,
    }
    source_hue = protocol["held_out_guard"]["source_hue_degrees"]
    for item in protocol["auxiliary_colors"].values():
        assert abs((item["hue_degrees"] - source_hue + 180.0) % 360.0 - 180.0) >= protocol["held_out_guard"]["minimum_hue_separation_degrees"]
