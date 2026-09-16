from __future__ import annotations

import numpy as np

from src.methods.colorpeel_ice import gamut_aware_lch_injection as gamut
from src.methods.colorpeel_ice import natural_subject_recolor_pilot as pilot


def test_mailbox_protocol_holds_out_orange_source_from_auxiliary_hues():
    import json
    from pathlib import Path

    protocol = json.loads((Path(__file__).parents[3] / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_mailbox_pilot_protocol_v1.json").read_text(encoding="utf-8"))
    assert protocol["held_out_guard"]["generated_palette_names"] == ["red", "green", "cyan", "blue", "magenta"]
    assert protocol["held_out_guard"]["excluded_palette_names"] == ["yellow"]
    palette = {item["name"]: item["hue_degrees"] for item in protocol["auxiliary_palette"]}
    distances = pilot.hue_distance_degrees(
        [palette[name] for name in protocol["held_out_guard"]["generated_palette_names"]],
        protocol["source"]["original_hue_degrees"],
    )
    assert (distances >= protocol["held_out_guard"]["minimum_hue_separation_degrees"]).all()


def test_recolor_preserves_background_lightness_and_fixed_hue():
    image = np.full((12, 12, 3), [80, 100, 120], dtype=np.uint8)
    mask = np.zeros((12, 12), dtype=np.uint8); mask[2:10, 2:10] = 255
    alpha = np.zeros((12, 12), dtype=np.uint16); alpha[2:10, 2:10] = 65535
    output, metrics, maps = pilot.recolor(image, mask, alpha, 120.0)
    assert np.array_equal(output[mask == 0], image[mask == 0])
    assert metrics["outside_mask_changed_pixel_count"] == 0 and metrics["rgb_clipping_pixel_count"] == 0
    before = gamut.linear_rgb_to_lab(gamut._srgb_to_linear(image[mask == 255]))
    after = gamut.linear_rgb_to_lab(gamut._srgb_to_linear(output[mask == 255]))
    assert np.max(np.abs(after[:, 0] - before[:, 0])) < 1.0
    assert metrics["hue_error_p95_degrees"] < 3.0 and maps["C_final"].shape == mask.shape


def test_recolor_reduces_only_chroma_for_gamut_and_soft_alpha_never_changes_outside():
    image = np.full((8, 8, 3), [200, 40, 40], dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8); mask[2:6, 2:6] = 255
    alpha = np.zeros((8, 8), dtype=np.uint16); alpha[2:6, 2:6] = 40000
    output, metrics, maps = pilot.recolor(image, mask, alpha, 240.0)
    inside = mask == 255
    assert np.array_equal(output[~inside], image[~inside])
    assert metrics["alpha_outside_nonzero_pixel_count"] == 0
    assert np.all(maps["C_final"][inside] <= maps["source_C"][inside] + 1e-5)
    assert metrics["gamut_reduced_pixel_ratio"] >= 0.0
