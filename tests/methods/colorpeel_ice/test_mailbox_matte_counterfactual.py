"""Material edit must reduce highlights without changing the source scene or seams."""

import numpy as np

from src.methods.colorpeel_ice.prepare_mailbox_matte_counterfactual import imagegen_matte, suppress_highlights


def test_highlight_suppression_preserves_background_and_dark_seam():
    image = np.full((512, 512, 3), 45, dtype=np.uint8)
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[96:416, 96:416] = 255
    image[96:416, 96:416] = (160, 75, 60)
    image[245:265, 130:380] = (245, 190, 175)
    image[290:294, 130:380] = (75, 35, 30)
    settings = {
        "gaussian_sigma_pixels": 16.0,
        "highlight_floor_l8": 3.0,
        "suppression_strength": 1.0,
        "edge_transition_pixels": 5.0,
    }

    output, metrics = suppress_highlights(image, mask, settings)

    assert np.array_equal(output[mask == 0], image[mask == 0])
    assert np.array_equal(output[291, 160], image[291, 160])
    assert output[250, 160].mean() < image[250, 160].mean() - 5
    assert metrics["changed_foreground_pixels"] > 0
    assert metrics["outside_mask_changed_pixels"] == 0


def test_imagegen_matte_uses_original_background_and_recolors_only_mask():
    image = np.full((512, 512, 3), (40, 30, 20), dtype=np.uint8)
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[200:400, 120:420] = 255
    reference = np.full((512, 512, 3), (220, 20, 30), dtype=np.uint8)
    reference[200:400, 120:420] = (100, 180, 40)

    green, green_metrics = imagegen_matte(image, mask, reference, "green")
    blue, blue_metrics = imagegen_matte(image, mask, reference, "blue")

    assert np.array_equal(green[mask == 0], image[mask == 0])
    assert np.array_equal(blue[mask == 0], image[mask == 0])
    assert not np.array_equal(green[mask == 255], blue[mask == 255])
    assert green_metrics["outside_mask_changed_pixels"] == 0
    assert blue_metrics["outside_mask_changed_pixels"] == 0
