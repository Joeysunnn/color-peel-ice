import numpy as np
import pytest

from experiments.color_geometry_lightness_v1.prepare import AB, remap
from src.methods.colorpeel_ice import neutral_lch_injection as color


def sample(flat=False):
    image = np.full((128, 128, 3), 100, dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[32:96, 32:96] = 255
    L = np.full(4096, 23.) if flat else np.tile(np.linspace(10, 40, 64), 64)
    lab = np.column_stack((L, np.zeros((4096, 2))))
    image[mask > 0] = color._linear_to_srgb_uint8(color.lab_to_linear_rgb(lab))
    return image, mask


def test_fixed_target_with_geometry_and_unchanged_background():
    image, mask = sample()
    output, audit = remap(image, mask)
    assert audit['numeric_pass']
    assert np.array_equal(output[mask == 0], image[mask == 0])
    lab = color.linear_rgb_to_lab(color._srgb_to_linear(output[mask > 0]))
    assert np.linalg.norm(lab[:, 1:] - AB, axis=1).max() < 1
    assert lab[:, 0].min() >= 41.8 and lab[:, 0].max() <= 58.2
    assert np.quantile(lab[:, 0], .95) - np.quantile(lab[:, 0], .05) > 6


def test_flat_source_does_not_become_fake_geometry():
    with pytest.raises(ValueError, match='insufficient lightness'):
        remap(*sample(flat=True))


def test_nonbinary_mask_is_rejected():
    image, mask = sample()
    mask[32, 32] = 128
    with pytest.raises(ValueError, match='binary'):
        remap(image, mask)
