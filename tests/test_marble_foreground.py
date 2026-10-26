"""Verify original background cannot enter the foreground-only image conditions."""

import unittest

import numpy as np
from PIL import Image

from experiments.subject_marble_foreground_v1.preprocess import NEUTRAL, mask_depth, prepare_inputs


class ForegroundInputsTest(unittest.TestCase):
    def setUp(self):
        self.mask = Image.new("L", (512, 512))
        self.mask.paste(255, (90, 125, 390, 300))
        self.first = Image.new("RGB", self.mask.size, (255, 0, 220))
        self.second = Image.new("RGB", self.mask.size, (0, 255, 20))
        foreground = Image.new("RGB", self.mask.size, (230, 80, 30))
        foreground.paste((20, 90, 210), (180, 160, 250, 220))
        self.first.paste(foreground, (0, 0), self.mask)
        self.second.paste(foreground, (0, 0), self.mask)

    def test_all_generation_images_and_visible_cutout_ignore_original_background(self):
        before = self.first.tobytes(), self.mask.tobytes()
        first, second = prepare_inputs(self.first, self.mask), prepare_inputs(self.second, self.mask)
        for name in ("target", "reference", "mask", "cutout", "original_neutral"):
            self.assertEqual(first[name].tobytes(), second[name].tobytes(), name)
        self.assertEqual(first["reference_box"], second["reference_box"])
        self.assertEqual(before, (self.first.tobytes(), self.mask.tobytes()))
        cutout = np.asarray(first["cutout"])
        inside = np.asarray(self.mask) != 0
        np.testing.assert_array_equal(cutout[inside, :3], np.asarray(self.first)[inside])
        self.assertTrue((cutout[~inside] == 0).all(), "transparent hidden RGB must also be removed")

    def test_masking_precedes_resizing_and_exterior_is_exact_neutral(self):
        actual = prepare_inputs(self.first, self.mask)
        expected = Image.composite(self.first, Image.new("RGB", self.first.size, NEUTRAL), self.mask)
        expected = expected.resize((1024, 1024), Image.Resampling.BICUBIC)
        expected = Image.composite(expected, Image.new("RGB", (1024, 1024), NEUTRAL), actual["mask"])
        self.assertEqual(actual["target"].tobytes(), expected.tobytes())
        outside = np.asarray(actual["mask"]) == 0
        self.assertTrue((np.asarray(actual["target"])[outside] == 127).all())
        self.assertEqual(set(np.unique(np.asarray(actual["mask"]))), {0, 255})
        wrong = self.first.resize((1024, 1024), Image.Resampling.BICUBIC)
        wrong = Image.composite(wrong, Image.new("RGB", wrong.size, NEUTRAL), actual["mask"])
        self.assertNotEqual(wrong.tobytes(), actual["target"].tobytes())

    def test_reference_uses_padded_bbox_and_preserves_aspect_ratio(self):
        actual = prepare_inputs(self.first, self.mask)
        self.assertEqual(actual["reference_box"], (120, 190, 840, 660))
        box = actual["reference_box"]
        crop = actual["target"].crop(box)
        expected_height = round(crop.height * 1024 / crop.width)
        offset = (1024 - expected_height) // 2
        expected = Image.new("RGB", (1024, 1024), NEUTRAL)
        expected.paste(crop.resize((1024, expected_height), Image.Resampling.BICUBIC), (0, offset))
        self.assertEqual(actual["reference"].tobytes(), expected.tobytes())
        self.assertTrue((np.asarray(actual["reference"])[:offset] == 127).all())
        self.assertEqual({name: (actual[name].mode, actual[name].size) for name in
            ("target", "reference", "mask", "cutout", "original_neutral")}, {
                "target": ("RGB", (1024, 1024)), "reference": ("RGB", (1024, 1024)),
                "mask": ("L", (1024, 1024)), "cutout": ("RGBA", (512, 512)),
                "original_neutral": ("RGB", (512, 512))})

    def test_depth_background_is_zero_without_altering_foreground(self):
        mask = prepare_inputs(self.first, self.mask)["mask"]
        depth = Image.new("RGB", mask.size, (50, 130, 210))
        depth.paste((25, 90, 180), (200, 250, 450, 450))
        original = depth.tobytes()
        result = mask_depth(depth, mask)
        inside = np.asarray(mask) != 0
        np.testing.assert_array_equal(np.asarray(result)[inside], np.asarray(depth)[inside])
        self.assertTrue((np.asarray(result)[~inside] == 0).all())
        self.assertEqual((result.mode, result.size), ("RGB", (1024, 1024)))
        self.assertEqual(original, depth.tobytes())
        gray = mask_depth(depth.convert("L"), mask)
        np.testing.assert_array_equal(np.asarray(gray)[inside], np.asarray(depth.convert("L").convert("RGB"))[inside])

    def test_rejects_misaligned_nonbinary_or_wrong_mode_inputs(self):
        invalid = ((self.first.resize((256, 256)), self.mask), (self.first.convert("RGBA"), self.mask),
                   (self.first, self.mask.convert("RGB")), (self.first, Image.new("L", self.mask.size, 128)))
        for image, mask in invalid:
            with self.assertRaises(ValueError):
                prepare_inputs(image, mask)
        with self.assertRaises(ValueError):
            mask_depth(Image.new("L", (512, 512)), self.mask)


if __name__ == "__main__":
    unittest.main()
