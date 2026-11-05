import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from experiments.subject_pink_input_repro_v1.prepare import prepare
from experiments.subject_trellis_counterfactual_v1.run import check_source_receipt
from src.methods.colorpeel_ice import gamut_aware_lch_injection as gamut
from src.methods.colorpeel_ice import single_image_subject as subject


class SubjectPinkInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        sources = self.root / 'sources'
        sources.mkdir()
        self.rgb = np.full((512, 512, 3), [40, 75, 100], dtype=np.uint8)
        self.mask = np.zeros((512, 512), dtype=np.uint8)
        self.mask[300:, 100:320] = 255  # Includes a cropped bottom and asymmetric placement.
        self.rgb[self.mask > 0] = [180, 100, 40]
        image_path, mask_path = sources / 'original.png', sources / 'sam.png'
        Image.fromarray(self.rgb).save(image_path)
        Image.fromarray(self.mask).save(mask_path)
        self.original = subject.prepare_inputs(image_path, mask_path, self.root / 'inputs')
        self.parent = self.root / 'inputs/source_receipt.json'
        self.output = self.root / 'pink_inputs'

    def test_condition_changes_hue_but_preserves_binary_alpha_and_original_photo_anchors(self):
        before = {path: Path(path).read_bytes() for path in self.original['source_locks']}
        before[str(self.parent)] = self.parent.read_bytes()
        receipt = prepare(self.parent, self.output)
        for key in ('source_path', 'source_sha256', 'image512_path', 'image512_sha256',
                    'mask_path', 'mask_sha256', 'original_mask_path', 'original_mask_sha256'):
            self.assertEqual(receipt[key], self.original[key])
        with Image.open(receipt['cutout_path']) as image:
            actual = np.asarray(image)
        np.testing.assert_array_equal(actual[..., 3], self.mask)
        self.assertTrue((actual[self.mask == 0, :3] == 0).all())
        self.assertTrue(np.any(actual[self.mask > 0, :3] != self.rgb[self.mask > 0]))
        core_lab = gamut.linear_rgb_to_lab(gamut._srgb_to_linear(actual[400:450, 150:200, :3].reshape(-1, 3)))
        hue = np.degrees(np.arctan2(core_lab[:, 2], core_lab[:, 1]))
        self.assertLess(float(np.abs(hue).max()), 1.0)
        self.assertEqual(receipt['conditioning_recolor']['recolor_metrics']['outside_mask_changed_pixel_count'], 0)
        self.assertGreater(receipt['conditioning_recolor']['recolor_metrics']['soft_edge_pixel_count'], 0)
        photos = subject.photo_rows(receipt, self.root / 'photos', 'test object')
        with Image.open(photos[0]['image']) as image:
            np.testing.assert_array_equal(np.asarray(image), self.rgb)
        self.assertEqual({path: Path(path).read_bytes() for path in before}, before)
        self.assertEqual(receipt['source_locks'][str(self.parent)], subject.sha256(self.parent))
        self.assertEqual(receipt['source_locks'][self.original['cutout_path']], self.original['cutout_sha256'])
        check_source_receipt(self.output / 'source_receipt.json')

    def test_parent_receipt_metadata_tampering_is_rejected_before_trellis(self):
        prepare(self.parent, self.output)
        changed = json.loads(self.parent.read_text())
        changed['segmentation'] = 'changed after pink preparation'
        subject.write_json(self.parent, changed)
        with self.assertRaisesRegex(ValueError, 'source lock hash mismatch'):
            check_source_receipt(self.output / 'source_receipt.json')

    def test_historical_five_hues_preserve_alpha_and_change_rgb_as_declared(self):
        hashes = set()
        for hue in (0.0, 120.0, 180.0, 240.0, 300.0):
            with self.subTest(hue=hue):
                receipt = prepare(self.parent, self.root / f'hue{hue:g}', hue)
                with Image.open(receipt['cutout_path']) as image:
                    actual = np.asarray(image)
                np.testing.assert_array_equal(actual[..., 3], self.mask)
                self.assertTrue((actual[self.mask == 0, :3] == 0).all())
                core = actual[400:450, 150:200, :3].reshape(-1, 3)
                lab = gamut.linear_rgb_to_lab(gamut._srgb_to_linear(core))
                measured = np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))
                error = np.abs((measured - hue + 180.0) % 360.0 - 180.0)
                self.assertLess(float(error.max()), 1.0)
                self.assertEqual(receipt['conditioning_recolor']['hue_degrees'], hue)
                self.assertEqual(receipt['image512_sha256'], self.original['image512_sha256'])
                hashes.add(receipt['cutout_sha256'])
        self.assertEqual(len(hashes), 5)

    def test_nonhistorical_hue_rejected_before_output_creation(self):
        for hue in (60.0, float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'five historical'):
                prepare(self.parent, self.output, hue)
            self.assertFalse(self.output.exists())

    def test_missing_current_mask_has_no_historical_fallback(self):
        Path(self.original['mask_path']).unlink()
        with self.assertRaises(FileNotFoundError):
            prepare(self.parent, self.output)
        self.assertFalse(self.output.exists())

    def test_source_folder_and_existing_output_are_not_overwritten(self):
        with self.assertRaisesRegex(ValueError, 'source folders'):
            prepare(self.parent, self.parent.parent / 'pink')
        prepare(self.parent, self.output)
        before = (self.output / 'source_receipt.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'fresh'):
            prepare(self.parent, self.output)
        self.assertEqual((self.output / 'source_receipt.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
