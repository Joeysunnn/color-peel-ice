import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.methods.colorpeel_ice import single_image_subject as subject


class SingleImageSubjectTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def source(self, size=(512, 512)):
        width, height = size
        rgb = np.full((height, width, 3), [180, 100, 40], dtype=np.uint8)
        rgb[:, :width // 4] = [20, 40, 60]
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[height // 4:3 * height // 4, width // 2:3 * width // 4] = 255
        image_path, mask_path = self.root / 'source.png', self.root / 'mask.png'
        Image.fromarray(rgb).save(image_path)
        Image.fromarray(mask).save(mask_path)
        return image_path, mask_path, rgb, mask

    def test_full_canvas_pair_preserves_asymmetric_coordinates_and_padding(self):
        image_path, mask_path, rgb, mask = self.source((200, 100))
        receipt = subject.prepare_inputs(image_path, mask_path, self.root / 'inputs')
        self.assertEqual(receipt['transform']['resized_size'], [512, 256])
        self.assertEqual(receipt['transform']['pad_left_top'], [0, 128])
        self.assertFalse(receipt['transform']['crop'])
        with Image.open(receipt['mask_path']) as opened:
            actual = np.asarray(opened)
        expected = np.zeros((512, 512), dtype=np.uint8)
        expected[192:320, 256:384] = 255
        np.testing.assert_array_equal(actual, expected)
        with Image.open(receipt['cutout_path']) as opened:
            rgba = np.asarray(opened)
        np.testing.assert_array_equal(rgba[..., 3], actual)
        self.assertTrue((rgba[actual == 0, :3] == 0).all())
        self.assertEqual(receipt['source_sha256'], subject.sha256(image_path))
        self.assertEqual(receipt['original_mask_sha256'], subject.sha256(mask_path))
        subject.check_locks(receipt['source_locks'])

    def test_512_original_pixels_and_background_are_unchanged_in_photo_rows(self):
        image_path, mask_path, original, mask = self.source()
        receipt = subject.prepare_inputs(image_path, mask_path, self.root / 'inputs')
        rows = subject.photo_rows(receipt, self.root / 'photos', 'number plate')
        self.assertEqual([row['id'] for row in rows],
                         ['photo_original', 'photo_lch_blue', 'photo_lch_green'])
        for row in rows:
            with Image.open(row['image']) as opened, Image.open(row['mask']) as opened_mask:
                actual, actual_mask = np.asarray(opened), np.asarray(opened_mask)
            np.testing.assert_array_equal(actual_mask, mask)
            np.testing.assert_array_equal(actual[mask == 0], original[mask == 0])
            self.assertEqual(row['caption'].count('<S*>'), 1)
            self.assertIn('number plate', row['caption'])
            if row['id'] == 'photo_original':
                np.testing.assert_array_equal(actual, original)
            else:
                self.assertTrue(np.any(actual[mask > 0] != original[mask > 0]))
                self.assertEqual(row['recolor_metrics']['outside_mask_changed_pixel_count'], 0)
        self.assertEqual([row['hue_degrees'] for row in rows[1:]], [240., 120.])

    def test_invalid_masks_and_reused_directory_are_rejected(self):
        image_path, mask_path, rgb, mask = self.source()
        image = Image.fromarray(rgb)
        for invalid in (mask[:128], np.full(mask.shape, 128), np.zeros_like(mask)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                subject.square_pair(image, invalid)
        subject.prepare_inputs(image_path, mask_path, self.root / 'inputs')
        with self.assertRaisesRegex(ValueError, 'fresh'):
            subject.prepare_inputs(image_path, mask_path, self.root / 'inputs')

    def test_source_mutation_breaks_receipt_locks(self):
        image_path, mask_path, _, _ = self.source()
        receipt = subject.prepare_inputs(image_path, mask_path, self.root / 'inputs')
        image_path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            subject.check_locks(receipt['source_locks'])

    def test_exact_eight_counterfactuals_with_dynamic_subject_category(self):
        rows = subject.appearance_rows('vase')
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({row['id'] for row in rows}), 8)
        self.assertEqual({(row['color'], row['metallic'], row['roughness']) for row in rows},
                         {(c, m, r) for c in ('blue', 'green') for m in (0, 1) for r in (.15, .85)})
        self.assertTrue(all(row['caption'].count('<S*>') == 1 and 'vase' in row['caption'] for row in rows))
        with self.assertRaises(ValueError):
            subject.appearance_rows('vase <C*>')

    def test_foreign_instance_asset_cannot_be_attached(self):
        image_path, mask_path, _, _ = self.source()
        subject.prepare_inputs(image_path, mask_path, self.root / 'inputs')
        trellis = self.root / 'trellis/repro_outputs'
        trellis.mkdir(parents=True)
        (trellis / 'status.json').write_text(json.dumps({'status': 'generated_pending_user_review',
            'generated_assets': 1, 'source_receipt': {'sha256': 'wrong-instance'}}))
        with self.assertRaisesRegex(ValueError, 'this instance'):
            subject.render_dataset(self.root / 'inputs', self.root / 'trellis', self.root / 'render', 'vase')
        self.assertFalse((self.root / 'render').exists())


if __name__ == '__main__':
    unittest.main()
