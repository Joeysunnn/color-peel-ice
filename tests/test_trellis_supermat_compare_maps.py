"""Common-ROI comparison must preserve alignment, float estimates and failure evidence."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from experiments.trellis_supermat_compare_v1 import compare_maps as compare


class CompareMapsTests(unittest.TestCase):
    def fixture(self, root, *, chw=True, tiny_overlap=False):
        sm, tr = root / 'sm', root / 'tr'
        sm.mkdir()
        tr.mkdir()
        source_mask = np.zeros((512, 512), bool)
        source_mask[32:480, 32:480] = True
        raw_mask = source_mask.astype(np.float32)
        if tiny_overlap:
            raw_mask[:] = 0
            raw_mask[120:140, 120:140] = 1
        roi = np.zeros((512, 512), np.uint8)
        roi[100:400, 100:400] = 255
        source = np.full((512, 512, 4), 120, np.uint8)
        source[..., 3] = source_mask.astype(np.uint8) * 255
        Image.fromarray(source).save(root / 'source.png')
        Image.fromarray(roi).save(root / 'roi.png')
        Image.new('RGB', (512, 512), (90, 150, 100)).save(sm / 'albedo.png')
        Image.new('L', (512, 512), 90).save(sm / 'roughness.png')
        Image.new('L', (512, 512), 10).save(sm / 'metallic.png')
        color = np.full((3, 512, 512) if chw else (512, 512, 3), 0.234567, np.float32)
        roughness = np.full((512, 512), 0.352345, np.float32)
        metallic = np.full((512, 512), -0.025, np.float32)
        np.savez(tr / 'attributes.npz', base_color=color, roughness=roughness, metallic=metallic, mask=raw_mask, alpha=raw_mask.copy())
        Image.new('RGB', (768, 512), (12, 34, 56)).save(root / 'actual_condition.png')
        fitted = raw_mask > 0.5
        Image.fromarray(fitted.astype(np.uint8) * 255).save(tr / 'fitted_mask.png')
        iou = float((source_mask & fitted).sum() / (source_mask | fitted).sum())
        (tr / 'review.json').write_text(json.dumps({'metrics': {'original512_mask_iou': iou},
                                                   'camera': {'status': 'estimated'}}), encoding='utf8')
        return sm, tr, root / 'source.png', root / 'roi.png', root / 'output'

    def test_raw_float_statistics_shared_roi_and_clipping_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            result = compare.compare(*args)
            self.assertEqual(result['status'], 'ready_for_review')
            self.assertEqual(result['coverage']['common_roi_pixels'], 296 * 296)
            self.assertAlmostEqual(result['coverage']['retention_fraction'], (296 * 296) / (300 * 300))
            tr = json.loads((args[-1] / 'trellis/material.json').read_text())
            sm = json.loads((args[-1] / 'supermat/material.json').read_text())
            self.assertAlmostEqual(tr['roughness'], float(np.float32(0.352345)), places=10)
            self.assertNotEqual(tr['roughness'], round(tr['roughness'] * 255) / 255)
            self.assertEqual(tr['metallic'], 0)
            self.assertLess(tr['distributions']['metallic']['raw_min'], 0)
            self.assertEqual(tr['distributions']['metallic']['clipping_fraction'], 1)
            self.assertAlmostEqual(sm['roughness'], 90 / 255)
            self.assertEqual(tr['common_roi_sha256'], sm['common_roi_sha256'])
            self.assertIn(str(args[1] / 'attributes.npz'), tr['source_locks'])
            self.assertTrue((args[-1] / 'panels/trellis_roi_overlay.png').is_file())
            self.assertEqual(result['conditioning']['trellis_original_size'], [768, 512])
            self.assertTrue((args[-1] / 'conditioning_comparison.png').is_file())
            with Image.open(args[-1] / 'comparison.png') as image:
                self.assertEqual(image.size, (2048, 1770))

    def test_insufficient_coverage_keeps_visuals_but_publishes_no_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary), chw=False, tiny_overlap=True)
            result = compare.compare(*args)
            self.assertEqual(result['status'], 'failed')
            self.assertGreater(result['coverage']['common_roi_pixels'], compare.MIN_PIXELS)
            self.assertLess(result['coverage']['retention_fraction'], compare.MIN_RETENTION)
            self.assertFalse(result['coverage']['roi_expanded'])
            self.assertTrue((args[-1] / 'comparison.png').is_file())
            self.assertTrue((args[-1] / 'common_roi.png').is_file())
            self.assertFalse((args[-1] / 'trellis/material.json').exists())
            self.assertFalse((args[-1] / 'supermat/material.json').exists())

    def test_transparent_region_is_excluded_and_raw_alpha_range_is_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            raw_path = args[1] / 'attributes.npz'
            with np.load(raw_path) as raw:
                values = {name: raw[name].copy() for name in raw.files}
            values['alpha'][:] = 0.4
            values['alpha'][110, 110] = -0.001
            values['alpha'][111, 111] = 1.001
            np.savez(raw_path, **values)
            result = compare.compare(*args)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['coverage']['common_roi_pixels'], 0)
            self.assertEqual(result['coverage']['trellis_raw_alpha_threshold'], 0.9)
            stats = result['coverage']['trellis_alpha_on_reviewed_roi']
            self.assertLess(stats['raw_min'], 0)
            self.assertGreater(stats['raw_max'], 1)
            self.assertGreater(stats['clipping_fraction'], 0)
            self.assertFalse((args[-1] / 'trellis/material.json').exists())
            values.pop('alpha')
            np.savez(raw_path, **values)
            with self.assertRaisesRegex(ValueError, 'alpha'):
                compare.load_trellis(raw_path)

    def test_raw_npz_required_and_camera_mask_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            raw = args[1] / 'attributes.npz'
            saved = raw.read_bytes()
            raw.unlink()
            with self.assertRaises(FileNotFoundError):
                compare.compare(*args)
            self.assertFalse(args[-1].exists())
            raw.write_bytes(saved)
            Image.new('L', (512, 512), 255).save(args[1] / 'fitted_mask.png')
            with self.assertRaisesRegex(ValueError, 'differs from raw mask'):
                compare.compare(*args)

    def test_mask_confidence_threshold_and_zero_padded_erosion(self):
        mask = np.ones((7, 7), bool)
        eroded = compare.erode(mask, 2)
        self.assertEqual(int(eroded.sum()), 9)
        self.assertTrue(eroded[2:5, 2:5].all())
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            with np.load(args[1] / 'attributes.npz') as raw:
                values = {name: raw[name].copy() for name in raw.files}
            values['mask'][100:400, 100:400] = 0.8
            np.savez(args[1] / 'attributes.npz', **values)
            result = compare.compare(*args)
            self.assertEqual(result['coverage']['common_roi_pixels'], 0)
            self.assertEqual(result['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
