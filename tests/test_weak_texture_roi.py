import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.methods.colorpeel_ice import single_image_cmt as cmt
from tests import test_single_image_cmt as fixtures


class WeakTextureRoiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    inputs = fixtures.SingleImageCmtTests.inputs
    maps = fixtures.SingleImageCmtTests.maps
    assets = fixtures.SingleImageCmtTests.assets

    def test_constant_surface_stays_uniform_without_invented_pattern(self):
        result, stats = cmt.extract_surface_patch(
            Image.new('RGB', (20, 15), (190, 90, 40)), Image.new('L', (20, 15), 255), [3, 4, 13, 11])
        self.assertEqual(result.mode, 'L')
        self.assertEqual(result.size, (10, 7))
        self.assertEqual(set(np.unique(result)), {128})
        self.assertEqual(stats['contrast_gain'], 1.0)
        self.assertEqual(stats['normalization'], 'median_center_only')
        self.assertEqual(stats['clipped_pixels'], 0)
        self.assertEqual(stats['output_uint8_range'], [128, 128])

    def test_weak_variation_keeps_unit_gain_instead_of_stretching_quantiles(self):
        levels = np.array([[128, 128, 129, 130], [129, 128, 130, 129]], dtype=np.uint8)
        pixels = np.repeat(levels[..., None], 3, axis=2)
        result, stats = cmt.extract_surface_patch(
            Image.fromarray(pixels), Image.new('L', (4, 2), 255), [0, 0, 4, 2])
        rgb = pixels.astype(np.float64) / 255
        linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
        luminance = linear @ np.array([.2126, .7152, .0722])
        expected = np.rint((.5 + luminance - np.median(luminance)) * 255).astype(np.uint8)
        np.testing.assert_array_equal(np.asarray(result), expected)
        self.assertGreater(int(expected.max()) - int(expected.min()), 0)
        self.assertLessEqual(int(expected.max()) - int(expected.min()), 5)
        self.assertGreater(int(expected.min()), 64)
        self.assertLess(int(expected.max()), 191)
        self.assertAlmostEqual(stats['input_median_luminance'], float(np.median(luminance)))
        self.assertEqual(stats['contrast_gain'], 1.0)

    def test_outside_highlights_and_edges_cannot_change_patch_or_statistics(self):
        pixels = np.full((10, 14, 3), 129, dtype=np.uint8)
        pixels[3:7, 4:9] = np.arange(5, dtype=np.uint8)[None, :, None] + 126
        box, mask = [4, 3, 9, 7], Image.new('L', (14, 10), 255)
        baseline, baseline_stats = cmt.extract_surface_patch(Image.fromarray(pixels), mask, box)
        changed = np.zeros_like(pixels)
        changed[:, ::2] = 255
        changed[3:7, 4:9] = pixels[3:7, 4:9]
        result, stats = cmt.extract_surface_patch(Image.fromarray(changed), mask, box)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(baseline))
        self.assertEqual(stats, baseline_stats)

    def test_invalid_or_noninterior_regions_are_rejected(self):
        image, full = Image.new('RGB', (12, 10), 128), Image.new('L', (12, 10), 255)
        for box in ([True, 2, 6, 5], [-1, 2, 6, 5], [2, 2, 13, 5], [2, 2, 6, 11],
                    [2, 2, 2, 5], [6, 2, 2, 5], [2, 2, 3, 3], [2, 2, 6], [2., 2, 6, 5]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                cmt.extract_surface_patch(image, full, box)
        partial = np.full((10, 12), 255, dtype=np.uint8)
        partial[3, 4] = 0
        for mask in (Image.new('L', (12, 10), 0), Image.fromarray(partial), Image.new('L', (11, 10), 255)):
            with self.subTest(size=mask.size), self.assertRaises(ValueError):
                cmt.extract_surface_patch(image, mask, [2, 2, 6, 5])

    def test_out_of_range_centered_luminance_is_rejected_instead_of_clipped(self):
        for majority, outlier in ((0, 255), (255, 0)):
            pixels = np.full((2, 3, 3), majority, dtype=np.uint8)
            pixels[0, 0] = outlier
            with self.subTest(majority=majority), self.assertRaises(ValueError):
                cmt.extract_surface_patch(Image.fromarray(pixels), Image.new('L', (3, 2), 255), [0, 0, 3, 2])

    def test_explicit_roi_flows_into_all_texture_rows_with_locked_source_crops(self):
        inputs = self.inputs()
        maps = self.maps(inputs)
        box = [100, 100, 140, 130]
        with patch.object(cmt, 'ASSET_ROOT', self.assets()), patch(
                'scripts.launch.colorpeel_run.git_info', return_value={'commit': 'test', 'branch': 'test'}):
            path = cmt.build_plan(inputs, maps, self.root / 'plan', branches=['texture'], texture_roi=box)
        plan = cmt.read(path)
        self.assertEqual(plan['failures'], {})
        self.assertEqual(plan['branches'], ['texture'])
        self.assertEqual(len(plan['records']), 16)
        self.assertEqual(plan['texture_extraction']['crop_xyxy'], box)
        self.assertEqual(plan['texture_extraction']['statistics']['normalization'], 'median_center_only')
        effective = plan['protocol']['texture_extraction']
        self.assertEqual(effective['method'], plan['texture_extraction']['statistics']['method'])
        self.assertEqual(effective['normalization'], 'median_center_only')
        self.assertEqual(effective['contrast_gain'], 1.0)
        self.assertEqual(effective['output_size'], [40, 30])
        self.assertFalse(effective['normal_bump_displacement'])
        self.assertEqual(effective['uv_repeats'], 1)
        self.assertFalse({'quantiles', 'output_range', 'size'} & set(effective))
        self.assertNotIn('material_target', plan)
        for name, original in (('texture_source_crop.png', cmt.read(inputs / 'input_receipt.json')['source']['image512_path']),
                               ('texture_albedo_crop.png', maps / 'albedo.png')):
            crop = path.parent / name
            self.assertEqual(plan['source_locks'][str(crop)], cmt.sha(crop))
            with Image.open(crop) as actual, Image.open(original) as source:
                self.assertEqual(actual.size, (40, 30))
                np.testing.assert_array_equal(np.asarray(actual.convert('RGB')), np.asarray(source.convert('RGB').crop(box)))
        texture = path.parent / 'texture.png'
        self.assertEqual({row['texture_path'] for row in plan['records']}, {str(texture)})
        self.assertEqual({row['texture_sha256'] for row in plan['records']}, {cmt.sha(texture)})
        self.assertTrue(all(row['branch'] == 'texture' and row['caption'].count('<T*>') == 1
                            for row in plan['records']))
        cmt.check_locks(plan['source_locks'])

    def test_invalid_explicit_roi_is_recorded_as_failed_without_texture_rows(self):
        inputs = self.inputs()
        maps = self.maps(inputs)
        with patch.object(cmt, 'ASSET_ROOT', self.assets()), patch(
                'scripts.launch.colorpeel_run.git_info', return_value={'commit': 'test', 'branch': 'test'}):
            path = cmt.build_plan(inputs, maps, self.root / 'invalid_plan',
                                  branches=['texture'], texture_roi=[0, 0, 10, 10])
        plan = cmt.read(path)
        self.assertEqual(plan['branches'], [])
        self.assertEqual(plan['records'], [])
        self.assertEqual(set(plan['failures']), {'texture'})
        self.assertIn('eroded object region', plan['failures']['texture'])
        self.assertFalse((path.parent / 'texture.png').exists())


if __name__ == '__main__':
    unittest.main()
