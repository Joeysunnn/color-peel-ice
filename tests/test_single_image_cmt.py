import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.methods.colorpeel_ice import single_image_cmt as cmt
from src.methods.colorpeel_ice.single_image_subject import prepare_inputs as shared_inputs


class SingleImageCmtTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def inputs(self):
        image = self.root / 'source.png'
        mask = self.root / 'mask.png'
        Image.new('RGB', (512, 512), (190, 90, 40)).save(image)
        binary = np.zeros((512, 512), dtype=np.uint8)
        binary[64:448, 64:448] = 255
        Image.fromarray(binary).save(mask)
        shared_inputs(image, mask, self.root / 'shared')
        cmt.prepare_inputs(self.root / 'shared', self.root / 'inputs')
        return self.root / 'inputs'

    def maps(self, inputs, *, wrong_source=False):
        maps = self.root / 'supermat/input'
        maps.mkdir(parents=True)
        Image.new('RGB', (512, 512), (128, 128, 128)).save(maps / 'albedo.png')
        for channel, value in (('roughness', 100), ('metallic', 30)):
            Image.new('L', (512, 512), value).save(maps / (channel + '.png'))
        source = inputs / 'supermat_input/input.png'
        cmt.write(maps.parent / 'inference_manifest.json', {'status': 'complete', 'outputs': [{
            'input': str(source), 'input_sha256': 'wrong' if wrong_source else cmt.sha(source),
            'maps': {channel: {'path': str(maps / (channel + '.png')),
                              'sha256': cmt.sha(maps / (channel + '.png'))}
                     for channel in ('albedo', 'roughness', 'metallic')}}]})
        return maps

    def assets(self):
        from scripts.methods.colorpeel_ice.run_d1_gamut_aware_lch_injection import ASSETS
        assets = self.root / 'assets'
        for relative in ASSETS.values():
            path = assets / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'unit test asset, not a renderer input')
        return assets

    def test_automatic_roi_erodes_and_supermat_receipt_locks_instance(self):
        inputs = self.inputs()
        receipt = cmt.read(inputs / 'input_receipt.json')
        with Image.open(receipt['region']) as image:
            region = np.asarray(image)
        self.assertEqual(int((region == 255).sum()), 374 ** 2)
        self.assertFalse(receipt['region_reviewed_by_user'])
        self.assertEqual(cmt.sha(receipt['supermat_input']), receipt['source']['cutout_sha256'])
        self.assertTrue(Path(receipt['roi_overlay']).is_file())
        self.assertEqual(receipt['overlay_use'], 'preview_only_never_model_input')
        cmt.check_locks(receipt['source_locks'])

    def test_rectangle_never_includes_outside_pixels_and_has_stable_tie_break(self):
        valid = np.zeros((12, 30), dtype=bool)
        valid[2:6, 3:9] = True
        valid[7:11, 20:26] = True
        self.assertEqual(cmt.largest_rectangle(valid), (3, 2, 9, 6))
        self.assertFalse(cmt.eroded_region(np.ones((12, 30), dtype=bool))[0].any())

    def test_color_target_ignores_other_objects_and_background(self):
        rgb = np.full((50, 70, 3), (240, 30, 40), dtype=np.uint8)
        valid = np.zeros((50, 70), dtype=bool)
        valid[10:40, 20:50] = True
        target = cmt.color_target(rgb, valid)
        rgb[~valid] = (20, 60, 250)
        self.assertEqual(cmt.color_target(rgb, valid), target)
        self.assertGreater(target['L_amplitude'], 0)
        self.assertFalse(target['chroma_reduction'])

    def test_color_remap_preserves_background_and_uses_measured_ab(self):
        from src.methods.colorpeel_ice import neutral_lch_injection as color
        levels = np.linspace(50, 220, 64).astype(np.uint8)
        rgb = np.repeat(np.broadcast_to(levels[None, :, None], (64, 64, 1)), 3, axis=2).copy()
        mask = np.zeros((64, 64), dtype=bool)
        mask[8:56, 8:56] = True
        target = cmt.color_target(np.full_like(rgb, (150, 100, 50)), mask)
        result, metrics = cmt.remap_color(rgb, mask, target)
        np.testing.assert_array_equal(result[~mask], rgb[~mask])
        lab = color.linear_rgb_to_lab(color._srgb_to_linear(result[mask]))
        self.assertLess(np.abs(lab[:, 1:] - [target['a'], target['b']]).max(), 1)
        self.assertEqual(metrics['outside_mask_changed_pixels'], 0)

    def test_valid_constant_albedo_remains_uniform_without_noise_amplification(self):
        gray, stats = cmt.extract_gray(Image.new('RGB', (40, 50), (120, 120, 120)),
                                       Image.new('L', (40, 50), 255))
        self.assertEqual(set(np.unique(gray)), {128})
        self.assertTrue(stats['normalization_degenerate'])
        self.assertFalse(stats['spatial_pattern_invented'])

    def test_plan_exact_counts_and_source_specific_material_uniform_texture(self):
        inputs = self.inputs()
        maps = self.maps(inputs)
        with patch.object(cmt, 'ASSET_ROOT', self.assets()), patch(
                'scripts.launch.colorpeel_run.git_info', return_value={'commit': 'test', 'branch': 'test'}):
            path = cmt.build_plan(inputs, maps, self.root / 'plan')
        plan = cmt.read(path)
        self.assertEqual(plan['failures'], {})
        for branch, count in cmt.BRANCH_COUNTS.items():
            rows = [row for row in plan['records'] if row['branch'] == branch]
            self.assertEqual(len(rows), count)
            token = {'color': '<C*>', 'material': '<M*>', 'texture': '<T*>'}[branch]
            self.assertTrue(all(row['caption'].count(token) == 1 for row in rows))
        self.assertAlmostEqual(plan['material_target']['roughness'], 100 / 255)
        self.assertAlmostEqual(plan['material_target']['metallic'], 30 / 255)
        self.assertTrue(plan['texture_extraction']['statistics']['normalization_degenerate'])
        self.assertTrue((path.parent / 'texture_roi_overlay.png').is_file())
        cmt.check_locks(plan['source_locks'])

    def test_supermat_output_from_another_instance_is_rejected(self):
        inputs = self.inputs()
        maps = self.maps(inputs, wrong_source=True)
        with patch('scripts.launch.colorpeel_run.git_info', return_value={}):
            with self.assertRaisesRegex(ValueError, 'this instance'):
                cmt.build_plan(inputs, maps, self.root / 'plan', branches=['material'])

    def test_color_only_plan_requires_no_supermat_files(self):
        inputs = self.inputs()
        with patch.object(cmt, 'ASSET_ROOT', self.assets()), patch(
                'scripts.launch.colorpeel_run.git_info', return_value={'commit': 'test', 'branch': 'test'}):
            path = cmt.build_plan(inputs, None, self.root / 'plan', branches=['color'])
        plan = cmt.read(path)
        self.assertEqual(plan['branches'], ['color'])
        self.assertEqual(len(plan['records']), 9)
        self.assertNotIn('material_target', plan)
        self.assertNotIn('texture_extraction', plan)


if __name__ == '__main__':
    unittest.main()
