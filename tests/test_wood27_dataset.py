"""Verify the exact24-render factorial and photograph-only LCh interventions."""

from itertools import product
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.subject_trellis_wood27_v1 import prepare
from src.methods.colorpeel_ice import gamut_aware_lch_injection as gamut


class Wood27DatasetTest(unittest.TestCase):
    def rows72(self):
        return [{'id': f'{color}_{metallic}_{roughness}_{view}_{light}', 'color': color,
                 'controls': {'metallic': metallic, 'roughness': roughness}, 'view': view, 'light': light}
                for color, metallic, roughness, view, light in product(
                    ('blue', 'green'), (0, 1), (.15, .50, .85), ('left', 'center', 'right'), ('studio', 'courtyard'))]

    def test_select24_is_complete_endpoint_studio_factorial_and_retains_source_order(self):
        source = self.rows72()
        selected = prepare.select24(source)
        expected = {(color, metallic, roughness, view, 'studio')
                    for color, metallic, roughness, view in product(
                        ('blue', 'green'), (0, 1), (.15, .85), ('left', 'center', 'right'))}
        actual = {(r['color'], r['controls']['metallic'], r['controls']['roughness'], r['view'], r['light'])
                  for r in selected}
        self.assertEqual(len(selected), 24)
        self.assertEqual(actual, expected)
        self.assertTrue(all(r['light'] != 'courtyard' and r['controls']['roughness'] != .50 for r in selected))
        self.assertEqual([source.index(row) for row in selected], sorted(source.index(row) for row in selected))

    def test_selection_rejects_missing_duplicate_cell_and_duplicate_id(self):
        source = self.rows72()
        selected = prepare.select24(source)
        missing = [r for r in source if r['id'] != selected[0]['id']]
        duplicate_cell = [dict(r) for r in selected]
        duplicate_cell[1] = {**duplicate_cell[0], 'id': 'another-id-same-cell'}
        duplicate_id = [dict(r) for r in selected]
        duplicate_id[1]['id'] = duplicate_id[0]['id']
        for rows in (missing, source + [selected[0]], duplicate_cell, duplicate_id):
            with self.subTest(rows=len(rows)):
                with self.assertRaises(ValueError):
                    prepare.select24(rows)

    def photo(self):
        rgb = np.full((12, 12, 3), [160, 115, 70], dtype=np.uint8)
        rgb[0, 0] = 255
        rgb[1, 1] = [30, 80, 140]
        alpha = np.full((12, 12), 255, dtype=np.uint8)
        alpha[1, 1] = 0
        alpha[2, 2] = 127
        alpha[3, 3] = 128
        return rgb, alpha

    def test_recolor_preserves_neutral_white_background_and_uint16_alpha(self):
        rgb, alpha = self.photo()
        before_rgb, before_alpha = rgb.copy(), alpha.copy()
        for color, hue in prepare.HUES.items():
            with self.subTest(color=color):
                with patch.object(prepare, 'recolor', wraps=prepare.recolor) as core:
                    changed, metrics, maps, editable = prepare.recolor_photo(rgb, alpha, hue)
                actual_mask, alpha16 = core.call_args.args[1:3]
                self.assertEqual(alpha16.dtype, np.uint16)
                self.assertEqual(alpha16[2, 2], 127 * 257)
                self.assertEqual(alpha16[3, 3], 128 * 257)
                self.assertEqual(alpha16[4, 4], 65535)
                self.assertEqual(alpha16[0, 0], 0)
                self.assertEqual(alpha16[1, 1], 0)
                self.assertEqual(set(np.unique(actual_mask)), {0, 255})
                self.assertTrue(np.array_equal(editable, actual_mask))
                self.assertTrue(np.array_equal(changed[0, 0], rgb[0, 0]))
                self.assertTrue(np.array_equal(changed[alpha == 0], rgb[alpha == 0]))
                self.assertEqual(metrics['pure_white_support_pixels_preserved'], 1)
                self.assertEqual(metrics['source_alpha_support_pixels'], int((alpha > 0).sum()))
                self.assertEqual(metrics['outside_mask_changed_pixel_count'], 0)
                self.assertEqual(metrics['rgb_clipping_pixel_count'], 0)
                opaque = (alpha == 255) & (editable == 255)
                old_lab = gamut.linear_rgb_to_lab(gamut._srgb_to_linear(rgb[opaque]))
                new_lab = gamut.linear_rgb_to_lab(gamut._srgb_to_linear(changed[opaque]))
                self.assertLess(float(np.max(np.abs(new_lab[:, 0] - old_lab[:, 0]))), 1.0)
                angles = np.degrees(np.arctan2(new_lab[:, 2], new_lab[:, 1])) % 360
                errors = np.abs((angles - hue + 180) % 360 - 180)
                self.assertLess(float(np.median(errors)), 3.0)
                self.assertTrue(np.all(maps['C_final'][editable == 255] <= maps['source_C'][editable == 255] + 1e-5))
        self.assertTrue(np.array_equal(rgb, before_rgb))
        self.assertTrue(np.array_equal(alpha, before_alpha))

    def test_photographic_training_mask_is_independent_of_editing_support(self):
        rgb, alpha = self.photo()
        object_mask = np.where(alpha >= 128, 255, 0).astype(np.uint8)
        for hue in prepare.HUES.values():
            _, _, _, editing_mask = prepare.recolor_photo(rgb, alpha, hue)
            self.assertEqual(object_mask[0, 0], 255)  # Pure white remains a training occupancy pixel.
            self.assertEqual(editing_mask[0, 0], 0)
            self.assertEqual(object_mask[2, 2], 0)  # A soft recoloring pixel need not be a training pixel.
            self.assertEqual(editing_mask[2, 2], 255)
            self.assertTrue(np.array_equal(object_mask, np.where(alpha >= 128, 255, 0).astype(np.uint8)))

    def test_stage_row_byte_copies_render_and_preserves_single_subject_caption(self):
        with tempfile.TemporaryDirectory(prefix='wood27-copy-', dir=os.environ.get('TMP')) as temporary:
            root = Path(temporary).resolve()
            image_source, mask_source = root / 'render.png', root / 'mask.png'
            Image.new('RGB', (512, 512), (30, 80, 140)).save(image_source)
            mask = np.zeros((512, 512), dtype=np.uint8)
            mask[80:450, 110:420] = 255
            Image.fromarray(mask).save(mask_source)
            row = {**prepare.select24(self.rows72())[0], 'caption': 'a photo of <S*> spoon in blue color',
                   'image_sha256': prepare.sha256(image_source), 'mask_sha256': prepare.sha256(mask_source)}
            result = prepare.stage_row(row, root / 'output', image_source=image_source, mask_source=mask_source)
            self.assertEqual(result['caption'], row['caption'])
            self.assertEqual(result['caption'].count('<S*>'), 1)
            self.assertFalse(result['training_eligible'])
            for field, source in (('image', image_source), ('mask', mask_source)):
                target = Path(result[field])
                self.assertEqual(target.read_bytes(), source.read_bytes())
                self.assertEqual(result[field + '_sha256'], row[field + '_sha256'])
                self.assertEqual(set(target.parent.iterdir()), {target})
            with self.assertRaisesRegex(ValueError, 'exactly one Subject'):
                prepare.stage_row({**row, 'id': 'bad-caption', 'caption': 'a photo of a spoon'},
                    root / 'output', image_source=image_source, mask_source=mask_source)
            image_source.write_bytes(b'changed render')
            with self.assertRaisesRegex(ValueError, 'copy differs'):
                prepare.stage_row({**row, 'id': 'bad-source'}, root / 'output',
                    image_source=image_source, mask_source=mask_source)


if __name__ == '__main__':
    unittest.main()
