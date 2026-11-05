import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.subject_conditioning_factorial_v1 import prepare as experiment
from experiments.subject_trellis_counterfactual_v1.run import check_source_receipt


def feather_fixture(support):
    # Tests factor separation; established recolor/EDT implementation has its own tests.
    return support.astype(np.uint16) * 65535, {'backend': 'fixture_full_weight'}


class SubjectConditioningFactorialTests(unittest.TestCase):
    def inputs(self):
        soft = np.zeros((512, 512), dtype=np.uint8)
        soft[290:512, 144:441] = 254
        soft[290, 144:441] = 128  # Soft halo is below the official >204 crop threshold.
        sam = np.zeros_like(soft)
        sam[291:512, 145:440] = 255
        rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        rgb[soft > 0] = [180, 110, 40]
        return rgb, sam, soft

    def build(self):
        with patch.object(experiment, 'derive_repaired_alpha', side_effect=feather_fixture):
            rows = experiment.build_variants(*self.inputs())
        return {row['name']: row for row in rows}

    def test_eight_cells_keep_independent_factors_and_scope_exterior(self):
        rgb, sam, soft = self.inputs()
        rows = self.build()
        self.assertEqual(len(rows), 8)
        for row in rows.values():
            geometry = sam if row['geometry_alpha'] == 'sam' else soft
            expected_crop = [145, 254, 439, 548] if row['geometry_alpha'] == 'sam' else [144, 253, 440, 549]
            self.assertEqual(row['auto_crop_xyxy'], expected_crop)
            self.assertEqual(experiment.expected_auto_condition(row['rgba']).size,
                             (expected_crop[2] - expected_crop[0], expected_crop[3] - expected_crop[1]))
            np.testing.assert_array_equal(row['rgba'][..., 3], geometry)
            self.assertFalse(row['rgba'][geometry == 0, :3].any())
            if row['color'] == 'original':
                np.testing.assert_array_equal(row['pre_alpha_rgb'], rgb)
            else:
                support = sam > 0 if row['recolor_support'] == 'sam' else soft > 0
                np.testing.assert_array_equal(row['pre_alpha_rgb'][~support], rgb[~support])
                self.assertTrue(np.any(row['pre_alpha_rgb'][support] != rgb[support]))
        for color in ('original', 'pink'):
            for scope in ('sam', 'biref'):
                np.testing.assert_array_equal(rows[f'{color}__{scope}__sam']['pre_alpha_rgb'],
                                              rows[f'{color}__{scope}__biref']['pre_alpha_rgb'])

    def test_original_pairs_are_identical_conditions_but_pink_scopes_can_differ(self):
        rows = self.build()
        for alpha in ('sam', 'biref'):
            images = [experiment.expected_auto_condition(rows[f'original__{scope}__{alpha}']['rgba'])
                      for scope in ('sam', 'biref')]
            self.assertEqual(images[0].tobytes(), images[1].tobytes())
        self.assertFalse(np.array_equal(rows['pink__sam__biref']['rgba'], rows['pink__biref__biref']['rgba']))

    def test_fractional_auto_crop_preserves_official_pillow_rounding_and_threshold(self):
        rgba = np.zeros((512, 512, 4), dtype=np.uint8)
        rgba[21:26, 11:17] = [100, 150, 200, 255]
        rgba[0, 0] = [100, 150, 200, 204]  # Must not expand the crop.
        self.assertEqual(experiment.official_crop_box(rgba[..., 3]), [11.5, 21., 15.5, 25.])
        expected = Image.fromarray(rgba).crop([11.5, 21., 15.5, 25.]).convert('RGB')
        actual = experiment.expected_auto_condition(rgba)
        self.assertEqual(actual.size, expected.size)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_reject_missing_sam_coverage_in_frozen_rgb(self):
        rgb, sam, soft = self.inputs()
        soft[300, 200] = 0
        with self.assertRaisesRegex(ValueError, 'cover all SAM'):
            experiment.build_variants(rgb, sam, soft)

    def test_prepare_receipts_lock_parent_and_keep_semantic_sam(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / 'inputs'
            inputs.mkdir()
            rgb, sam, soft = self.inputs()
            paths = {name: inputs / (name + '.png') for name in ('source', 'mask', 'cutout', 'conditioning_alpha')}
            for name, values in [('source', rgb), ('mask', sam), ('cutout', np.dstack([rgb, soft])), ('conditioning_alpha', soft)]:
                Image.fromarray(values).save(paths[name])
            receipt = {'conditioning_alpha': {'method': 'pinned_BiRefNet_full_native_RGBA_then_joint_LANCZOS/v1'}}
            for name, path in paths.items():
                receipt[name + '_path'], receipt[name + '_sha256'] = str(path), experiment.sha256(path)
            parent = inputs / 'receipt.json'
            parent.write_text(json.dumps(receipt))
            with patch.object(experiment, 'derive_repaired_alpha', side_effect=feather_fixture), \
                    patch.object(experiment, 'git_info', return_value={'commit': 'fixture'}):
                manifest = experiment.prepare(parent, root / 'prepared')
                with self.assertRaisesRegex(ValueError, 'fresh'):
                    experiment.prepare(parent, root / 'prepared')
            self.assertEqual(len(manifest['variants']), 8)
            for row in manifest['variants']:
                checked = check_source_receipt(row['source_receipt'])
                self.assertIn(str(parent), checked['sha256'])
                self.assertEqual(checked['mask_path'], str(paths['mask']))
            parent.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'source lock hash mismatch'):
                check_source_receipt(manifest['variants'][0]['source_receipt'])


if __name__ == '__main__':
    unittest.main()
