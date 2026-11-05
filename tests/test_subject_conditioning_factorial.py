import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.subject_conditioning_factorial_v1 import prepare as experiment
from experiments.subject_trellis_counterfactual_v1.run import check_source_receipt, fixed_crop_condition


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
            rows, crop = experiment.build_variants(*self.inputs())
        return {row['name']: row for row in rows}, crop

    def test_eight_cells_keep_independent_factors_and_scope_exterior(self):
        rgb, sam, soft = self.inputs()
        rows, crop = self.build()
        self.assertEqual(len(rows), 8)
        self.assertEqual(crop, [144, 253, 440, 549])
        for row in rows.values():
            geometry = sam if row['geometry_alpha'] == 'sam' else soft
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
        rows, crop = self.build()
        for alpha in ('sam', 'biref'):
            images = [fixed_crop_condition(Image.fromarray(rows[f'original__{scope}__{alpha}']['rgba']), crop)
                      for scope in ('sam', 'biref')]
            self.assertEqual(images[0].tobytes(), images[1].tobytes())
        self.assertFalse(np.array_equal(rows['pink__sam__biref']['rgba'], rows['pink__biref__biref']['rgba']))

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
