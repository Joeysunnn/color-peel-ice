"""Reject wrong surface/normalization/data when reusing the accepted Mailbox T16."""

import copy
import unittest

from experiments.paper_mailbox_legacy_texture_v1 import run


class LegacyTextureTests(unittest.TestCase):
    def fixture(self):
        files = {name: {'path': '/legacy/' + name, 'sha256': digest}
                 for name, digest in [('input.png', run.INPUT_SHA), ('mask.png', run.MASK_SHA),
                                      ('texture.png', run.TEXTURE_SHA)]}
        extraction = {'selected_region': 'small_painted_lid', 'files': files,
                      'texture_sampling': {'crop_xyxy': [335, 399, 387, 412], 'no_inpainting': True},
                      'texture_statistics': {'method': 'linear_srgb_rec709_grayscale_robust_quantiles',
                                             'quantiles': [0.02, 0.98], 'output_range': [0.25, 0.75]}}
        locks = {r['path']: r['sha256'] for r in files.values()}
        rows = []
        for i in range(16):
            row = {'id': f'legacy{i}', 'source_id': 'mailbox', 'branch': 'texture',
                   'concept_id': 'mailbox_texture', 'prompt': 'a photo of a cube with <T*> texture'}
            for field in ('image', 'mask'):
                row[field] = f'/legacy/{field}{i}.png'
                row[field + '_sha256'] = str(i).zfill(64)
                locks[row[field]] = row[field + '_sha256']
            rows.append(row)
        return extraction, rows, locks

    def test_accepts_frozen_data_without_rewriting_rows(self):
        extraction, rows, locks = self.fixture()
        before = copy.deepcopy((extraction, rows, locks))
        run.validate_legacy(extraction, rows, locks)
        self.assertEqual((extraction, rows, locks), before)
        for digest in (run.INPUT_SHA, run.MASK_SHA, run.TEXTURE_SHA):
            self.assertEqual(len(digest), 64)

    def test_rejects_whole_object_or_new_backplate_roi(self):
        for roi in ([179, 419, 367, 507], [225, 350, 295, 390]):
            extraction, rows, locks = self.fixture()
            extraction['texture_sampling']['crop_xyxy'] = roi
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                run.validate_legacy(extraction, rows, locks)

    def test_rejects_new_gain1_method_or_changed_normalization(self):
        for key, value in [('method', 'explicit_surface_roi_linear_luminance_unit_gain'),
                           ('quantiles', [0.05, 0.95]), ('output_range', [0, 1])]:
            extraction, rows, locks = self.fixture()
            extraction['texture_statistics'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                run.validate_legacy(extraction, rows, locks)

    def test_rejects_other_concept_missing_image_lock_and_duplicate_rows(self):
        for kind in ('other_object', 'missing_lock', 'duplicate'):
            extraction, rows, locks = self.fixture()
            if kind == 'other_object':
                rows[0]['source_id'] = 'wood_spoon'
            elif kind == 'missing_lock':
                del locks[rows[0]['image']]
            else:
                rows[-1] = rows[0]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                run.validate_legacy(extraction, rows, locks)


if __name__ == '__main__':
    unittest.main()
