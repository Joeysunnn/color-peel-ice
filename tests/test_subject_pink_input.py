import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.subject_pink_input_repro_v1.prepare import prepare
from experiments.subject_pink_input_repro_v1 import prepare as pink
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

    def test_exact_white_is_preserved_without_removing_sam_foreground(self):
        sources = self.root / 'white_sources'
        sources.mkdir()
        for all_white in (False, True):
            with self.subTest(all_white=all_white):
                rgb = self.rgb.copy()
                rgb[self.mask > 0] = 255 if all_white else [180, 100, 40]
                rgb[400:410, 150:160] = 255
                image_path, mask_path = sources / f'{all_white}.png', sources / f'{all_white}_mask.png'
                Image.fromarray(rgb).save(image_path)
                Image.fromarray(self.mask).save(mask_path)
                folder = self.root / f'white_{all_white}'
                subject.prepare_inputs(image_path, mask_path, folder / 'input')
                receipt = prepare(folder / 'input/source_receipt.json', folder / 'pink')
                with Image.open(receipt['cutout_path']) as image:
                    actual = np.asarray(image)
                white = (self.mask > 0) & np.all(rgb == 255, axis=-1)
                np.testing.assert_array_equal(actual[..., 3], self.mask)
                np.testing.assert_array_equal(actual[white, :3], rgb[white])
                self.assertTrue((actual[self.mask == 0, :3] == 0).all())
                metrics = receipt['conditioning_recolor']['recolor_metrics']
                self.assertEqual(metrics['pure_white_support_pixels_preserved'], int(white.sum()))
                if all_white:
                    self.assertEqual(metrics['status'], 'unchanged_all_foreground_exact_white')
                else:
                    self.assertEqual(metrics['status'], 'recolored')
                    editable = (self.mask > 0) & ~white
                    self.assertTrue(np.any(actual[editable, :3] != rgb[editable]))
                check_source_receipt(folder / 'pink/source_receipt.json')

    def test_missing_current_mask_has_no_historical_fallback(self):
        Path(self.original['mask_path']).unlink()
        with self.assertRaises(FileNotFoundError):
            prepare(self.parent, self.output)
        self.assertFalse(self.output.exists())

    def matting_fixture(self, name='matting'):
        folder = self.root / name
        folder.mkdir()
        alpha = np.where(self.mask > 0, 254, 0).astype(np.uint8)
        alpha[299, 150:170] = 1  # Soft support can extend beyond the semantic mask.
        alpha[300:303, 150:170] = 64
        alpha[303:306, 150:170] = 128
        alpha[400:410, 150:160] = 255
        rgb = np.full(self.rgb.shape, [73, 140, 190], dtype=np.uint8)
        rgb[400:410, 150:160] = 255
        rgba = np.dstack([np.where(alpha[..., None] > 0, rgb, 0), alpha])
        cutout, alpha_path = folder / 'cutout512.png', folder / 'conditioning_alpha512.png'
        Image.fromarray(rgba).save(cutout)
        Image.fromarray(alpha).save(alpha_path)
        record = {**self.original, 'cutout_path': str(cutout), 'cutout_sha256': subject.sha256(cutout),
            'conditioning_alpha_path': str(alpha_path), 'conditioning_alpha_sha256': subject.sha256(alpha_path),
            'semantic_receipt_path': str(self.parent), 'semantic_receipt_sha256': subject.sha256(self.parent),
            'source_locks': {**self.original['source_locks'], str(self.parent): subject.sha256(self.parent),
                str(cutout): subject.sha256(cutout), str(alpha_path): subject.sha256(alpha_path)}}
        receipt = folder / 'source_receipt.json'
        subject.write_json(receipt, record)
        return receipt, record, rgba

    def test_soft_alpha_pink_uses_matting_rgb_and_preserves_semantic_inputs(self):
        parent, original, rgba = self.matting_fixture()
        with patch.object(pink, 'recolor', wraps=pink.recolor) as recolor:
            receipt = prepare(parent, self.output)
        np.testing.assert_array_equal(recolor.call_args.args[0], rgba[..., :3])
        actual = np.array(Image.open(receipt['cutout_path']))
        np.testing.assert_array_equal(actual[..., 3], rgba[..., 3])
        np.testing.assert_array_equal(actual[400:410, 150:160, :3], rgba[400:410, 150:160, :3])
        self.assertTrue((actual[rgba[..., 3] == 0, :3] == 0).all())
        self.assertTrue(np.any(actual[rgba[..., 3] > 0, :3] != rgba[rgba[..., 3] > 0, :3]))
        for key in ('mask_path', 'mask_sha256', 'image512_path', 'image512_sha256',
                    'original_mask_path', 'original_mask_sha256', 'conditioning_alpha_path',
                    'conditioning_alpha_sha256', 'semantic_receipt_path', 'semantic_receipt_sha256'):
            self.assertEqual(receipt[key], original[key])
        self.assertIn('conditioning_alpha>0', receipt['conditioning_recolor']['method'])
        checked = check_source_receipt(self.output / 'source_receipt.json')
        self.assertFalse(receipt['alpha_matches_mask'])
        self.assertTrue(receipt['alpha_matches_conditioning_alpha'])
        self.assertFalse(checked['alpha_matches_mask'])
        self.assertTrue(checked['alpha_matches_conditioning_alpha'])
        self.assertEqual(checked['conditioning_alpha_path'], original['conditioning_alpha_path'])
        rows = subject.photo_rows(receipt, self.root / 'photos', 'test object')
        np.testing.assert_array_equal(np.array(Image.open(rows[0]['image'])), self.rgb)
        for row in rows:
            np.testing.assert_array_equal(np.array(Image.open(row['mask'])), self.mask)
        self.parent.write_bytes(b'semantic receipt changed')
        with self.assertRaisesRegex(ValueError, 'source lock hash mismatch'):
            check_source_receipt(self.output / 'source_receipt.json')

    def test_conditioning_alpha_requires_coupled_fields_and_correct_hash(self):
        parent, record, _ = self.matting_fixture()
        for field in ('conditioning_alpha_path', 'conditioning_alpha_sha256'):
            broken = dict(record)
            del broken[field]
            subject.write_json(parent, broken)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'both path and sha256'):
                check_source_receipt(parent)
        record['conditioning_alpha_sha256'] = '0' * 64
        subject.write_json(parent, record)
        with self.assertRaisesRegex(ValueError, 'input hash mismatch'):
            check_source_receipt(parent)

    def test_conditioning_alpha_mode_size_semantic_mask_and_cutout_are_checked(self):
        for case in ('mode', 'size', 'semantic', 'cutout'):
            parent, record, rgba = self.matting_fixture(case)
            if case in ('mode', 'size'):
                alpha = Image.fromarray(rgba[..., 3])
                alpha = alpha.convert('RGB') if case == 'mode' else alpha.resize((256, 256))
                path = Path(record['conditioning_alpha_path'])
                alpha.save(path)
                record['conditioning_alpha_sha256'] = subject.sha256(path)
                message = 'L512'
            elif case == 'semantic':
                path = parent.parent / 'nonbinary_mask.png'
                Image.fromarray(rgba[..., 3]).save(path)
                record['mask_path'], record['mask_sha256'] = str(path), subject.sha256(path)
                message = 'binary semantic mask'
            else:
                path = Path(record['cutout_path'])
                rgba[350, 200, 3] = 128
                Image.fromarray(rgba).save(path)
                record['cutout_sha256'] = subject.sha256(path)
                message = 'differs from conditioning alpha'
            record['source_locks'][str(path)] = subject.sha256(path)
            subject.write_json(parent, record)
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, message):
                check_source_receipt(parent)

    def test_legacy_soft_mask_receipts_remain_valid_without_new_fields(self):
        parent, record, _ = self.matting_fixture()
        record['mask_path'] = record.pop('conditioning_alpha_path')
        record['mask_sha256'] = record.pop('conditioning_alpha_sha256')
        subject.write_json(parent, record)
        checked = check_source_receipt(parent)
        self.assertTrue(checked['alpha_matches_mask'])
        self.assertNotIn('alpha_matches_conditioning_alpha', checked)

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
