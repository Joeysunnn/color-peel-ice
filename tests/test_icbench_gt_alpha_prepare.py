import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.icbench_gt_alpha_four_v1 import prepare as prep
from experiments.subject_trellis_counterfactual_v1.run import check_source_receipt
from src.methods.colorpeel_ice.single_image_subject import prepare_inputs


class GTAlphaPrepareTests(unittest.TestCase):
    def setUp(self):
        git = patch.object(prep, 'git_info', return_value={'commit': 'test', 'branch': 'test'})
        git.start()
        self.addCleanup(git.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source = self.root / 'source'
        source.mkdir()
        self.rgb = np.full((512, 512, 3), [40, 75, 100], dtype=np.uint8)
        self.mask = np.zeros((512, 512), dtype=np.uint8)
        self.mask[420:512, :121] = 255
        self.rgb[self.mask > 0] = [180, 100, 40]
        self.rgb[450:452, 30:32] = 255
        Image.fromarray(self.rgb).save(source / 'image.png')
        Image.fromarray(self.mask).save(source / 'gt.png')
        self.parent = prepare_inputs(source / 'image.png', source / 'gt.png', self.root / 'semantic',
            segmentation='supplied ICBench GT binary mask')
        self.receipt = self.root / 'semantic/source_receipt.json'

    @staticmethod
    def read(path):
        with Image.open(path) as image:
            return np.array(image)

    @staticmethod
    def segmenter(image):
        alpha = np.full((image.height, image.width), 254, dtype=np.uint8)
        alpha[::5, ::7] = 0
        alpha[::3, ::4] = 128
        return Image.fromarray(np.dstack([np.asarray(image), alpha]))

    def test_hard_preserves_binary_gt_and_never_loads_biref(self):
        before = {p: Path(p).read_bytes() for p in self.parent['source_locks']}
        with patch.object(prep.subject_alpha, 'load_segmenter', side_effect=AssertionError('must not load')):
            result = prep.prepare(self.receipt, self.root / 'hard')
        self.assertEqual([x['name'] for x in result['variants']], ['original__hard', 'pink__hard'])
        for variant in result['variants']:
            receipt = json.loads(Path(variant['source_receipt']).read_text())
            np.testing.assert_array_equal(self.read(receipt['conditioning_alpha_path']), self.mask)
            self.assertEqual(receipt['segmentation'], self.parent['segmentation'])
            self.assertEqual(receipt['image512_path'], self.parent['image512_path'])
            self.assertIn(str(self.receipt), receipt['source_locks'])
            check_source_receipt(variant['source_receipt'])
        np.testing.assert_array_equal(self.read(result['variants'][0]['pre_alpha_rgb_path']), self.rgb)
        pink = self.read(result['variants'][1]['pre_alpha_rgb_path'])
        np.testing.assert_array_equal(pink[self.mask == 0], self.rgb[self.mask == 0])
        np.testing.assert_array_equal(pink[450:452, 30:32], self.rgb[450:452, 30:32])
        self.assertTrue(np.any(pink[self.mask > 0] != self.rgb[self.mask > 0]))
        self.assertEqual({p: Path(p).read_bytes() for p in before}, before)

    def test_soft_crop_clips_gt_preserves_values_and_rgb_matches_hard(self):
        hard = prep.prepare(self.receipt, self.root / 'hard')
        calls = []

        def predict(image):
            calls.append(np.array(image))
            return self.segmenter(image)

        soft = prep.prepare(self.receipt, self.root / 'soft', soft=True, segmenter=predict)
        self.assertEqual(soft['conditioning_alpha']['crop_xyxy_exclusive'], [0, 407, 134, 512])
        self.assertEqual(soft['conditioning_alpha']['padding_pixels'], 13)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0], self.rgb[407:512, :134])
        raw = self.read(self.root / 'soft/biref_raw_alpha512.png')
        actual = self.read(self.root / 'soft/gt_clipped_alpha512.png')
        self.assertTrue((actual[self.mask == 0] == 0).all())
        np.testing.assert_array_equal(actual[self.mask > 0], raw[self.mask > 0])
        self.assertTrue(((actual > 0) & (actual < 255)).any())
        for h, s in zip(hard['variants'], soft['variants']):
            np.testing.assert_array_equal(self.read(h['pre_alpha_rgb_path']), self.read(s['pre_alpha_rgb_path']))
            receipt = json.loads(Path(s['source_receipt']).read_text())
            cutout = self.read(receipt['cutout_path'])
            np.testing.assert_array_equal(cutout[..., 3], actual)
            np.testing.assert_array_equal(cutout[actual > 0, :3], self.read(s['pre_alpha_rgb_path'])[actual > 0])
            self.assertTrue((cutout[actual == 0, :3] == 0).all())
            check_source_receipt(s['source_receipt'])

    def test_crop_uses_half_open_bbox_ceil_and_clips_all_canvas_edges(self):
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[1:102, 410:512] = 255
        self.assertEqual(prep.padded_gt_crop(mask), ([399, 0, 512, 113], 11))

    def test_invalid_predictions_keep_raw_failure_evidence_and_no_variant_receipts(self):
        cases = [(0, 'is empty'), (128, 'no values > 204'), (255, 'no soft values')]
        for value, message in cases:
            def predict(image, value=value):
                return Image.fromarray(np.dstack([np.asarray(image),
                    np.full((image.height, image.width), value, dtype=np.uint8)]))
            out = self.root / f'failure_{value}'
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                prep.prepare(self.receipt, out, soft=True, segmenter=predict)
            self.assertTrue((out / 'biref_raw_alpha_crop.png').is_file())
            self.assertTrue((out / 'biref_raw_alpha512.png').is_file())
            state = json.loads((out / 'status.json').read_text())
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['variants'], [])
            self.assertFalse(list(out.glob('*/source_receipt.json')))

    def test_changed_prediction_rgb_rejected_with_raw_alpha_preserved(self):
        def predict(image):
            rgba = np.array(self.segmenter(image))
            rgba[0, 0, 0] ^= 1
            return Image.fromarray(rgba)
        out = self.root / 'rgb_changed'
        with self.assertRaisesRegex(ValueError, 'changed source crop RGB'):
            prep.prepare(self.receipt, out, soft=True, segmenter=predict)
        self.assertTrue((out / 'biref_raw_alpha_crop.png').is_file())

    def test_fresh_external_outputs_and_parent_lock_are_enforced(self):
        with self.assertRaisesRegex(ValueError, 'outside all source'):
            prep.prepare(self.receipt, self.receipt.parent / 'derived')
        out = self.root / 'hard'
        result = prep.prepare(self.receipt, out)
        with self.assertRaisesRegex(ValueError, 'fresh'):
            prep.prepare(self.receipt, out)
        self.receipt.write_text(self.receipt.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'source lock hash mismatch'):
            check_source_receipt(result['variants'][0]['source_receipt'])


if __name__ == '__main__':
    unittest.main()
