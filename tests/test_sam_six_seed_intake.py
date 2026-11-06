from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.sam_six_seed_v1 import intake
from experiments.subject_trellis_counterfactual_v1.run import check_source_receipt


class SAMSixIntakeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.revision = {'commit': 'test', 'branch': 'test'}
        git = patch.object(intake, 'git_info', return_value=self.revision)
        git.start()
        self.addCleanup(git.stop)
        self.selection = {'schema': 'icbench_gt36_selection/v1', 'objects': []}
        fixed = [(0, 160), (1, 8), (3, 207), (9, 205), (46, 97), (81, 130)]
        others = [number for number in range(40) if number not in {sample for sample, _ in fixed}][:23]
        for sample, mask in fixed + [(sample, 1) for sample in others] + [(sample, 2) for sample in others[:7]]:
            folder = self.root / 'sources' / str(sample)
            folder.mkdir(parents=True, exist_ok=True)
            image_path, mask_path = folder / 'img.jpg', folder / f'{mask}.png'
            if not image_path.exists():
                Image.new('RGB', (512, 512), (180, 100, 40)).save(image_path)
            gt = np.zeros((512, 512), dtype=np.uint8)
            gt[100:150, 200:250] = 255
            Image.fromarray(gt).save(mask_path)
            self.selection['objects'].append({'sample_id': sample, 'mask_id': mask,
                'object_key': f'image_{sample:03d}__mask_{mask:03d}', 'stable_id': f'D1GT:{sample}/{mask}.png',
                'image': {'path': str(image_path), 'sha256': intake.sha256(image_path)},
                'mask': {'path': str(mask_path), 'sha256': intake.sha256(mask_path)}})
        self.selection_path = self.root / 'selection.json'
        intake.write_json(self.selection_path, self.selection)

    @staticmethod
    def predict(image, box):
        masks = np.zeros((3, image.height, image.width), dtype=bool)
        masks[0, 100:150, 200:250] = True  # Exact GT, but lower SAM predicted score.
        masks[1, 80:170, 180:270] = True  # Deliberately extends beyond prompt/GT.
        masks[2, 110:130, 210:230] = True
        return masks, np.array([0.8, 0.95, 0.7], dtype=np.float32)

    def test_single_model_load_predicted_score_selection_and_unclipped_source_receipt(self):
        calls = []

        def predict(image, box):
            calls.append((np.asarray(image).copy(), box))
            return self.predict(image, box)

        with patch.object(intake, 'BoxSAM', return_value=predict) as factory:
            state = intake.run(self.selection_path, self.root / 'output')
        factory.assert_called_once_with()
        self.assertEqual(state['status'], 'prepared')
        self.assertEqual(len(calls), 6)
        self.assertEqual([row['object_key'] for row in state['objects']], list(intake.OBJECT_KEYS))
        for call, row in zip(calls, state['objects']):
            self.assertEqual(call[1], [200, 100, 250, 150])
            folder = self.root / 'output' / row['object_key']
            report = intake.read(folder / 'segmentation.json')
            self.assertEqual(report['selected_index'], 1)
            self.assertEqual(len(report['candidates']), 3)
            with np.load(folder / 'sam_outputs.npz', allow_pickle=False) as raw:
                np.testing.assert_array_equal(raw['masks'][1], self.predict(Image.fromarray(call[0]), call[1])[0][1])
                self.assertEqual(raw['predicted_iou_scores'].argmax(), 1)
            receipt = intake.read(row['source_receipt'])
            self.assertIn('GT-box-prompted SAM', receipt['segmentation'])
            with Image.open(receipt['mask_path']) as opened:
                mask = np.asarray(opened)
            self.assertEqual(mask[80, 180], 255)  # Outside GT and its bounding box.
            self.assertEqual(int((mask > 0).sum()), 90 * 90)
            with Image.open(receipt['image512_path']) as opened:
                np.testing.assert_array_equal(opened, call[0])
            self.assertTrue(check_source_receipt(row['source_receipt'])['alpha_matches_mask'])
            self.assertIn(str(folder / 'sam_outputs.npz'), state['source_locks'])
        intake.check_locks(state['source_locks'])

    def test_empty_gt_inference_error_and_empty_selected_mask_are_isolated_and_retained(self):
        source = self.selection['objects'][0]['mask']
        Image.new('L', (512, 512), 0).save(source['path'])
        source['sha256'] = intake.sha256(source['path'])
        intake.write_json(self.selection_path, self.selection)
        count = 0

        def predict(image, box):
            nonlocal count
            count += 1
            if count == 1:
                raise RuntimeError('injected inference failure')
            masks, scores = self.predict(image, box)
            if count == 2:
                masks[1] = False  # Still highest score; do not fall back to GT-perfect candidate0.
            return masks, scores

        output = self.root / 'output'
        state = intake.run(self.selection_path, output, segmenter=predict)
        self.assertEqual(state['status'], 'partial')
        self.assertEqual(count, 5)
        self.assertEqual([row['status'] for row in state['objects']], ['blocked'] * 3 + ['prepared'] * 3)
        for row, message in zip(state['objects'], ['empty GT', 'injected inference failure', 'selected SAM mask is empty']):
            self.assertIn(message, row['blocked_reason'])
            self.assertNotIn('source_receipt', row)
            self.assertTrue((output / row['object_key'] / 'segmentation.json').is_file())
        empty = output / intake.OBJECT_KEYS[2]
        self.assertTrue((empty / 'sam_outputs.npz').is_file())
        self.assertTrue((empty / 'sam_candidate_0.png').is_file())
        self.assertTrue((empty / 'sam_selected.png').is_file())
        self.assertFalse((empty / 'input').exists())

    def test_source_misbinding_hash_change_and_missing_target_rejected_before_model_load(self):
        for mode in ('misbound', 'hash', 'missing'):
            selection = deepcopy(self.selection)
            if mode == 'misbound':
                selection['objects'][0]['mask'] = selection['objects'][1]['mask']
            elif mode == 'hash':
                selection['objects'][0]['image']['sha256'] = '0' * 64
            else:
                row = selection['objects'][0]
                old = Path(row['mask']['path'])
                replacement = old.with_name('999.png')
                replacement.write_bytes(old.read_bytes())
                row.update(mask_id=999, object_key='image_000__mask_999', stable_id='D1GT:0/999.png')
                row['mask']['path'] = str(replacement)
            intake.write_json(self.selection_path, selection)
            with self.subTest(mode=mode), patch.object(intake, 'BoxSAM') as factory, self.assertRaises(ValueError):
                intake.run(self.selection_path, self.root / ('output_' + mode))
            factory.assert_not_called()

    def test_model_load_failure_keeps_six_explicit_blocked_targets(self):
        with patch.object(intake, 'BoxSAM', side_effect=RuntimeError('cached SAM unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'cached SAM unavailable'):
                intake.run(self.selection_path, self.root / 'output')
        state = intake.read(self.root / 'output/inputs.json')
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(len(state['objects']), 6)
        self.assertTrue(all('cached SAM unavailable' in row['blocked_reason'] for row in state['objects']))


if __name__ == '__main__':
    unittest.main()
