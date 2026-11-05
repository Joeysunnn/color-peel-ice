from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.icbench_gt_mesh_v1 import run as batch
from experiments.subject_trellis_counterfactual_v1.run import check_source_receipt


class ICBenchGTMeshTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.selection = {'schema': 'icbench_gt36_selection/v1', 'objects': []}
        for index in range(36):
            sample, mask = index % 29, index + 1
            folder = self.root / 'sources' / str(sample)
            folder.mkdir(parents=True, exist_ok=True)
            image_path, mask_path = folder / 'img.jpg', folder / f'{mask}.png'
            if not image_path.exists():
                Image.new('RGB', (512, 512), (180, 100, 40)).save(image_path)
            alpha = np.zeros((512, 512), dtype=np.uint8)
            alpha[30:430, 40:240] = 255
            alpha[150:200, 100:150] = 0  # Retain a genuine GT hole.
            alpha[490:500, 490:500] = 255  # Retain a disconnected GT component.
            Image.fromarray(alpha).save(mask_path)
            self.selection['objects'].append({'sample_id': sample, 'mask_id': mask,
                'object_key': f'image_{sample:03d}__mask_{mask:03d}', 'stable_id': f'D1GT:{sample}/{mask}.png',
                'image': {'path': str(image_path), 'sha256': batch.sha256(image_path)},
                'mask': {'path': str(mask_path), 'sha256': batch.sha256(mask_path)}})

    def test_selection_rejects_wrong_count_duplicate_identity_and_misbound_mask(self):
        self.assertEqual(len(batch.validate_selection(self.selection)), 65)
        for case in ('missing', 'duplicate', 'wrong_mask', 'wrong_hash'):
            value = deepcopy(self.selection)
            if case == 'missing':
                value['objects'].pop()
            elif case == 'duplicate':
                value['objects'][-1] = value['objects'][0]
            elif case == 'wrong_mask':
                value['objects'][0]['mask'] = value['objects'][1]['mask']
            else:
                value['objects'][0]['mask']['sha256'] = '0' * 64
            with self.subTest(case=case), self.assertRaises(ValueError):
                batch.validate_selection(value)

    def test_gt_pink_retains_exact_binary_mask_holes_components_and_provenance(self):
        row = self.selection['objects'][0]
        batch.prepare_inputs(row['image']['path'], row['mask']['path'], self.root / 'input',
                             segmentation=f'ICBench GT {row["stable_id"]}')
        receipt = batch.prepare_pink(self.root / 'input/source_receipt.json', self.root / 'pink')
        with Image.open(row['mask']['path']) as source, Image.open(receipt['cutout_path']) as cutout:
            mask, rgba = np.array(source), np.array(cutout)
        np.testing.assert_array_equal(rgba[..., 3], mask)
        self.assertEqual(set(np.unique(rgba[..., 3])), {0, 255})
        self.assertTrue((rgba[mask == 0, :3] == 0).all())
        self.assertIn(row['stable_id'], receipt['segmentation'])
        self.assertNotIn('SAM', receipt['conditioning_recolor']['mask_policy'])
        self.assertEqual(receipt['conditioning_recolor']['hue_degrees'], 0.)
        self.assertEqual(receipt['conditioning_recolor']['recolor_metrics']['outside_mask_changed_pixel_count'], 0)
        self.assertTrue(check_source_receipt(self.root / 'pink/source_receipt.json')['alpha_matches_mask'])

    def test_nonbinary_gt_is_rejected_without_softening_or_thresholding(self):
        row = self.selection['objects'][0]
        Image.new('L', (512, 512), 128).save(row['mask']['path'])
        with self.assertRaisesRegex(ValueError, 'binary'):
            batch.prepare_inputs(row['image']['path'], row['mask']['path'], self.root / 'input',
                                 segmentation='ICBench GT')
        self.assertFalse((self.root / 'input').exists())

    def test_generation_failure_keeps_other_objects_and_four_view_render_jobs(self):
        selection_path = self.root / 'selection.json'
        batch.write_json(selection_path, self.selection)
        output = self.root / 'output'
        revision = {'commit': 'fixture', 'dirty': False}
        commands = []

        def prepare_pink(parent, folder, hue_degrees):
            folder.mkdir()
            batch.write_json(folder / 'source_receipt.json', {'test_fixture': True})

        def execute(command, log, gpu):
            commands.append(command)
            directory = Path(command[command.index('--output') + 1])
            if command[3] == 'experiments.subject_trellis_counterfactual_v1.run':
                if directory.parent.name == self.selection['objects'][0]['object_key']:
                    raise RuntimeError('injected generation failure')
                directory.mkdir()
                (directory / 'repro_outputs').mkdir()
                receipt = command[command.index('--source-receipt') + 1]
                batch.write_json(directory / 'repro_outputs/status.json', {
                    'status': 'generated_pending_user_review', 'generated_assets': 1, 'git': revision,
                    'official_commit': batch.OFFICIAL, 'seed': 42,
                    'models': {'revisions': batch.read(batch.MODEL_REVISIONS)},
                    'source_receipt': {'sha256': batch.sha256(receipt)}})
            else:
                directory.mkdir()
                batch.write_json(directory / 'status.json', {'status': 'completed_pending_visual_review',
                    'git': revision, 'completed_renders': 4})

        with patch.object(batch, 'git_info', return_value=revision), \
                patch.object(batch, 'fresh', side_effect=lambda path: (path.mkdir() or path)), \
                patch.object(batch, 'prepare_inputs'), patch.object(batch, 'prepare_pink', side_effect=prepare_pink), \
                patch.object(batch, 'execute', side_effect=execute):
            with self.assertRaisesRegex(RuntimeError, 'partial'):
                batch.run(selection_path, output, [0, 1])
        state = batch.read(output / 'status.json')
        self.assertEqual(state['status'], 'partial')
        self.assertEqual(state['completed_objects'], 35)
        self.assertIn('injected generation failure', state['objects'][0]['error'])
        self.assertFalse(state['quality_accepted_by_user'])
        generation = [cmd for cmd in commands if '--seed' in cmd]
        renders = [cmd for cmd in commands if '--assets' in cmd]
        self.assertEqual(len(generation), 36)
        self.assertEqual(len(renders), 35)
        for command in generation:
            self.assertEqual(command[command.index('--seed') + 1], '42')
            self.assertNotIn('--fixed-crop', command)
        for command in renders:
            self.assertEqual(command.index('--labels') - command.index('--assets'), 2)
        self.assertIn('image_000__mask_001', (output / 'index.html').read_text(encoding='utf8'))


if __name__ == '__main__':
    unittest.main()
