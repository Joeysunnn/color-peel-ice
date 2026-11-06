from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.icbench_gt_alpha_four_v1 import run as batch


class GTAlphaFourRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.selection = {'schema': 'icbench_gt36_selection/v1', 'objects': []}
        for index in range(36):
            sample, mask = index % 29, index + 1
            folder = self.root / 'sources' / str(sample)
            folder.mkdir(parents=True, exist_ok=True)
            image, alpha = folder / 'img.jpg', folder / f'{mask}.png'
            image.write_bytes(b'mocked source image')
            alpha.write_bytes(b'mocked GT binary mask')
            self.selection['objects'].append({'sample_id': sample, 'mask_id': mask,
                'object_key': f'image_{sample:03d}__mask_{mask:03d}', 'stable_id': f'D1GT:{sample}/{mask}.png',
                'image': {'path': str(image), 'sha256': batch.sha256(image)},
                'mask': {'path': str(alpha), 'sha256': batch.sha256(alpha)}})
        self.selection_path = self.root / 'selection.json'
        batch.write_json(self.selection_path, self.selection)
        self.revision = {'commit': 'fixture', 'dirty': False}
        self.commands, self.first_slots = [], None
        self.fail_semantic = set()
        self.fail_prepare, self.fail_generation, self.fail_render = set(), set(), set()

    def inputs(self, image, mask, output, *, segmentation):
        self.assertIn('ICBench GT', segmentation)
        if output.parent.name in self.fail_semantic:
            raise ValueError('fixture invalid GT')
        output.mkdir()
        batch.write_json(output / 'source_receipt.json', {'semantic': True})

    def execute(self, command, log, gpu):
        self.commands.append(command)
        output = Path(command[command.index('--output') + 1])
        if command[3] == batch.PREPARE_MODULE:
            kind = 'soft' if '--soft' in command else 'hard'
            self.assertEqual(command[0], batch.RUNTIME['trellis_python' if kind == 'soft' else 'python'])
            if (output.parent.name, kind) in self.fail_prepare:
                raise RuntimeError('fixture ' + kind + ' preparation failure')
            for color in ('original', 'pink'):
                folder = output / (color + '__' + kind)
                folder.mkdir(parents=True)
                batch.write_json(folder / 'source_receipt.json', {'fixture_variant': folder.name})
        elif command[3] == 'experiments.subject_trellis_counterfactual_v1.run':
            identity = (output.parents[2].name, output.parent.name)
            if identity in self.fail_generation:
                raise RuntimeError('fixture generation failure')
            (output / 'repro_outputs').mkdir(parents=True)
            (output / 'review').mkdir()
            (output / 'native_asset.pt').write_bytes(b'native fixture')
            batch.write_json(output / 'review/review.json', {'camera': output.parent.name,
                'metrics': {'original512_mask_iou': .01}})  # No score gate belongs to this runner.
            receipt = command[command.index('--source-receipt') + 1]
            batch.write_json(output / 'repro_outputs/status.json', {
                'status': 'generated_pending_user_review', 'generated_assets': 1, 'git': self.revision,
                'official_commit': batch.OFFICIAL, 'seed': 42,
                'models': {'revisions': batch.read(batch.MODEL_REVISIONS)},
                'source_receipt': {'sha256': batch.sha256(receipt)}})
        else:
            self.assertEqual(command[3], 'experiments.mailbox_alpha_control_v1.render')
            identity = (output.parents[2].name, output.parent.name)
            if identity in self.fail_render:
                raise RuntimeError('fixture render failure')
            self.assertEqual(Path(command[command.index('--reference-review') + 1]),
                             output.parent / 'trellis/review/review.json')
            self.assertEqual(command.index('--labels') - command.index('--assets'), 2)
            output.mkdir()
            batch.write_json(output / 'status.json', {'status': 'completed_pending_visual_review',
                'git': self.revision, 'completed_renders': 4})

    def run_batch(self, partial=False):
        output = self.root / 'output'
        original_write = batch.write_json
        def write(path, value):
            if Path(path) == output / 'status.json' and value['status'] == 'running':
                if self.first_slots is None:
                    self.first_slots = [v['status'] for row in value['objects'] for v in row['variants']]
                return  # Avoid repeated large fixture status writes; retain the final real JSON.
            original_write(path, value)
        with patch.object(batch, 'git_info', return_value=self.revision), \
                patch.object(batch, 'fresh', side_effect=lambda path: (path.mkdir() or path)), \
                patch.object(batch, 'prepare_inputs', side_effect=self.inputs), \
                patch.object(batch, 'execute', side_effect=self.execute), \
                patch.object(batch, 'write_json', side_effect=write), \
                patch.object(batch, 'gallery'), patch.object(batch, 'object_sheet', return_value={}):
            if partial:
                with self.assertRaisesRegex(RuntimeError, 'partial'):
                    batch.run(self.selection_path, output, [0, 1, 2, 3])
            else:
                batch.run(self.selection_path, output, [0, 1, 2, 3])
        return batch.read(output / 'status.json')

    def test_complete_plan_has_144_slots_576_views_and_per_mesh_references(self):
        state = self.run_batch()
        self.assertEqual(self.first_slots, ['pending'] * 144)
        self.assertEqual(state['expected_meshes'], 144)
        self.assertEqual(state['expected_view_renders'], 576)
        self.assertEqual(state['generated_meshes'], 144)
        self.assertEqual(state['completed_variants'], 144)
        self.assertEqual(state['completed_view_renders'], 576)
        self.assertEqual(state['completed_objects'], 36)
        self.assertFalse(state['training_started'])
        self.assertFalse(state['quality_accepted_by_user'])
        prepares = [command for command in self.commands if command[3] == batch.PREPARE_MODULE]
        generations = [command for command in self.commands if '--seed' in command]
        renders = [command for command in self.commands if '--assets' in command]
        self.assertEqual((len(prepares), len(generations), len(renders)), (72, 144, 144))
        for command in self.commands:
            self.assertNotIn('--fixed-crop', command)
            self.assertNotIn('train', command)
        for command in generations:
            self.assertEqual(command[command.index('--seed') + 1], '42')
        for row in state['objects']:
            self.assertEqual([variant['name'] for variant in row['variants']], list(batch.VARIANTS))
            for variant in row['variants']:
                self.assertEqual(state['source_locks'][variant['source_receipt']], variant['source_receipt_sha256'])

    def test_semantic_group_generation_and_render_failures_are_isolated(self):
        keys = [row['object_key'] for row in self.selection['objects']]
        self.fail_semantic.add(keys[0])
        self.fail_prepare.update(((keys[1], 'soft'), (keys[2], 'hard')))
        self.fail_generation.add((keys[3], 'original__hard'))
        self.fail_render.add((keys[4], 'pink__soft'))
        state = self.run_batch(partial=True)
        variants = [variant for row in state['objects'] for variant in row['variants']]
        self.assertEqual(len(variants), 144)
        self.assertEqual(sum(variant['status'] == 'blocked' for variant in variants), 8)
        self.assertEqual(sum(variant['status'] == 'failed' for variant in variants), 2)
        self.assertEqual(state['generated_meshes'], 135)
        self.assertEqual(state['completed_variants'], 134)
        self.assertEqual(state['completed_view_renders'], 536)
        self.assertEqual(state['completed_objects'], 31)
        self.assertEqual([variant['status'] for variant in state['objects'][1]['variants']],
                         ['completed', 'completed', 'blocked', 'blocked'])
        self.assertEqual([variant['status'] for variant in state['objects'][2]['variants']],
                         ['blocked', 'blocked', 'completed', 'completed'])
        self.assertTrue(state['objects'][4]['variants'][3]['generated'])
        self.assertEqual(state['status'], 'partial')

    def test_partial_object_sheet_keeps_hard_successes_and_soft_placeholders(self):
        folder = self.root / 'image_000__mask_001'
        folder.mkdir()
        variants = []
        for index, name in enumerate(batch.VARIANTS):
            item = {'name': name, 'status': 'completed' if index < 2 else 'blocked'}
            if index < 2:
                path = folder / (name + '.png')
                Image.new('RGB', (2048, 544), (100 + index * 50, 20, 30)).save(path)
                item['comparison_clay'] = str(path)
            else:
                item['error'] = 'soft preparation failed'
            variants.append(item)
        row = {'object_key': folder.name, 'status': 'partial', 'variants': variants}
        row.update(batch.object_sheet(self.root, row))
        with Image.open(row['comparison_clay']) as image:
            self.assertEqual(image.size, (2048, 2176))
            pixels = np.asarray(image)
            np.testing.assert_array_equal(pixels[100, 100], [100, 20, 30])
            np.testing.assert_array_equal(pixels[644, 100], [150, 20, 30])
            self.assertTrue((pixels[1088:] != 238).any())  # Visible placeholder text in both missing rows.
        batch.gallery(self.root, {'objects': [row]})
        html = (self.root / 'index.html').read_text(encoding='utf8')
        self.assertIn(folder.name + '/comparison_clay.png', html)
        self.assertIn('original__soft: blocked', html)
        self.assertIn('independently fitted', html)


if __name__ == '__main__':
    unittest.main()
