from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.sam_six_seed_v1 import run as batch


class SamSixSeedRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.revision = {'commit': 'fixture', 'branch': 'test'}
        self.inputs = {'schema': 'sam_six_seed_inputs/v1', 'git': self.revision, 'objects': []}
        for key in batch.OBJECT_KEYS:
            receipt = self.root / (key + '.json')
            batch.write_json(receipt, {'segmentation': 'SAM fixture'})
            self.inputs['objects'].append({'object_key': key, 'source_receipt': str(receipt),
                                          'source_receipt_sha256': batch.sha256(receipt)})
        self.commands, self.first_slots = [], None
        self.fail_prepare, self.fail_render = set(), set()
        self.bad_generation = {}
        self.rgb_mismatch = set()

    def test_manifest_requires_fixed_six_ids_same_git_and_unchanged_receipts(self):
        self.assertEqual(len(batch.validate_inputs(self.inputs, self.revision)), 6)
        for case in ('missing', 'duplicate', 'wrong_git', 'hash', 'empty_reason'):
            inputs = deepcopy(self.inputs)
            if case == 'missing':
                inputs['objects'].pop()
            elif case == 'duplicate':
                inputs['objects'][-1] = inputs['objects'][0]
            elif case == 'wrong_git':
                inputs['git']['commit'] = 'other'
            elif case == 'hash':
                inputs['objects'][0]['source_receipt_sha256'] = '0' * 64
            else:
                inputs['objects'][0]['blocked_reason'] = ''
            with self.subTest(case=case), self.assertRaises(ValueError):
                batch.validate_inputs(inputs, self.revision)

    def execute(self, command, log, gpu):
        self.commands.append(command)
        output = Path(command[command.index('--output') + 1])
        if command[3] == batch.PREPARE_MODULE:
            kind = 'soft' if '--soft' in command else 'hard'
            self.assertEqual(command[command.index('--mask-kind') + 1], 'SAM')
            self.assertEqual(command[0], batch.RUNTIME['trellis_python' if kind == 'soft' else 'python'])
            if (output.parent.name, kind) in self.fail_prepare:
                raise RuntimeError('fixture ' + kind + ' preparation failure')
            for color in ('original', 'pink'):
                folder = output / (color + '__' + kind)
                folder.mkdir(parents=True)
                batch.write_json(folder / 'source_receipt.json', {'fixture_variant': folder.name})
                value = (100, 120, 140) if color == 'original' else (180, 100, 140)
                if output.parent.name in self.rgb_mismatch and kind == 'soft' and color == 'original':
                    value = (101, 120, 140)
                Image.new('RGB', (2, 2), value).save(folder / 'pre_alpha_rgb512.png',
                                                   compress_level=0 if kind == 'hard' else 9)
        elif command[3] == 'experiments.subject_trellis_counterfactual_v1.run':
            seed = int(command[command.index('--seed') + 1])
            identity = (output.parents[3].name, output.parents[1].name, seed)
            failure = self.bad_generation.get(identity)
            if failure == 'error':
                raise RuntimeError('fixture generation failure')
            (output / 'repro_outputs').mkdir(parents=True)
            (output / 'review').mkdir()
            (output / 'native_asset.pt').write_bytes(b'native fixture')
            batch.write_json(output / 'review/review.json', {'camera': seed, 'metrics': {'original512_mask_iou': .01}})
            batch.write_json(output / 'pipeline.json', {'args': {
                'shape_slat_normalization': [0, 1], 'tex_slat_normalization': [0, 1]}})
            receipt = command[command.index('--source-receipt') + 1]
            batch.write_json(output / 'repro_outputs/status.json', {
                'status': 'generated_pending_user_review', 'generated_assets': 1, 'git': self.revision,
                'official_commit': batch.OFFICIAL, 'seed': 42 if failure == 'wrong_seed' else seed,
                'models': {'revisions': batch.read(batch.MODEL_REVISIONS)},
                'pipeline_type': '1024_cascade', 'encoder_receipt_sha256': 'encoder',
                'sampling': {'steps': 13 if failure == 'config' else 12},
                'flow_dtype': {'shape': 'float16'}, 'sparse_algorithm': 'explicit_gemm',
                'source_receipt': {'sha256': batch.sha256(receipt)}})
        else:
            self.assertEqual(command[3], 'experiments.mailbox_alpha_control_v1.render')
            seed = int(output.parent.name.removeprefix('seed_'))
            identity = (output.parents[3].name, output.parents[1].name, seed)
            if identity in self.fail_render:
                raise RuntimeError('fixture render failure')
            self.assertEqual(Path(command[command.index('--reference-review') + 1]),
                             output.parent / 'trellis/review/review.json')
            output.mkdir()
            batch.write_json(output / 'status.json', {'status': 'completed_pending_visual_review',
                'git': self.revision, 'completed_renders': 4})

    def run_batch(self, partial=False):
        inputs_path, output = self.root / 'inputs.json', self.root / 'output'
        batch.write_json(inputs_path, self.inputs)
        original_write = batch.write_json
        def write(path, value):
            if Path(path) == output / 'status.json' and value['status'] == 'running':
                if self.first_slots is None:
                    self.first_slots = [item['status'] for row in value['objects']
                                        for variant in row['variants'] for item in variant['runs']]
                return
            original_write(path, value)
        with patch.object(batch, 'git_info', return_value=self.revision), \
                patch.object(batch, 'fresh', side_effect=lambda path: (path.mkdir() or path)), \
                patch.object(batch, 'execute', side_effect=self.execute), \
                patch.object(batch, 'write_json', side_effect=write), \
                patch.object(batch, 'gallery'), patch.object(batch, 'variant_sheet', return_value={}):
            if partial:
                with self.assertRaisesRegex(RuntimeError, 'partial'):
                    batch.run(inputs_path, output, [0, 1, 2, 3])
            else:
                batch.run(inputs_path, output, [0, 1, 2, 3])
        return batch.read(output / 'status.json')

    def test_96_slots_share_one_preparation_per_condition_across_four_seeds(self):
        state = self.run_batch()
        self.assertEqual(self.first_slots, ['pending'] * 96)
        self.assertEqual(state['expected_meshes'], 96)
        self.assertEqual(state['generated_meshes'], 96)
        self.assertEqual(state['completed_view_renders'], 384)
        self.assertEqual(state['completed_variants'], 24)
        self.assertEqual(state['completed_objects'], 6)
        self.assertFalse(state['training_started'])
        prepares = [command for command in self.commands if command[3] == batch.PREPARE_MODULE]
        generations = [command for command in self.commands if '--seed' in command]
        renders = [command for command in self.commands if '--assets' in command]
        self.assertEqual((len(prepares), len(generations), len(renders)), (12, 96, 96))
        for row in state['objects']:
            self.assertEqual(row['rgb_parity']['status'], 'passed')
            self.assertTrue(all(color['decoded_pixels_equal'] and not color['png_files_equal']
                                for color in row['rgb_parity']['colors'].values()))
            for variant in row['variants']:
                self.assertEqual([item['seed'] for item in variant['runs']], list(batch.SEEDS))
                self.assertEqual({item['source_receipt'] for item in variant['runs']}, {variant['source_receipt']})
                self.assertEqual({item['source_receipt_sha256'] for item in variant['runs']}, {variant['source_receipt_sha256']})
        for seed in batch.SEEDS:
            self.assertEqual(sum(command[command.index('--seed') + 1] == str(seed) for command in generations), 24)
        for command in self.commands:
            self.assertNotIn('--fixed-crop', command)
            self.assertNotIn('train', command)

    def test_mask_group_seed_configuration_and_render_failures_remain_isolated(self):
        keys = batch.OBJECT_KEYS
        self.inputs['objects'][0] = {'object_key': keys[0], 'blocked_reason': 'upstream SAM target not found'}
        self.fail_prepare.update(((keys[1], 'soft'), (keys[2], 'hard')))
        self.bad_generation.update({(keys[3], 'original__hard', 43): 'wrong_seed',
                                    (keys[3], 'original__hard', 44): 'config',
                                    (keys[3], 'original__hard', 45): 'error'})
        self.fail_render.add((keys[4], 'pink__soft', 44))
        state = self.run_batch(partial=True)
        runs = [item for row in state['objects'] for variant in row['variants'] for item in variant['runs']]
        self.assertEqual(len(runs), 96)
        self.assertEqual(sum(item['status'] == 'blocked' for item in runs), 32)
        self.assertEqual(sum(item['status'] == 'failed' for item in runs), 4)
        self.assertEqual(state['generated_meshes'], 61)
        self.assertEqual(state['completed_mesh_runs'], 60)
        self.assertEqual(state['completed_view_renders'], 240)
        self.assertEqual(state['completed_variants'], 14)
        self.assertEqual(state['completed_objects'], 1)
        self.assertEqual([variant['status'] for variant in state['objects'][1]['variants']],
                         ['completed', 'completed', 'blocked', 'blocked'])
        self.assertEqual(state['objects'][3]['variants'][0]['runs'][0]['status'], 'completed')
        self.assertIn('seed or input', state['objects'][3]['variants'][0]['runs'][1]['error'])
        self.assertIn('non-seed', state['objects'][3]['variants'][0]['runs'][2]['error'])
        self.assertTrue(state['objects'][4]['variants'][3]['runs'][2]['generated'])
        self.assertEqual(state['objects'][2]['rgb_parity']['status'], 'unavailable')

    def test_rgb_pixel_mismatch_blocks_soft_seeds_and_preserves_hard(self):
        self.rgb_mismatch.add(batch.OBJECT_KEYS[0])
        state = self.run_batch(partial=True)
        row = state['objects'][0]
        self.assertEqual(row['rgb_parity']['status'], 'mismatch')
        self.assertFalse(row['rgb_parity']['colors']['original']['decoded_pixels_equal'])
        self.assertTrue(row['rgb_parity']['colors']['pink']['decoded_pixels_equal'])
        self.assertEqual([variant['status'] for variant in row['variants']],
                         ['completed', 'completed', 'blocked', 'blocked'])
        self.assertTrue(all(item['status'] == 'blocked' for variant in row['variants'][2:] for item in variant['runs']))
        self.assertEqual(state['generated_meshes'], 88)
        self.assertEqual(state['completed_view_renders'], 352)

    def test_seed_sheet_keeps_successful_rows_and_failed_placeholders(self):
        row = {'object_key': batch.OBJECT_KEYS[0], 'status': 'partial', 'variants': []}
        variant = {'name': 'original__hard', 'status': 'partial', 'runs': []}
        row['variants'].append(variant)
        for index, seed in enumerate(batch.SEEDS):
            item = {'name': f'seed_{seed}', 'seed': seed, 'status': 'completed' if index < 2 else 'blocked'}
            if index < 2:
                path = self.root / f'seed_{seed}.png'
                Image.new('RGB', (2048, 544), (100 + index * 50, 20, 30)).save(path)
                item['comparison_clay'] = str(path)
            else:
                item['error'] = 'fixture blocked seed'
            variant['runs'].append(item)
        variant.update(batch.variant_sheet(self.root, row, variant))
        with Image.open(variant['comparison_clay']) as image:
            self.assertEqual(image.size, (2048, 2176))
            pixels = np.asarray(image)
            np.testing.assert_array_equal(pixels[100, 100], [100, 20, 30])
            np.testing.assert_array_equal(pixels[644, 100], [150, 20, 30])
            self.assertTrue((pixels[1088:] != 238).any())
        batch.gallery(self.root, {'objects': [row]})
        html = (self.root / 'index.html').read_text(encoding='utf8')
        self.assertIn('original__hard/comparison_clay.png', html)
        self.assertIn('seed 44: blocked', html)
        self.assertIn('independently fitted', html)


if __name__ == '__main__':
    unittest.main()
