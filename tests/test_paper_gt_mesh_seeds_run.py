from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.paper_gt_mesh_seeds_v1 import run as batch


class PaperGtMeshSeedRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.object_root = self.root / batch.OBJECT_KEY
        self.object_root.mkdir()
        source = self.root / 'dataset' / '0'
        source.mkdir(parents=True)
        image, mask = source / 'img.jpg', source / '160.png'
        Image.new('RGB', (8, 8), (100, 120, 140)).save(image)
        Image.new('L', (8, 8), 255).save(mask)
        self.identity = {'object_key': batch.OBJECT_KEY, 'source_image': str(image),
            'source_image_sha256': batch.sha256(image), 'target_gt_mask': str(mask),
            'target_gt_mask_sha256': batch.sha256(mask)}
        batch.write_json(self.object_root / 'REVIEW_STATE.json', self.identity)
        self.revision = {'commit': 'fixture', 'branch': 'test'}
        self.models = self.root / 'model_revisions.json'
        batch.write_json(self.models, {'fixture': 'pinned'})
        self.commands = []
        self.fail_prepare = set()
        self.fail_generate = set()

    def camera_fixture(self, trellis):
        folder = trellis / 'review'
        folder.mkdir(parents=True)
        parameters = [0, .2, 0, 2.8, .5, .5]
        camera = {'fitting_mode': 'coarse72_then_powell',
            'optimizer_budget': {'maxfev': 350, 'maxiter': 6},
            'parameters': dict(zip(batch.PARAMETERS, parameters)),
            'fov_degrees': 40, 'fit_resolution': 128, 'output_resolution': 512,
            'objective': '1-soft_mask_IoU + 0.05*RGB_BaseColor_MAE',
            'evaluations': 73, 'loss': .9, 'refinement_success': False,
            'extrinsics': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2.8], [0, 0, 0, 1]],
            'intrinsics': [[1, 0, .5], [0, 1, .5], [0, 0, 1]]}
        batch.write_json(folder / 'camera_fit_trajectory.json', {
            'fitting_mode': camera['fitting_mode'], 'optimizer_budget': camera['optimizer_budget'],
            'evaluations': [{'loss': .9, 'parameters': parameters}] * 73})
        review = {'camera': camera, 'metrics': {'original512_mask_iou': .01},
            'artifacts': {'camera_fit_trajectory': str(folder / 'camera_fit_trajectory.json')}}
        batch.write_json(folder / 'review.json', review)
        return review

    def prepare_semantic(self, image, mask, output, *, segmentation):
        self.assertEqual(Path(image), Path(self.identity['source_image']))
        self.assertEqual(Path(mask), Path(self.identity['target_gt_mask']))
        self.assertIn('GT', segmentation)
        output.mkdir()
        receipt = {'segmentation': segmentation, 'source_locks': {}}
        batch.write_json(output / 'source_receipt.json', receipt)
        return receipt

    def execute(self, command, log, gpu):
        self.commands.append(command)
        output = Path(command[command.index('--output') + 1])
        if command[3] == batch.PREPARE_MODULE:
            kind = 'soft' if '--soft' in command else 'hard'
            self.assertEqual(command[command.index('--mask-kind') + 1], 'GT')
            if kind in self.fail_prepare:
                raise RuntimeError('fixture ' + kind + ' preparation failure')
            for color in ('original', 'pink'):
                folder = output / (color + '__' + kind)
                folder.mkdir(parents=True)
                batch.write_json(folder / 'source_receipt.json', {'fixture_variant': folder.name})
                value = (100, 120, 140) if color == 'original' else (180, 100, 140)
                Image.new('RGB', (2, 2), value).save(folder / 'pre_alpha_rgb512.png',
                    compress_level=0 if kind == 'hard' else 9)
        elif command[3] == 'experiments.subject_trellis_counterfactual_v1.run':
            seed = int(command[command.index('--seed') + 1])
            if (output.parents[1].name, seed) in self.fail_generate:
                raise RuntimeError('fixture generation failure')
            (output / 'repro_outputs').mkdir(parents=True)
            (output / 'native_asset.pt').write_bytes(b'native fixture')
            self.camera_fixture(output)
            batch.write_json(output / 'pipeline.json', {'args': {
                'shape_slat_normalization': [0, 1], 'tex_slat_normalization': [0, 1]}})
            receipt = command[command.index('--source-receipt') + 1]
            batch.write_json(output / 'repro_outputs/status.json', {
                'status': 'generated_pending_user_review', 'generated_assets': 1, 'git': self.revision,
                'official_commit': batch.OFFICIAL, 'seed': seed, 'models': {'revisions': {'fixture': 'pinned'}},
                'pipeline_type': '1024_cascade', 'encoder_receipt_sha256': 'encoder',
                'sampling': {'steps': 12}, 'flow_dtype': {'shape': 'float16'},
                'sparse_algorithm': 'explicit_gemm', 'source_receipt': {'sha256': batch.sha256(receipt)}})
        else:
            self.assertEqual(command[3], 'experiments.mailbox_alpha_control_v1.render')
            reference = Path(command[command.index('--reference-review') + 1])
            self.assertEqual(reference, output.parent / 'trellis/review/review.json')
            output.mkdir()
            camera = batch.read(reference)['camera']
            batch.write_json(output / 'status.json', {'status': 'completed_pending_visual_review',
                'git': self.revision, 'completed_renders': 4, 'reference_review': str(reference),
                'cameras': {'reference': {key: camera[key] for key in ('extrinsics', 'intrinsics')}}})

    def run_batch(self, partial=False):
        output = self.object_root / 'mesh_candidates' / 'fixture'
        with patch.object(batch, 'git_info', return_value=self.revision), \
                patch.object(batch, 'MODEL_REVISIONS', self.models), \
                patch.object(batch, 'fresh', side_effect=lambda path: (path.mkdir(parents=True) or path)), \
                patch.object(batch, 'prepare_inputs', side_effect=self.prepare_semantic) as semantic, \
                patch.object(batch, 'execute', side_effect=self.execute), \
                patch.object(batch, 'gallery'), patch.object(batch, 'object_sheet', return_value={}):
            if partial:
                with self.assertRaisesRegex(RuntimeError, 'partial'):
                    batch.run(self.object_root, output, [0, 1])
            else:
                batch.run(self.object_root, output, [0, 1])
        semantic.assert_called_once()
        self.assertEqual(Path(semantic.call_args.args[2]), output / 'input')
        return batch.read(output / 'status.json')

    def test_sixteen_meshes_use_gt_shared_inputs_and_each_own_automatic_camera(self):
        state = self.run_batch()
        self.assertEqual(state['expected_meshes'], 16)
        self.assertEqual(state['generated_meshes'], 16)
        self.assertEqual(state['completed_view_renders'], 64)
        self.assertFalse(state['training_started'])
        self.assertEqual(state['source'], self.identity)
        self.assertEqual(state['rgb_parity']['status'], 'passed')
        prepares = [command for command in self.commands if command[3] == batch.PREPARE_MODULE]
        generations = [command for command in self.commands if '--seed' in command]
        renders = [command for command in self.commands if '--assets' in command]
        self.assertEqual((len(prepares), len(generations), len(renders)), (2, 16, 16))
        self.assertEqual({Path(command[command.index('--output') + 1]).name for command in prepares},
                         {'hard_inputs', 'soft_inputs'})
        self.assertEqual({variant['name'] for variant in state['variants']}, set(batch.VARIANTS))
        for variant in state['variants']:
            self.assertEqual([item['seed'] for item in variant['runs']], [42, 43, 44, 45])
            receipts = {command[command.index('--source-receipt') + 1] for command in generations
                        if Path(command[command.index('--output') + 1]).parents[1].name == variant['name']}
            self.assertEqual(receipts, {variant['source_receipt']})
            self.assertTrue(all(item['status'] == 'completed' for item in variant['runs']))
        for command in self.commands:
            self.assertNotIn('--fixed-crop', command)
            self.assertNotIn('--initial-review', command)

    def test_soft_preparation_failure_blocks_only_eight_soft_meshes(self):
        self.fail_prepare.add('soft')
        state = self.run_batch(partial=True)
        self.assertEqual(state['generated_meshes'], 8)
        self.assertEqual(state['completed_view_renders'], 32)
        for variant in state['variants']:
            expected = 'blocked' if variant['name'].endswith('__soft') else 'completed'
            self.assertEqual({item['status'] for item in variant['runs']}, {expected})

    def test_one_generation_failure_preserves_fifteen_other_meshes(self):
        self.fail_generate.add(('original__hard', 43))
        state = self.run_batch(partial=True)
        runs = [item for variant in state['variants'] for item in variant['runs']]
        self.assertEqual(sum(item['status'] == 'failed' for item in runs), 1)
        self.assertEqual(sum(item['status'] == 'completed' for item in runs), 15)
        self.assertEqual(state['generated_meshes'], 15)
        self.assertEqual(state['completed_view_renders'], 60)

    def test_source_rejects_wrong_identity_and_changed_bytes(self):
        batch.validate_source(self.object_root)
        for field, value in (('object_key', 'image_081__mask_130'),
                             ('source_image_sha256', '0' * 64),
                             ('target_gt_mask_sha256', '0' * 64)):
            identity = deepcopy(self.identity)
            identity[field] = value
            batch.write_json(self.object_root / 'REVIEW_STATE.json', identity)
            with self.subTest(field=field), self.assertRaises(ValueError):
                batch.validate_source(self.object_root)

    def test_camera_rejects_warm_start_and_another_mesh_reference(self):
        trellis = self.root / 'trellis'
        review = self.camera_fixture(trellis)
        batch.check_camera(trellis)
        rendered = {'reference_review': str(trellis / 'review/review.json'),
                    'cameras': {'reference': {key: review['camera'][key]
                                             for key in ('extrinsics', 'intrinsics')}}}
        batch.check_camera(trellis, rendered)
        rendered['reference_review'] = str(self.root / 'other_mesh/review/review.json')
        with self.assertRaises(ValueError):
            batch.check_camera(trellis, rendered)
        review['camera']['fitting_mode'] = 'single_warm_start_powell'
        batch.write_json(trellis / 'review/review.json', review)
        with self.assertRaises(ValueError):
            batch.check_camera(trellis)


if __name__ == '__main__':
    unittest.main()
