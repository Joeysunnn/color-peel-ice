import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.methods.colorpeel_ice import single_image_pipeline as pipeline
from src.methods.colorpeel_ice import single_image_training as training
from scripts.launch.colorpeel_run import read_config


class SingleImagePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        environment = patch.dict(os.environ, {'COLORPEEL_RUN_ROOT': str(self.root),
                                             'COLORPEEL_OUTPUT_ROOT': str(self.root)})
        environment.start()
        self.addCleanup(environment.stop)
        self.revision = {'commit': 'a' * 40, 'branch': 'codex/single-image'}
        revision = patch.object(pipeline, 'git_info', return_value=self.revision)
        revision.start()
        self.addCleanup(revision.stop)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        pipeline.write_json(path, value)
        return path

    def prepared(self):
        objects = []
        for index, available in ((1, ['color', 'texture']), (2, ['color'])):
            folder = self.root / 'inputs' / f'object_{index:03d}'
            item = {'id': folder.name, 'datasets': {name: {} for name in available},
                    'failures': {'texture': 'initializer unavailable'} if index == 2 else {}}
            manifest = self.write(folder / 'object.json', item)
            objects.append({**item, 'object_manifest': str(manifest),
                            'object_manifest_sha256': pipeline.sha256(manifest)})
        return self.write(self.root / 'inputs/prepared.json', {
            'status': 'partial', 'git': self.revision, 'branches': ['color', 'texture'], 'objects': objects})

    def test_step_parser_and_training_plan_reject_invalid_overrides(self):
        self.assertEqual(pipeline.parse_steps(['subject=1000', 'texture=600']), {'subject': 1000, 'texture': 600})
        for values in (['color=1', 'color=2'], ['color'], ['color=abc']):
            with self.subTest(values=values), self.assertRaises(ValueError):
                pipeline.parse_steps(values)
        with self.assertRaises(ValueError):
            pipeline.training_plan(['color'], pipeline.parse_steps(['color=-3']))

    def test_train_queue_isolates_failed_and_missing_branches_and_continues(self):
        prepared = self.prepared()
        calls = []

        def execute(command, log, gpu):
            command = [str(value) for value in command]
            calls.append(command)
            output = Path(command[command.index('--output') + 1])
            branch = command[command.index('--branches') + 1]
            output.mkdir(parents=True)
            if branch == 'texture':
                self.write(output / 'status.json', {'status': 'failed', 'error': 'finite-loss check failed'})
                raise subprocess.CalledProcessError(1, command)
            self.write(output / 'status.json', {'status': 'trained', 'completed_steps': 70,
                                               'checkpoint_index': str(output / 'ckpts/index.json')})

        with patch.object(pipeline, 'execute', side_effect=execute):
            output = pipeline.train(prepared, self.root / 'training',
                                    {'color': 70, 'texture': 400}, [0], pipeline.RUNTIME)
        state = pipeline.read(output)
        outcomes = {(job['object_id'], job['branch']): job['status'] for job in state['objects']}
        self.assertEqual(outcomes, {('object_001', 'color'): 'trained', ('object_001', 'texture'): 'failed',
                                   ('object_002', 'color'): 'trained', ('object_002', 'texture'): 'blocked'})
        self.assertEqual(len(calls), 3)
        self.assertEqual(state['status'], 'partial')
        self.assertTrue((output.parent / 'object_001/texture_result.json').exists())
        self.assertTrue((output.parent / 'train_outputs/COMMANDS.md').exists())

    def test_only_selected_branch_is_queued(self):
        prepared = self.prepared()
        calls = []

        def execute(command, log, gpu):
            command = [str(value) for value in command]
            calls.append(command)
            output = Path(command[command.index('--output') + 1])
            self.write(output / 'status.json', {'status': 'trained', 'completed_steps': 150})

        with patch.object(pipeline, 'execute', side_effect=execute):
            result = pipeline.train(prepared, self.root / 'selected', {'color': 150}, [0, 2], pipeline.RUNTIME)
        state = pipeline.read(result)
        self.assertEqual(state['status'], 'succeeded')
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(command[command.index('--branches') + 1] == 'color' for command in calls))

    def test_changed_git_or_object_manifest_blocks_before_execution(self):
        for name in ('git', 'manifest'):
            with self.subTest(name=name):
                prepared = self.prepared()
                value = pipeline.read(prepared)
                if name == 'git':
                    value['git'] = {'commit': 'b' * 40, 'branch': 'codex/changed'}
                    pipeline.write_json(prepared, value)
                else:
                    Path(value['objects'][0]['object_manifest']).write_text('{}')
                with patch.object(pipeline, 'execute') as execute, self.assertRaises(ValueError):
                    pipeline.train(prepared, self.root / name, {'color': 150}, [0], pipeline.RUNTIME)
                execute.assert_not_called()

    def test_train_worker_config_roundtrip_and_actual_step_export(self):
        source = self.write(self.root / 'object.json', {'id': 'object_004', 'record': {},
                             'initializer_manifest': 'initializer.json', 'datasets': {'texture': {}}})
        config = {'schema_version': 1, 'stage': 'train',
                  'run': {'study': training.STUDY, 'variant': 'object_004_texture_steps630', 'seed': 42},
                  'environment': {'CUDA_VISIBLE_DEVICES': '2'},
                  'args': {'learning_rate': 1e-5, 'kv_learning_rate': 5e-5, 'adam_epsilon': 1e-8,
                           'max_train_steps': 630, 'checkpoint_steps': [630]}}

        def launched(command, **kwargs):
            path = Path(command[command.index('--config') + 1])
            self.assertEqual(read_config(path), config)
            run_dir = Path(command[command.index('--run-dir') + 1])
            self.assertIn('__object_004_texture_steps630__aaaaaaa__42', run_dir.name)

        with patch.object(training, 'prepare') as prepare, patch.object(training, 'verify_prepared'), \
                patch.object(training, 'build_config', return_value=config), \
                patch.object(training, 'export', return_value={'steps': {630: {'steps': 630, 'paired': True}}}) as export, \
                patch.object(pipeline.subprocess, 'run', side_effect=launched):
            state = pipeline.train_object(source, self.root / 'worker', 'texture', 630, 2)
        self.assertEqual(state['status'], 'trained')
        self.assertEqual(state['checkpoint']['steps'], 630)
        self.assertEqual(state['completed_steps'], 630)
        self.assertEqual(prepare.call_args.kwargs['branches'], ['texture'])
        self.assertEqual(prepare.call_args.kwargs['steps'], {'texture': 630})
        self.assertEqual(export.call_args.kwargs['gpu'], 2)

    def test_all_stage_passes_selected_plan_and_distinct_output_roots(self):
        image = self.root / 'image.jpg'
        image.write_bytes(b'test input')
        prepared = self.write(self.root / 'fixture_prepared.json', {'status': 'prepared'})
        checkpoints = self.write(self.root / 'fixture_checkpoints.json', {'status': 'succeeded'})
        with patch.object(pipeline, 'prepare', return_value=prepared) as prepare, \
                patch.object(pipeline, 'train', return_value=checkpoints) as train:
            code = pipeline.main(['--image', str(image), '--output', str(self.root / 'all'),
                                  '--branches', 'color', 'material', '--steps', 'material=250', '--gpus', '1', '3'])
        self.assertEqual(code, 0)
        self.assertEqual(prepare.call_args.args[1], self.root / 'all/prepared')
        self.assertEqual(prepare.call_args.args[2:4], (['color', 'material'], 1))
        self.assertEqual(train.call_args.args[1:4], (self.root / 'all/training',
                                                   {'color': 150, 'material': 250}, [1, 3]))

    def test_prepare_retains_color_when_texture_initializer_unavailable(self):
        from src.methods.colorpeel_ice import single_image_subject
        image = self.root / 'input.jpg'
        image.write_bytes(b'test image')
        source_hash = pipeline.sha256(image)
        calls = []

        def execute(command, log, gpu):
            command = [str(value) for value in command]
            calls.append(command)
            output = Path(command[command.index('--output') + 1])
            output.mkdir(parents=True)
            if '--worker' in command:
                worker = command[command.index('--worker') + 1]
                if worker == 'discover':
                    self.write(output / 'categories.json', {'status': 'verified'})
                elif worker == 'segment':
                    self.write(output / 'objects.json', {'status': 'succeeded', 'failures': [],
                        'source': {'image_sha256': source_hash, 'canonical_image_sha256': source_hash},
                        'objects': [{'id': 'object_001', 'source_image': str(image),
                                     'object_mask': str(self.root / 'mask.png'), 'vlm_input': str(image)}]})
                elif worker == 'initialize':
                    self.write(output / 'initializers.json', {'selected_branches': ['color'],
                        'subject_category': 'vase', 'unavailable_branches': ['texture: invalid BPE']})
                else:
                    self.fail('unexpected worker')
            else:
                self.assertIn('single_image_cmt', ' '.join(command))
                stage = command[command.index('--stage') + 1]
                if stage == 'plan':
                    self.assertEqual(command[command.index('--branches') + 1:], ['color'])
                if stage == 'finalize':
                    self.write(output / 'datasets.json', {'branches': {'color': {'rows': []}}, 'failures': {}})

        with patch.object(single_image_subject, 'prepare_inputs'), \
                patch.object(pipeline, 'execute', side_effect=execute):
            result = pipeline.prepare(image, self.root / 'prepare', ['color', 'texture'], 0, pipeline.RUNTIME)
        state = pipeline.read(result)
        self.assertEqual(state['status'], 'partial')
        self.assertEqual(set(state['objects'][0]['datasets']), {'color'})
        self.assertIn('texture', state['objects'][0]['failures'])
        self.assertFalse(any('trellis' in ' '.join(command).lower() for command in calls))
        self.assertFalse(any('run_supermat_batch' in ' '.join(command) for command in calls))

    def test_subject_uses_soft_alpha_pink_while_color_keeps_original_input(self):
        from PIL import Image, ImageDraw
        from experiments.subject_pink_input_repro_v1.prepare import prepare as prepare_pink
        sources = self.root / 'sources'
        sources.mkdir()
        image, mask = sources / 'source.png', sources / 'sam.png'
        Image.new('RGB', (512, 512), (180, 100, 40)).save(image)
        alpha = Image.new('L', (512, 512))
        ImageDraw.Draw(alpha).rectangle((120, 80, 300, 450), fill=255)
        alpha.save(mask)
        original_bytes = {path: path.read_bytes() for path in (image, mask)}
        source_hash = pipeline.sha256(image)
        output = self.root / 'prepare'
        folder = output / 'objects/object_001'
        original_receipt = folder / 'input/source_receipt.json'
        matte_receipt = folder / 'trellis_alpha/source_receipt.json'
        pink_receipt = folder / 'trellis_input/source_receipt.json'
        stages = []

        def execute(command, log, gpu):
            command = [str(value) for value in command]
            target = Path(command[command.index('--output') + 1])
            if 'src.methods.colorpeel_ice.subject_alpha' in command:
                self.assertEqual(command[0], pipeline.RUNTIME['trellis_python'])
                self.assertEqual(Path(command[command.index('--source-receipt') + 1]), original_receipt)
                self.assertEqual(Path(command[command.index('--objects-json') + 1]), output / 'segmentation/objects.json')
                self.assertEqual(command[command.index('--object-id') + 1], 'object_001')
                target.mkdir()
                soft = alpha.point(lambda value: 254 if value else 0)
                soft_path, cutout = target / 'alpha512.png', target / 'cutout512.png'
                soft.save(soft_path)
                with Image.open(image) as opened:
                    rgba = opened.convert('RGBA')
                    rgba.putalpha(soft)
                    rgba.paste((0, 0, 0, 0), (0, 0, 512, 512), soft.point(lambda value: 0 if value else 255))
                    rgba.save(cutout)
                receipt = pipeline.read(original_receipt)
                locks = {**receipt['source_locks'], str(original_receipt): pipeline.sha256(original_receipt),
                         str(soft_path): pipeline.sha256(soft_path), str(cutout): pipeline.sha256(cutout)}
                self.write(matte_receipt, {**receipt, 'source_locks': locks,
                    'cutout_path': str(cutout), 'cutout_sha256': locks[str(cutout)],
                    'conditioning_alpha_path': str(soft_path), 'conditioning_alpha_sha256': locks[str(soft_path)],
                    'semantic_receipt_path': str(original_receipt), 'semantic_receipt_sha256': locks[str(original_receipt)]})
                stages.append('matte')
                return
            if 'experiments.subject_pink_input_repro_v1.prepare' in command:
                self.assertEqual(Path(command[command.index('--source-receipt') + 1]), matte_receipt)
                self.assertEqual(command[command.index('--hue-degrees') + 1], '0')
                prepare_pink(matte_receipt, target, 0)
                stages.append('pink')
                return
            target.mkdir(parents=True)
            if '--worker' in command:
                worker = command[command.index('--worker') + 1]
                if worker == 'discover':
                    self.write(target / 'categories.json', {'status': 'verified'})
                elif worker == 'segment':
                    self.write(target / 'objects.json', {'status': 'succeeded', 'failures': [],
                        'source': {'image_sha256': source_hash, 'canonical_image_sha256': source_hash},
                        'objects': [{'id': 'object_001', 'source_image': str(image),
                                     'object_mask': str(mask), 'vlm_input': str(image)}]})
                elif worker == 'initialize':
                    self.assertEqual(command[command.index('--image') + 1], str(image))
                    self.write(target / 'initializers.json', {'selected_branches': ['subject', 'color'],
                        'subject_category': 'spoon', 'unavailable_branches': []})
                else:
                    self.fail('unexpected training worker')
            elif 'experiments.subject_trellis_counterfactual_v1.run' in command:
                self.assertEqual(Path(command[command.index('--source-receipt') + 1]), pink_receipt)
                stages.append('trellis')
            elif 'src.methods.colorpeel_ice.single_image_subject' in command:
                self.assertEqual(Path(command[command.index('--inputs') + 1]), pink_receipt.parent)
                self.assertEqual(Path(command[command.index('--trellis-run') + 1]), folder / 'trellis')
                self.write(target / 'subject.json', {'rows': []})
                stages.append('subject')
            else:
                self.assertIn('single_image_cmt', ' '.join(command))
                stage = command[command.index('--stage') + 1]
                if stage == 'inputs':
                    self.assertEqual(Path(command[command.index('--prepared') + 1]), original_receipt.parent)
                    stages.append('color_original')
                elif stage == 'finalize':
                    self.write(target / 'datasets.json', {'branches': {'color': {'rows': []}}, 'failures': {}})

        with patch.object(pipeline, 'execute', side_effect=execute):
            result = pipeline.prepare(image, output, ['subject', 'color'], 0, pipeline.RUNTIME)
        state = pipeline.read(result)
        self.assertEqual(state['status'], 'prepared', state['objects'][0]['failures'])
        self.assertEqual(stages, ['matte', 'pink', 'trellis', 'subject', 'color_original'])
        self.assertFalse(state['training_started'])
        self.assertEqual(state['subject_conditioning_policy'], 'sam_semantics_biref_soft_alpha_pink/v1')
        conditioning = state['objects'][0]['trellis_conditioning']
        self.assertEqual(conditioning['source_receipt_sha256'], pipeline.sha256(pink_receipt))
        self.assertEqual(conditioning['source_receipt'], str(pink_receipt))
        original, pink = pipeline.read(original_receipt), pipeline.read(pink_receipt)
        self.assertNotEqual(original['cutout_sha256'], pink['cutout_sha256'])
        for key in ('image512_path', 'image512_sha256', 'mask_path', 'mask_sha256'):
            self.assertEqual(original[key], pink[key])
        with Image.open(pink['cutout_path']) as opened, Image.open(pink['conditioning_alpha_path']) as soft:
            self.assertEqual(opened.getchannel('A').tobytes(), soft.tobytes())
            self.assertNotEqual(opened.getchannel('A').tobytes(), alpha.tobytes())
        self.assertEqual({path: path.read_bytes() for path in original_bytes}, original_bytes)


if __name__ == '__main__':
    unittest.main()
