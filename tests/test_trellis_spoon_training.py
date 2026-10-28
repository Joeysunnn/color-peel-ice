"""CPU checks for frozen spoon recipes, subject audits and transfer prompts."""

import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from experiments.subject_marble_color30_v1 import inference as shared_inference
from experiments.subject_multicolor_multimaterial_b import training as baseline
from experiments.subject_trellis_dataset_v1 import inference as previous
from experiments.subject_trellis_spoons_v1 import inference, training


class TrellisSpoonTrainingTest(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parents[1]
        self.temporary = tempfile.TemporaryDirectory(prefix='test-spoon-training-', dir=workspace)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / 'dataset'
        self.dataset.mkdir()

    def prepare(self, subject='metal_spoon', count=180):
        audit = {'subject_id': subject, 'rows': count}
        for name, value in (('dataset_audit.json', audit), ('concepts.json', []),
                            ('source_locks.json', {}), ('training_assets_manifest.jsonl', {})):
            baseline._write(self.dataset / name, value)
        return audit

    def test_all_four_configs_only_change_data_dose_and_initialization(self):
        for subject in ('metal_spoon', 'wood_spoon'):
            for count in (180, 72):
                self.prepare(subject, count)
                for method in training.METHODS:
                    source = baseline.read_json(baseline.BASELINES[method])
                    config = training.build_config(method, self.dataset)
                    differences = {key for key in source['args'] if source['args'][key] != config['args'][key]}
                    self.assertEqual(differences, baseline.ARG_CHANGES | {'initializer_token'})
                    self.assertEqual(config['subject_id'], subject)
                    self.assertEqual(config['args']['initializer_token'], 'spoon')
                    self.assertEqual(config['args']['modifier_token'], '<S*>')
                    self.assertEqual(config['args']['max_train_steps'], 1000)
                    self.assertEqual(config['args']['checkpoint_steps'], [500, 750, 1000])
                    self.assertEqual(config['run']['study'], training.STUDY)
                    self.assertIn(subject, config['run']['variant'])
                    self.assertIn('blue_green72' if count == 72 else 'all180', config['run']['variant'])
                    self.assertEqual(config['run']['seed'], 42)
                    if method == 'lora_r64':
                        self.assertEqual(config['args']['subject_lora_rank'], 64)
                        self.assertEqual(config['args']['subject_lora_alpha'], 64)
                self.assertEqual(baseline.build_config('fullrank', self.dataset)['args']['initializer_token'], 'mailbox')

    def test_bundle_requires_explicit_initializer_exception_and_rejects_optimizer_changes(self):
        audit = self.prepare()
        with self.assertRaisesRegex(ValueError, 'allowed'):
            baseline.build_configuration_bundle(self.dataset, self.root / 'rejected',
                verify_dataset=lambda _: audit, config_builder=training.build_config)
        self.assertFalse((self.root / 'rejected').exists())
        with patch.object(training, 'verify_dataset', return_value=audit):
            bundle = training.build_configuration_bundle(self.dataset, self.root / 'accepted')
        for method in training.METHODS:
            self.assertEqual(set(bundle['comparison']['comparison'][method]['args_diff']), training.ARG_CHANGES)
            self.assertEqual(baseline.read_config(Path(bundle['paths'][method])), bundle['configs'][method])
        def changed(method, dataset):
            config = training.build_config(method, dataset)
            config['args']['learning_rate'] *= 2
            return config
        with self.assertRaisesRegex(ValueError, 'allowed'):
            baseline.build_configuration_bundle(self.dataset, self.root / 'optimizer-change',
                verify_dataset=lambda _: audit, config_builder=changed, allowed_arg_changes=training.ARG_CHANGES)

    def test_full_training_hook_audits_the_actual_spoon_identity(self):
        audit = self.prepare('wood_spoon', 72)
        captured = []
        def launch(command, **kwargs):
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '2')
            if command[-1] == '--dry-run':
                output = Path(command[command.index('--run-dir') + 1])
                output.mkdir(parents=True)
                (output / 'config.yaml').write_bytes(Path(command[command.index('--config') + 1]).read_bytes())
                baseline._write(output / 'manifest.json', {'status': 'dry-run'})
            return types.SimpleNamespace(returncode=0)
        def verify(recipe, run, config):
            captured.append(recipe)
            self.assertEqual(config['args']['initializer_token'], 'spoon')
            return {'status': 'verified', 'steps': 1000}
        with patch.object(training, 'verify_dataset', return_value=audit), \
             patch.object(baseline, 'ROOT', self.root / 'project'), \
             patch.object(baseline, 'git_info', return_value={'commit': 'a' * 40}), \
             patch.object(baseline, 'resolve_run_dir', side_effect=lambda path: path.resolve()), \
             patch.object(baseline.shared, 'verify_model', return_value={}), \
             patch.object(baseline.subprocess, 'run', side_effect=launch), \
             patch.object(baseline.shared, 'audit_run', side_effect=verify):
            result = training.run(self.dataset, self.root / 'training', 'lora_r64', 2, True)
        self.assertEqual(result['status'], 'training_verified')
        self.assertEqual(captured, [{'method': 'lora_r64', 'branch': 'subject',
                                    'concept_id': 'wood_spoon', 'token': '<S*>'}])

    def test_probes_match_previous_rows_except_subject_noun(self):
        for step in inference.STEPS:
            for builder, old_builder, count in ((inference.prompt_rows, previous.prompt_rows, 20),
                    (inference.blue_green_prompt_rows, previous.blue_green_prompt_rows, 36)):
                actual, old = builder(step), old_builder(step)
                self.assertEqual(len(actual), count)
                self.assertEqual(len({row['id'] for row in actual}), count)
                self.assertEqual(len({row['image_path'] for row in actual}), count)
                for row, before in zip(actual, old):
                    self.assertEqual(row, {**before, 'prompt': before['prompt'].replace('<S*> mailbox', '<S*> spoon')})
                    self.assertNotIn('mailbox', row['prompt'])
                    self.assertEqual(row['prompt'].count('<S*>'), 1)
        self.assertEqual(inference.SAMPLING, previous.SAMPLING)

    def test_inference_routes_both_sizes_to_the_spoon_probes(self):
        for subject in ('metal_spoon', 'wood_spoon'):
            for count, outputs in ((180, 20), (72, 36)):
                self.prepare(subject, count)
                self.assertEqual(inference.expected_images(self.dataset), outputs)
                with patch.object(shared_inference, 'run', return_value={}) as run:
                    inference.run(self.dataset, self.root / f'{subject}-{count}', 750)
                self.assertIs(run.call_args.kwargs['training_module'], training)
                self.assertEqual(run.call_args.kwargs['reference_count'], count)
                self.assertEqual(len(run.call_args.kwargs['probe_rows'](750)), outputs)
                self.assertIn(subject, run.call_args.kwargs['reference_label'])

    def test_snapshot_selection_audit_uses_the_actual_spoon_identity(self):
        self.prepare('metal_spoon', 180)
        selections = []
        configs = {}
        for method in training.METHODS:
            output = self.root / method
            output.mkdir()
            config = training.build_config(method, self.dataset)
            configs[method] = config
            baseline._write(output / 'config.yaml', config)
            selections.append({'method': method, 'run': str(output), 'status': 'verified',
                'token': '<S*>', 'steps': 1000, 'branch': 'subject', 'hashes': {'fake': 'lock'}})
        index = self.root / 'index.json'
        baseline._write(index, {'git': {'commit': 'test'}, 'runs': selections})
        recipes = []
        def audit(recipe, run, expected):
            recipes.append(recipe)
            self.assertEqual(expected, configs[recipe['method']])
            return next(item for item in selections if item['method'] == recipe['method'])
        with patch.object(shared_inference, 'git_info', return_value={'commit': 'test'}), \
             patch.object(shared_inference, '_locks'), \
             patch.object(shared_inference.shared, 'audit_run', side_effect=audit):
            actual = shared_inference._selections(index, self.dataset, training_module=training)
        self.assertEqual(actual, selections)
        self.assertEqual({recipe['concept_id'] for recipe in recipes}, {'metal_spoon'})


if __name__ == '__main__':
    unittest.main()
