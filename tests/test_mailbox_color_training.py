"""Check Color configuration, launcher audit routing, and real paired C exports."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.mailbox_color_sd15_v1 import training
from experiments.subject_multicolor_multimaterial_b import training as baseline


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.shape, self.ndim = self.value.shape, self.value.ndim

    def __getitem__(self, index):
        return Tensor(self.value[index])

    def clone(self):
        return Tensor(self.value.copy())

    def contiguous(self):
        return self


class MailboxColorTrainingTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='mailbox-color-training-', dir=os.environ.get('TMP'))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.dataset = self.root / 'dataset'
        self.dataset.mkdir()
        for name in ('concepts.json', 'training_assets_manifest.jsonl', 'dataset_audit.json', 'source_locks.json'):
            (self.dataset / name).write_text('{}', encoding='utf8')

    def test_both_configs_change_only_declared_arguments_and_leave_baselines_unchanged(self):
        hashes = {method: training.sha256(path) for method, path in baseline.BASELINES.items()}
        for method in training.METHODS:
            original = baseline.read_json(baseline.BASELINES[method])
            config = training.build_config(method, self.dataset)
            args = config['args']
            differences = {key for key in set(original['args']) | set(args)
                           if original['args'].get(key) != args.get(key)}
            self.assertEqual(differences, training.ARG_CHANGES)
            self.assertEqual(len(differences), 8)
            self.assertEqual(args['modifier_token'], '<C*>')
            self.assertEqual(args['initializer_token'], 'orange')
            self.assertEqual(args['max_train_steps'], 1000)
            self.assertEqual(args['checkpoint_steps'], [400, 600, 800, 1000])
            self.assertEqual(args['checkpointing_steps'], 200)
            self.assertEqual(args['adam_weight_decay'], 0)
            self.assertEqual(args['cos_weight'], 0)
            self.assertEqual(args['freeze_model'], 'crossattn_kv')
            self.assertTrue(args['hflip'])
            self.assertEqual(config['run']['seed'], 42)
            if method == 'fullrank':
                self.assertTrue(args['token_local_kv'])
                self.assertEqual(args['kv_learning_rate'], 1e-5)
            else:
                self.assertEqual(args['subject_lora_mode'], 'token_local_kv')
                self.assertEqual((args['subject_lora_rank'], args['subject_lora_alpha']), (64, 64))
                self.assertEqual(args['kv_learning_rate'], 5e-5)
            self.assertEqual(training.sha256(baseline.BASELINES[method]), hashes[method])
        with self.assertRaisesRegex(ValueError, 'unsupported Color'):
            training.build_config('unsupported', self.dataset)

    def test_configuration_bundle_preserves_exact_argument_comparison(self):
        with patch.object(training.data, 'verify_dataset', return_value={'rows': 9}):
            bundle = training.build_configuration_bundle(self.dataset, self.root / 'plan')
        self.assertEqual(bundle['comparison']['checkpoint_steps'], [400, 600, 800, 1000])
        for method in training.METHODS:
            self.assertEqual(set(bundle['comparison']['comparison'][method]['args_diff']), training.ARG_CHANGES)
            self.assertEqual(training.read_config(Path(bundle['paths'][method])), bundle['configs'][method])

    def test_real_shared_run_audits_color_recipe_and_indexes_dataset(self):
        calls = []

        def launch(command, **kwargs):
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '2')
            if command[-1] == '--dry-run':
                output = Path(command[command.index('--run-dir') + 1])
                output.mkdir(parents=True)
                (output / 'config.yaml').write_bytes(Path(command[command.index('--config') + 1]).read_bytes())
                baseline._write(output / 'manifest.json', {'status': 'dry-run'})
            return SimpleNamespace(returncode=0)

        def audit(recipe, run, config):
            calls.append(recipe)
            self.assertEqual(recipe, training.recipe('lora_r64'))
            self.assertEqual(config['args']['modifier_token'], '<C*>')
            return {**recipe, 'status': 'verified', 'steps': 1000}

        output = self.root / 'training'
        with patch.object(training.data, 'verify_dataset', return_value={'rows': 9}), \
             patch.object(baseline, 'ROOT', self.root / 'code'), \
             patch.object(baseline, 'git_info', return_value={'commit': 'a' * 40}), \
             patch.object(baseline, 'resolve_run_dir', side_effect=lambda path: path.resolve()), \
             patch.object(baseline.shared, 'verify_model', return_value={}), \
             patch.object(baseline.subprocess, 'run', side_effect=launch), \
             patch.object(baseline.shared, 'audit_run', side_effect=audit):
            result = training.run(self.dataset, output, 'lora_r64', 2, True)
        self.assertEqual((result['status'], result['completed_steps']), ('training_verified', 1000))
        self.assertEqual(calls, [training.recipe('lora_r64')])
        index = baseline.read_json(output / 'training_index.json')
        self.assertEqual(index['dataset'], str(self.dataset))
        self.assertEqual(index['runs'][0]['branch'], 'color')
        self.assertEqual(index['runs'][0]['token'], '<C*>')

    def snapshot_fixture(self, step, method='fullrank'):
        run = self.root / f'run-{method}-{step}'
        checkpoint = run / 'checkpoints'
        directory = checkpoint / f'checkpoint-{step}'
        directory.mkdir(parents=True)
        config = training.build_config(method, self.dataset)
        baseline._write(run / 'config.yaml', config)
        for name in training.snapshots.STATE_FILES:
            (directory / name).write_bytes(b'actual snapshot file')
        baseline._write(checkpoint / 'embedding_update_audit.json', {
            'observed_optimization_steps': 1000,
            'modifier_tokens': [{'token': '<C*>', 'token_id': 1, 'exposure_steps': 1000,
                'nonzero_gradient_steps': 1000, 'initial_final_l2_delta': 1.0}]})
        value = step / 1000
        adapter = {'delta': Tensor(np.full((2, 768), value))}
        table = Tensor(np.stack([np.zeros(768), np.full(768, value)]))
        final = {'delta': Tensor(np.ones((2, 768)))}
        empty = {'self.processor': {}} if method == 'fullrank' else {}
        final.update(empty)
        final_token = {'<C*>': Tensor(np.ones(768))}
        weight = 'pytorch_token_local_kv_weights.bin' if method == 'fullrank' else 'pytorch_lora_kv_weights.bin'
        loads, exported = [], []

        def load(path, **kwargs):
            name = Path(path).name
            loads.append(name)
            return {'pytorch_model.bin': adapter, 'pytorch_model_1.bin': {training.snapshots.EMBEDDING_KEY: table},
                    weight: final, '<C*>.bin': final_token}[name]

        def write_export(torch, actual_adapter, embedding, actual_empty, selection, actual_step, record, output):
            output.mkdir()
            exported.append((actual_step, actual_adapter, embedding))
            return {'step': actual_step, 'hashes': {}}

        torch = SimpleNamespace(Tensor=Tensor, load=load, isfinite=lambda item: np.isfinite(item.value),
                                equal=lambda left, right: np.array_equal(left.value, right.value))
        selection = {**training.recipe(method), 'status': 'verified', 'steps': 1000,
            'run': str(run), 'checkpoint': str(checkpoint), 'weight': weight,
            'hashes': {str(run / 'config.yaml'): training.sha256(run / 'config.yaml')}}
        return selection, torch, final_token, loads, exported, write_export, empty

    def export_fixture(self, fixture, step, output):
        selection, torch, _, _, _, writer, empty = fixture
        with patch.dict('sys.modules', {'torch': torch}), \
             patch.object(training.snapshots, 'adapter_schema', return_value=({'delta': (2, 768)}, empty)), \
             patch.object(training.snapshots, '_write_export', side_effect=writer), \
             patch.object(training.snapshots, '_embedding_audit', wraps=training.snapshots._embedding_audit) as audit:
            result = training.export_step(selection, step, output)
        self.assertEqual(audit.call_args.args[1:], (1000, '<C*>'))
        return result

    def test_real_color_export_pairs_all_four_steps_for_both_methods(self):
        for method in training.METHODS:
            for step in training.STEPS:
                with self.subTest(method=method, step=step):
                    fixture = self.snapshot_fixture(step, method)
                    result = self.export_fixture(fixture, step, self.root / f'export-{method}-{step}')
                    self.assertEqual((result['steps'], result['training_steps']), (step, 1000))
                    self.assertTrue(result['paired_embedding_finite'])
                    exported = fixture[4][0]
                    self.assertEqual(exported[0], step)
                    np.testing.assert_array_equal(exported[2].value, np.full(768, step / 1000))
                    self.assertEqual('<C*>.bin' in fixture[3], step == 1000)

    def test_color_export_rejects_subject_selection_wrong_steps_and_tampered_locks(self):
        fixture = self.snapshot_fixture(400)
        selection, torch = fixture[:2]
        with patch.dict('sys.modules', {'torch': torch}):
            for invalid in (200, 500, 750, 400.0, True):
                with self.subTest(step=invalid), self.assertRaisesRegex(ValueError, 'audited Color'):
                    training.export_step(selection, invalid, self.root / 'invalid')
            for key, value in (('token', '<S*>'), ('branch', 'subject'), ('status', 'pending'), ('steps', 600)):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'audited Color'):
                    training.export_step({**selection, key: value}, 400, self.root / 'invalid')
            Path(selection['run'], 'config.yaml').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'source artifact changed'):
                training.export_step(selection, 400, self.root / 'invalid')

    def test_final_color_export_rejects_wrong_final_embedding(self):
        fixture = self.snapshot_fixture(1000)
        fixture[2]['<C*>'] = Tensor(np.zeros(768))
        with self.assertRaisesRegex(ValueError, 'final Color snapshot differs'):
            self.export_fixture(fixture, 1000, self.root / 'bad-final')


if __name__ == '__main__':
    unittest.main()
