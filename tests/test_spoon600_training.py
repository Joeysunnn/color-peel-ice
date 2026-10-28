"""CPU checks for exact600-step spoon recipes and shared reporting."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.subject_multicolor_multimaterial_b import training as baseline
from experiments.subject_trellis_spoons_v1 import training as previous
from experiments.subject_trellis_spoons600_v1 import training


class Spoon600TrainingTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='spoon600-training-', dir=os.environ.get('TMP'))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.dataset = self.root / 'dataset'
        self.dataset.mkdir()

    def prepare(self, subject='metal_spoon', count=72):
        audit = {'subject_id': subject, 'rows': count}
        for name, value in (('dataset_audit.json', audit), ('concepts.json', []),
                ('source_locks.json', {}), ('training_assets_manifest.jsonl', {})):
            baseline._write(self.dataset / name, value)
        return audit

    def test_both_subjects_and_methods_only_change_the_training_dose(self):
        for subject in ('metal_spoon', 'wood_spoon'):
            self.prepare(subject)
            for method in training.METHODS:
                old = previous.build_config(method, self.dataset)
                config = training.build_config(method, self.dataset)
                self.assertEqual({key for key in old['args'] if old['args'][key] != config['args'][key]},
                    {'max_train_steps', 'checkpoint_steps'})
                source = baseline.read_json(baseline.BASELINES[method])
                self.assertEqual({key for key in source['args'] if source['args'][key] != config['args'][key]},
                    baseline.ARG_CHANGES | {'initializer_token'})
                self.assertEqual(config['args']['max_train_steps'], 600)
                self.assertEqual(config['args']['checkpoint_steps'], [200, 400, 600])
                self.assertEqual(config['args']['initializer_token'], 'spoon')
                self.assertEqual(config['args']['modifier_token'], '<S*>')
                self.assertEqual(config['subject_id'], subject)
                self.assertEqual(config['run']['seed'], 42)
                self.assertEqual(config['run']['study'], training.STUDY)
                self.assertTrue(config['run']['variant'].endswith('_steps600'))
                if method == 'lora_r64':
                    self.assertEqual((config['args']['subject_lora_rank'], config['args']['subject_lora_alpha']), (64, 64))

    def test_the600_recipe_rejects_all180(self):
        self.prepare(count=180)
        with self.assertRaisesRegex(ValueError, 'blue/green72'):
            training.build_config('fullrank', self.dataset)

    def test_config_comparison_records_actual600_snapshots_and_old_default(self):
        audit = self.prepare()
        with patch.object(training, 'verify_dataset', return_value=audit):
            bundle = training.build_configuration_bundle(self.dataset, self.root / 'new-plan')
        self.assertEqual(bundle['comparison']['checkpoint_steps'], [200, 400, 600])
        for method in training.METHODS:
            self.assertEqual(set(bundle['comparison']['comparison'][method]['args_diff']), training.ARG_CHANGES)
            self.assertEqual(baseline.read_config(Path(bundle['paths'][method])), bundle['configs'][method])
        old = baseline.build_configuration_bundle(self.dataset, self.root / 'old-plan', verify_dataset=lambda _: audit)
        self.assertEqual(old['comparison']['checkpoint_steps'], [500, 750, 1000])

    def fake_run(self, steps, *, old_default=False, execute=True):
        audit = self.prepare('wood_spoon')
        captures = []

        def launch(command, **kwargs):
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '2')
            if command[-1] == '--dry-run':
                output = Path(command[command.index('--run-dir') + 1])
                output.mkdir(parents=True)
                (output / 'config.yaml').write_bytes(Path(command[command.index('--config') + 1]).read_bytes())
                baseline._write(output / 'manifest.json', {'status': 'dry-run'})
            return SimpleNamespace(returncode=0)

        def audit_run(recipe, run, config):
            captures.append((recipe, config))
            self.assertEqual(config['args']['max_train_steps'], steps)
            self.assertEqual(config['args']['checkpoint_steps'], [500, 750, 1000] if old_default else [200, 400, 600])
            return {'status': 'verified', 'steps': steps}

        output = self.root / 'run'
        with patch.object(training, 'verify_dataset', return_value=audit), \
             patch.object(baseline, 'ROOT', self.root / 'code'), \
             patch.object(baseline, 'git_info', return_value={'commit': 'a' * 40}), \
             patch.object(baseline, 'resolve_run_dir', side_effect=lambda path: path.resolve()), \
             patch.object(baseline.shared, 'verify_model', return_value={}), \
             patch.object(baseline.subprocess, 'run', side_effect=launch), \
             patch.object(baseline.shared, 'audit_run', side_effect=audit_run):
            if old_default:
                result = baseline.run(self.dataset, output, 'lora_r64', 2, execute,
                    verify_dataset=lambda _: audit)
            else:
                result = training.run(self.dataset, output, 'lora_r64', 2, execute)
        return result, output, captures

    def test_shared_training_executes_and_reports600(self):
        state, output, captures = self.fake_run(600)
        self.assertEqual((state['status'], state['max_steps'], state['completed_steps']), ('training_verified', 600, 600))
        self.assertEqual(state['checkpoint_steps'], [200, 400, 600])
        self.assertEqual(captures[0][0]['concept_id'], 'wood_spoon')
        summary = (output / 'train_outputs/SUMMARY.md').read_text()
        self.assertIn('600/600', summary)
        self.assertNotIn('1000', summary)
        self.assertEqual(json.loads((output / 'train_outputs/status.json').read_text())['max_steps'], 600)

    def test_shared_dry_run_reports600_without_training(self):
        state, output, captures = self.fake_run(600, execute=False)
        self.assertEqual((state['status'], state['max_steps'], state['completed_steps']), ('dry_run_verified', 600, 0))
        self.assertFalse(captures)
        self.assertIn('0/600', (output / 'train_outputs/SUMMARY.md').read_text())

    def test_shared_old_default_still_executes_and_reports1000(self):
        state, output, captures = self.fake_run(1000, old_default=True)
        self.assertEqual((state['max_steps'], state['completed_steps']), (1000, 1000))
        self.assertEqual(state['checkpoint_steps'], [500, 750, 1000])
        self.assertEqual(captures[0][0]['concept_id'], 'mailbox')
        self.assertIn('1000/1000', (output / 'train_outputs/SUMMARY.md').read_text())


if __name__ == '__main__':
    unittest.main()
