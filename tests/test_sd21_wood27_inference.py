"""Ensure the27-image contract reaches paired export and the shared Subject probes."""

from functools import partial
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.sd21_wood_branches_v1 import inference as shared, snapshots
from experiments.sd21_wood_subject27_v1 import contract, inference
from tests import test_sd21_wood_inference as fixtures


class SD21Wood27InferenceTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SD21WoodInferenceTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        dataset = self.root / 'source27' / 'dataset'
        dataset.mkdir(parents=True)
        image, mask = dataset / 'actual27.png', dataset / 'mask27.png'
        Image.new('RGB', (32, 32), 'green').save(image)
        Image.new('L', (32, 32), 255).save(mask)
        rows = [{'id': f'subject27_{number}', 'image': str(image), 'mask': str(mask),
                 'image_sha256': self.fixture.sha(image), 'mask_sha256': self.fixture.sha(mask),
                 'caption': f'a photo of <S*> spoon training example{number}'} for number in range(27)]
        manifest = dataset / 'training_assets_manifest.jsonl'
        manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.data = {'rows': 27, 'references': rows, 'source_locks': {
            str(path): self.fixture.sha(path) for path in (image, mask, manifest)}}
        for item in (patch.object(contract, 'DATASET', dataset),
                     patch.object(contract, 'SOURCE_ROOT', dataset.parent),
                     patch.object(contract, 'verify_dataset', return_value=self.data)):
            item.start()
            self.addCleanup(item.stop)

    def training_run(self, method='fullrank', smoke=False):
        # Keep the old fixture's72-image recipe intact, then write the actual new contract.
        run = self.fixture.training_run('subject', method, smoke)
        config = contract.build_config('subject', method, model_dir=self.fixture.model, smoke=smoke)
        (run / 'config.yaml').write_text(json.dumps(config))
        path = run / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['run'] = config['run']
        path.write_text(json.dumps(manifest))
        return run

    def test_wrapper_injects_new_contract_into_the_actual_export_callback(self):
        with patch.object(inference.shared, 'run', return_value={'sentinel': True}) as runner:
            result = inference.run(Path('new_training'), Path('new_output'), 'lora_r64', execute=True, smoke=True)
        self.assertEqual(result, {'sentinel': True})
        self.assertEqual(runner.call_args.args, ('subject', 'lora_r64', Path('new_training'), Path('new_output')))
        self.assertEqual(runner.call_args.kwargs['seeds'], tuple(range(30, 61)))
        self.assertTrue(runner.call_args.kwargs['execute'])
        self.assertTrue(runner.call_args.kwargs['smoke'])
        exporter = runner.call_args.kwargs['snapshot_exporter']
        self.assertIsInstance(exporter, partial)
        self.assertIs(exporter.func, snapshots.audit_export)
        self.assertIs(exporter.keywords['experiment_contract'], contract)

    def test_new_config_requires_injection_and_both_methods_export_four_pairs_with27_references(self):
        f = self.fixture
        with patch.dict(sys.modules, {'torch': f.torch}):
            for method in contract.METHODS:
                run = self.training_run(method)
                with self.subTest(method=method), self.assertRaisesRegex(ValueError, 'training recipe'):
                    snapshots.audit_export(run, 'subject', method, self.root / ('wrong_default_' + method))
                before = {str(path): f.sha(path) for path in run.rglob('*') if path.is_file()}
                result = snapshots.audit_export(run, 'subject', method, self.root / ('exports27_' + method),
                                                experiment_contract=contract)
                self.assertEqual(result['training_images'], 27)
                self.assertEqual(result['references'], self.data['references'])
                self.assertEqual(list(result['steps']), [400, 600, 800, 1000])
                self.assertEqual(result['training']['steps'], 1000)
                self.assertEqual(json.loads((run / 'config.yaml').read_text())['run']['study'], contract.STUDY)
                old_image = f.data['subject']['references'][0]['image']
                self.assertNotIn(old_image, result['source_locks'])
                self.assertTrue(set(self.data['source_locks']) <= set(result['source_locks']))
                for step, selection in result['steps'].items():
                    self.assertEqual((selection['step'], selection['steps'], selection['training_steps']), (step, step, 1000))
                    self.assertEqual(selection['embedding_dim'], 1024)
                    self.assertEqual(selection['method'], method)
                    self.assertTrue(selection['paired_embedding_finite'])
                    token = f.registry[str(Path(selection['checkpoint']) / '<S*>.bin')]['<S*>']
                    self.assertTrue(np.all(token.value == step / 1000))
                    snapshot = run / 'checkpoints' / f'checkpoint-{step}'
                    self.assertEqual(set(selection['source_snapshot_hashes']), {
                        str(snapshot / 'pytorch_model.bin'), str(snapshot / 'pytorch_model_1.bin')})
                    self.assertEqual(selection['final_snapshot_matches_export'], step == 1000)
                    snapshots.check_locks(selection['hashes'])
                self.assertEqual(before, {str(path): f.sha(path) for path in run.rglob('*') if path.is_file()})
        self.assertEqual(len(f.data['subject']['references']), 72)

    def test_full_wrapper_dry_run_reaches_new_contract_and_displays27_actual_references(self):
        f = self.fixture
        run = self.training_run('lora_r64')
        output = self.root / 'wrapper27_full'
        with patch.dict(sys.modules, {'torch': f.torch}), \
             patch.object(snapshots, 'audit_export', wraps=snapshots.audit_export) as export, \
             patch.object(contract, 'build_config', wraps=contract.build_config) as config, \
             patch.object(shared, 'load_pipeline') as load:
            proof = inference.run(run, output, 'lora_r64')
        self.assertIs(export.call_args.kwargs['experiment_contract'], contract)
        config.assert_called_once_with('subject', 'lora_r64', model_dir=f.model, smoke=False)
        load.assert_not_called()
        self.assertEqual((proof['status'], proof['training_images'], proof['expected_images'], proof['generated_images']),
                         ('dry_run_verified', 27, 1116, 0))
        self.assertEqual(proof['completed_steps'], [400, 600, 800, 1000])
        refs = json.loads((output / 'training_references.json').read_text())['images']
        self.assertEqual(len(refs), 27)
        self.assertEqual([{key: value for key, value in row.items() if key != 'review_image'} for row in refs],
                         self.data['references'])
        self.assertEqual(len(list((output / 'training_references').glob('*.png'))), 27)
        self.assertIn('27 actual training RGB images', (output / 'training_references.html').read_text(encoding='utf8'))
        self.assertTrue((output / 'training_references_sheet.png').is_file())
        for step in contract.STEPS:
            rows = [json.loads(line) for line in (output / f'step{step}/generation_manifest.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 279)
            self.assertEqual({row['step'] for row in rows}, {step})
            self.assertEqual({row['seed'] for row in rows}, set(range(30, 61)))
            self.assertTrue(all(row['branch'] == 'subject' and row['method'] == 'lora_r64' for row in rows))
            self.assertEqual(len({row['prompt'] for row in rows}), 9)
            self.assertTrue((output / f'ckpts/step{step}/receipt.json').is_file())

    def test_both_wrapper_smokes_load_only_subject_token_kv_and27_reference_images(self):
        f = self.fixture
        with patch.dict(sys.modules, f.modules()):
            for method in contract.METHODS:
                f.calls.clear()
                f.loads.clear()
                f.installs.clear()
                run = self.training_run(method, True)
                output = self.root / 'wrapper27_smoke' / method
                proof = inference.run(run, output, method, execute=True, smoke=True)
                self.assertEqual((proof['training_images'], proof['expected_images'], proof['generated_images']), (27, 1, 1))
                self.assertEqual(proof['completed_steps'], [2])
                self.assertEqual(len(f.calls), 1)
                tokens, prompt, kwargs = f.calls[0]
                self.assertEqual(tokens, ('<S*>',))
                self.assertEqual(kwargs['generator'].seed, 42)
                self.assertEqual((kwargs['num_inference_steps'], kwargs['guidance_scale']), (100, 3.5))
                self.assertEqual(f.installs[0][0], 'fullrank' if method == 'fullrank' else 'lora')
                mask = kwargs['cross_attention_kwargs']['modifier_token_mask']
                self.assertEqual((int(mask[0].sum()), int(mask[1].sum())), (0, 1))
                refs = json.loads((output / 'training_references.json').read_text())['images']
                self.assertEqual(len(refs), 27)
                self.assertTrue(all(Path(row['image']).name == 'actual27.png' for row in refs))
                record = json.loads((output / 'step2/generation_status.jsonl').read_text())
                self.assertTrue(record['subject_adapter_loaded'])
                self.assertFalse(record['material_adapter_loaded'])
                self.assertFalse(record['texture_adapter_loaded'])

    def test_formal_nine_prompt_grid_keeps_all31_seeds_and_four_saved_steps_per_method(self):
        for method in contract.METHODS:
            matrices = [shared.prompt_rows('subject', method, step) for step in contract.STEPS]
            self.assertEqual([len(rows) for rows in matrices], [279] * 4)
            self.assertEqual(sum(map(len, matrices)), 1116)
            for rows in matrices:
                self.assertEqual(len({row['id'] for row in rows}), 279)
                self.assertEqual({row['seed'] for row in rows}, set(range(30, 61)))
                self.assertEqual(len({row['prompt'] for row in rows}), 9)
                self.assertTrue(all(row['prompt'].count('<S*>') == 1 and '<T*>' not in row['prompt']
                                    and '<M*>' not in row['prompt'] for row in rows))


if __name__ == '__main__':
    unittest.main()
