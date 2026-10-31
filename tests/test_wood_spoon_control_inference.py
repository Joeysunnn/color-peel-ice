"""CPU controls for native, single-S, single-T and dual S+T routing at seed42."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

from experiments.wood_spoon_composition_v1 import control_inference as controls
from tests import test_wood_spoon_composition_inference as fixtures


class WoodSpoonControlInferenceTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.WoodSpoonCompositionInferenceTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.texture500_audit()
        self.root, self.audit_path = self.fixture.root, self.fixture.audit_path

    def test_exact_eight_prompts_and_loaded_branch_flags(self):
        expected = {
            'mug_without_t': 'a photo of a mug',
            'mug_with_t': 'a photo of a mug with <T*> texture',
            'spoon_without_t': 'a photo of a spoon',
            'spoon_with_t': 'a photo of a spoon with <T*> texture',
            'wood_spoon_without_t': 'a photo of a wood spoon',
            'wood_spoon_with_t': 'a photo of a wood spoon with <T*> texture',
            'subject_without_t': 'a photo of <S*> spoon',
            'subject_with_t': 'a photo of <S*> spoon with <T*> texture',
        }
        self.assertEqual({key: value['prompt'] for key, value in controls.CASES.items()}, expected)
        self.assertEqual(controls.SAMPLING, {**controls.base.SAMPLING, 'seeds': [42]})
        for case, prompt in expected.items():
            row = controls.prompt_row(case)
            self.assertEqual((row['id'], row['case_id'], row['method'], row['seed']), (case, case, 'lora_r64', 42))
            self.assertEqual(row['prompt'], prompt)
            self.assertEqual(row['subject_adapter_loaded'], '<S*>' in prompt)
            self.assertEqual(row['texture_adapter_loaded'], '<T*>' in prompt)
            self.assertEqual(row['attribute_adapter_loaded'], '<T*>' in prompt)
            self.assertEqual(row['subject_step'], 600 if '<S*>' in prompt else None)
            self.assertEqual(row['attribute_step'], 500 if '<T*>' in prompt else None)
        with self.assertRaises(ValueError):
            controls.prompt_row('unrequested_control')

    def test_real_shared_pipeline_routes_all_eight_controls_and_copies_only_loaded_references(self):
        f = self.fixture
        subject = f.audit['sources']['subject']['lora_r64']
        texture = f.audit['sources']['texture']['lora_r64']
        with patch.dict(sys.modules, f.modules()), \
             patch.object(controls.base.evaluate, 'read_json', side_effect=f.model_json):
            for case, spec in controls.CASES.items():
                with self.subTest(case=case):
                    f.calls.clear()
                    f.loads.clear()
                    f.installs.clear()
                    f.embeddings.clear()
                    output = self.root / 'controls' / case
                    proof = controls.run(self.audit_path, output, case, 'cpu', True)
                    selected = ([subject] if spec['use_subject'] else []) + ([texture] if spec['use_texture'] else [])
                    tokens = [item['token'] for item in selected]
                    self.assertEqual(f.loads, [str(Path(item['checkpoint']) / item['weight']) for item in selected])
                    self.assertEqual(f.embeddings, [(item['checkpoint'], item['token'] + '.bin') for item in selected])
                    self.assertEqual(len(f.installs), int(bool(selected)))
                    if selected:
                        self.assertEqual(f.installs[0][0], 'dual_lora' if len(selected) == 2 else 'single_lora')
                    self.assertEqual(len(f.calls), 1)
                    loaded, prompt, kwargs = f.calls[0]
                    self.assertEqual(loaded, tuple(token + '.bin' for token in tokens))
                    self.assertEqual(prompt, spec['prompt'])
                    self.assertEqual((kwargs['generator'], kwargs['num_inference_steps'], kwargs['guidance_scale'],
                                      kwargs['width'], kwargs['height']), (42, 100, 3.5, 512, 512))
                    if not selected:
                        self.assertNotIn('cross_attention_kwargs', kwargs)
                    else:
                        mask = kwargs['cross_attention_kwargs']['modifier_token_mask']
                        if len(selected) == 2:
                            self.assertEqual(set(mask), {'subject', 'material'})
                            self.assertFalse(np.any(mask['subject'] & mask['material']))
                            masks = mask.values()
                        else:
                            self.assertNotIsInstance(mask, dict)
                            masks = [mask]
                        for mask in masks:
                            self.assertEqual(mask.dtype, np.bool_)
                            self.assertEqual(mask.shape, (2, 77))
                            self.assertEqual((int(mask[0].sum()), int(mask[1].sum())), (0, 1))
                    self.assertEqual((proof['status'], proof['expected_images'], proof['generated_images']), ('succeeded', 1, 1))
                    self.assertEqual(proof['subject_training'], subject if spec['use_subject'] else None)
                    self.assertEqual(proof['texture_training'], texture if spec['use_texture'] else None)
                    expected_counts = {name: count for name, count, enabled in
                        (('subject', 72, spec['use_subject']), ('texture', 16, spec['use_texture'])) if enabled}
                    self.assertEqual(proof['training_images'], expected_counts)
                    refs = json.loads((output / 'training_references.json').read_text())['references']
                    self.assertEqual({name: len(items) for name, items in refs.items()}, expected_counts)
                    self.assertEqual(len(list(output.glob('training_references/*/*.png'))), sum(expected_counts.values()))
                    if not selected:
                        self.assertFalse((output / 'training_references').exists())
                    manifest = [json.loads(line) for line in (output / 'generation_manifest.jsonl').read_text().splitlines()]
                    self.assertEqual(manifest, [controls.prompt_row(case)])
                    records = [json.loads(line) for line in (output / 'generation_status.jsonl').read_text().splitlines()]
                    self.assertEqual(len(records), 1)
                    record = records[0]
                    self.assertFalse(record['copied'])
                    self.assertEqual(record['subject_adapter_loaded'], spec['use_subject'])
                    self.assertEqual(record['texture_adapter_loaded'], spec['use_texture'])
                    self.assertEqual(record['attribute_adapter_loaded'], spec['use_texture'])
                    self.assertEqual(record['image_sha256'], controls.base.sha256(output / record['image_path']))
                    self.assertTrue((output / 'index.html').is_file())

    def test_dry_run_requires_source_audit_but_never_loads_a_model(self):
        output = self.root / 'dry_run'
        with patch.object(controls.base.evaluate, 'load_pipeline') as load:
            proof = controls.run(self.audit_path, output, 'spoon_without_t')
        self.assertEqual((proof['status'], proof['generated_images']), ('dry_run', 0))
        self.assertEqual(proof['training_images'], {})
        self.assertFalse((output / 'generation_status.jsonl').exists())
        load.assert_not_called()

    def test_even_native_control_rejects_wrong_source_step_or_changed_adapter_before_loading(self):
        texture = self.fixture.audit['sources']['texture']['lora_r64']
        texture['steps'] = 1000
        controls.base.write(self.audit_path, self.fixture.audit)
        with patch.object(controls.base.evaluate, 'load_pipeline') as load:
            with self.assertRaisesRegex(ValueError, 'attribute500'):
                controls.run(self.audit_path, self.root / 'wrong_step', 'spoon_without_t', 'cpu', True)
            texture['steps'] = 500
            controls.base.write(self.audit_path, self.fixture.audit)
            (Path(texture['checkpoint']) / texture['weight']).write_text('changed after audit')
            with self.assertRaisesRegex(ValueError, 'source locks'):
                controls.run(self.audit_path, self.root / 'changed_adapter', 'spoon_with_t', 'cpu', True)
            load.assert_not_called()

    def test_fresh_output_stays_outside_all_source_runs_and_checkpoints(self):
        texture = self.fixture.audit['sources']['texture']['lora_r64']
        for output in (Path(texture['run']) / 'control', Path(texture['checkpoint']) / 'control', self.root / 'sources'):
            with self.subTest(output=output), self.assertRaises(ValueError):
                controls.run(self.audit_path, output, 'spoon_without_t')
        existing = self.root / 'existing'
        existing.mkdir()
        with self.assertRaises(ValueError):
            controls.run(self.audit_path, existing, 'spoon_without_t')

    def test_failure_is_retained_and_not_retried(self):
        output = self.root / 'failed'
        with patch.dict(sys.modules, self.fixture.modules()), \
             patch.object(controls.base.evaluate, 'load_pipeline', side_effect=RuntimeError('CUDA failed')) as load:
            with self.assertRaisesRegex(RuntimeError, 'CUDA failed'):
                controls.run(self.audit_path, output, 'subject_without_t', 'cpu', True)
        self.assertEqual(load.call_count, 1)
        proof = json.loads((output / 'provenance.json').read_text())
        self.assertEqual((proof['status'], proof['generated_images']), ('failed', 0))
        self.assertTrue((output / 'index.html').is_file())


if __name__ == '__main__':
    unittest.main()
