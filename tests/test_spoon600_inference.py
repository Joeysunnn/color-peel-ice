"""Check real600-step snapshot validation and inherited spoon transfer probes."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.subject_marble_color30_v1 import inference as shared
from experiments.subject_multicolor_multimaterial_b import inference as snapshots
from experiments.subject_trellis_spoons_v1 import inference as previous
from experiments.subject_trellis_spoons600_v1 import inference, training


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.shape, self.ndim = self.value.shape, self.value.ndim

    def __getitem__(self, item):
        return Tensor(self.value[item])

    def clone(self):
        return Tensor(self.value.copy())

    def contiguous(self):
        return self


class Spoon600InferenceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='spoon600-inference-', dir=os.environ.get('TMP'))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def test_all_three_prompt_matrices_only_change_the_checkpoint_step(self):
        old = previous.blue_green_prompt_rows(500)
        for step in (200, 400, 600):
            rows = inference.prompt_rows(step)
            self.assertEqual(rows, [{**row, 'step': step} for row in old])
            self.assertEqual(len(rows), 36)
            self.assertEqual(len({row['id'] for row in rows}), 36)
            self.assertEqual(len({row['image_path'] for row in rows}), 36)
            self.assertTrue(all('<S*> spoon' in row['prompt'] for row in rows))
        for invalid in (500, 750, 1000, 200.0, True):
            with self.subTest(step=invalid), self.assertRaises(ValueError):
                inference.prompt_rows(invalid)
        self.assertEqual(inference.SAMPLING, previous.SAMPLING)

    def snapshot_fixture(self, total, step):
        run = self.root / f'run-{total}-{step}'
        checkpoint = run / 'checkpoints'
        directory = checkpoint / f'checkpoint-{step}'
        directory.mkdir(parents=True)
        plan = [200, 400, 600] if total == 600 else [500, 750, 1000]
        config = {'stage': 'train', 'run': {'study': 'cpu_snapshot', 'variant': 'paired', 'seed': 42},
            'args': {'max_train_steps': total, 'checkpoint_steps': plan}}
        snapshots.training._write(run / 'config.yaml', config)
        for name in snapshots.snapshots.STATE_FILES:
            (directory / name).write_bytes(b'paired snapshot file')
        record = {'observed_optimization_steps': total, 'modifier_tokens': [
            {'token': '<S*>', 'token_id': 1, 'exposure_steps': total,
             'nonzero_gradient_steps': total, 'initial_final_l2_delta': 1.0}]}
        snapshots.training._write(checkpoint / 'embedding_update_audit.json', record)
        adapter = {'delta': Tensor(np.full((2, 768), step / total))}
        table = Tensor(np.stack([np.zeros(768), np.full(768, step / total)]))
        final = {'delta': Tensor(np.ones((2, 768))), 'self.processor': {}}
        final_token = {'<S*>': Tensor(np.ones(768))}
        weight = 'pytorch_token_local_kv_weights.bin'
        loads = []
        def load(path, **kwargs):
            name = Path(path).name
            loads.append(name)
            return {'pytorch_model.bin': adapter, 'pytorch_model_1.bin': {snapshots.snapshots.EMBEDDING_KEY: table},
                weight: final, '<S*>.bin': final_token}[name]
        torch = SimpleNamespace(Tensor=Tensor, load=load,
            isfinite=lambda item: np.isfinite(item.value),
            equal=lambda left, right: np.array_equal(left.value, right.value))
        selection = {'token': '<S*>', 'branch': 'subject', 'method': 'fullrank', 'status': 'verified',
            'steps': total, 'weight': weight, 'run': str(run), 'checkpoint': str(checkpoint),
            'hashes': {str(run / 'config.yaml'): snapshots.sha256(run / 'config.yaml')}}
        return selection, torch, loads, final_token

    def export(self, total, step, *, explicit=True, mismatch=False):
        selection, torch, loads, final_token = self.snapshot_fixture(total, step)
        if mismatch:
            final_token['<S*>'] = Tensor(np.zeros(768))
        output = self.root / f'export-{total}-{step}'
        def write_export(torch, adapter, embedding, empty, selection, step, record, output):
            output.mkdir()
            return {'step': step, 'hashes': {}}
        kwargs = {'steps': (200, 400, 600), 'total_steps': 600} if explicit else {}
        with patch.dict('sys.modules', {'torch': torch}), \
             patch.object(snapshots.snapshots, 'adapter_schema', return_value=({'delta': (2, 768)}, {'self.processor': {}})), \
             patch.object(snapshots.snapshots, '_write_export', side_effect=write_export), \
             patch.object(snapshots.snapshots, '_embedding_audit', wraps=snapshots.snapshots._embedding_audit) as audit:
            result = snapshots.export_subject_step(selection, step, output, **kwargs)
        self.assertEqual(audit.call_args.args[1:], (total, '<S*>'))
        return result, loads

    def test_real_export_supports600_embedding_audit_and_final_weight_pairing(self):
        for step in (200, 400, 600):
            with self.subTest(step=step):
                result, loads = self.export(600, step)
                self.assertEqual(result['training_steps'], 600)
                self.assertEqual(result['steps'], step)
                self.assertTrue(result['paired_embedding_finite'])
                self.assertEqual('<S*>.bin' in loads, step == 600)

    def test600_final_export_rejects_mismatched_final_subject_embedding(self):
        with self.assertRaisesRegex(ValueError, 'final Subject snapshot differs'):
            self.export(600, 600, mismatch=True)

    def test_original_export_defaults_remain1000_and_reject_new_plan(self):
        result, _ = self.export(1000, 500, explicit=False)
        self.assertEqual(result['training_steps'], 1000)
        selection, torch, _, _ = self.snapshot_fixture(600, 200)
        with patch.dict('sys.modules', {'torch': torch}), self.assertRaisesRegex(ValueError, 'outside the audited plan'):
            snapshots.export_subject_step(selection, 200, self.root / 'wrong-default')

    def prepare_dataset(self):
        data = self.root / 'dataset'
        data.mkdir()
        rgb, mask = self.root / 'actual.png', self.root / 'mask.png'
        Image.new('RGB', (512, 512), 'blue').save(rgb)
        Image.new('L', (512, 512), 255).save(mask)
        audit = {'rows': 72, 'subject_id': 'metal_spoon', 'source_mode': 'synthetic_native_geometry'}
        refs = [{'id': f'actual_{i}', 'kind': 'synthetic_material_counterfactual', 'caption': 'a photo of <S*> spoon',
            'image': str(rgb), 'mask': str(mask), 'image_sha256': snapshots.sha256(rgb),
            'mask_sha256': snapshots.sha256(mask)} for i in range(72)]
        (data / 'training_assets_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in refs))
        for name, value in [('concepts.json', []), ('source_locks.json', {}), ('dataset_audit.json', audit)]:
            snapshots.training._write(data / name, value)
        return data, rgb, audit

    def test_execution_uses_custom_exporter_for_both_methods_and_writes36_images(self):
        data, rgb, audit = self.prepare_dataset()
        index = self.root / 'index.json'
        index.write_text('{}')
        selections = [{'method': method, 'run': str(self.root / method)} for method in inference.METHODS]
        class Generator:
            def __init__(self, **kwargs):
                pass
            def manual_seed(self, seed):
                return self
        fake_torch = SimpleNamespace(Generator=Generator, cuda=SimpleNamespace(empty_cache=lambda: None))
        def pipe(prompt, **kwargs):
            self.assertEqual((kwargs['num_inference_steps'], kwargs['guidance_scale']), (100, 3.5))
            return SimpleNamespace(images=[Image.new('RGB', (512, 512), 'gray')], nsfw_content_detected=[False])
        with patch.object(training, 'verify_dataset', return_value=audit), \
             patch.object(training, 'ROOT', self.root / 'code'), patch.object(shared, 'git_info', return_value={'commit': 'cpu'}), \
             patch.object(shared, '_selections', return_value=selections), \
             patch.object(inference, 'snapshot_exporter', return_value={'hashes': {str(rgb): snapshots.sha256(rgb)}}) as export, \
             patch.object(shared, 'export_subject_step', side_effect=AssertionError('legacy exporter must not run')), \
             patch.object(shared.evaluate, 'load_pipeline', return_value=pipe), \
             patch.object(shared.evaluate, 'attention_masks', return_value={}), patch.dict('sys.modules', {'torch': fake_torch}):
            proof = inference.run(data, self.root / 'step400', 400, index, True)
        self.assertEqual([call.args[1] for call in export.call_args_list], [400, 400])
        self.assertEqual((proof['generated_images'], proof['expected_images'], proof['training_images']), (36, 36, 72))
        self.assertEqual(len(proof['contact_sheets']), 3)

    def test_selection_loader_accepts_the_declared600_recipe_and_rejects1000(self):
        data = self.root / 'dataset'
        data.mkdir()
        index = self.root / 'index.json'
        selections = []
        configs = {}
        for method in inference.METHODS:
            folder = self.root / method
            folder.mkdir()
            config = {'stage': 'train', 'run': {'study': 'cpu_snapshot', 'variant': 'paired', 'seed': 42},
                'args': {'max_train_steps': 600}, 'subject_id': 'metal_spoon'}
            configs[method] = config
            snapshots.training._write(folder / 'config.yaml', config)
            selections.append({'method': method, 'run': str(folder), 'status': 'verified', 'token': '<S*>',
                'steps': 600, 'branch': 'subject', 'hashes': {'fake': 'lock'}})
        snapshots.training._write(index, {'git': {'commit': 'cpu'}, 'runs': selections})
        def audit(recipe, run, config):
            return next(row for row in selections if row['method'] == recipe['method'])
        with patch.object(shared, 'git_info', return_value={'commit': 'cpu'}), patch.object(shared, '_locks'), \
             patch.object(training, 'build_config', side_effect=lambda method, _: configs[method]), \
             patch.object(shared.shared, 'audit_run', side_effect=audit):
            self.assertEqual(shared._selections(index, data, training_module=training), selections)
            selections[0]['steps'] = 1000
            snapshots.training._write(index, {'git': {'commit': 'cpu'}, 'runs': selections})
            with self.assertRaisesRegex(ValueError, 'differs from the declared'):
                shared._selections(index, data, training_module=training)


if __name__ == '__main__':
    unittest.main()
