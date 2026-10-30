"""Exercise real selected-source receipts, hashes and paired tensor rejection on CPU."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.wood_spoon_composition_v1 import source


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.shape, self.ndim = self.value.shape, self.value.ndim

    def __getitem__(self, key):
        return Tensor(self.value[key])

    def clone(self):
        return Tensor(self.value.copy())

    def contiguous(self):
        return self


class WoodCompositionSourceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='wood-composition-source-', dir=os.environ.get('TMP'))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        # Windows cannot create the Linux checkpoint names <S*>.bin etc.; keep
        # only those payload bytes virtual while exercising all real receipts.
        self.virtual_files = {}
        original_sha256 = source.sha256
        def digest(path):
            path = Path(path)
            return (hashlib.sha256(self.virtual_files[path]).hexdigest()
                    if path in self.virtual_files else original_sha256(path))
        for module in (source, source.step_snapshots):
            patcher = patch.object(module, 'sha256', side_effect=digest)
            patcher.start()
            self.addCleanup(patcher.stop)

    def fixture(self, branch='subject', method='fullrank'):
        step = 600 if branch == 'subject' else 1000
        token = {'subject': '<S*>', 'material': '<M*>', 'texture': '<T*>'}[branch]
        run = self.root / 'training'
        checkpoint = run / 'checkpoints'
        snapshot = checkpoint / f'checkpoint-{step}'
        derived_dir = self.root / 'paired'
        snapshot.mkdir(parents=True)
        derived_dir.mkdir()
        weight = 'pytorch_token_local_kv_weights.bin' if method == 'fullrank' else 'pytorch_lora_kv_weights.bin'
        adaptation = {'modifier_tokens': [token], 'adaptation_mode': 'token_local_kv' if method == 'fullrank'
            else 'lora_material_kv' if branch == 'material' else 'lora_subject_kv'}
        if method == 'lora_r64':
            adaptation.update(rank=64, alpha=64, mode='token_local_kv')
        config = {'stage': 'train', 'run': {'study': 'cpu_pairing', 'variant': 'wood', 'seed': 42},
            'args': {'max_train_steps': step, 'checkpoint_steps': [step]}}
        audit = {'observed_optimization_steps': step, 'modifier_tokens': [{'token': token, 'token_id': 1,
            'exposure_steps': step, 'nonzero_gradient_steps': step, 'initial_final_l2_delta': 1.0}]}
        for path, data in ((run / 'config.yaml', config), (run / 'manifest.json', {}),
                           (checkpoint / 'adaptation_config.json', adaptation),
                           (checkpoint / 'embedding_update_audit.json', audit),
                           (derived_dir / 'adaptation_config.json', adaptation)):
            path.write_text(json.dumps(data))
        payloads = {
            snapshot / 'pytorch_model.bin': {'delta': Tensor(np.ones((2, 768)))},
            snapshot / 'pytorch_model_1.bin': {source.snapshots.EMBEDDING_KEY:
                Tensor(np.stack([np.zeros(768), np.ones(768)]))},
            checkpoint / weight: {'delta': Tensor(np.ones((2, 768)))},
            checkpoint / (token + '.bin'): {token: Tensor(np.ones(768))},
            derived_dir / weight: {'delta': Tensor(np.ones((2, 768)))},
            derived_dir / (token + '.bin'): {token: Tensor(np.ones(768))}}
        for path in payloads:
            if '<' in path.name:
                self.virtual_files[path] = b'locked binary payload'
            else:
                path.write_bytes(b'locked binary payload')
        files = [run / 'config.yaml', run / 'manifest.json', checkpoint / weight,
                 checkpoint / (token + '.bin'), checkpoint / 'adaptation_config.json',
                 checkpoint / 'embedding_update_audit.json']
        parent = {'method': method, 'branch': branch, 'token': token, 'weight': weight,
            'concept_id': 'wood_spoon' if branch == 'subject' else 'wood_spoon_' + branch,
            'status': 'verified', 'run': str(run), 'checkpoint': str(checkpoint), 'steps': step,
            'hashes': {str(path): source.sha256(path) for path in files}}
        snapshot_hashes = {str(snapshot / name): source.sha256(snapshot / name)
                           for name in ('pytorch_model.bin', 'pytorch_model_1.bin')}
        derived_hashes = {str(derived_dir / name): source.sha256(derived_dir / name)
                          for name in (weight, token + '.bin', 'adaptation_config.json')}
        derived = {**parent, 'checkpoint': str(derived_dir), 'path': str(derived_dir), 'step': step,
            'training_steps': step, 'source_checkpoint': str(checkpoint), 'source_hashes': parent['hashes'],
            'source_snapshot_hashes': snapshot_hashes,
            'source_adapter_sha256': snapshot_hashes[str(snapshot / 'pytorch_model.bin')],
            'source_text_encoder_sha256': snapshot_hashes[str(snapshot / 'pytorch_model_1.bin')],
            'derived_adapter_sha256': derived_hashes[str(derived_dir / weight)],
            'derived_token_sha256': derived_hashes[str(derived_dir / (token + '.bin'))],
            'hashes': {**parent['hashes'], **snapshot_hashes, **derived_hashes},
            'paired_embedding_finite': True, 'final_snapshot_matches_export': True}
        (derived_dir / 'receipt.json').write_text(json.dumps(derived))
        torch = SimpleNamespace(Tensor=Tensor, load=lambda path, **kwargs: payloads[Path(path)],
            isfinite=lambda value: np.isfinite(value.value),
            equal=lambda left, right: np.array_equal(left.value, right.value))
        return parent, derived, derived_dir, payloads, torch

    def verify(self, fixture):
        parent, derived, directory, _, torch = fixture
        with patch.dict('sys.modules', {'torch': torch}), patch.object(source.snapshots, 'adapter_schema',
                return_value=({'delta': (2, 768)}, {})), patch.object(source.step_snapshots, 'adapter_schema',
                return_value=({'delta': (2, 768)}, {})):
            locks = {}
            result = source.verify_pair(parent, derived, parent['steps'], directory, locks)
        self.assertIn(str(directory / 'receipt.json'), locks)
        return result

    def test_subject600_and_natural1000_use_actual_snapshot_and_existing_export(self):
        for branch in ('subject', 'material', 'texture'):
            for method in source.METHODS:
                with self.subTest(branch=branch, method=method):
                    # Each fixture owns fresh immutable paths.
                    child = self.root / (branch + method)
                    child.mkdir()
                    original = self.root
                    self.root = child
                    fixture = self.fixture(branch, method)
                    self.assertEqual(self.verify(fixture), fixture[1])
                    self.root = original

    def test_wrong_step_and_source_checkpoint_are_rejected(self):
        fixture = self.fixture()
        for key, value in (('steps', 400), ('source_checkpoint', '/another/checkpoint')):
            with self.subTest(key=key):
                altered = list(fixture)
                altered[1] = {**fixture[1], key: value}
                with self.assertRaisesRegex(ValueError, 'source/step'):
                    self.verify(altered)

    def test_binary_hash_corruption_is_rejected(self):
        fixture = self.fixture()
        (fixture[2] / fixture[0]['weight']).write_bytes(b'changed adapter')
        with self.assertRaisesRegex(ValueError, 'hashes'):
            self.verify(fixture)

    def test_forged_consistent_receipt_cannot_hide_different_adapter_tensor(self):
        fixture = self.fixture('material')
        parent, derived, directory, payloads, _ = fixture
        path = directory / parent['weight']
        path.write_bytes(b'forged but consistently rehashed adapter')
        payloads[path] = {'delta': Tensor(np.zeros((2, 768)))}
        derived['derived_adapter_sha256'] = source.sha256(path)
        derived['hashes'][str(path)] = source.sha256(path)
        (directory / 'receipt.json').write_text(json.dumps(derived))
        with self.assertRaisesRegex(ValueError, 'tensors differ'):
            self.verify(fixture)

    def test_matching_receipt_does_not_hide_wrong_subject_token_embedding(self):
        fixture = self.fixture()
        parent, _, directory, payloads, _ = fixture
        payloads[directory / (parent['token'] + '.bin')] = {parent['token']: Tensor(np.zeros(768))}
        with self.assertRaisesRegex(ValueError, 'tensors differ'):
            self.verify(fixture)

    def test_selection_filter_rejects_duplicate_method_and_ignores_unselected_source(self):
        parent = self.fixture('texture')[0]
        second = {**copy.deepcopy(parent), 'method': 'lora_r64'}
        unrelated = {**copy.deepcopy(parent), 'concept_id': 'mailbox_texture'}
        self.assertEqual(set(source._methods([parent, second, unrelated], 'texture', '<T*>', 1000)), set(source.METHODS))
        with self.assertRaisesRegex(ValueError, 'exactly two'):
            source._methods([parent, parent], 'texture', '<T*>', 1000)


class WoodMaterialLegacyPlanTest(unittest.TestCase):
    def plan(self):
        return {'git': source.MATERIAL_GIT, 'recipes': [row for row in source.natural_config.recipes()
                if row['concept_id'] == 'wood_spoon_material']}

    def test_pinned_wood_only_legacy_plan_and_explicit_wood_identity_are_accepted(self):
        legacy = self.plan()
        self.assertNotIn('source_id', legacy)
        self.assertTrue(source._wood_material_plan(legacy))
        self.assertTrue(source._wood_material_plan({**legacy, 'source_id': 'wood_spoon'}))

    def test_missing_identity_in_other_revision_or_conflicting_recipe_is_rejected(self):
        legacy = self.plan()
        for changes in ({'git': {'commit': 'another'}}, {'source_id': 'mailbox'}, {'source_id': None},
                        {'recipes': legacy['recipes'][:1]}, {'recipes': []}):
            with self.subTest(changes=changes):
                self.assertFalse(source._wood_material_plan({**legacy, **changes}))
        changed = copy.deepcopy(legacy)
        changed['recipes'][1]['source_id'] = 'mailbox'
        self.assertFalse(source._wood_material_plan(changed))


if __name__ == '__main__':
    unittest.main()
