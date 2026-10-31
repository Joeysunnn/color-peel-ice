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

    def fixture(self, branch='subject', method='fullrank', step=None):
        total = 600 if branch == 'subject' else 1000
        step = total if step is None else step
        level = step / total
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
            'args': {'max_train_steps': total, 'checkpoint_steps': [500, 750, 1000] if total == 1000 else [200, 400, 600]}}
        audit = {'observed_optimization_steps': total, 'modifier_tokens': [{'token': token, 'token_id': 1,
            'exposure_steps': total, 'nonzero_gradient_steps': total, 'initial_final_l2_delta': 1.0}]}
        for path, data in ((run / 'config.yaml', config), (run / 'manifest.json', {}),
                           (checkpoint / 'adaptation_config.json', adaptation),
                           (checkpoint / 'embedding_update_audit.json', audit),
                           (derived_dir / 'adaptation_config.json', adaptation)):
            path.write_text(json.dumps(data))
        payloads = {
            snapshot / 'pytorch_model.bin': {'delta': Tensor(np.ones((2, 768)) * level)},
            snapshot / 'pytorch_model_1.bin': {source.snapshots.EMBEDDING_KEY:
                Tensor(np.stack([np.zeros(768), np.ones(768) * level]))},
            checkpoint / weight: {'delta': Tensor(np.ones((2, 768)))},
            checkpoint / (token + '.bin'): {token: Tensor(np.ones(768))},
            derived_dir / weight: {'delta': Tensor(np.ones((2, 768)) * level)},
            derived_dir / (token + '.bin'): {token: Tensor(np.ones(768) * level)}}
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
            'status': 'verified', 'run': str(run), 'checkpoint': str(checkpoint), 'steps': total,
            'hashes': {str(path): source.sha256(path) for path in files}}
        snapshot_hashes = {str(snapshot / name): source.sha256(snapshot / name)
                           for name in ('pytorch_model.bin', 'pytorch_model_1.bin')}
        derived_hashes = {str(derived_dir / name): source.sha256(derived_dir / name)
                          for name in (weight, token + '.bin', 'adaptation_config.json')}
        derived = {**parent, 'checkpoint': str(derived_dir), 'path': str(derived_dir), 'steps': step, 'step': step,
            'training_steps': total, 'source_checkpoint': str(checkpoint), 'source_hashes': parent['hashes'],
            'source_snapshot_hashes': snapshot_hashes,
            'source_adapter_sha256': snapshot_hashes[str(snapshot / 'pytorch_model.bin')],
            'source_text_encoder_sha256': snapshot_hashes[str(snapshot / 'pytorch_model_1.bin')],
            'derived_adapter_sha256': derived_hashes[str(derived_dir / weight)],
            'derived_token_sha256': derived_hashes[str(derived_dir / (token + '.bin'))],
            'hashes': {**parent['hashes'], **snapshot_hashes, **derived_hashes},
            'paired_embedding_finite': True, 'final_snapshot_matches_export': step == total}
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
            result = source.verify_pair(parent, derived, derived['step'], directory, locks)
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

    def test_texture500_uses_its_own_embedding_and_never_loads_final1000_tensors(self):
        fixture = self.fixture('texture', 'lora_r64', 500)
        _, derived, directory, payloads, torch = fixture
        loaded = []
        def load(path, **kwargs):
            loaded.append(Path(path))
            return payloads[Path(path)]
        torch.load = load
        self.assertEqual(self.verify(fixture), derived)
        self.assertEqual(derived['training_steps'], 1000)
        self.assertFalse(derived['final_snapshot_matches_export'])
        self.assertTrue(any('checkpoint-500' in str(path) for path in loaded))
        self.assertFalse(any('checkpoint-1000' in str(path) for path in loaded))
        self.assertTrue(np.all(payloads[directory / '<T*>.bin']['<T*>'].value == 0.5))
        self.assertFalse(any(path.parent == Path(fixture[0]['checkpoint']) for path in loaded))

    def test_texture500_rejects_final1000_embedding_and_wrong_final_flag(self):
        fixture = self.fixture('texture', step=500)
        _, derived, directory, payloads, _ = fixture
        original = payloads[directory / '<T*>.bin']
        payloads[directory / '<T*>.bin'] = {'<T*>': Tensor(np.ones(768))}
        with self.assertRaisesRegex(ValueError, 'tensors differ'):
            self.verify(fixture)
        payloads[directory / '<T*>.bin'] = original
        derived['final_snapshot_matches_export'] = True
        (directory / 'receipt.json').write_text(json.dumps(derived))
        with self.assertRaisesRegex(ValueError, 'snapshot provenance'):
            self.verify(fixture)

    def test_texture500_requires_membership_in_actual_checkpoint_plan(self):
        fixture = self.fixture('texture', step=500)
        parent, _, _, _, _ = fixture
        path = Path(parent['run']) / 'config.yaml'
        config = json.loads(path.read_text())
        config['args']['checkpoint_steps'] = [750, 1000]
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'source/step'):
            self.verify(fixture)

    def test_subject200_and400_use_actual_same_step_encoder_and_reject_final600_token(self):
        original_root = self.root
        for step in (200, 400):
            for method in source.METHODS:
                with self.subTest(step=step, method=method):
                    self.root = original_root / f'{step}-{method}'
                    self.root.mkdir()
                    fixture = self.fixture('subject', method, step)
                    parent, derived, directory, payloads, _ = fixture
                    self.assertEqual(self.verify(fixture), derived)
                    self.assertEqual(derived['steps'], step)
                    self.assertEqual(derived['training_steps'], 600)
                    payloads[directory / '<S*>.bin'] = payloads[Path(parent['checkpoint']) / '<S*>.bin']
                    with self.assertRaisesRegex(ValueError, 'tensors differ'):
                        self.verify(fixture)
        self.root = original_root

    def test_texture750_pairs_its_actual_snapshot_and_rejects_final_token(self):
        original_root = self.root
        for method in source.METHODS:
            with self.subTest(method=method):
                self.root = original_root / method
                self.root.mkdir()
                fixture = self.fixture('texture', method, 750)
                parent, derived, directory, payloads, _ = fixture
                self.assertEqual(self.verify(fixture), derived)
                self.assertEqual(derived['steps'], 750)
                self.assertEqual(derived['training_steps'], 1000)
                self.assertFalse(derived['final_snapshot_matches_export'])
                payloads[directory / '<T*>.bin'] = payloads[Path(parent['checkpoint']) / '<T*>.bin']
                with self.assertRaisesRegex(ValueError, 'tensors differ'):
                    self.verify(fixture)
        self.root = original_root

    def _prepare_texture500(self, subject_step=600, texture_step=500, include_subject=True):
        natural = self.root / 'natural'
        material = self.root / 'material-must-not-be-read'
        inference_git = {'commit': 'inference', 'branch': 'cpu'}
        parents = [{**recipe, 'status': 'verified', 'steps': 1000,
            'run': str(natural / 'training' / (recipe['concept_id'] + '_' + recipe['method']))}
            for recipe in source.natural_config.recipes() if recipe['concept_id'] == 'wood_spoon_texture']
        derived = [{**row, 'steps': texture_step, 'step': texture_step, 'training_steps': 1000} for row in parents]
        files = {'status.json': {'status': 'execution_verified', 'git': source.NATURAL_GIT,
                     'max_steps_per_run': 1000, 'source_dataset': str(self.root / 'data')},
            'protocol.json': {}, 'training/training_index.json': {'git': source.NATURAL_GIT, 'runs': parents},
            f'step{texture_step}/training_index.json': {'git': source.NATURAL_GIT, 'training_steps': texture_step,
                'source_training_steps': 1000, 'runs': derived}}
        for name, data in files.items():
            path = natural / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        observed = []
        read = source._read
        def read_selected(locks, path):
            path = Path(path)
            self.assertFalse(path.is_relative_to(material))
            if not include_subject:
                self.assertFalse(path.is_relative_to(source.SUBJECT_ROOT))
            observed.append(path)
            return read(locks, path)
        subject = {method: {'method': method, 'steps': subject_step, 'training_steps': 600}
                   for method in source.METHODS}
        with patch.object(source, 'NATURAL_ROOT', natural), patch.object(source, 'MATERIAL_ROOT', material), \
             patch.object(source, 'git_info', return_value=inference_git), \
             patch.object(source, 'resolve_run_dir', side_effect=Path), \
             patch.object(source, '_subject', return_value=(subject, [{'id': i} for i in range(72)])) as subject_reader, \
             patch.object(source, 'verify_dataset', return_value=({}, {}, {})), \
             patch.object(source, '_read', side_effect=read_selected), \
             patch.object(source, '_references', return_value=[{'id': i} for i in range(16)]) as references, \
             patch.object(source.natural_config, 'build_config', return_value={}), \
             patch.object(source.shared, 'audit_run', side_effect=lambda recipe, run, expected, **kw:
                 next(row for row in parents if row['run'] == str(run))) as audit, \
             patch.object(source, 'verify_pair', side_effect=lambda parent, item, *args: item) as pairs:
            kwargs = {} if subject_step == 600 else {'subject_step': subject_step}
            if not include_subject:
                subject_reader.side_effect = AssertionError('Literal baseline must not audit Subject')
                kwargs['include_subject'] = False
            result = source.prepare(self.root / 'selected-audit', texture_step=texture_step, include_material=False, **kwargs)
        branches = {'subject', 'texture'} if include_subject else {'texture'}
        self.assertEqual(set(result['sources']), branches)
        self.assertEqual(set(result['references']), branches)
        self.assertEqual(set(result['training_gits']), branches)
        if include_subject:
            self.assertEqual(result['sources']['subject'], subject)
            self.assertEqual(subject_reader.call_args.kwargs, {} if subject_step == 600 else {'step': subject_step})
        else:
            subject_reader.assert_not_called()
        self.assertEqual(audit.call_count, 2)
        self.assertEqual(pairs.call_count, 2)
        self.assertEqual(references.call_count, 1)
        self.assertEqual(references.call_args.args[2], 16)
        self.assertTrue(all(call.args[2] == texture_step for call in pairs.call_args_list))
        self.assertTrue(all(f'step{texture_step}' in str(call.args[3]) for call in pairs.call_args_list))
        self.assertIn(natural / f'step{texture_step}/training_index.json', observed)
        self.assertTrue((self.root / 'selected-audit/audit.json').is_file())

    def test_prepare_selects_texture500_without_reading_material_source(self):
        self._prepare_texture500()

    def test_literal_texture500_does_not_audit_subject_or_read_subject_references(self):
        self._prepare_texture500(include_subject=False)

    def test_prepare_selects_texture750_without_material_source(self):
        self._prepare_texture500(subject_step=400, texture_step=750)

    def test_prepare_forwards_subject200_and400_with_texture500_only(self):
        original_root = self.root
        for step in (200, 400):
            with self.subTest(step=step):
                self.root = original_root / str(step)
                self.root.mkdir()
                self._prepare_texture500(step)
        self.root = original_root

    def test_subject_loader_reads_selected_step_but_retains600_step_parent_audit(self):
        subject_root = self.root / 'frozen_subject'
        output = self.root / 'audit'
        output.mkdir()
        parents = [{'method': method, 'branch': 'subject', 'concept_id': 'wood_spoon',
                    'token': '<S*>', 'status': 'verified', 'steps': 600} for method in source.METHODS]
        files = {'training_index.json': {'git': source.SUBJECT_GIT, 'runs': parents},
            'step200/provenance.json': {'git': source.SUBJECT_GIT, 'training': parents, 'status': 'succeeded',
                'step': 200, 'training_images': 72, 'generated_images': 36},
            **{'training_dataset/' + name: {} for name in
                ('dataset_audit.json', 'source_locks.json', 'approval.json', 'concepts.json')}}
        for parent in parents:
            files[f"step200/checkpoints/{parent['method']}/receipt.json"] = {
                **parent, 'steps': 200, 'step': 200, 'training_steps': 600}
        for name, data in files.items():
            path = subject_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        process = SimpleNamespace(returncode=0, stdout='WOOD_COMPOSITION_AUDIT=' +
            json.dumps({'parents': parents, 'git': source.SUBJECT_GIT}) + '\n')
        with patch.object(source, 'SUBJECT_ROOT', subject_root), \
             patch.object(source.subprocess, 'run', return_value=process), \
             patch.object(source, '_references', return_value=[{'id': i} for i in range(72)]), \
             patch.object(source, 'verify_pair', side_effect=lambda parent, item, *args: item) as pairs:
            selections, references = source._subject(output, {}, step=200)
        self.assertEqual(len(references), 72)
        self.assertEqual(len(selections), 2)
        for call in pairs.call_args_list:
            self.assertEqual(call.args[0]['steps'], 600)
            self.assertEqual(call.args[1]['steps'], 200)
            self.assertEqual(call.args[2], 200)
            self.assertEqual(call.args[3], subject_root / 'step200/checkpoints' / call.args[0]['method'])


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
