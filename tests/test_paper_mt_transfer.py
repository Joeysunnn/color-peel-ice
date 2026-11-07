"""Protect the exact paper M/T prompts and the selected paired exports."""

from collections import Counter
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.paper_mt_transfer_v1 import run


class MaterialTextureTransferTests(unittest.TestCase):
    def test_four_checkpoints_have_exact_prompts_and_inclusive_seed_range(self):
        expected_steps = {'material': (150, 200), 'texture': (300, 500)}
        self.assertEqual(run.STEPS, expected_steps)
        self.assertEqual(run.SEEDS, tuple(range(20, 61)))
        total = 0
        for branch, steps in expected_steps.items():
            token = '<M*>' if branch == 'material' else '<T*>'
            prompts = {f'a photo of a {obj} with a <M*> surface' if branch == 'material'
                       else f'a photo of a {obj} with <T*> texture'
                       for obj in ('chair', 'mug', 'vase', 'sofa')}
            for step in steps:
                with self.subTest(branch=branch, step=step):
                    rows = run.plan(branch, step)
                    total += len(rows)
                    self.assertEqual(len(rows), 164)
                    self.assertEqual(len({row['id'] for row in rows}), 164)
                    self.assertEqual(len({row['image_path'] for row in rows}), 164)
                    self.assertEqual(Counter(row['prompt'] for row in rows), dict.fromkeys(prompts, 41))
                    for prompt in prompts:
                        self.assertEqual({row['seed'] for row in rows if row['prompt'] == prompt},
                                         set(range(20, 61)))
                    for row in rows:
                        self.assertEqual(row['branch_steps'], {branch: step})
                        self.assertEqual(row['mode'], branch + '_only')
                        self.assertEqual(row['negative_prompt'], '')
                        self.assertEqual(row['guidance_scale'], 3.5)
                        self.assertEqual(row['prompt'].count(token), 1)
                        self.assertFalse(any(other in row['prompt'] for other in
                                             ('<S*>', '<C*>', '<M*>', '<T*>') if other != token))
                        self.assertEqual(row['image_path'], f'images/{row["case"]}/seed{row["seed"]}.png')
        self.assertEqual(total, 656)

    def test_unsupported_branch_or_checkpoint_is_rejected(self):
        for branch, step in (('subject', 1000), ('color', 150), ('material', 300),
                             ('material', 250), ('texture', 200), ('texture', 400)):
            with self.subTest(branch=branch, step=step), self.assertRaises(ValueError):
                run.plan(branch, step)

    def test_training_source_accepts_only_a_contained_single_directory_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / 'object'
            self.assertEqual(run.training_source(root, 'texture', run.TRAINING_RUN),
                             root / 'training' / run.TRAINING_RUN / 'object_003' / 'texture')
            self.assertEqual(run.training_source(root, 'texture', 'legacy_T16-v1'),
                             root / 'training' / 'legacy_T16-v1' / 'object_003' / 'texture')
            for name in ('', '.', '..', '../outside', 'run/../../outside', 'run/sub',
                         r'run\sub', '/absolute', r'C:\absolute', 'bad name', None):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    run.training_source(root, 'texture', name)

    def test_training_source_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            training = root / 'object' / 'training'
            training.mkdir(parents=True)
            outside = root / 'outside'
            outside.mkdir()
            try:
                (training / 'legacy').symlink_to(outside, target_is_directory=True)
            except OSError as error:
                self.skipTest('symlink creation unavailable: ' + str(error))
            with self.assertRaisesRegex(ValueError, 'escapes'):
                run.training_source(root / 'object', 'texture', 'legacy')

    def test_run_forwards_default_and_explicit_training_provenance(self):
        explicit = {'training_run': 'legacy_T16',
                    'source_git': {'branch': 'codex/legacy-test', 'commit': 'a' * 40},
                    'source_image_sha': 'b' * 64}
        for overrides in ({}, explicit):
            expected = overrides or {'training_run': run.TRAINING_RUN,
                                     'source_git': run.SOURCE_GIT,
                                     'source_image_sha': run.CANONICAL_SHA}
            with self.subTest(overrides=overrides), patch.dict(os.environ), \
                    patch.object(run, 'require_idle'), \
                    patch.object(run, 'prepare', side_effect=RuntimeError('stop before loading')) as prepare:
                with self.assertRaisesRegex(RuntimeError, '^stop before loading$'):
                    run.run(Path('object'), 'texture', 300, 0, **overrides)
                prepare.assert_called_once_with(Path('object'), 'texture', 300, 0, **expected)

    def fixture(self, root, branch='material', step=150):
        source = root / branch
        checkpoint = source / 'ckpts' / f'step{step}'
        checkpoint.mkdir(parents=True)
        manifest_path = root / (branch + '_initializers.json')
        initializer = 'material' if branch == 'material' else 'raised'
        token_id = 7118 if branch == 'material' else 6027
        manifest = {'branches': {branch: {'initializer': initializer, 'initializer_token_id': token_id}}}
        manifest_path.write_text(json.dumps(manifest), encoding='utf8')
        audit_path = source / 'initializer_audit.json'
        audit_path.write_text('{"initialized_equal_base": true}\n', encoding='utf8')
        token = '<M*>' if branch == 'material' else '<T*>'
        weight = 'pytorch_lora_kv_weights.bin'
        # Windows cannot create the production token filenames. The selection
        # contract uses their declared locks; prepare verifies bytes on Linux.
        item = {'branch': branch, 'step': step, 'steps': step,
                'training_steps': 200 if branch == 'material' else 500,
                'method': 'lora_r64', 'token': token, 'embedding_dim': 1024,
                'base_model': 'fixture-sd21', 'status': 'verified',
                'paired_embedding_finite': True, 'weight': weight, 'checkpoint': str(checkpoint),
                'hashes': {str(checkpoint / weight): 'a' * 64,
                           str(checkpoint / (token + '.bin')): 'b' * 64,
                           str(audit_path): run.sha256(audit_path)},
                'initializer_provenance': {
                    'initializer_manifest': str(manifest_path),
                    'initializer_manifest_sha256': run.sha256(manifest_path),
                    'initializer': initializer, 'initializer_token_id': token_id,
                    'initializer_audit': str(audit_path),
                    'initializer_audit_sha256': run.sha256(audit_path)}}
        source_audit = {'initializer_manifest': str(manifest_path), 'source_locks': {}}
        return item, source, source_audit, manifest

    def validate(self, item, branch, step, source, source_audit, manifest):
        return run.validate_selection(item, branch, step, source, source_audit, manifest, 'fixture-sd21')

    def test_accepts_each_new_export_without_requiring_old_pretraining_audit_lock(self):
        for branch, steps in {'material': (150, 200), 'texture': (300, 500)}.items():
            for step in steps:
                with self.subTest(branch=branch, step=step), tempfile.TemporaryDirectory() as temporary:
                    item, source, audit, manifest = self.fixture(Path(temporary), branch, step)
                    original = copy.deepcopy(item)
                    self.validate(item, branch, step, source, audit, manifest)
                    self.assertEqual(item, original)
                    self.assertEqual(audit['source_locks'], {})

    def test_wrong_export_identity_or_pairing_is_rejected(self):
        wrong = {'branch': 'texture', 'step': 200, 'steps': 200, 'training_steps': 500,
                 'method': 'fullrank', 'token': '<T*>', 'embedding_dim': 768,
                 'base_model': 'sd15', 'status': 'unverified', 'paired_embedding_finite': False,
                 'weight': 'other_adapter.bin'}
        with tempfile.TemporaryDirectory() as temporary:
            item, source, audit, manifest = self.fixture(Path(temporary))
            for key, value in wrong.items():
                changed = {**item, key: value}
                with self.subTest(field=key), self.assertRaises(ValueError):
                    self.validate(changed, 'material', 150, source, audit, manifest)

    def test_foreign_or_wrong_step_directory_and_missing_pair_locks_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            item, source, audit, manifest = self.fixture(Path(temporary))
            for path in (source / 'ckpts/step200', source.parent / 'old_producer/ckpts/step150'):
                changed = {**item, 'checkpoint': str(path)}
                with self.subTest(path=path), self.assertRaises(ValueError):
                    self.validate(changed, 'material', 150, source, audit, manifest)
            for filename in (item['weight'], item['token'] + '.bin'):
                changed = copy.deepcopy(item)
                del changed['hashes'][str(Path(item['checkpoint']) / filename)]
                with self.subTest(missing=filename), self.assertRaises(ValueError):
                    self.validate(changed, 'material', 150, source, audit, manifest)

    def test_initializer_provenance_and_actual_copy_audit_must_remain_locked(self):
        with tempfile.TemporaryDirectory() as temporary:
            item, source, audit, manifest = self.fixture(Path(temporary))
            wrong = {'initializer_manifest': str(source / 'old_initializers.json'),
                     'initializer_manifest_sha256': '0' * 64, 'initializer': 'painted',
                     'initializer_token_id': 1, 'initializer_audit_sha256': '0' * 64}
            for key, value in wrong.items():
                changed = copy.deepcopy(item)
                changed['initializer_provenance'][key] = value
                with self.subTest(field=key), self.assertRaises(ValueError):
                    self.validate(changed, 'material', 150, source, audit, manifest)
            changed = copy.deepcopy(item)
            del changed['hashes'][item['initializer_provenance']['initializer_audit']]
            with self.assertRaises(ValueError):
                self.validate(changed, 'material', 150, source, audit, manifest)

    def test_generation_error_remains_primary_when_collecting_partial_evidence_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows = run.plan('material', 150)
            audit = {'inference_git': {'commit': 'fixture-inference', 'branch': 'codex/test'}}
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'fixture-original'}), \
                    patch.object(run, 'require_idle'), \
                    patch.object(run, 'prepare', return_value=(output, rows, audit)), \
                    patch.object(run, 'run_group', side_effect=RuntimeError('generation failed')), \
                    patch.object(run, 'collect', side_effect=ValueError('bad ledger')):
                with self.assertRaisesRegex(RuntimeError, '^generation failed$'):
                    run.run(output / 'object', 'material', 150, 0)
            state = json.loads((output / 'status.json').read_text(encoding='utf8'))
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['error'], 'RuntimeError: generation failed')
            self.assertEqual(state['collection_error'], 'ValueError: bad ledger')
            self.assertEqual(state['generated_images'], 0)
            self.assertTrue((output / 'index.html').is_file())


if __name__ == '__main__':
    unittest.main()
