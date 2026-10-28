"""CPU checks for immutable fullrank reuse with explicit, truthful source revisions."""

import copy
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from experiments.subject_trellis_spoons_v1 import inference, reuse, training
from experiments.texture_material_disentangled_v1 import train as shared


class FullrankReuseTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='spoon-reuse-', dir=os.environ.get('TMP'))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source_root = self.root / 'old'
        self.dataset = self.source_root / 'metal_spoon/all180/training_dataset'
        self.dataset.mkdir(parents=True)
        self.source = self.source_root / 'metal_spoon/all180/training/fullrank'
        self.source.mkdir(parents=True)
        self.audit = {'subject_id': 'metal_spoon', 'rows': 180}
        training._write(self.dataset / 'dataset_audit.json', self.audit)
        self.current = {'commit': 'b' * 40, 'branch': 'codex/spoon-fix'}
        self.fullrank = self.selection('fullrank', self.source / 'runs/fullrank', reuse.OLD_GIT)
        self.rank64 = self.selection('lora_r64', self.root / 'new/runs/rank64', self.current)
        training._write(self.source / 'status.json', {'status': 'training_verified',
            'method': 'fullrank', 'completed_steps': 1000, 'git': reuse.OLD_GIT})
        training._write(self.source / 'training_index.json', {'git': reuse.OLD_GIT, 'runs': [self.fullrank]})
        for step in (500, 750, 1000):
            for name in reuse.STATE_FILES:
                path = Path(self.fullrank['checkpoint']) / f'checkpoint-{step}' / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(f'{step}/{name}'.encode())
        patches = [patch.object(reuse, 'SOURCE_ROOT', self.source_root),
            patch.object(training, 'verify_dataset', return_value=self.audit),
            patch.object(reuse, 'git_info', return_value=self.current),
            patch.object(shared, 'audit_run', side_effect=self.actual_audit)]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def selection(self, method, run, revision):
        checkpoint = run / 'checkpoints'
        checkpoint.mkdir(parents=True)
        config = training.build_config(method, self.dataset)
        training._write(run / 'config.yaml', config)
        training._write(run / 'manifest.json', {'status': 'succeeded', 'returncode': 0, 'git': revision})
        weight = 'pytorch_token_local_kv_weights.bin' if method == 'fullrank' else 'pytorch_lora_kv_weights.bin'
        paths = [run / 'config.yaml', run / 'manifest.json']
        for name in (weight, 'token_embedding.bin', 'adaptation_config.json', 'training_metrics.jsonl', 'embedding_update_audit.json'):
            path = checkpoint / name
            path.write_bytes(name.encode())
            paths.append(path)
        return {'method': method, 'branch': 'subject', 'concept_id': 'metal_spoon', 'token': '<S*>',
            'status': 'verified', 'steps': 1000, 'run': str(run), 'checkpoint': str(checkpoint),
            'weight': weight, 'hashes': {str(path): shared.sha256(path) for path in paths}}

    def actual_audit(self, recipe, run, expected, **kwargs):
        method = recipe['method']
        selection = self.fullrank if method == 'fullrank' else self.rank64
        self.assertEqual(Path(selection['run']), run)
        self.assertEqual(recipe['concept_id'], 'metal_spoon')
        self.assertEqual(expected, training.build_config(method, self.dataset))
        self.assertEqual(kwargs, {'expected_git': reuse.OLD_GIT} if method == 'fullrank' else {})
        return selection

    def mixed_index(self):
        _, receipt = reuse.completed_fullrank(self.dataset, self.source)
        return {'git': self.current, 'source_gits': {'fullrank': reuse.OLD_GIT, 'lora_r64': self.current},
            'runs': [self.fullrank, self.rank64], 'reuse_receipt': receipt}

    def test_completed_source_is_read_only_and_locks_every_saved_state(self):
        before = {str(path): shared.sha256(path) for path in self.source.rglob('*') if path.is_file()}
        selection, receipt = reuse.completed_fullrank(self.dataset, self.source)
        self.assertEqual(selection, self.fullrank)
        self.assertFalse(receipt['retrained'])
        self.assertEqual(receipt['source_git'], reuse.OLD_GIT)
        self.assertEqual(sum('checkpoint-' in key for key in receipt['locks']), 15)
        self.assertEqual(before, {str(path): shared.sha256(path) for path in self.source.rglob('*') if path.is_file()})

    def test_wrong_source_subject_or_source_revision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'same frozen'):
            reuse.completed_fullrank(self.dataset, self.root / 'another')
        with patch.object(training, 'verify_dataset', return_value={'subject_id': 'wood_spoon', 'rows': 180}), \
             self.assertRaisesRegex(ValueError, 'same frozen'):
            reuse.completed_fullrank(self.dataset, self.source)
        state = training.read_json(self.source / 'status.json')
        state['git'] = self.current
        training._write(self.source / 'status.json', state)
        with self.assertRaisesRegex(ValueError, 'fixed original revision'):
            reuse.completed_fullrank(self.dataset, self.source)

    def test_changed_weight_or_empty_paired_snapshot_is_rejected(self):
        weight = Path(self.fullrank['checkpoint']) / self.fullrank['weight']
        original = weight.read_bytes()
        weight.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            reuse.completed_fullrank(self.dataset, self.source)
        weight.write_bytes(original)
        snapshot = Path(self.fullrank['checkpoint']) / 'checkpoint-750/pytorch_model_1.bin'
        snapshot.write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'paired snapshot'):
            reuse.completed_fullrank(self.dataset, self.source)

    def test_mixed_index_uses_real_old_and_new_revisions(self):
        index = self.root / 'mixed.json'
        training._write(index, self.mixed_index())
        self.assertEqual(reuse.selection_loader(index, self.dataset), [self.fullrank, self.rank64])

    def test_mixed_index_rejects_wrong_methods_revisions_and_reuse_receipt(self):
        original = self.mixed_index()
        index = self.root / 'mixed.json'
        for changed in ('methods', 'revision', 'receipt', 'source_path'):
            value = copy.deepcopy(original)
            if changed == 'methods':
                value['runs'][1]['method'] = 'fullrank'
            elif changed == 'revision':
                value['source_gits']['fullrank'] = self.current
            elif changed == 'receipt':
                value['reuse_receipt']['subject_id'] = 'wood_spoon'
            else:
                value['reuse_receipt']['source_dir'] = str(self.root / 'wrong-source')
            training._write(index, value)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                reuse.selection_loader(index, self.dataset)

    def test_default_inference_keeps_strict_loader_and_explicit_hook_is_forwarded(self):
        with patch.object(inference.baseline, 'run', return_value={}) as run:
            inference.run(self.dataset, self.root / 'default', 500)
            self.assertIsNone(run.call_args.kwargs['selection_loader'])
            inference.run(self.dataset, self.root / 'mixed', 500, selection_loader=reuse.selection_loader)
            self.assertIs(run.call_args.kwargs['selection_loader'], reuse.selection_loader)


class AuditRevisionGuardTest(unittest.TestCase):
    def test_default_guard_rejects_old_manifest_and_explicit_source_revision_reaches_next_guard(self):
        # Run the real audit through its revision guard; stop before tensor operations.
        config = {'args': {'max_train_steps': 1000}}
        manifest = {'status': 'succeeded', 'returncode': 0, 'git': reuse.OLD_GIT}
        class ReachedTensorAudit(Exception):
            pass
        def read(path):
            if Path(path).name == 'manifest.json':
                return manifest
            raise ReachedTensorAudit()
        recipe = {'method': 'fullrank', 'token': '<S*>'}
        current = {'commit': 'b' * 40, 'branch': 'codex/spoon-fix'}
        with patch.object(shared, 'read_config', return_value=config), patch.object(shared, 'read_json', side_effect=read), \
             patch.object(shared, 'git_info', return_value=current), patch.dict('sys.modules', {'torch': types.SimpleNamespace()}):
            with self.assertRaisesRegex(ValueError, 'revision'):
                shared.audit_run(recipe, Path('fixture-run'), config)
            with self.assertRaises(ReachedTensorAudit):
                shared.audit_run(recipe, Path('fixture-run'), config, expected_git=reuse.OLD_GIT)
            with self.assertRaisesRegex(ValueError, 'revision'):
                shared.audit_run(recipe, Path('fixture-run'), config, expected_git=current)


if __name__ == '__main__':
    unittest.main()
