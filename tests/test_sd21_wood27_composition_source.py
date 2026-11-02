"""Check paused-transfer reuse, immutable identity guards and cached T600 pairing."""

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.sd21_wood_subject27_v1 import composition_source as source


class CompositionSourceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='wood27-source-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.subject, self.texture = self.root / 'subject', self.root / 'texture'
        self.model = {'id': 'same-sd21-model'}
        self.files, self.configs = {}, {}
        self.receipts, self.indices = {}, {}
        self.reads = []
        self.digest = lambda path: hashlib.sha256(str(Path(path)).encode()).hexdigest()
        for branch, root, git in (('subject', self.subject, source.SUBJECT_GIT),
                                  ('texture', self.texture, source.TEXTURE_GIT)):
            jobs = []
            for method in source.contract.METHODS:
                group = root / 'methods' / method if branch == 'subject' else root / branch / method
                run = group / 'training' / 'original_run'
                config = {'run': {'study': branch + '_study', 'variant': method, 'seed': 42},
                          'args': {'max_train_steps': 1000}}
                self.configs[run / 'config.yaml'] = config
                self.files[run / 'manifest.json'] = {'status': 'succeeded', 'returncode': 0,
                    'stage': 'train', 'git': git, 'run': config['run']}
                jobs.append({'branch': branch, 'method': method, 'run': str(run)})
                index = {'git': git, 'smoke': False, 'steps': {}}
                self.files[group / 'inference/ckpts/index.json'] = index
                self.indices[branch, method] = index
                for step in ((400, 600) if branch == 'subject' else (600,)):
                    checkpoint = group / 'inference/ckpts' / f'step{step}'
                    snapshot = run / 'checkpoints' / f'checkpoint-{step}'
                    token = '<S*>' if branch == 'subject' else '<T*>'
                    weight = 'pytorch_token_local_kv_weights.bin' if method == 'fullrank' else 'pytorch_lora_kv_weights.bin'
                    paths = [run / 'config.yaml', run / 'manifest.json', checkpoint / weight,
                        checkpoint / (token + '.bin'), checkpoint / 'adaptation_config.json',
                        run / 'checkpoints/embedding_update_audit.json', run / 'checkpoints/training_metrics.jsonl',
                        snapshot / 'pytorch_model.bin', snapshot / 'pytorch_model_1.bin']
                    item = {'branch': branch, 'method': method, 'step': step, 'steps': step,
                        'training_steps': 1000, 'token': token, 'weight': weight, 'embedding_dim': 1024,
                        'status': 'verified', 'paired_embedding_finite': True, 'final_snapshot_matches_export': False,
                        'concept_id': 'wood_spoon' if branch == 'subject' else 'wood_spoon_texture',
                        'run': str(run), 'checkpoint': str(checkpoint), 'source_checkpoint': str(run / 'checkpoints'),
                        'base_model': self.model, 'hashes': {str(path): self.digest(path) for path in paths},
                        'source_snapshot_hashes': {str(snapshot / name): self.digest(snapshot / name)
                            for name in ('pytorch_model.bin', 'pytorch_model_1.bin')},
                        'source_adapter_sha256': self.digest(snapshot / 'pytorch_model.bin'),
                        'source_text_encoder_sha256': self.digest(snapshot / 'pytorch_model_1.bin'),
                        'derived_adapter_sha256': self.digest(checkpoint / weight),
                        'derived_token_sha256': self.digest(checkpoint / (token + '.bin'))}
                    self.files[checkpoint / 'receipt.json'] = item
                    self.receipts[branch, method, step] = item
                    index['steps'][str(step)] = item
                    self.files[checkpoint / 'adaptation_config.json'] = {'modifier_tokens': [token],
                        'adaptation_mode': 'token_local_kv' if method == 'fullrank' else 'lora_subject_kv',
                        'rank': 64, 'alpha': 64, 'mode': 'token_local_kv'}
            self.files[root / 'experiment.json'] = {'git': git, 'jobs': jobs,
                'status': 'failed', 'error': 'user SIGINT interrupted transfer; training completed'}
        def read(path):
            self.reads.append(Path(path))
            return self.files[Path(path)]
        self.patches = [
            patch.object(source, 'SUBJECT_ROOT', self.subject), patch.object(source, 'TEXTURE_ROOT', self.texture),
            patch.object(source, 'read_json', side_effect=read), patch.object(source, 'sha256', side_effect=self.digest),
            patch.object(source.previous.snapshots, 'sha256', side_effect=self.digest),
            patch.object(source, 'read_config', side_effect=lambda path: self.configs[Path(path)]),
            patch.object(source, 'git_info', return_value={'commit': 'new', 'branch': 'new'}),
            patch.object(source.previous, 'verify_model', return_value={'model': self.model}),
            patch.object(source.previous.snapshots, 'model_locks', return_value={str(self.root / 'model'): self.digest(self.root / 'model')}),
            patch.object(source.contract, 'build_config', side_effect=lambda branch, method: {'run': {'study': branch + '_study', 'variant': method, 'seed': 42}, 'args': {'max_train_steps': 1000}}),
            patch.object(source.previous.contract, 'build_config', side_effect=lambda branch, method: {'run': {'study': branch + '_study', 'variant': method, 'seed': 42}, 'args': {'max_train_steps': 1000}})]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def prepare(self):
        data = lambda branch, count: {'rows': count, 'references': [{'id': i} for i in range(count)],
            'source_locks': {str(self.root / (branch + '_data')): self.digest(self.root / (branch + '_data'))}}
        with patch.object(source.contract, 'verify_dataset', return_value=data('subject', 27)) as s_data, \
             patch.object(source.previous.contract, 'verify_dataset', return_value=data('texture', 16)) as t_data, \
             patch.object(source.previous, 'check_pair') as pair:
            result = source.prepare()
        return result, pair, s_data, t_data

    def test_paused_generation_does_not_block_successful_training_pairs_or_enter_locks(self):
        result, pair, s_data, t_data = self.prepare()
        self.assertEqual(set(result), {400, 600})
        self.assertEqual(pair.call_count, 6)  #T600 methods once, S400/600 methods once each
        s_data.assert_called_once_with('subject')
        t_data.assert_called_once_with('texture')
        for step, audit in result.items():
            self.assertEqual(len(audit['references']['subject']), 27)
            self.assertEqual(len(audit['references']['texture']), 16)
            self.assertEqual(audit['training_gits']['subject'], source.SUBJECT_GIT)
            self.assertEqual(audit['training_gits']['texture'], source.TEXTURE_GIT)
            self.assertEqual(audit['sources']['subject']['fullrank']['step'], step)
            self.assertTrue(all(Path(path).name not in ('experiment.json', 'status.json', 'provenance.json')
                                for path in audit['source_locks']))
        self.assertTrue(all(path.name not in ('status.json', 'provenance.json') for path in self.reads))

    def test_wrong_step_method_or_empty_payload_locks_are_rejected(self):
        item = self.receipts['subject', 'fullrank', 400]
        saved = copy.deepcopy(item)
        for key, value in (('steps', 600), ('method', 'lora_r64'), ('hashes', {})):
            with self.subTest(key=key):
                item.clear()
                item.update(copy.deepcopy(saved))
                item[key] = value
                with self.assertRaisesRegex(ValueError, 'training or paired export differs'):
                    self.prepare()

    def test_wrong_historical_git_or_failed_training_manifest_are_rejected(self):
        index = self.indices['subject', 'fullrank']
        index['git'] = source.TEXTURE_GIT
        with self.assertRaisesRegex(ValueError, 'training or paired export differs'):
            self.prepare()
        index['git'] = source.SUBJECT_GIT
        run = Path(self.receipts['subject', 'fullrank', 400]['run'])
        self.files[run / 'manifest.json']['status'] = 'failed'
        with self.assertRaisesRegex(ValueError, 'training or paired export differs'):
            self.prepare()


if __name__ == '__main__':
    unittest.main()
