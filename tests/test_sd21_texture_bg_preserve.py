"""CPU checks for scientific scope and the nontrivial prefix-evidence inference."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.sd21_texture_bg_preserve_v1 import contract
from experiments.sd21_texture_bg_preserve_v1.inference import plan
from experiments.sd21_texture_bg_preserve_v1.resume_source import validated_prefix
from src.train.resume_utils import METRIC_FIELDS


class TextureBackgroundExperimentTests(unittest.TestCase):
    def evidence(self):
        rows = [{**{name: 0.1 for name in METRIC_FIELDS}, 'step': step,
                 'present_modifier_tokens': ['<T*>'], 'losses_finite': {'all': True}}
                for step in range(1, 1001)]
        audit = {'observed_optimization_steps': 1000, 'non_modifier_embedding_drift': {'changed_rows': 0},
                 'modifier_tokens': [{'token': '<T*>', 'token_id': 49408, 'exposure_steps': 1000,
                                      'nonzero_gradient_steps': 1000}]}
        return rows, audit

    def test_prefix_retains_original_bytes(self):
        rows, audit = self.evidence()
        lines = [(json.dumps(row, separators=(',', ':')) + '\r\n').encode() for row in rows]
        prefix, token_id = validated_prefix(b''.join(lines), audit)
        self.assertEqual(prefix, b''.join(lines[:400]))
        self.assertEqual(token_id, 49408)

    def test_prefix_requires_every_step_proof(self):
        rows, audit = self.evidence()
        payload = ('\n'.join(json.dumps(row) for row in rows) + '\n').encode()
        for field in ('exposure_steps', 'nonzero_gradient_steps'):
            malformed = copy.deepcopy(audit)
            malformed['modifier_tokens'][0][field] = 999
            with self.assertRaises(ValueError):
                validated_prefix(payload, malformed)
        rows[999]['total_loss'] = float('nan')
        with self.assertRaises(ValueError):
            validated_prefix(('\n'.join(json.dumps(row) for row in rows) + '\n').encode(), audit)
        rows[999]['total_loss'] = 0.1
        rows[401]['step'] = 401
        with self.assertRaises(ValueError):
            validated_prefix(('\n'.join(json.dumps(row) for row in rows) + '\n').encode(), audit)

    def test_only_authorized_recipe_changes(self):
        parent = {'run': {'study': 'old', 'variant': 'old', 'seed': 42}, 'data_lineage': {},
                  'args': {'max_train_steps': 1000, 'checkpoint_steps': [400, 600, 800, 1000],
                           'learning_rate': 1e-5, 'subject_lora_lr': 5e-5, 'mixed_precision': 'no',
                           'lr_scheduler': 'constant', 'train_batch_size': 1, 'dataloader_num_workers': 0}}
        path = Path.cwd() / 'fresh' / 'checkpoints' / 'checkpoint-400'
        with patch.object(contract.original, 'build_config', return_value=parent):
            left, right = (contract.build_config(arm, path) for arm in contract.ARMS)
            for config in (left, right):
                changed = {key for key in set(parent['args']) | set(config['args'])
                           if parent['args'].get(key) != config['args'].get(key)}
                self.assertEqual(changed, {'max_train_steps', 'checkpoint_steps', 'resume_from_checkpoint',
                                          'texture_background_preservation_weight'})
                self.assertEqual(config['args']['max_train_steps'], 600)
                self.assertEqual(config['args']['checkpoint_steps'], [500, 600])
            differences = {key for key in left['args'] if left['args'][key] != right['args'][key]}
            self.assertEqual(differences, {'texture_background_preservation_weight'})
            with self.assertRaises(ValueError):
                contract.build_config('lambda1', path, method='fullrank')
            with self.assertRaises(ValueError):
                contract.build_config('lambda1', path, smoke=True)
        self.assertNotIn('resume_from_checkpoint', parent['args'])

    def test_exact48_images_and_prompt_tokens(self):
        rows = plan()
        self.assertEqual(len(rows), 48)
        self.assertEqual(len({row['id'] for row in rows}), 48)
        self.assertEqual(len({row['image_path'] for row in rows}), 48)
        self.assertEqual({row['method'] for row in rows}, {'lora_r64'})
        for arm in contract.ARMS:
            for step in (500, 600):
                selected = [row for row in rows if row['arm'] == arm and row['step'] == step]
                self.assertEqual(len(selected), 12)
                self.assertEqual({row['seed'] for row in selected}, {31, 38})
                self.assertEqual(sum(row['mode'] == 'sct' for row in selected), 6)
                self.assertEqual([(row['scene'], row['seed']) for row in selected if row['scene'] in ('chair', 'sofa')],
                                 [('chair', 38), ('sofa', 31)])
                for row in selected:
                    self.assertEqual(row['prompt'].count('<T*>'), 1)
                    self.assertEqual(row['prompt'].count('<S*>'), int(row['mode'] == 'sct'))
                    self.assertEqual(row['prompt'].count('<C*>'), int(row['mode'] == 'sct'))


if __name__ == '__main__':
    unittest.main()
