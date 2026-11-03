"""Verify the paired comparison changes only its declared scientific variables."""

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.sd21_tokenverse_isolation_v1 import contract


class IsolationContractTests(unittest.TestCase):
    def setUp(self):
        self.manifest = Path.cwd().resolve() / 'fresh_bank/manifest.json'
        self.parent = {'run': {'study': 'old', 'variant': 'old', 'seed': 42},
            'environment': {'HF_HUB_OFFLINE': '1'}, 'data_lineage': {'rows': 16},
            'args': {'max_train_steps': 1500, 'checkpoint_steps': [500, 1000, 1500],
                'learning_rate': 1e-5, 'kv_learning_rate': 5e-5, 'cos_weight': 0.0,
                'mixed_precision': 'no', 'dataloader_num_workers': 2, 'hflip': False,
                'train_batch_size': 1, 'gradient_accumulation_steps': 1,
                'subject_lora_mode': 'token_local_kv', 'subject_lora_rank': 64,
                'adam_weight_decay': 0.0, 'seed': 42}}

    def test_fresh_paired_args_preserve_original_foreground_recipe(self):
        for branch in contract.BRANCHES:
            untouched = copy.deepcopy(self.parent)
            with patch.object(contract.RECIPES[branch], 'build_config', return_value=self.parent):
                configs = {arm: contract.build_config(branch, arm, self.manifest) for arm in contract.ARMS}
            self.assertEqual(self.parent, untouched)
            for arm, config in configs.items():
                allowed = {'max_train_steps', 'checkpoint_steps', 'gradient_checkpointing', 'concept_isolation_weight'}
                if arm == 'isolation':
                    allowed.add('concept_isolation_manifest')
                self.assertTrue({key for key in set(self.parent['args']) | set(config['args'])
                                 if self.parent['args'].get(key) != config['args'].get(key)} <= allowed)
                self.assertEqual(config['args']['max_train_steps'], contract.STEPS[branch][-1])
                self.assertEqual(config['args']['checkpoint_steps'], list(contract.STEPS[branch]))
                self.assertTrue(config['args']['gradient_checkpointing'])
                self.assertEqual(config['environment']['CUDA_VISIBLE_DEVICES'], str(contract.GPUS[branch]))
                self.assertEqual(config['args']['concept_isolation_weight'], contract.ARMS[arm])
                self.assertFalse(config['data_lineage']['old_weights_reused'])
                self.assertNotIn('resume_from_checkpoint', config['args'])
            self.assertNotIn('concept_isolation_manifest', configs['control']['args'])
            changed = {key for key in set(configs['control']['args']) | set(configs['isolation']['args'])
                       if configs['control']['args'].get(key) != configs['isolation']['args'].get(key)}
            self.assertEqual(changed, {'concept_isolation_weight', 'concept_isolation_manifest'})

    def test_fixed_scope_and_checkpoint_plan(self):
        self.assertEqual(contract.BRANCHES, ('subject', 'color', 'texture'))
        self.assertEqual(contract.METHODS, ('lora_r64',))
        self.assertEqual(contract.ARMS, {'control': 0.0, 'isolation': 1.0})
        self.assertEqual(contract.STEPS, {'subject': (500, 1000), 'color': (100, 150), 'texture': (500, 1000)})
        for branch in contract.BRANCHES:
            recipe = contract.recipe(branch, 'isolation', self.manifest)
            self.assertEqual(recipe.BRANCHES, (branch,))
            self.assertEqual(recipe.STEPS, contract.STEPS[branch])
            self.assertEqual(recipe.TOKENS[branch], contract.TOKENS[branch])

    def test_rejects_outside_scope_or_resume(self):
        for options in ({'branch': 'material'}, {'arm': 'other'}, {'method': 'fullrank'},
                        {'smoke': True}, {'manifest': Path('relative.json')}, {'model_dir': Path('wrong')}):
            values = {'branch': 'texture', 'arm': 'control', 'manifest': self.manifest, **options}
            with self.subTest(options=options), self.assertRaises(ValueError):
                contract.build_config(**values)
        self.parent['args']['resume_from_checkpoint'] = '/old/checkpoint-400'
        with patch.object(contract.texture, 'build_config', return_value=self.parent), self.assertRaises(ValueError):
            contract.build_config('texture', 'isolation', self.manifest)


if __name__ == '__main__':
    unittest.main()
