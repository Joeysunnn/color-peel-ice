"""Pure CPU checks for the fixed paired continuation and 56-image evaluation."""

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.sd21_texture_bg_preserve_v1.inference import plan as previous_plan
from experiments.sd21_texture_external_preserve_v1 import contract
from experiments.sd21_texture_external_preserve_v1.inference import plan


class ExternalPreservationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.checkpoint = Path.cwd().resolve() / 'fresh' / 'checkpoints' / 'checkpoint-400'
        self.manifest = Path.cwd().resolve() / 'fresh' / 'assets' / 'manifest.json'
        self.parent = {
            'status': 'original', 'run': {'study': 'old', 'variant': 'old', 'seed': 42},
            'data_lineage': {'source': {'rows': 16, 'caption_sha256': 'fixed'}},
            'args': {'max_train_steps': 1000, 'checkpoint_steps': [400, 600, 800, 1000],
                     'learning_rate': 1e-5, 'subject_lora_lr': 5e-5, 'mixed_precision': 'no',
                     'lr_scheduler': 'constant', 'train_batch_size': 1, 'dataloader_num_workers': 0,
                     'modifier_token': '<T*>', 'initializer_token': 'texture',
                     'concepts_list': '/fixed/original/texture16/concepts.json',
                     'subject_lora_mode': 'token_local_kv', 'subject_lora_rank': 64,
                     'with_prior_preservation': False, 'seed': 42}}

    def test_exact56_cases_and_no_extra_concepts(self):
        rows = plan()
        self.assertEqual(contract.ARMS, {'control': 0.0, 'preservation': 10.0})
        self.assertEqual(contract.STEPS, (500, 600))
        self.assertEqual(contract.SEEDS, (31, 38))
        self.assertEqual(len(rows), 56)
        self.assertEqual(len({row['id'] for row in rows}), 56)
        self.assertEqual(len({row['image_path'] for row in rows}), 56)
        self.assertEqual({row['method'] for row in rows}, {'lora_r64'})
        self.assertEqual({row['mode'] for row in rows}, {'sct', 'texture_only'})
        for arm in ('control', 'preservation'):
            for step in (500, 600):
                selected = [row for row in rows if (row['arm'], row['step']) == (arm, step)]
                self.assertEqual(len(selected), 14)
                scenes = [(row['scene'], row['seed']) for row in selected if row['mode'] == 'sct']
                self.assertEqual(set(scenes), {(scene, seed) for scene in ('street', 'snow', 'flowers', 'beach')
                                              for seed in (31, 38)})
                self.assertEqual(len(scenes), 8)
                texture = [row for row in selected if row['mode'] == 'texture_only']
                self.assertEqual([(row['scene'], row['seed']) for row in texture],
                                 [('mug', 31), ('mug', 38), ('vase', 31), ('vase', 38), ('chair', 38), ('sofa', 31)])
                for row in selected:
                    self.assertEqual(row['prompt'].count('<T*>'), 1)
                    self.assertEqual(row['prompt'].count('<S*>'), int(row['mode'] == 'sct'))
                    self.assertEqual(row['prompt'].count('<C*>'), int(row['mode'] == 'sct'))
                    self.assertNotIn('<M*>', row['prompt'])
                    if row['scene'] == 'beach':
                        self.assertTrue(row['prompt'].endswith(' at the beach'))

    def test_previous48_rows_and_schema_are_retained_except_arm_names(self):
        expected = []
        for old in previous_plan():
            row = dict(old)
            arm = {'lambda0': 'control', 'lambda1': 'preservation'}[old['arm']]
            row['arm'] = arm
            row['id'] = old['id'].replace(old['arm'] + '__', arm + '__', 1)
            row['image_path'] = old['image_path'].replace('images/' + old['arm'] + '/', 'images/' + arm + '/', 1)
            expected.append(row)
        self.assertEqual([row for row in plan() if row['scene'] != 'beach'], expected)

    def test_only_allowed_arguments_change_and_source_config_is_untouched(self):
        untouched = copy.deepcopy(self.parent)
        common_changes = {'max_train_steps', 'checkpoint_steps', 'resume_from_checkpoint',
                          'texture_external_preservation_weight'}
        with patch.object(contract.original, 'build_config', return_value=self.parent) as build:
            configs = {arm: contract.build_config(arm, self.checkpoint, self.manifest) for arm in contract.ARMS}
        self.assertEqual(build.call_args_list, [unittest.mock.call('texture', 'lora_r64')] * 2)
        self.assertEqual(self.parent, untouched)
        for arm, config in configs.items():
            allowed = common_changes | ({'texture_external_preservation_manifest'} if arm == 'preservation' else set())
            changed = {key for key in set(self.parent['args']) | set(config['args'])
                       if self.parent['args'].get(key) != config['args'].get(key)}
            self.assertEqual(changed, allowed)
            for key in set(self.parent['args']) - allowed:
                self.assertEqual(config['args'][key], self.parent['args'][key])
            self.assertEqual(config['args']['max_train_steps'], 600)
            self.assertEqual(config['args']['checkpoint_steps'], [500, 600])
            self.assertEqual(config['args']['resume_from_checkpoint'], str(self.checkpoint))
            self.assertEqual(config['data_lineage']['resume_step'], 400)
            self.assertEqual(config['data_lineage']['source'], self.parent['data_lineage']['source'])
            self.assertEqual(config['run']['seed'], 42)
        self.assertNotIn('texture_external_preservation_manifest', configs['control']['args'])
        self.assertEqual(configs['preservation']['args']['texture_external_preservation_manifest'], str(self.manifest))
        self.assertEqual(configs['control']['args']['texture_external_preservation_weight'], 0.0)
        self.assertEqual(configs['preservation']['args']['texture_external_preservation_weight'], 10.0)

    def test_rejects_wrong_branch_method_model_resume_and_relative_manifest(self):
        with patch.object(contract.original, 'build_config', return_value=self.parent):
            for options in ({'branch': 'subject'}, {'branch': 'color'}, {'branch': 'material'},
                            {'method': 'fullrank'}, {'smoke': True}, {'model_dir': self.checkpoint.parent}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    contract.build_config('control', self.checkpoint, self.manifest, **options)
            for arm, checkpoint, manifest in (
                    ('other', self.checkpoint, self.manifest),
                    ('control', Path('checkpoint-400'), self.manifest),
                    ('control', self.checkpoint.with_name('checkpoint-600'), self.manifest),
                    ('preservation', self.checkpoint, Path('manifest.json'))):
                with self.subTest(arm=arm, checkpoint=checkpoint, manifest=manifest), self.assertRaises(ValueError):
                    contract.build_config(arm, checkpoint, manifest)

    def test_recipe_exports_texture_only_and_fixed_checkpoints(self):
        recipe = contract.recipe('preservation', self.checkpoint, self.manifest)
        self.assertEqual(recipe.BRANCHES, ('texture',))
        self.assertEqual(recipe.METHODS, ('lora_r64',))
        self.assertEqual(recipe.STEPS, (500, 600))
        self.assertEqual(recipe.TOKENS, {'texture': '<T*>'})
        self.assertEqual(recipe.CONCEPT_IDS, {'texture': 'wood_spoon_texture'})
        self.assertIs(recipe.verify_dataset, contract.original.verify_dataset)
        with patch.object(contract.original, 'build_config', return_value=self.parent):
            config = recipe.build_config('texture', 'lora_r64')
        self.assertEqual(config['args']['resume_from_checkpoint'], str(self.checkpoint))
        self.assertEqual(config['args']['texture_external_preservation_manifest'], str(self.manifest))


if __name__ == '__main__':
    unittest.main()
