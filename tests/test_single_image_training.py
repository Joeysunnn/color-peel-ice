import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from src.methods.colorpeel_ice import single_image_training as training


class Tokenizer:
    def __init__(self):
        self.tokens = {}

    def add_tokens(self, tokens):
        added = sum(token not in self.tokens for token in tokens)
        self.tokens.update({token: index + 100 for index, token in enumerate(tokens)})
        return added

    def encode(self, text, **kwargs):
        return [0] + [self.tokens.get(word, 5) for word in text.split()] + [1]

    def convert_tokens_to_ids(self, token):
        return self.tokens[token]


class SingleImageTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.record = {'id': 'object_007'}
        for key in ('source_image', 'object_mask', 'vlm_input'):
            path = self.source / (key + '.png')
            Image.new('L' if key == 'object_mask' else 'RGB', (512, 512), 255).save(path)
            self.record.update({key: str(path), key + '_sha256': training.sha256(path)})
        image = self.source / 'derived_rgb.png'
        mask = self.source / 'derived_mask.png'
        Image.new('RGB', (512, 512), (130, 60, 10)).save(image)
        Image.new('L', (512, 512), 255).save(mask)
        self.datasets = {}
        for branch in training.BRANCHES:
            self.datasets[branch] = {
                'source_image_sha256': self.record['source_image_sha256'],
                'source_mask_sha256': self.record['object_mask_sha256'],
                'source_locks': {str(image): training.sha256(image), str(mask): training.sha256(mask)},
                'rows': [{'id': f'{branch}_{index}', 'image': str(image),
                          'mask': None if branch == 'color' else str(mask),
                          'caption': f'a photo of {training.TOKENS[branch]} vase'}
                         for index in range(training.COUNTS[branch])]}

    def test_defaults_and_selected_branch_step_override(self):
        self.assertEqual(training.training_plan(), {'subject': 1000, 'color': 150,
                                                    'material': 200, 'texture': 400})
        self.assertEqual(training.training_plan(['texture'], {'texture': 600, 'color': 80}), {'texture': 600})
        for branches, steps in (([], None), (['subject', 'subject'], None), (['unknown'], None),
                                (None, {'color': 0}), (None, {'subject': True}),
                                (None, {'texture': 3.2}), (None, {'new': 10})):
            with self.subTest(branches=branches, steps=steps), self.assertRaises(ValueError):
                training.training_plan(branches, steps)

    def test_only_selected_data_staged_and_byte_changes_detected(self):
        output = self.root / 'staged'
        receipt = training.stage_datasets(output, self.record, self.datasets, ['color'], Tokenizer())
        self.assertEqual(receipt['counts'], {'color': 9})
        self.assertFalse((output / 'subject').exists())
        audit = training.verify_dataset(output, 'color')
        self.assertEqual(audit['rows'], 9)
        self.assertTrue(all(row['mask'] is None for row in audit['references']))
        staged = Path(audit['references'][0]['image'])
        self.assertEqual(staged.read_bytes(), Path(self.datasets['color']['rows'][0]['image']).read_bytes())
        staged.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            training.verify_dataset(output, 'color')

    def test_checkpoint_schedule_requires_ordered_distinct_steps_and_final_step(self):
        plan = {'material': 200, 'texture': 500}
        self.assertEqual(training.checkpoint_plan(plan), {'material': [200], 'texture': [500]})
        self.assertEqual(training.checkpoint_plan(plan, {'material': [150, 200]}),
                         {'material': [150, 200], 'texture': [500]})
        invalid = [[], [0, 200], [-1, 200], [True, 200], [1.5, 200], [150, 150, 200],
                   [200, 150], [150, 201], [150], '150,200', None]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                training.checkpoint_plan(plan, {'material': values})
        for branch in ('subject', 'unknown'):
            with self.subTest(branch=branch), self.assertRaisesRegex(ValueError, 'selected'):
                training.checkpoint_plan(plan, {branch: [200]})

    def test_wrong_source_mask_hash_and_wrong_count_are_rejected(self):
        for index, mutate in enumerate((
            lambda data: data['subject'].update(source_mask_sha256='0' * 64),
            lambda data: data['subject']['rows'].pop(),
            lambda data: data['subject']['rows'][0].update(caption='a photo of <C*> vase'),
            lambda data: data['subject']['rows'][0].update(caption='a ' * 80 + '<S*> vase'),
        )):
            datasets = copy.deepcopy(self.datasets)
            mutate(datasets)
            with self.subTest(index=index), self.assertRaises(ValueError):
                training.stage_datasets(self.root / f'invalid_{index}', self.record,
                                        datasets, ['subject'], Tokenizer())

    def test_empty_training_mask_is_not_accepted(self):
        path = self.source / 'empty_mask.png'
        Image.new('L', (512, 512), 0).save(path)
        data = copy.deepcopy(self.datasets)
        data['subject']['rows'][0]['mask'] = str(path)
        data['subject']['source_locks'][str(path)] = training.sha256(path)
        with self.assertRaisesRegex(ValueError, 'nonempty binary'):
            training.stage_datasets(self.root / 'invalid_mask', self.record, data, ['subject'], Tokenizer())

    def audit(self):
        return {'object': self.record, 'branches': list(training.BRANCHES),
                'steps': dict(training.DEFAULT_STEPS), 'initializer_manifest': str(self.root / 'initializers.json'),
                'initializers': {'branches': {branch: {'initializer': word, 'initializer_token_id': index}
                    for index, (branch, word) in enumerate(zip(training.BRANCHES,
                                                            ('vase', 'orange', 'glossy', 'uniform')))}}}

    def test_material_target_bound_to_actual_render_metadata(self):
        data = copy.deepcopy(self.datasets['material'])
        data['material_target'] = {'roughness': .35, 'metallic': .1}
        for index, row in enumerate(data['rows']):
            row.update(data['material_target'])
            row['source_image'], row['source_mask'] = row['image'], row['mask']
            metadata = {key: row[key] for key in ('image', 'mask', 'roughness', 'metallic')}
            metadata.update({key + '_sha256': training.sha256(Path(row[key])) for key in ('image', 'mask')})
            path = self.source / f'metadata_{index}.json'
            training._write(path, metadata)
            row.update(metadata=str(path), metadata_sha256=training.sha256(path))
            data['source_locks'][str(path)] = training.sha256(path)
        self.assertEqual(training.audit_material_rows(data), data['material_target'])
        data['rows'][0]['metallic'] = .9
        with self.assertRaisesRegex(ValueError, 'numeric target'):
            training.audit_material_rows(data)
        data['rows'][0]['metallic'] = .1
        path = Path(data['rows'][0]['metadata'])
        metadata = json.loads(path.read_text())
        metadata['roughness'] = .8
        training._write(path, metadata)
        data['rows'][0]['metadata_sha256'] = data['source_locks'][str(path)] = training.sha256(path)
        with self.assertRaisesRegex(ValueError, 'numeric target'):
            training.audit_material_rows(data)

    def test_configs_use_instance_source_and_final_configurable_step(self):
        audit = self.audit()
        audit['steps']['texture'] = 630
        training._write(self.root / 'source_audit.json', audit)
        for branch in training.BRANCHES:
            config = training.build_config(self.root, branch, gpu=3)
            args = config['args']
            self.assertEqual(args['max_train_steps'], audit['steps'][branch])
            self.assertEqual(args['checkpoint_steps'], [audit['steps'][branch]])
            self.assertEqual(args['initializer_token'], audit['initializers']['branches'][branch]['initializer'])
            self.assertEqual(args['learning_rate'], 1e-5)
            self.assertEqual(args['kv_learning_rate'], 5e-5)
            self.assertEqual(args['subject_lora_rank'], 64)
            self.assertEqual(args['subject_lora_alpha'], 64)
            self.assertEqual(args['adam_weight_decay'], 0)
            self.assertEqual(config['environment']['CUDA_VISIBLE_DEVICES'], '3')
            self.assertEqual(config['data_lineage']['source_mask_sha256'], self.record['object_mask_sha256'])
            self.assertNotIn('mailbox', json.dumps(config).lower())
            self.assertEqual(Path(args['concepts_list']), self.root / 'dataset' / branch / 'concepts.json')
        recipe = training.recipe(self.root, 'texture', gpu=3)
        self.assertEqual(recipe.STEPS, (630,))
        self.assertEqual(recipe.CONCEPT_IDS['texture'], 'object_007_texture')
        self.assertEqual(recipe.build_config('texture', 'lora_r64')['args']['checkpoint_steps'], [630])

    def test_prepared_plan_cannot_change_without_new_preparation(self):
        audit = self.audit()
        audit['new_training_git'] = {'commit': 'abc', 'branch': 'codex/test'}
        audit['source_locks'] = training.source_locks(self.record)
        training._write(self.root / 'source_audit.json', audit)
        training._write(self.root / 'prepared.json', {'git': audit['new_training_git'],
            'source_audit_sha256': training.sha256(self.root / 'source_audit.json')})
        with patch.object(training, 'git_info', return_value=audit['new_training_git']):
            self.assertEqual(training.verify_prepared(self.root)['steps'], training.DEFAULT_STEPS)
            audit['steps']['texture'] = 900
            training._write(self.root / 'source_audit.json', audit)
            with self.assertRaisesRegex(ValueError, 'recipe changed'):
                training.verify_prepared(self.root)

    def test_material_texture_schedules_match_config_export_and_smoke(self):
        audit = self.audit()
        audit['steps']['texture'] = 500
        audit['checkpoint_steps'] = {'material': [150, 200], 'texture': [300, 500]}
        training._write(self.root / 'source_audit.json', audit)
        for branch, expected in audit['checkpoint_steps'].items():
            with self.subTest(branch=branch):
                config = training.build_config(self.root, branch)
                recipe = training.recipe(self.root, branch)
                self.assertEqual(config['args']['checkpoint_steps'], expected)
                self.assertEqual(config['args']['max_train_steps'], expected[-1])
                self.assertEqual(recipe.STEPS, tuple(expected))
                self.assertEqual(recipe.build_config(branch, 'lora_r64'), config)
                smoke = recipe.build_config(branch, 'lora_r64', smoke=True)
                self.assertEqual(smoke['args']['checkpoint_steps'], [2])
                self.assertEqual(smoke['args']['max_train_steps'], 2)

    def test_export_retains_initializer_provenance_at_every_requested_step(self):
        audit = self.audit()
        audit['checkpoint_steps'] = {'material': [150, 200]}
        training._write(self.root / 'source_audit.json', audit)
        manifest = Path(audit['initializer_manifest'])
        training._write(manifest, {'selected_branches': ['material']})
        selected = audit['initializers']['branches']['material']
        initial = {'initializer_manifest': str(manifest), 'initializer_manifest_sha256': training.sha256(manifest),
                   'initializer': selected['initializer'], 'initializer_token_id': selected['initializer_token_id'],
                   'modifier_token': '<M*>', 'initialized_equal_base': True, 'initializer_unchanged': True,
                   'base_initializer_sha256': 'abc', 'modifier_after_copy_sha256': 'abc'}
        (self.root / 'run/checkpoints').mkdir(parents=True)
        for name in ('initializer_audit.json', 'adaptation_config.json'):
            training._write(self.root / 'run/checkpoints' / name, initial)

        def export(*args, experiment_contract):
            self.assertEqual(experiment_contract.STEPS, (150, 200))
            self.assertEqual(experiment_contract.build_config('material', 'lora_r64')['args']['checkpoint_steps'],
                             [150, 200])
            for step in (150, 200):
                (args[3] / f'step{step}').mkdir(parents=True)
            return {'source_locks': {}, 'steps': {150: {'hashes': {}}, 200: {'hashes': {}}}}

        with patch.object(training, 'verify_prepared', return_value=audit), \
                patch.object(training.snapshots, 'audit_export', side_effect=export):
            result = training.export(self.root, 'material', self.root / 'run', self.root / 'exports')
        for step in (150, 200):
            receipt = training.read_json(self.root / f'exports/step{step}/receipt.json')
            self.assertEqual(receipt, result['steps'][step])
            self.assertEqual(receipt['initializer_provenance']['initializer'], selected['initializer'])
            self.assertEqual(receipt['hashes'][str(manifest)], training.sha256(manifest))


if __name__ == '__main__':
    unittest.main()
