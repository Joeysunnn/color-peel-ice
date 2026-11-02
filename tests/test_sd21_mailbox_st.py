"""CPU checks for the authorized Mailbox and wood Texture compositions."""

from contextlib import ExitStack
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import call, patch

from experiments.sd21_mailbox_st_v1 import run as composition


class MailboxCompositionTest(unittest.TestCase):
    def test_matching_methods_have_exactly22_unique_mailbox_rows(self):
        rows = [row for method in composition.METHODS for row in composition.prompt_rows(method)]
        self.assertEqual(composition.METHODS, ('fullrank', 'lora_r64'))
        self.assertEqual(len(rows), 22)
        self.assertEqual(len({row['id'] for row in rows}), 22)
        for method in composition.METHODS:
            self.assertEqual([row['seed'] for row in rows if row['method'] == method], list(range(30, 41)))
        for row in rows:
            self.assertEqual(row['prompt'], 'a photo of <S*> mailbox with <T*> texture')
            self.assertEqual((row['subject_step'], row['attribute_step'], row['subject_images']), (1500, 600, 11))
            self.assertEqual(row['texture_source'], 'wood_spoon_texture')
        with self.assertRaises(ValueError):
            composition.prompt_rows('mixed')

    def test_s1000_t1000_rows_and_invalid_pairs(self):
        rows = [row for method in composition.METHODS
                for row in composition.prompt_rows(method, 1000, 1000)]
        self.assertEqual(len(rows), 22)
        self.assertEqual(len({row['id'] for row in rows}), 22)
        self.assertEqual({(row['subject_step'], row['attribute_step']) for row in rows}, {(1000, 1000)})
        self.assertTrue(all('S1000_wood_T1000' in row['id'] for row in rows))
        self.assertEqual({row['prompt'] for row in rows}, {composition.PROMPT})
        for pair in ((1500, 1000), (1000, 600), (500, 1000)):
            with self.assertRaisesRegex(ValueError, 'authorized'):
                composition.prompt_rows('fullrank', *pair)

    def test_prepare_forwards_distinct_recipes_and_historical_sources(self):
        data = lambda count: {'rows': count, 'references': list(range(count)), 'source_locks': {}}
        model, revision = {'id': 'sd21'}, {'commit': 'new', 'branch': 'new'}
        with patch.dict(os.environ), \
             patch.object(composition, 'git_info', return_value=revision), \
             patch.object(composition, 'read_json', return_value={'status': 'completed', 'attempts': []}), \
             patch.object(composition.shared, 'verify_model', return_value={'model': model}), \
             patch.object(composition.shared.snapshots, 'model_locks', return_value={}), \
             patch.object(composition.subject_contract, 'verify_dataset', return_value=data(11)), \
             patch.object(composition.shared.contract, 'verify_dataset', return_value=data(16)), \
             patch.object(composition.source, '_pair', return_value=({'paired': True}, {})) as pair:
            audit = composition.prepare(1000, 1000)
            self.assertEqual(os.environ[composition.subject_contract.DATASET_ENV], str(composition.SUBJECT_ROOT / 'dataset'))
        expected = []
        for method in composition.METHODS:
            expected += [call(composition.SUBJECT_ROOT, 'subject', method, 1000,
                              composition.SUBJECT_GIT, model, recipe_contract=composition.subject_contract),
                         call(composition.source.TEXTURE_ROOT, 'texture', method, 1000,
                              composition.source.TEXTURE_GIT, model, recipe_contract=composition.shared.contract)]
        self.assertEqual(pair.call_args_list, expected)
        self.assertEqual({key: len(value) for key, value in audit['references'].items()}, {'subject': 11, 'texture': 16})
        self.assertEqual(audit['texture_source'], 'wood_spoon_texture')
        self.assertEqual((audit['subject_step'], audit['texture_step']), (1000, 1000))

    def test_gpu_idle_checks_precede_both_serial_methods(self):
        with tempfile.TemporaryDirectory(prefix='mailbox-st-') as temporary:
            events = []
            revision = {'commit': 'new', 'branch': 'new'}
            def guard():
                self.assertNotIn('fullrank', events)
                events.append('idle')
            def generate(audit, output, method, execute):
                self.assertTrue(execute)
                events.append(method)
                return {'status': 'succeeded', 'generated_images': 11}
            with patch.dict(os.environ), \
                 patch.object(composition, 'resolve_run_dir', side_effect=lambda path: path.resolve()), \
                 patch.object(composition, 'require_idle_gpu0', side_effect=guard), \
                 patch.object(composition, 'prepare', return_value={'inference_git': revision, 'source_locks': {}}), \
                 patch.object(composition, 'generate', side_effect=generate), \
                 patch.object(composition.shared.snapshots, 'check_locks'), \
                 patch.object(composition, 'git_info', return_value=revision):
                state = composition.run(Path(temporary) / 'new', execute=True)
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '0')
            self.assertEqual(events, ['idle', 'idle', 'fullrank', 'lora_r64'])
            self.assertEqual((state['status'], state['generated_images']), ('succeeded', 22))

    def test_final_subject_pair_uses1500_and_rejects_stale_wood_identity(self):
        source = composition.source
        with tempfile.TemporaryDirectory(prefix='mailbox-pair-') as temporary:
            for method in composition.METHODS:
                root = Path(temporary).resolve()
                group = root / 'methods' / method
                run = group / 'training/original'
                checkpoint = group / 'inference/ckpts/step1500'
                snapshot = run / 'checkpoints/checkpoint-1500'
                weight = 'pytorch_token_local_kv_weights.bin' if method == 'fullrank' else 'pytorch_lora_kv_weights.bin'
                config = {'run': {'variant': method}, 'args': {'max_train_steps': 1500}}
                recipe = SimpleNamespace(build_config=lambda *args: config, TOKENS={'subject': '<S*>'},
                                         CONCEPT_IDS={'subject': 'mailbox'})
                paths = [run / 'config.yaml', run / 'manifest.json', checkpoint / weight,
                         checkpoint / '<S*>.bin', checkpoint / 'adaptation_config.json',
                         run / 'checkpoints/embedding_update_audit.json', run / 'checkpoints/training_metrics.jsonl',
                         snapshot / 'pytorch_model.bin', snapshot / 'pytorch_model_1.bin']
                digest = lambda path: str(path)
                item = {'branch': 'subject', 'method': method, 'step': 1500, 'steps': 1500,
                        'training_steps': 1500, 'token': '<S*>', 'weight': weight, 'embedding_dim': 1024,
                        'status': 'verified', 'paired_embedding_finite': True, 'final_snapshot_matches_export': True,
                        'concept_id': 'mailbox', 'run': str(run), 'checkpoint': str(checkpoint),
                        'source_checkpoint': str(run / 'checkpoints'), 'base_model': {'id': 'sd21'},
                        'hashes': {str(path): digest(path) for path in paths},
                        'source_snapshot_hashes': {str(path): digest(path) for path in paths[-2:]},
                        'source_adapter_sha256': digest(snapshot / 'pytorch_model.bin'),
                        'source_text_encoder_sha256': digest(snapshot / 'pytorch_model_1.bin'),
                        'derived_adapter_sha256': digest(checkpoint / weight),
                        'derived_token_sha256': digest(checkpoint / '<S*>.bin')}
                files = {checkpoint / 'receipt.json': item,
                         group / 'inference/ckpts/index.json': {'git': composition.SUBJECT_GIT, 'smoke': False, 'steps': {'1500': item}},
                         run / 'manifest.json': {'status': 'succeeded', 'returncode': 0, 'stage': 'train',
                                                 'git': composition.SUBJECT_GIT, 'run': config['run']},
                         checkpoint / 'adaptation_config.json': {'modifier_tokens': ['<S*>'],
                             'adaptation_mode': 'token_local_kv' if method == 'fullrank' else 'lora_subject_kv',
                             'rank': 64, 'alpha': 64, 'mode': 'token_local_kv'}}
                with ExitStack() as stack:
                    stack.enter_context(patch.object(source, '_job', return_value=(group, run)))
                    stack.enter_context(patch.object(source, 'read_json', side_effect=lambda path: files[Path(path)]))
                    stack.enter_context(patch.object(source, 'read_config', return_value=config))
                    stack.enter_context(patch.object(source, 'sha256', side_effect=digest))
                    stack.enter_context(patch.object(source.previous.snapshots, 'check_locks'))
                    check = stack.enter_context(patch.object(source.previous, 'check_pair'))
                    source._pair(root, 'subject', method, 1500, composition.SUBJECT_GIT, {'id': 'sd21'}, recipe_contract=recipe)
                    check.assert_called_once_with(item, 'subject', method, step=1500, training_steps=1500, token='<S*>')
                    saved = copy.deepcopy(item)
                    for key, value in (('training_steps', 1000), ('concept_id', 'wood_spoon'),
                                       ('final_snapshot_matches_export', False)):
                        with self.subTest(method=method, field=key):
                            item.clear()
                            item.update(copy.deepcopy(saved))
                            item[key] = value
                            with self.assertRaisesRegex(ValueError, 'training or paired export differs'):
                                source._pair(root, 'subject', method, 1500, composition.SUBJECT_GIT,
                                             {'id': 'sd21'}, recipe_contract=recipe)


if __name__ == '__main__':
    unittest.main()
