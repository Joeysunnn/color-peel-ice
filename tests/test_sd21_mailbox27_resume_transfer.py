import unittest
from copy import deepcopy
from unittest.mock import patch

from experiments.sd21_mailbox27_resume1000_v1 import contract, inference


class ResumeTransferTest(unittest.TestCase):
    def test_resume_changes_only_endpoint_and_snapshot_arguments(self):
        for method in contract.METHODS:
            original = {'args': {'max_train_steps': 600, 'checkpoint_steps': [200, 400, 600],
                'concepts_list': str(contract.DATASET / 'concepts.json'),
                'pretrained_model_name_or_path': str(contract.MODEL_DIR),
                'learning_rate': 1e-5, 'kv_learning_rate': 1e-5 if method == 'fullrank' else 5e-5,
                'lr_scheduler': 'constant', 'train_batch_size': 1, 'seed': 42},
                'run': {'study': 'old', 'variant': method, 'seed': 42}, 'data_lineage': {}}
            manifest = {'status': 'succeeded', 'returncode': 0, 'git': {'commit': contract.OLD_COMMIT}}
            with patch.object(contract, 'read_config', return_value=deepcopy(original)), \
                    patch.object(contract, 'read_json', side_effect=[original, manifest]):
                actual = contract.build_config('subject', method)
            expected_args = {**original['args'], 'max_train_steps': 1000, 'checkpoint_steps': [800, 1000],
                'resume_from_checkpoint': str(contract.source_run(method) / 'checkpoints/checkpoint-600')}
            self.assertEqual(actual['args'], expected_args)

    def test_paired_800_1000_matte_and_all_seeds(self):
        rows = [row for method in contract.METHODS for step in contract.STEPS
                for row in inference.prompt_rows(method, step)]
        self.assertEqual(len(rows), 396)
        self.assertEqual(len({row['id'] for row in rows}), 396)
        matte = [row for row in rows if row['label'] == 'matte']
        self.assertEqual(len(matte), 44)
        self.assertEqual({row['seed'] for row in matte}, set(range(30, 41)))
        self.assertEqual({row['prompt'] for row in matte}, {'a photo of <S*> mailbox with a matte surface'})
        for method in contract.METHODS:
            for step in contract.STEPS:
                self.assertEqual(len([r for r in matte if r['method'] == method and r['step'] == step]), 11)

    def test_no_old_step_or_unknown_method(self):
        for method, step in [('fullrank', 600), ('other', 800), ('lora_r64', 800.0)]:
            with self.assertRaises(ValueError):
                inference.prompt_rows(method, step)
