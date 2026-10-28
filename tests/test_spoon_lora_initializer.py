"""Exercise the actual trainer's LoRA guard without importing CUDA/model modules."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class SpoonLoraInitializerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / 'src/train/train_colorpeel.py'
        tree = ast.parse(source.read_text(encoding='utf8'))
        block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
            and isinstance(node.test, ast.Attribute) and node.test.attr == 'subject_lora_mode'
            and any(isinstance(child, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == 'valid_concept' for target in child.targets) for child in node.body))
        # Compile the original assignments and rejection guard, stopping before
        # trainer imports/processor construction. No validation logic is copied.
        guard = next(index for index, node in enumerate(block.body) if isinstance(node, ast.If)
            and any(isinstance(child, ast.Raise) for child in node.body))
        cls.validation = compile(ast.Module(body=block.body[:guard + 1], type_ignores=[]), str(source), 'exec')

    def validate(self, token='<S*>', initializer='spoon', **changes):
        values = dict(modifier_token=[token], initializer_token=[initializer], subject_lora_mode='token_local_kv',
            token_local_kv=False, perfusion_subject=False, enable_xformers_memory_efficient_attention=False,
            freeze_model='crossattn_kv', k_learning_rate=None, v_learning_rate=None,
            subject_lora_rank=64, subject_lora_alpha=64)
        values.update(changes)
        scope = {'args': SimpleNamespace(**values)}
        exec(self.validation, scope)
        self.assertTrue(scope['valid_concept'])

    def test_spoon_and_existing_mailbox_subjects_pass(self):
        for initializer in ('spoon', 'mailbox'):
            for mode in ('full_kv', 'token_local_kv', 'full_v', 'token_local_v'):
                with self.subTest(initializer=initializer, mode=mode):
                    self.validate(initializer=initializer, subject_lora_mode=mode)

    def test_arbitrary_subject_initializers_and_multiple_tokens_are_rejected(self):
        for initializer in ('wood', 'metal', 'texture', '', 'arbitrary'):
            with self.subTest(initializer=initializer), self.assertRaisesRegex(ValueError, 'LoRA requires'):
                self.validate(initializer=initializer)
        for changes in ({'modifier_token': ['<S*>', '<M*>']}, {'initializer_token': ['spoon', 'mailbox']},
                        {'modifier_token': None}, {'initializer_token': None}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'LoRA requires'):
                self.validate(**changes)

    def test_material_and_texture_rules_are_preserved(self):
        for initializer in ('metal', 'material', 'wood'):
            self.validate('<M*>', initializer)
        self.validate('<T*>', 'texture')
        for token, initializer in (('<M*>', 'spoon'), ('<T*>', 'spoon'), ('<S*>', 'texture'), ('<X*>', 'spoon')):
            with self.subTest(token=token, initializer=initializer), self.assertRaisesRegex(ValueError, 'LoRA requires'):
                self.validate(token, initializer)
        for token, initializer in (('<M*>', 'metal'), ('<T*>', 'texture')):
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, 'LoRA requires'):
                self.validate(token, initializer, subject_lora_mode='full_kv')

    def test_frozen_base_positive_rank_and_other_adapter_guards_remain(self):
        for changes in ({'freeze_model': 'crossattn'}, {'subject_lora_rank': 0}, {'subject_lora_alpha': 0},
                        {'subject_lora_alpha': -1}, {'token_local_kv': True}, {'perfusion_subject': True},
                        {'enable_xformers_memory_efficient_attention': True},
                        {'k_learning_rate': 1e-5}, {'v_learning_rate': 1e-5}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'LoRA requires'):
                self.validate(**changes)


if __name__ == '__main__':
    unittest.main()
