"""Exercise the real trainer guard and adapter metadata without CUDA imports."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ColorTokenLocalGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        trainer = ROOT / 'src/train/train_colorpeel.py'
        tree = ast.parse(trainer.read_text(encoding='utf8'))
        block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
            and isinstance(node.test, ast.Attribute) and node.test.attr == 'subject_lora_mode'
            and any(isinstance(child, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == 'valid_concept' for target in child.targets) for child in node.body))
        guard = next(index for index, node in enumerate(block.body) if isinstance(node, ast.If)
            and any(isinstance(child, ast.Raise) for child in node.body))
        cls.validation = compile(ast.Module(body=block.body[:guard + 1], type_ignores=[]), str(trainer), 'exec')
        metadata = next(node for node in ast.walk(tree) if isinstance(node, ast.Dict)
            and any(isinstance(key, ast.Constant) and key.value == 'adaptation_mode'
                    and isinstance(value, ast.IfExp) for key, value in zip(node.keys, node.values)))
        cls.metadata = compile(ast.Expression(body=metadata), str(trainer), 'eval')
        audit = ROOT / 'experiments/texture_material_disentangled_v1/train.py'
        audit_tree = ast.parse(audit.read_text(encoding='utf8'))
        mode = next(node.value for node in ast.walk(audit_tree) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'expected_mode' for target in node.targets))
        cls.audit_mode = compile(ast.Expression(body=mode), str(audit), 'eval')

    def validate(self, **changes):
        values = dict(modifier_token=['<C*>'], initializer_token=['orange'], subject_lora_mode='token_local_kv',
            token_local_kv=False, perfusion_subject=False, enable_xformers_memory_efficient_attention=False,
            freeze_model='crossattn_kv', k_learning_rate=None, v_learning_rate=None,
            subject_lora_rank=64, subject_lora_alpha=64)
        values.update(changes)
        scope = {'args': SimpleNamespace(**values)}
        exec(self.validation, scope)
        self.assertTrue(scope['valid_concept'])

    def test_orange_color_is_allowed_only_in_token_local_mode(self):
        self.validate()
        for mode in ('full_kv', 'full_v', 'token_local_v'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'LoRA requires'):
                self.validate(subject_lora_mode=mode)
        for initializer in ('purple', 'mailbox', 'spoon', 'texture', '', 'arbitrary'):
            with self.subTest(initializer=initializer), self.assertRaisesRegex(ValueError, 'LoRA requires'):
                self.validate(initializer_token=[initializer])

    def test_color_does_not_relax_single_token_or_adapter_guards(self):
        for changes in ({'modifier_token': ['<S*>', '<C*>']}, {'initializer_token': ['orange', 'orange']},
                        {'modifier_token': None}, {'initializer_token': None}, {'freeze_model': 'crossattn'},
                        {'subject_lora_rank': 0}, {'subject_lora_alpha': 0}, {'token_local_kv': True},
                        {'perfusion_subject': True}, {'enable_xformers_memory_efficient_attention': True},
                        {'k_learning_rate': 1e-5}, {'v_learning_rate': 1e-5}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'LoRA requires'):
                self.validate(**changes)

    def test_existing_subject_material_and_texture_modes_are_preserved(self):
        for initializer in ('mailbox', 'spoon'):
            for mode in ('full_kv', 'token_local_kv', 'full_v', 'token_local_v'):
                self.validate(modifier_token=['<S*>'], initializer_token=[initializer], subject_lora_mode=mode)
        for token, initializers in (('<M*>', ('metal', 'material', 'wood')), ('<T*>', ('texture',))):
            for initializer in initializers:
                self.validate(modifier_token=[token], initializer_token=[initializer])
                with self.assertRaisesRegex(ValueError, 'LoRA requires'):
                    self.validate(modifier_token=[token], initializer_token=[initializer], subject_lora_mode='full_kv')

    def test_export_metadata_and_shared_audit_agree_for_all_tokens(self):
        for token, expected in (('<C*>', 'lora_color_kv'), ('<S*>', 'lora_subject_kv'),
                                ('<M*>', 'lora_material_kv'), ('<T*>', 'lora_subject_kv')):
            with self.subTest(token=token):
                args = SimpleNamespace(modifier_token=[token], subject_lora_mode='token_local_kv',
                                       subject_lora_rank=64, subject_lora_alpha=64)
                metadata = eval(self.metadata, {'args': args,
                    'SUBJECT_LORA_KV_WEIGHT_NAME': 'pytorch_lora_kv_weights.bin'})
                self.assertEqual(metadata['adaptation_mode'], expected)
                self.assertEqual(metadata['modifier_tokens'], [token])
                self.assertEqual((metadata['mode'], metadata['rank'], metadata['alpha']), ('token_local_kv', 64, 64))
                self.assertEqual(eval(self.audit_mode, {'method': 'lora_r64', 'token': token}), expected)
                self.assertEqual(eval(self.audit_mode, {'method': 'fullrank', 'token': token}), 'token_local_kv')


if __name__ == '__main__':
    unittest.main()
