"""Check the bounded diagnostic plan and that ablations only change copied gates."""

import unittest

import numpy as np

from experiments.sd21_background_diagnosis_v1 import run as diagnosis

try:
    import torch
except ModuleNotFoundError:
    torch = None


class BackgroundDiagnosisTest(unittest.TestCase):
    def test_exactly20_residual_cases_and_two_literal_stock_cases(self):
        rows = diagnosis.plan()
        self.assertEqual(len(rows), 22)
        self.assertEqual(len({row['image_path'] for row in rows}), 22)
        for method in ('fullrank', 'lora_r64'):
            for condition in diagnosis.DISABLED:
                selected = [row for row in rows if row['method'] == method and row['condition'] == condition]
                self.assertEqual([row['seed'] for row in selected], [31, 38])
                self.assertEqual({row['prompt'] for row in selected},
                                 {'a photo of <S*> mailbox in <C*> color with <T*> texture surrounded by flowers'})
        stock = [row for row in rows if row['method'] == 'stock']
        self.assertEqual([row['seed'] for row in stock], [31, 38])
        self.assertEqual({row['prompt'] for row in stock}, {'a photo of a mailbox surrounded by flowers'})

    def test_every_ablation_preserves_other_slots_shape_negative_cfg_and_original(self):
        original = {'modifier_token_mask': {branch: np.zeros((2, 77), dtype=bool)
                                             for branch in ('subject', 'color', 'texture')}}
        for branch, position in zip(original['modifier_token_mask'], (4, 7, 10)):
            original['modifier_token_mask'][branch][1, position] = True
        for condition, disabled in diagnosis.DISABLED.items():
            result = diagnosis.condition_gates(original, condition)
            for branch, mask in result['modifier_token_mask'].items():
                self.assertEqual(mask.shape, (2, 77))
                self.assertEqual(mask.dtype, np.dtype(bool))
                self.assertFalse(mask[0].any())
                self.assertEqual(int(mask.sum()), 0 if branch in disabled else 1)
                self.assertFalse(np.shares_memory(mask, original['modifier_token_mask'][branch]))
                self.assertEqual(int(original['modifier_token_mask'][branch].sum()), 1)
        with self.assertRaisesRegex(ValueError, 'unknown'):
            diagnosis.condition_gates(original, 'erase_embeddings')

    @unittest.skipIf(torch is None, 'requires server torch')
    def test_real_torch_gate_copies_keep_boolean_masks_and_leave_inputs_unchanged(self):
        original = {'modifier_token_mask': {name: torch.tensor([[False, False], [False, True]])
                                             for name in ('subject', 'color', 'texture')}}
        for condition, disabled in diagnosis.DISABLED.items():
            result = diagnosis.condition_gates(original, condition)
            for branch, mask in result['modifier_token_mask'].items():
                source = original['modifier_token_mask'][branch]
                expected = torch.zeros_like(source) if branch in disabled else source
                self.assertEqual(mask.dtype, torch.bool)
                self.assertTrue(torch.equal(mask, expected))
                self.assertNotEqual(mask.data_ptr(), source.data_ptr())
                self.assertEqual(int(source.sum()), 1)


if __name__ == '__main__':
    unittest.main()
