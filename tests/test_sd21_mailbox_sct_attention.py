"""Numerical S/C/T gating and loading checks, using real torch when available."""

import copy
from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from experiments.sd21_mailbox_sct_v1 import attention
    from experiments.lora_kv_subject_v1.attention import SubjectLoraKVAttnProcessor
    from src.train.custom_attention.attention_processor_custom import TokenLocalKVAttnProcessor


@unittest.skipIf(torch is None, 'requires server torch')
class SCTAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.attn = SimpleNamespace(to_k=torch.nn.Linear(64, 64, bias=False),
                                    to_v=torch.nn.Linear(64, 64, bias=False))
        self.hidden = torch.randn(2, 6, 64)
        self.masks = {branch: torch.zeros(2, 6, dtype=torch.bool) for branch in attention.TOKENS}
        for position, mask in enumerate(self.masks.values()):
            mask[1, position] = True

    def processor(self, method):
        processor = attention.SCTTokenLocalKVAttnProcessor(64, 64, method)
        with torch.no_grad():
            for parameter in processor.parameters():
                parameter.normal_(std=0.03)
        return processor

    def test_each_slot_matches_existing_single_adapter_and_triple_preserves_stock_positions(self):
        for method in attention.METHODS:
            processor = self.processor(method)
            base = (self.attn.to_k(self.hidden), self.attn.to_v(self.hidden))
            expected = [tensor.clone() for tensor in base]
            for branch in attention.TOKENS:
                old = (TokenLocalKVAttnProcessor(64, 64) if method == 'fullrank' else
                       SubjectLoraKVAttnProcessor(64, 64, 'token_local_kv', 64, 64))
                for projection in ('k', 'v'):
                    layer = getattr(old, f'delta_{projection}' if method == 'fullrank' else f'to_{projection}_lora')
                    layer.load_state_dict(getattr(processor, projection + '_residuals')[branch].state_dict())
                gates = {name: mask if name == branch else torch.zeros_like(mask)
                         for name, mask in self.masks.items()}
                actual = processor.project_kv(self.attn, self.hidden, gates)
                single = old.project_kv(self.attn, self.hidden, self.masks[branch])
                for index in (0, 1):
                    torch.testing.assert_close(actual[index], single[index], rtol=0, atol=0)
                    expected[index][self.masks[branch]] = single[index][self.masks[branch]]
            actual = processor.project_kv(self.attn, self.hidden, self.masks)
            for index in (0, 1):
                torch.testing.assert_close(actual[index], expected[index], rtol=0, atol=0)
                torch.testing.assert_close(actual[index][0], base[index][0], rtol=0, atol=0)
            zero = processor.project_kv(self.attn, self.hidden,
                                       {name: torch.zeros_like(mask) for name, mask in self.masks.items()})
            for got, wanted in zip(zero, base):
                torch.testing.assert_close(got, wanted, rtol=0, atol=0)

    def test_overlap_and_missing_branch_masks_are_rejected(self):
        processor = self.processor('fullrank')
        bad = dict(self.masks, texture=self.masks['color'])
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            processor.project_kv(self.attn, self.hidden, bad)
        with self.assertRaisesRegex(ValueError, 'separate'):
            processor.project_kv(self.attn, self.hidden, {'subject': self.masks['subject']})

    def test_loader_preserves_each_source_and_rejects_extra_keys_before_installing(self):
        cross, self_name = 'block.attn2.processor', 'block.attn1.processor'
        for method in attention.METHODS:
            original = self.processor(method)
            states = {branch: {} for branch in attention.TOKENS}
            for branch, state in states.items():
                if method == 'fullrank':
                    state[self_name] = {}
                for projection in ('k', 'v'):
                    prefix = f'delta_{projection}' if method == 'fullrank' else f'to_{projection}_lora'
                    state.update({f'{cross}.{prefix}.{key}': value.clone()
                                  for key, value in getattr(original, projection + '_residuals')[branch].state_dict().items()})
            installed = []
            unet = SimpleNamespace(attn_processors={self_name: None, cross: None},
                                   get_submodule=lambda name: self.attn,
                                   set_attn_processor=installed.append)
            base_weights = [self.attn.to_k.weight.detach().clone(), self.attn.to_v.weight.detach().clone()]
            attention.install_sct_token_local_kv(unet, states, method)
            loaded = installed[0][cross]
            self.assertFalse(any(parameter.requires_grad for parameter in loaded.parameters()))
            for actual, expected in zip(loaded.project_kv(self.attn, self.hidden, self.masks),
                                        original.project_kv(self.attn, self.hidden, self.masks)):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for actual, expected in zip((self.attn.to_k.weight, self.attn.to_v.weight), base_weights):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            bad = copy.deepcopy(states)
            bad['color']['unexpected'] = torch.zeros(1)
            with self.assertRaisesRegex(ValueError, 'unexpected'):
                attention.install_sct_token_local_kv(unet, bad, method)
            self.assertEqual(len(installed), 1)

    def test_three_distinct_masks_have_zero_cfg_negative_half(self):
        class Tokenizer:
            model_max_length = 6
            def get_vocab(self):
                return {'<S*>': 11, '<C*>': 12, '<T*>': 13}
            def convert_tokens_to_ids(self, token):
                return self.get_vocab()[token]
            def __call__(self, *args, **kwargs):
                return SimpleNamespace(input_ids=torch.tensor([[0, 11, 12, 13, 0, 0]]))
        pipe = SimpleNamespace(tokenizer=Tokenizer(), unet=SimpleNamespace(device='cpu'))
        prompt = 'a photo of <S*> mailbox in <C*> color with <T*> texture'
        masks = attention.attention_masks(pipe, prompt)['modifier_token_mask']
        self.assertEqual(set(masks), {'subject', 'color', 'texture'})
        for position, branch in enumerate(attention.TOKENS, start=1):
            self.assertEqual(masks[branch].shape, (2, 6))
            self.assertEqual(masks[branch].nonzero().tolist(), [[1, position]])
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            attention.attention_masks(pipe, prompt + ' <C*>')


if __name__ == '__main__':
    unittest.main()
