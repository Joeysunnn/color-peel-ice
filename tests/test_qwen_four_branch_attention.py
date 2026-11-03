"""Numerical independence, CFG and atomic export validation for SCMT LoRA64."""

import copy
from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from experiments.sd21_qwen_initializers_v1 import attention
    from experiments.lora_kv_subject_v1.attention import SubjectLoraKVAttnProcessor


@unittest.skipIf(torch is None, 'requires server torch')
class FourBranchAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.attn = SimpleNamespace(to_k=torch.nn.Linear(1024, 64, bias=False),
                                   to_v=torch.nn.Linear(1024, 64, bias=False))
        self.hidden = torch.randn(2, 7, 1024)
        self.processor = attention.FourBranchLoraKVAttnProcessor(64, 1024)
        with torch.no_grad():
            for parameter in self.processor.parameters():
                parameter.normal_(std=0.02)
        self.masks = {branch: torch.zeros(2, 7, dtype=torch.bool) for branch in attention.TOKENS}
        for position, mask in enumerate(self.masks.values(), start=1):
            mask[1, position] = True

    def states(self):
        return {branch: {f'block.attn2.processor.to_{projection}_lora.{key}': value.clone()
                        for projection in ('k', 'v')
                        for key, value in getattr(self.processor, projection + '_residuals')[branch].state_dict().items()}
                for branch in attention.TOKENS}

    def test_each_slot_matches_single_adapter_and_four_slots_never_overwrite(self):
        base = (self.attn.to_k(self.hidden), self.attn.to_v(self.hidden))
        expected = [value.clone() for value in base]
        for branch in attention.TOKENS:
            single = SubjectLoraKVAttnProcessor(64, 1024, 'token_local_kv', 64, 64)
            for projection in ('k', 'v'):
                getattr(single, 'to_' + projection + '_lora').load_state_dict(
                    getattr(self.processor, projection + '_residuals')[branch].state_dict())
            gates = {name: mask if name == branch else torch.zeros_like(mask)
                     for name, mask in self.masks.items()}
            isolated = self.processor.project_kv(self.attn, self.hidden, gates)
            wanted = single.project_kv(self.attn, self.hidden, self.masks[branch])
            for index in (0, 1):
                torch.testing.assert_close(isolated[index], wanted[index], rtol=0, atol=0)
                expected[index][self.masks[branch]] = wanted[index][self.masks[branch]]
        actual = self.processor.project_kv(self.attn, self.hidden, self.masks)
        for got, wanted, stock in zip(actual, expected, base):
            torch.testing.assert_close(got, wanted, rtol=0, atol=0)
            torch.testing.assert_close(got[0], stock[0], rtol=0, atol=0)
            torch.testing.assert_close(got[1, -1], stock[1, -1], rtol=0, atol=0)

    def test_zero_gates_match_stock_complete_attention(self):
        from diffusers.models.attention_processor import Attention, AttnProcessor
        attn = Attention(query_dim=64, cross_attention_dim=1024, heads=4, dim_head=16)
        attn.eval()
        hidden = torch.randn(2, 5, 64)
        zero = {branch: torch.zeros_like(mask) for branch, mask in self.masks.items()}
        expected = AttnProcessor()(attn, hidden, encoder_hidden_states=self.hidden)
        actual = self.processor(attn, hidden, encoder_hidden_states=self.hidden, modifier_token_mask=zero)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_cfg_overlap_missing_and_repeated_gates_fail(self):
        cases = []
        missing = dict(self.masks)
        del missing['material']
        cases.append(missing)
        cases.append(dict(self.masks, material=self.masks['subject']))
        negative = copy.deepcopy(self.masks)
        negative['texture'][0, 3] = True
        cases.append(negative)
        repeated = copy.deepcopy(self.masks)
        repeated['color'][1, 6] = True
        cases.append(repeated)
        for masks in cases:
            with self.assertRaises(ValueError):
                self.processor.project_kv(self.attn, self.hidden, masks)

    def test_loader_validates_all_four_states_before_atomic_install(self):
        installed = []
        unet = SimpleNamespace(attn_processors={'block.attn1.processor': None, 'block.attn2.processor': None},
                               get_submodule=lambda name: self.attn, set_attn_processor=installed.append)
        states, base = self.states(), self.attn.to_k.weight.detach().clone()
        attention.install_four_branch_lora(unet, states)
        loaded = installed[0]['block.attn2.processor']
        self.assertFalse(any(parameter.requires_grad for parameter in loaded.parameters()))
        for got, wanted in zip(loaded.project_kv(self.attn, self.hidden, self.masks),
                               self.processor.project_kv(self.attn, self.hidden, self.masks)):
            torch.testing.assert_close(got, wanted, rtol=0, atol=0)
        torch.testing.assert_close(self.attn.to_k.weight, base, rtol=0, atol=0)
        key = next(iter(states['texture']))
        for mutation in ('missing', 'extra', 'shape', 'nan', 'dtype'):
            bad = copy.deepcopy(states)
            if mutation == 'missing':
                del bad['texture'][key]
            elif mutation == 'extra':
                bad['material']['unexpected'] = torch.zeros(1)
            elif mutation == 'shape':
                bad['color'][key] = torch.zeros(1)
            elif mutation == 'nan':
                bad['subject'][key].flatten()[0] = float('nan')
            else:
                bad['texture'][key] = bad['texture'][key].half()
            with self.assertRaises(ValueError):
                attention.install_four_branch_lora(unet, bad)
        self.assertEqual(len(installed), 1)

    def test_masks_are_prompt_specific_distinct_and_refuse_truncation(self):
        class Tokenizer:
            model_max_length = 77
            def get_vocab(self):
                return dict(zip(attention.TOKENS.values(), (11, 12, 13, 14)))
            def convert_tokens_to_ids(self, token):
                return self.get_vocab()[token]
            def encode(self, prompt, add_special_tokens=True):
                return [0] + [self.get_vocab().get(word, 5) for word in prompt.split()] + [1]
            def __call__(self, prompt, **kwargs):
                ids = self.encode(prompt)
                return SimpleNamespace(input_ids=torch.tensor([ids + [1] * (77 - len(ids))]))
        pipe = SimpleNamespace(tokenizer=Tokenizer(), unet=SimpleNamespace(device='cpu'),
                               maybe_convert_prompt=lambda prompt, tokenizer: prompt)
        for prompt in ('<S*> <C*> <M*> <T*>', 'scene <T*> <M*> <C*> <S*>'):
            gates = attention.attention_masks(pipe, prompt)['modifier_token_mask']
            for branch, token in attention.TOKENS.items():
                self.assertEqual(gates[branch].nonzero().tolist(), [[1, prompt.split().index(token) + 1]])
            self.assertEqual(sum(int(mask.sum()) for mask in gates.values()), 4)
        with self.assertRaisesRegex(ValueError, 'CLIP'):
            attention.attention_masks(pipe, 'word ' * 77 + '<S*> <C*> <M*> <T*>')
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            attention.attention_masks(pipe, '<S*> <C*> <M*> <T*> <M*>')


if __name__ == '__main__':
    unittest.main()
