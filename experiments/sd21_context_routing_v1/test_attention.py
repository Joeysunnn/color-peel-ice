"""CPU projection/output invariants for the fixed Q1 routing experiment."""

import math
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from experiments.sd21_mailbox_sct_v1.attention import SCTTokenLocalKVAttnProcessor, TOKENS
    from experiments.sd21_context_routing_v1.attention import TextureContextRoutingProcessor, install_context_routing

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_cross, self.heads = None, 2
            self.to_q = torch.nn.Linear(64, 64, bias=False)
            self.to_k = torch.nn.Linear(64, 64, bias=False)
            self.to_v = torch.nn.Linear(64, 64, bias=False)
            self.to_out = torch.nn.ModuleList([torch.nn.Linear(64, 64, bias=False), torch.nn.Dropout(0)])
            self.requires_grad_(False)
            self.eval()

        def prepare_attention_mask(self, mask, sequence_length, batch_size):
            return mask

        def head_to_batch_dim(self, tensor):
            batch, length, width = tensor.shape
            return tensor.reshape(batch, length, self.heads, width // self.heads).permute(0, 2, 1, 3).reshape(batch * self.heads, length, width // self.heads)

        def batch_to_head_dim(self, tensor):
            batch_heads, length, width = tensor.shape
            return tensor.reshape(batch_heads // self.heads, self.heads, length, width).permute(0, 2, 1, 3).reshape(batch_heads // self.heads, length, width * self.heads)

        def get_attention_scores(self, query, key, mask):
            scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(query.shape[-1])
            return torch.softmax(scores if mask is None else scores + mask, dim=-1)


@unittest.skipIf(torch is None, 'requires server torch; CPU only')
class RoutingTests(unittest.TestCase):
    def setup_case(self, method='lora_r64'):
        torch.manual_seed(42)
        self.attn = Attention()
        self.original = SCTTokenLocalKVAttnProcessor(64, 64, method)
        with torch.no_grad():
            for parameter in self.original.parameters():
                parameter.normal_(std=0.04)
        self.original.requires_grad_(False).eval()
        self.routed = TextureContextRoutingProcessor(self.original)
        self.hidden = torch.randn(2, 77, 64)
        self.queries = torch.randn(2, 5, 64)
        self.masks = {branch: torch.zeros(2, 77, dtype=torch.bool) for branch in TOKENS}
        for position, branch in zip((3, 8, 12), TOKENS):
            self.masks[branch][1, position] = True

    def output(self, processor, auxiliary=None, masks=None):
        kwargs = {} if auxiliary is None else {'texture_encoder_hidden_states': auxiliary}
        return processor(self.attn, self.queries, encoder_hidden_states=self.hidden,
                         modifier_token_mask=self.masks if masks is None else masks, **kwargs)

    def test_none_and_joint_override_preserve_projection_and_attention_outputs(self):
        for method in ('fullrank', 'lora_r64'):
            self.setup_case(method)
            expected_kv = self.original.project_kv(self.attn, self.hidden, self.masks)
            expected_output = self.output(self.original)
            expected_probs = self.attn.attn_probs.clone()
            for auxiliary in (None, self.hidden):
                with self.subTest(method=method, override=auxiliary is not None):
                    actual_kv = self.routed.project_kv(self.attn, self.hidden, self.masks, auxiliary)
                    for expected, actual in zip(expected_kv, actual_kv):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(self.output(self.routed, auxiliary), expected_output, rtol=0, atol=0)
                    torch.testing.assert_close(self.attn.attn_probs, expected_probs, rtol=0, atol=0)

    def test_q1_changes_only_conditional_texture_kv_and_preserves_unconditional_output(self):
        self.setup_case()
        auxiliary = torch.randn_like(self.hidden) + 3
        before = self.original.project_kv(self.attn, self.hidden, self.masks)
        after = self.routed.project_kv(self.attn, self.hidden, self.masks, auxiliary)
        for index, projection in enumerate(('k', 'v')):
            expected = getattr(self.attn, 'to_' + projection)(self.hidden)
            for branch in TOKENS:
                source = auxiliary if branch == 'texture' else self.hidden
                delta = getattr(self.original, projection + '_residuals')[branch](source)
                expected = torch.where(self.masks[branch].unsqueeze(-1), expected + delta, expected)
            torch.testing.assert_close(after[index], expected, rtol=0, atol=0)
            torch.testing.assert_close(after[index][~self.masks['texture']], before[index][~self.masks['texture']], rtol=0, atol=0)
            self.assertGreater(float((after[index][self.masks['texture']] - before[index][self.masks['texture']]).abs().max()), 0)
        expected_output = self.output(self.original)
        actual_output = self.output(self.routed, auxiliary)
        torch.testing.assert_close(actual_output[0], expected_output[0], rtol=0, atol=0)
        self.assertGreater(float((actual_output[1] - expected_output[1]).abs().max()), 0)

    def test_zero_texture_gate_supports_true_sc_and_ignores_auxiliary(self):
        self.setup_case()
        masks = {branch: value.clone() for branch, value in self.masks.items()}
        masks['texture'].zero_()
        auxiliary = torch.randn_like(self.hidden)
        expected = self.original.project_kv(self.attn, self.hidden, masks)
        actual = self.routed.project_kv(self.attn, self.hidden, masks, auxiliary)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(self.output(self.original, masks=masks),
                                   self.output(self.routed, auxiliary, masks), rtol=0, atol=0)

    def test_wrapper_reuses_existing_layers_and_leaves_self_attention_untouched(self):
        self.setup_case()
        self_processor = SCTTokenLocalKVAttnProcessor(method='lora_r64')
        unet = SimpleNamespace(attn_processors={'block.attn1.processor': self_processor,
                                               'block.attn2.processor': self.original})
        unet.set_attn_processor = lambda value: setattr(unet, 'attn_processors', value)
        initial = {name: value.clone() for name, value in self.original.state_dict().items()}
        parameter_ids = {id(parameter) for parameter in self.original.parameters()}
        install_context_routing(unet)
        routed = unet.attn_processors['block.attn2.processor']
        self.assertIs(routed.original, self.original)
        self.assertIs(unet.attn_processors['block.attn1.processor'], self_processor)
        self.assertEqual({id(parameter) for parameter in routed.parameters()}, parameter_ids)
        self.assertFalse(routed.training)
        for name, value in self.original.state_dict().items():
            torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            install_context_routing(unet)

    def test_invalid_auxiliary_cfg_gates_and_norm_fail_explicitly(self):
        self.setup_case()
        for auxiliary in (self.hidden[:, :-1], self.hidden.double(), 'not a tensor'):
            with self.subTest(auxiliary=type(auxiliary)), self.assertRaises(ValueError):
                self.routed.project_kv(self.attn, self.hidden, self.masks, auxiliary)
        for wrong in ('negative', 'overlap', 'missing'):
            masks = {key: value.clone() for key, value in self.masks.items()}
            if wrong == 'negative':
                masks['texture'][0, 12] = True
            elif wrong == 'overlap':
                masks['texture'][1, 3] = True
            else:
                del masks['texture']
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                self.routed.project_kv(self.attn, self.hidden, masks, self.hidden)
        self.attn.norm_cross = True
        with self.assertRaisesRegex(ValueError, 'norm_cross'):
            self.output(self.routed, self.hidden)


if __name__ == '__main__':
    unittest.main()
