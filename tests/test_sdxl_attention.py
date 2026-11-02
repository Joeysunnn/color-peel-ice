import unittest

import torch
from torch import nn
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from experiments.sdxl_mailbox_v1.attention import (
    TokenLocalKVProcessor, adapter_parameters, adapter_state_dict,
    install_token_local_kv, load_adapter_state_dict,
)


class TinyUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = Attention(query_dim=8, heads=2, dim_head=4)
        self.attn2 = Attention(query_dim=8, cross_attention_dim=12, heads=2, dim_head=4)

    @property
    def attn_processors(self):
        return {"attn1.processor": self.attn1.processor, "attn2.processor": self.attn2.processor}

    def set_attn_processor(self, processors):
        for name, processor in processors.items():
            self.get_submodule(name.removesuffix(".processor")).set_processor(processor)


class SDXLAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)

    def test_zero_adapter_matches_native_sdpa_including_4d_norm_residual(self):
        for method in ("fullrank", "lora_r64"):
            for spatial in (False, True):
                attn = Attention(query_dim=8, cross_attention_dim=12, heads=2, dim_head=4,
                                 norm_num_groups=2, cross_attention_norm="layer_norm",
                                 residual_connection=True, rescale_output_factor=2.0)
                states = torch.randn(2, 8, 2, 3) if spatial else torch.randn(2, 6, 8)
                text = torch.randn(2, 5, 12)
                gate = torch.zeros(2, 5, dtype=torch.bool)
                gate[1, 2] = True
                expected = AttnProcessor2_0()(attn, states, text)
                processor = TokenLocalKVProcessor(8, 12, method, rank=2, alpha=2)
                actual = processor(attn, states, text, modifier_token_mask=gate)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_nonzero_residual_only_changes_target_cfg_positions(self):
        for method in ("fullrank", "lora_r64"):
            attn = Attention(query_dim=8, cross_attention_dim=12, heads=2, dim_head=4)
            attn.requires_grad_(False)
            processor = TokenLocalKVProcessor(8, 12, method, rank=2, alpha=4)
            with torch.no_grad():
                for parameter in processor.parameters():
                    parameter.fill_(0.1)
            text = torch.randn(2, 5, 12, requires_grad=True)
            gate = torch.zeros(2, 5, dtype=torch.bool)
            gate[1, 2] = True
            key, value = processor.project_kv(attn, text, gate)
            torch.testing.assert_close(key[~gate], attn.to_k(text)[~gate], atol=0, rtol=0)
            torch.testing.assert_close(value[~gate], attn.to_v(text)[~gate], atol=0, rtol=0)
            self.assertFalse(torch.equal(key[gate], attn.to_k(text)[gate]))
            (key.sum() + value.sum()).backward()
            self.assertTrue(torch.all(text.grad[~gate].abs().sum(-1) > 0))
            self.assertTrue(all(parameter.grad is not None for parameter in processor.parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in attn.parameters()))
            if method == "lora_r64":
                expected = processor.k_up(processor.k_down(text)) * 2
                torch.testing.assert_close((key - attn.to_k(text))[gate], expected[gate])

    def test_install_preserves_self_attention_and_round_trips_exact_weights(self):
        for method in ("fullrank", "lora_r64"):
            source, target = TinyUnet(), TinyUnet()
            original_self = source.attn1.processor
            install_token_local_kv(source, method, rank=2, alpha=2)
            install_token_local_kv(target, method, rank=2, alpha=2)
            self.assertIs(source.attn1.processor.delegate, original_self)
            states = torch.randn(1, 5, 8)
            expected_self = original_self(source.attn1, states)
            actual_self = source.attn1(states, modifier_token_mask=torch.zeros(1, 3, dtype=torch.bool))
            torch.testing.assert_close(actual_self, expected_self, atol=0, rtol=0)
            adapters = adapter_parameters(source)
            self.assertEqual(set(map(id, adapters)), {id(p) for p in source.parameters() if p.requires_grad})
            self.assertTrue(all(parameter.dtype == torch.float32 for parameter in adapters))
            with torch.no_grad():
                for parameter in adapters:
                    parameter.normal_()
            state = adapter_state_dict(source)
            load_adapter_state_dict(target, state)
            for key, value in adapter_state_dict(target).items():
                torch.testing.assert_close(value, state[key], atol=0, rtol=0)
            bad = dict(state)
            bad.pop(next(iter(bad)))
            with self.assertRaises(ValueError):
                load_adapter_state_dict(target, bad)
            bad = dict(state)
            bad[next(iter(bad))] = torch.tensor([float("nan")])
            with self.assertRaises(ValueError):
                load_adapter_state_dict(target, bad)

    def test_fp32_master_parameters_work_under_autocast(self):
        model = TinyUnet()
        install_token_local_kv(model, "lora_r64", rank=2, alpha=2)
        gate = torch.tensor([[False, True, False]])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model.attn2(torch.randn(1, 4, 8), torch.randn(1, 3, 12), modifier_token_mask=gate)
            loss = output.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.dtype == torch.float32 and p.grad is not None
                            and torch.isfinite(p.grad).all() for p in adapter_parameters(model)))

    def test_gate_mismatch_is_rejected(self):
        model = TinyUnet()
        processor = TokenLocalKVProcessor(8, 12, "fullrank")
        for gate in (None, torch.zeros(1, 3), torch.zeros(1, 2, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                processor.project_kv(model.attn2, torch.randn(1, 3, 12), gate)


if __name__ == "__main__":
    unittest.main()
