from types import SimpleNamespace
import unittest

import torch
from torch import nn
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

from experiments.sdxl_mailbox_v1.conditioning import (
    FrozenTokenEmbedding, encode_conditioning, install_learned_tokens, make_time_ids,
)


class TinyTokenizer:
    eos_token_id = 10
    model_max_length = 8

    def __init__(self):
        self.vocab = {word: i for i, word in enumerate(
            ("bos", "pad", "mailbox", "a", "photo", "of", "in", "orange", "color", "unused", "eos"))}
        self.insert_before_target = False

    def get_vocab(self):
        return dict(self.vocab)

    def __len__(self):
        return len(self.vocab)

    def encode(self, prompt, add_special_tokens=False):
        return [self.vocab[word] for word in prompt.split()]

    def add_tokens(self, token):
        self.vocab[token] = len(self.vocab)
        return 1

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def __call__(self, prompts, padding, max_length, truncation, return_tensors):
        rows = []
        for prompt in prompts:
            tokens = self.encode(prompt)
            if self.insert_before_target and 11 in tokens:
                tokens.insert(tokens.index(11), 9)
            tokens = ([0] + tokens)[:max_length - 1] + [10]
            rows.append(tokens + [1] * (max_length - len(tokens)))
        return SimpleNamespace(input_ids=torch.tensor(rows))


def models_and_tokenizers():
    common = dict(vocab_size=11, intermediate_size=24, num_hidden_layers=2,
                  num_attention_heads=2, max_position_embeddings=8, eos_token_id=2,
                  bos_token_id=0, pad_token_id=1, attention_dropout=0.0)
    encoders = [CLIPTextModel(CLIPTextConfig(hidden_size=8, **common)),
                CLIPTextModelWithProjection(CLIPTextConfig(hidden_size=12, projection_dim=12, **common))]
    return [TinyTokenizer(), TinyTokenizer()], [encoder.eval() for encoder in encoders]


class SDXLConditioningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_only_appended_vector_changes_after_adamw_updates(self):
        base = nn.Embedding(11, 8)
        original = base.weight.detach().clone()
        wrapper = FrozenTokenEmbedding(base, 11, 2)
        torch.testing.assert_close(wrapper.vector, original[2], atol=0, rtol=0)
        optimizer = torch.optim.AdamW([wrapper.vector], lr=0.1, weight_decay=0)
        for _ in range(3):
            optimizer.zero_grad()
            wrapper(torch.tensor([[0, 11, 10]])).square().sum().backward()
            optimizer.step()
        torch.testing.assert_close(wrapper.base.weight, original, atol=0, rtol=0)
        self.assertFalse(torch.equal(wrapper.vector, original[2]))
        self.assertIsNone(wrapper.base.weight.grad)

    def test_native_conditioning_equivalence_and_eos_after_token_append(self):
        tokenizers, encoders = models_and_tokenizers()
        ids = tokenizers[0](["a mailbox"], padding="max_length", max_length=8,
                            truncation=True, return_tensors="pt").input_ids
        before = [encoder(ids, output_hidden_states=True) for encoder in encoders]
        wrappers = install_learned_tokens(tokenizers, encoders, "<S*>", "mailbox")
        hidden, pooled, gate = encode_conditioning(tokenizers, encoders, ["a mailbox"], "<S*>")
        torch.testing.assert_close(hidden, torch.cat([output.hidden_states[-2] for output in before], -1), atol=0, rtol=0)
        torch.testing.assert_close(pooled, before[1].text_embeds, atol=0, rtol=0)
        self.assertFalse(gate.any())
        hidden, pooled, gate = encode_conditioning(tokenizers, encoders, ["a <S*> mailbox"], "<S*>")
        self.assertEqual(hidden.shape, (1, 8, 20))
        self.assertEqual(pooled.shape, (1, 12))
        self.assertEqual(gate.nonzero().tolist(), [[0, 2]])
        ids = tokenizers[1](["a <S*> mailbox"], padding="max_length", max_length=8,
                            truncation=True, return_tensors="pt").input_ids
        output = encoders[1](ids, output_hidden_states=True)
        expected = encoders[1].text_projection(output.last_hidden_state[:, 4])
        wrong = encoders[1].text_projection(output.last_hidden_state[:, 2])
        torch.testing.assert_close(pooled, expected, atol=0, rtol=0)
        self.assertFalse(torch.allclose(pooled, wrong))
        (hidden.square().mean() + pooled.square().mean()).backward()
        for wrapper in wrappers:
            self.assertGreater(wrapper.vector.grad.abs().sum().item(), 0)
            self.assertIsNone(wrapper.base.weight.grad)
        self.assertEqual(sum(p.requires_grad for encoder in encoders for p in encoder.parameters()), 2)

    def test_token_alignment_and_truncation_fail_loudly(self):
        tokenizers, encoders = models_and_tokenizers()
        install_learned_tokens(tokenizers, encoders, "<S*>", "mailbox")
        tokenizers[1].insert_before_target = True
        with self.assertRaisesRegex(ValueError, "disagree"):
            encode_conditioning(tokenizers, encoders, ["a <S*> mailbox"], "<S*>")
        tokenizers[1].insert_before_target = False
        with self.assertRaisesRegex(ValueError, "truncated"):
            encode_conditioning(tokenizers, encoders, ["a a a a a a a a <S*>"], "<S*>")

    def test_time_ids_use_actual_source_and_target_sizes_in_order(self):
        actual = make_time_ids((512, 768), (1024, 1024), (0, 16), 2, dtype=torch.float16)
        self.assertEqual(actual.tolist(), [[512, 768, 0, 16, 1024, 1024]] * 2)
        self.assertEqual(actual.dtype, torch.float16)
        with self.assertRaises(ValueError):
            make_time_ids((0, 512), (1024, 1024))


if __name__ == "__main__":
    unittest.main()
