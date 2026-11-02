import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from diffusers.models.attention_processor import Attention

from experiments.sdxl_mailbox_v1.attention import (
    adapter_parameters, adapter_state_dict, install_token_local_kv,
)
from experiments.sdxl_mailbox_v1.conditioning import FrozenTokenEmbedding
from experiments.sdxl_mailbox_v1.infer import check_snapshot
from experiments.sdxl_mailbox_v1.train import checkpoint, masked_loss, tensor_hash


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


class SDXLTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)

    def test_masked_loss_sums_channels_and_normalizes_each_sample(self):
        prediction = torch.stack((torch.full((4, 2, 2), 2.), torch.full((4, 2, 2), 3.)))
        prediction.requires_grad_(True)
        mask = torch.tensor([[[[1., 0.], [1., 0.]]], [[[0., 1.], [0., 0.]]]])
        loss = masked_loss(prediction, torch.zeros_like(prediction), mask)
        # Per-sample values are 4*2^2 and 4*3^2, even with unequal mask areas.
        torch.testing.assert_close(loss, torch.tensor(26.))
        loss.backward()
        self.assertTrue(torch.equal(prediction.grad[~mask.expand_as(prediction).bool()], torch.zeros(20)))
        self.assertTrue((prediction.grad[mask.expand_as(prediction).bool()] != 0).all())

    def test_loss_promotes_half_error_to_fp32_before_square(self):
        prediction = torch.full((1, 4, 1, 1), 300., dtype=torch.float16, requires_grad=True)
        loss = masked_loss(prediction, torch.zeros_like(prediction), torch.ones(1, 1, 1, 1))
        self.assertEqual(loss.dtype, torch.float32)
        torch.testing.assert_close(loss, torch.tensor(360000.))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_checkpoint_pairs_current_adapters_vectors_and_optimizer_step(self):
        unet = TinyUnet()
        install_token_local_kv(unet, "fullrank")
        embeddings = [FrozenTokenEmbedding(nn.Embedding(5, dimension), 5, 2) for dimension in (4, 8)]
        initial_base = [embedding.base.weight.detach().clone() for embedding in embeddings]
        original_unet = {name: value.detach().clone() for name, value in unet.named_parameters()
                         if not value.requires_grad}
        parameters = adapter_parameters(unet) + [embedding.vector for embedding in embeddings]
        optimizer = torch.optim.AdamW(parameters, lr=0.01, weight_decay=0)
        for _ in range(2):
            optimizer.zero_grad()
            text = torch.cat([embedding(torch.tensor([[0, 5, 3]])) for embedding in embeddings], dim=-1)
            prediction = unet.attn2(torch.randn(1, 4, 8), text,
                                    modifier_token_mask=torch.tensor([[False, True, False]]))
            prediction.square().mean().backward()
            optimizer.step()
        for embedding, original in zip(embeddings, initial_base):
            torch.testing.assert_close(embedding.base.weight, original, atol=0, rtol=0)
            self.assertIsNone(embedding.base.weight.grad)
        for name, parameter in unet.named_parameters():
            if name in original_unet:
                torch.testing.assert_close(parameter, original_unet[name], atol=0, rtol=0)
                self.assertIsNone(parameter.grad)
        config = {"model_dir": "/frozen/model", "method": "fullrank", "branch": "subject",
                  "image_count": 27, "token": "<S*>", "initializer": "mailbox"}
        hashes = [tensor_hash(embedding.base.weight) for embedding in embeddings]
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.get_rng_state_all", return_value=[]):
            path = Path(directory) / "checkpoint-2.pt"
            checkpoint(path, config, 2, SimpleNamespace(unet=unet), embeddings, optimizer, scaler, hashes)
            state = torch.load(path, map_location="cpu", weights_only=False)
        check_snapshot(state, config, 2, embeddings)
        self.assertEqual(state["next_row"], 2)
        self.assertTrue(all(float(value["step"]) == 2 for value in state["optimizer"]["state"].values()))
        for actual, embedding in zip(state["embeddings"], embeddings):
            torch.testing.assert_close(actual, embedding.vector.detach(), atol=0, rtol=0)
        for key, actual in state["adapter"].items():
            torch.testing.assert_close(actual, adapter_state_dict(unet)[key], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            check_snapshot(state, config, 1, embeddings)
        with self.assertRaises(ValueError):
            check_snapshot(state, dict(config, method="lora_r64"), 2, embeddings)
        with self.assertRaises(ValueError):
            check_snapshot(state, dict(config, model_dir="/different/base"), 2, embeddings)
        bad = copy.deepcopy(state)
        bad["embeddings"][1] = torch.full_like(bad["embeddings"][1], float("nan"))
        with self.assertRaises(ValueError):
            check_snapshot(bad, config, 2, embeddings)
        bad = copy.deepcopy(state)
        bad["embeddings"] = bad["embeddings"][:1]
        with self.assertRaises(ValueError):
            check_snapshot(bad, config, 2, embeddings)
        bad = copy.deepcopy(state)
        bad["ordinary_embedding_hashes"][0] = "different"
        with self.assertRaises(ValueError):
            check_snapshot(bad, config, 2, embeddings)


if __name__ == "__main__":
    unittest.main()
