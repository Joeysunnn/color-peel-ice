import ast
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "src/train/texture_external_preservation.py"
TRAINER = ROOT / "src/train/train_colorpeel.py"


class ConfigurationTests(unittest.TestCase):
    def validate(self, **changes):
        node = next(node for node in ast.parse(HELPER.read_text(encoding="utf8")).body
                    if isinstance(node, ast.FunctionDef) and node.name == "validate_texture_external_preservation")
        scope = {"math": math, "Path": Path}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(HELPER), "exec"), scope)
        values = dict(texture_external_preservation_weight=0.1, texture_external_preservation_manifest="bank.json",
                      modifier_token="<T*>", initializer_token="texture", token_local_kv=False,
                      subject_lora_mode="token_local_kv", freeze_model="crossattn_kv", mixed_precision="no",
                      with_prior_preservation=False, texture_background_preservation_weight=0,
                      joint_two_object_binding=False, perfusion_subject=False, strict_unpaired_modifier_updates=False,
                      enable_xformers_memory_efficient_attention=False, cos_weight=0, lambda_attention=0,
                      train_batch_size=1, gradient_accumulation_steps=1, max_train_steps=600,
                      resume_from_checkpoint="/original/checkpoint-400")
        values.update(changes)
        return scope[node.name](SimpleNamespace(**values))

    def test_zero_weight_leaves_other_training_modes_alone_and_reads_no_manifest(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("unexpected manifest read")):
            self.assertFalse(self.validate(texture_external_preservation_weight=0, modifier_token="<S*>",
                                           texture_external_preservation_manifest="missing.json", mixed_precision="fp16"))

    def test_scope_and_weights_are_strict(self):
        self.assertTrue(self.validate())
        self.assertTrue(self.validate(token_local_kv=True, subject_lora_mode=None))
        for changes in ({"texture_external_preservation_weight": -1}, {"texture_external_preservation_weight": float("nan")},
                        {"texture_external_preservation_weight": float("inf")}, {"modifier_token": "<S*>"},
                        {"initializer_token": "wood"}, {"mixed_precision": "fp16"}, {"with_prior_preservation": True},
                        {"texture_background_preservation_weight": 1}, {"subject_lora_mode": "full_kv"},
                        {"joint_two_object_binding": True}, {"perfusion_subject": True},
                        {"texture_external_preservation_manifest": None}, {"max_train_steps": 1000},
                        {"resume_from_checkpoint": "/original/checkpoint-500"}, {"gradient_accumulation_steps": 2}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.validate(**changes)

    def test_cache_precedes_resume_and_external_backward_precedes_clip(self):
        tree = ast.parse(TRAINER.read_text(encoding="utf8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        create = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "ExternalTexturePreservation")
        restore = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "load_state")
        backwards = {node.args[0].id: node.lineno for node in calls
                     if isinstance(node.func, ast.Attribute) and node.func.attr == "backward"
                     and node.args and isinstance(node.args[0], ast.Name)
                     and node.args[0].id in ("loss", "weighted_external_loss")}
        clip = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "clip_grad_norm_")
        flag = next(node for node in calls if node.args and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "--texture_external_preservation_weight")
        self.assertLess(create.lineno, restore.lineno)
        self.assertEqual(set(backwards), {"loss", "weighted_external_loss"})
        self.assertLess(backwards["loss"], backwards["weighted_external_loss"])
        self.assertLess(backwards["weighted_external_loss"], clip.lineno)
        self.assertEqual(next(item.value.value for item in flag.keywords if item.arg == "default"), 0.0)


if torch is not None:
    from torch.utils.checkpoint import checkpoint
    from src.train.texture_external_preservation import ExternalTexturePreservation, preservation_rng

    class TinyTokenizer:
        model_max_length = 77
        def encode(self, prompt, add_special_tokens=True):
            return [0, 2, 3, 4, 1] if add_special_tokens else [2, 3, 4]
        def __call__(self, prompt, **kwargs):
            ids = self.encode(prompt)
            return SimpleNamespace(input_ids=torch.tensor([ids + [1] * (77 - len(ids))]),
                                   attention_mask=torch.tensor([[1] * len(ids) + [0] * (77 - len(ids))]))

    class TinyEncoder(torch.nn.Module):
        config = SimpleNamespace(use_attention_mask=False)
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(6, 1024)
            self.dropout = torch.nn.Dropout(0.2)
            with torch.no_grad():
                self.embedding.weight[5].copy_(self.embedding.weight[2])
        def get_input_embeddings(self):
            return self.embedding
        def forward(self, ids):
            random.random()
            np.random.random()
            torch.rand(1, device=ids.device)
            return (self.dropout(self.embedding(ids)),)

    class TinyVae(torch.nn.Module):
        config = SimpleNamespace(scaling_factor=0.18215)
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        def encode(self, pixels):
            mean = pixels.mean().expand(1, 4, 4, 4)
            return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: mean + torch.randn_like(mean)))

    class TinyUnet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(1024, 4, bias=False).requires_grad_(False)
            self.k = torch.nn.Linear(1024, 4, bias=False)
            self.v = torch.nn.Linear(1024, 4, bias=False)
            self.dropout = torch.nn.Dropout(0.2)
            self.checkpointed = False
            self.calls = []
        def block(self, noisy, hidden, gate):
            random.random()
            np.random.random()
            projected = self.base(hidden) + gate[..., None] * (self.k(hidden) + self.v(hidden))
            return projected.mean(1)[:, :, None, None].expand_as(noisy) + self.dropout(noisy) * 0.1
        def forward(self, noisy, timesteps, encoder_hidden_states, cross_attention_kwargs):
            gate = cross_attention_kwargs["modifier_token_mask"]
            self.calls.append((gate.detach().clone(), encoder_hidden_states.requires_grad, noisy.requires_grad, self.training))
            output = (checkpoint(self.block, noisy, encoder_hidden_states, gate)
                      if self.training and self.checkpointed else self.block(noisy, encoder_hidden_states, gate))
            return SimpleNamespace(sample=output)


@unittest.skipIf(torch is None, "PyTorch is not installed on this host")
class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        records = []
        for i in range(200):
            path = cls.folder / f"{i:03}.jpg"
            Image.new("RGB", (512, 512), (i, 50, 100)).save(path)
            records.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        cls.manifest = {"schema": 1, "prompt": "a photo at the beach", "seed": 42, "records": records}
        cls.path = cls.folder / "manifest.json"
        cls.path.write_text(json.dumps(cls.manifest), encoding="utf8")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        torch.manual_seed(42)
        self.encoder, self.unet, self.vae = TinyEncoder(), TinyUnet(), TinyVae()
        self.bank = ExternalTexturePreservation(self.path, TinyTokenizer(), self.encoder, 5, 2)
        self.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000),
                                         add_noise=lambda latents, noise, t: latents + noise * 0.5)

    def test_whole_frame_channel_sum_scale(self):
        class ConstantUnet(TinyUnet):
            def forward(self, noisy, timesteps, encoder_hidden_states, cross_attention_kwargs):
                value = int(cross_attention_kwargs["modifier_token_mask"].any())
                return SimpleNamespace(sample=torch.ones_like(noisy) * value)
        with self.bank.loss_context(ConstantUnet(), self.vae, self.scheduler, 401, "cpu") as (loss, metrics):
            self.assertEqual(loss.item(), 4.0)
            self.assertEqual(metrics["texture_external_preservation_seed"], 100443)
            self.assertEqual(metrics["texture_external_preservation_image_sha256"], self.manifest["records"][0]["sha256"])

    def test_all77_gate_only_adapter_gradients_and_detached_base_context(self):
        cached = self.bank.hidden_states.clone()
        with torch.no_grad():
            self.encoder.embedding.weight.add_(10)
        with self.bank.loss_context(self.unet, self.vae, self.scheduler, 600, "cpu") as (loss, metrics):
            loss.backward()
            self.assertFalse(metrics["teacher_requires_grad"])
            self.assertFalse(metrics["hidden_requires_grad"])
            self.assertEqual(metrics["texture_external_preservation_image_path"], self.manifest["records"][199]["path"])
        self.assertTrue(torch.equal(cached, self.bank.hidden_states))
        self.assertTrue(all(parameter.grad is not None and bool(parameter.grad.norm() > 0)
                            for layer in (self.unet.k, self.unet.v) for parameter in layer.parameters()))
        self.assertIsNone(self.encoder.embedding.weight.grad)
        self.assertIsNone(self.unet.base.weight.grad)
        teacher, student = self.unet.calls
        self.assertFalse(bool(teacher[0].any()))
        self.assertFalse(teacher[1] or teacher[2] or teacher[3])
        self.assertTrue(bool(student[0].all()))
        self.assertEqual(tuple(student[0].shape), (1, 77))
        self.assertFalse(student[1])
        self.assertTrue(student[2] and student[3])

    def test_rng_modes_and_reentrant_backward_are_isolated(self):
        self.unet.checkpointed = True
        self.encoder.dropout.eval()
        modules = (self.unet, self.encoder, self.vae)
        modes = [[module.training for module in owner.modules()] for owner in modules]
        cpu, py, np_state = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
        with self.bank.loss_context(self.unet, self.vae, self.scheduler, 401, "cpu") as (loss, _):
            loss.backward()
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
        self.assertEqual(py, random.getstate())
        self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]))
        self.assertEqual(np_state[2:], np.random.get_state()[2:])
        self.assertEqual(modes, [[module.training for module in owner.modules()] for owner in modules])
        self.assertGreater(self.unet.k.weight.grad.norm().item(), 0)
        self.assertGreater(self.unet.v.weight.grad.norm().item(), 0)
        self.assertIsNone(self.encoder.embedding.weight.grad)
        self.assertIsNone(self.unet.base.weight.grad)

    def test_step_replay_is_independent_of_foreground_rng_and_exceptions_restore(self):
        self.unet.eval()
        def evaluate():
            with self.bank.loss_context(self.unet, self.vae, self.scheduler, 401, "cpu") as (loss, metrics):
                return loss.item(), metrics
        first = evaluate()
        torch.rand(13)
        random.random()
        np.random.random()
        self.assertEqual(first, evaluate())
        before = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with preservation_rng(401, "cpu"):
                torch.rand(3)
                raise RuntimeError("test failure")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for step in (400, 601):
            with self.assertRaises(ValueError):
                with preservation_rng(step, "cpu"):
                    pass

    def test_manifest_hash_dimensions_and_pre_resume_cache_are_enforced(self):
        bad = json.loads(json.dumps(self.manifest))
        bad["records"][0]["sha256"] = "0" * 64
        path = self.folder / "bad.json"
        path.write_text(json.dumps(bad), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "SHA"):
            ExternalTexturePreservation(path, TinyTokenizer(), self.encoder, 5, 2)
        wrong = self.folder / "wrong.jpg"
        Image.new("RGB", (511, 512)).save(wrong)
        bad["records"][0] = {"path": str(wrong), "sha256": hashlib.sha256(wrong.read_bytes()).hexdigest()}
        path.write_text(json.dumps(bad), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "512px RGB"):
            ExternalTexturePreservation(path, TinyTokenizer(), self.encoder, 5, 2)
        with torch.no_grad():
            self.encoder.embedding.weight[5].add_(1)
        with self.assertRaisesRegex(ValueError, "before loading"):
            ExternalTexturePreservation(self.path, TinyTokenizer(), self.encoder, 5, 2)


if __name__ == "__main__":
    unittest.main()
