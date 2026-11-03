import ast
import hashlib
import json
from pathlib import Path
import random
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.train.concept_isolation import combine_prompt, validate_concept_isolation

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "src/train/train_colorpeel.py"


class ConfigurationTests(unittest.TestCase):
    def args(self, **changes):
        values = dict(concept_isolation_weight=1.0, concept_isolation_manifest="bank.json", modifier_token="<T*>",
            initializer_token="texture", max_train_steps=1000, subject_lora_mode="token_local_kv", token_local_kv=False,
            freeze_model="crossattn_kv", mixed_precision="no", train_batch_size=1, gradient_accumulation_steps=1,
            with_prior_preservation=False, texture_background_preservation_weight=0, texture_external_preservation_weight=0,
            joint_two_object_binding=False, perfusion_subject=False, strict_unpaired_modifier_updates=False,
            enable_xformers_memory_efficient_attention=False, lambda_attention=0, cos_weight=0, adam_weight_decay=0,
            resume_from_checkpoint=None, gradient_checkpointing=True, seed=42)
        values.update(changes)
        return SimpleNamespace(**values)

    def test_fig4_template_and_explicit_period_normalization(self):
        self.assertEqual(combine_prompt(" a <T*> texture. ", " a beach... "),
            "A picture divided into two parts. The left part is a <T*> texture. The right part is a beach.")
        with self.assertRaises(ValueError):
            combine_prompt("...", "a beach")

    def test_zero_weight_needs_no_assets_or_new_protocol_fields(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("asset read")):
            self.assertFalse(validate_concept_isolation(SimpleNamespace(concept_isolation_weight=0)))

    def test_valid_three_branches_and_reject_mixed_objectives(self):
        for token, initializer, steps in (("<S*>", "mailbox", 1000), ("<C*>", "orange", 150), ("<T*>", "texture", 1000)):
            self.assertTrue(validate_concept_isolation(self.args(modifier_token=token, initializer_token=initializer,
                                                               max_train_steps=steps)))
        for change in ({"concept_isolation_weight": -1}, {"concept_isolation_weight": float("nan")},
                       {"concept_isolation_weight": float("inf")}, {"modifier_token": "<M*>"},
                       {"initializer_token": "wood"}, {"mixed_precision": "fp16"}, {"with_prior_preservation": True},
                       {"texture_external_preservation_weight": 0.1}, {"texture_background_preservation_weight": 1},
                       {"token_local_kv": True}, {"max_train_steps": 600}, {"gradient_checkpointing": False},
                       {"adam_weight_decay": 0.01}, {"resume_from_checkpoint": "/x/checkpoint-400"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_concept_isolation(self.args(**change))

    def test_dataset_draws_caption_once_and_backward_precedes_optimizer(self):
        tree = ast.parse(TRAINER.read_text(encoding="utf8"))
        dataset = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CustomDiffusionDataset")
        getitem = next(node for node in dataset.body if isinstance(node, ast.FunctionDef) and node.name == "__getitem__")
        choices = [node for node in ast.walk(getitem) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == "choice"]
        self.assertEqual(len(choices), 1)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        backward = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "backward"
                        and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "weighted_isolation_loss")
        foreground = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "backward"
                          and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "loss")
        clip = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "clip_grad_norm_")
        self.assertLess(foreground.lineno, backward.lineno)
        self.assertLess(backward.lineno, clip.lineno)


if torch is not None:
    from torch.utils.checkpoint import checkpoint
    from src.train.concept_isolation import ConceptIsolation, isolation_rng

    class TinyTokenizer:
        model_max_length = 77
        def __init__(self):
            self.vocab = {"<S*>": 3, "<C*>": 4, "<T*>": 5, "mailbox": 6, "orange": 7, "texture": 8}
        def encode(self, text, add_special_tokens=True):
            words = re.findall(r"<[SCTM]\*>|[A-Za-z0-9_]+|[^\w\s]", text)
            ids = [self.vocab.setdefault(word, len(self.vocab) + 10) for word in words]
            return [0] + ids + [1] if add_special_tokens else ids
        def convert_ids_to_tokens(self, index):
            return next(key for key, value in self.vocab.items() if value == index)
        def __call__(self, text, **kwargs):
            ids = self.encode(text)
            return SimpleNamespace(input_ids=torch.tensor([ids + [1] * (77 - len(ids))]),
                attention_mask=torch.tensor([[1] * len(ids) + [0] * (77 - len(ids))]))

    class TinyEncoder(torch.nn.Module):
        config = SimpleNamespace(use_attention_mask=False)
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(256, 1024)
            self.scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.dropout = torch.nn.Dropout(0.1)
            self.calls = []
            with torch.no_grad():
                self.embedding.weight[5].copy_(self.embedding.weight[8])
        def get_input_embeddings(self):
            return self.embedding
        def forward(self, ids):
            random.random()
            np.random.random()
            torch.rand(1, device=ids.device)
            self.calls.append(ids.detach().clone())
            return (self.dropout(self.embedding(ids).cumsum(1) / 77) * self.scale,)

    class TinyVae(torch.nn.Module):
        config = SimpleNamespace(scaling_factor=0.18215)
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.canvas = None
        def encode(self, pixels):
            self.canvas = pixels.detach().clone()
            mean = pixels.mean().expand(1, 4, 64, 128)
            return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: mean + torch.randn_like(mean)))

    class TinyUnet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(1024, 4, bias=False).requires_grad_(False)
            self.k_down = torch.nn.Linear(1024, 2, bias=False)
            self.k_up = torch.nn.Linear(2, 4, bias=False)
            self.v_down = torch.nn.Linear(1024, 2, bias=False)
            self.v_up = torch.nn.Linear(2, 4, bias=False)
            self.dropout = torch.nn.Dropout(0.1)
            self.checkpointed, self.attn_probs, self.calls = False, None, []
        def block(self, noisy, hidden, gate):
            random.random()
            np.random.random()
            values = self.base(hidden) + gate[..., None] * (self.k_up(self.k_down(hidden)) + self.v_up(self.v_down(hidden)))
            self.attn_probs = values
            return values.mean(1)[:, :, None, None].expand_as(noisy) + self.dropout(noisy) * 0.1
        def forward(self, noisy, timesteps, encoder_hidden_states, cross_attention_kwargs):
            gate = cross_attention_kwargs["modifier_token_mask"]
            self.calls.append({"gate": gate.detach().clone(), "hidden_requires_grad": encoder_hidden_states.requires_grad,
                "noisy": noisy.detach().clone(), "t": timesteps.detach().clone(), "cache_empty": self.attn_probs is None})
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
        for index in range(25):
            path = cls.folder / f"{index:02}.png"
            Image.new("RGB", (512, 512), (index, 70, 120)).save(path)
            records.append({"id": str(index), "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "caption": f"a beach scene{index}", "size": [512, 512], "mode": "RGB"})
        cls.manifest = {"schema": 1, "records": records}
        cls.path = cls.folder / "manifest.json"
        cls.path.write_text(json.dumps(cls.manifest), encoding="utf8")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        torch.manual_seed(42)
        self.tokenizer, self.encoder, self.unet, self.vae = TinyTokenizer(), TinyEncoder(), TinyUnet(), TinyVae()
        self.prompt = "a photo of a blue cube with <T*> texture"
        self.isolation = ConceptIsolation(self.path, self.tokenizer, 5, 8, [self.prompt], 1000)
        self.isolation.bind_text_encoder(self.encoder)
        self.step = self.isolation.steps[0]
        self.pixels = torch.full((1, 3, 512, 512), -0.25)
        self.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000),
                                         add_noise=lambda latents, noise, t: latents + noise * 0.5)

    def context(self, **changes):
        values = dict(unet=self.unet, vae=self.vae, text_encoder=self.encoder, scheduler=self.scheduler,
                      pixel_values=self.pixels, prompt=self.prompt, step=self.step, device="cpu", dtype=torch.float32)
        values.update(changes)
        return self.isolation.loss_context(**values)

    def test_schedule_is_exact_half_reproducible_and_not_locked_to_half_of_t16(self):
        expected = sorted(random.Random(9042).sample(range(1, 1001), 500))
        self.assertEqual(self.isolation.audit()["isolation_steps"], expected)
        self.assertEqual({(step - 1) % 16 for step in expected}, set(range(16)))
        self.assertEqual(sum(self.isolation.enabled(step) for step in range(1, 1001)), 500)
        with self.assertRaises(ValueError):
            self.isolation.enabled(0)

    def test_canvas_pixels_right_region_loss_and_teacher_full_caption_reencoding(self):
        class SpatialUnet(TinyUnet):
            def forward(self, noisy, timesteps, encoder_hidden_states, cross_attention_kwargs):
                output = torch.zeros_like(noisy)
                if bool(cross_attention_kwargs["modifier_token_mask"].any()):
                    output[..., :64] = 1000
                    output[..., 64:] = 1
                return SimpleNamespace(sample=output)
        with self.context(unet=SpatialUnet()) as (loss, metrics):
            self.assertEqual(loss.item(), 4)
            self.assertEqual(metrics["concept_isolation_right_pixels"], 4096)
            self.assertFalse(metrics["teacher_requires_grad"] or metrics["teacher_hidden_requires_grad"])
            self.assertTrue(metrics["student_hidden_requires_grad"])
        self.assertEqual(tuple(self.vae.canvas.shape), (1, 3, 512, 1024))
        self.assertTrue(torch.equal(self.vae.canvas[..., :512], self.pixels))
        record = next(row for row in self.manifest["records"] if row["id"] == metrics["concept_isolation_bank_id"])
        expected = torch.from_numpy(np.array(Image.open(record["path"]))).permute(2, 0, 1)[None].float() / 127.5 - 1
        self.assertTrue(torch.equal(self.vae.canvas[..., 512:], expected))
        teacher, student = self.encoder.calls[-2:]
        self.assertEqual(int((student == 5).sum()), 1)
        expected_ids = student.clone()
        expected_ids[expected_ids == 5] = 8
        self.assertTrue(torch.equal(teacher, expected_ids))

    def test_checkpoint_backward_updates_modifier_and_adapters_without_teacher_or_base_grad(self):
        self.unet.checkpointed = True
        with torch.no_grad():
            self.encoder.embedding.weight[5].add_(0.1)
        self.encoder.dropout.eval()
        modules = (self.unet, self.vae, self.encoder)
        modes = [[module.training for module in owner.modules()] for owner in modules]
        before = self.encoder.embedding.weight.detach().clone()
        cpu, py, np_state = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
        with self.context() as (loss, _):
            loss.backward()
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
        self.assertEqual(py, random.getstate())
        self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]))
        self.assertEqual(np_state[2:], np.random.get_state()[2:])
        self.assertEqual(modes, [[module.training for module in owner.modules()] for owner in modules])
        self.assertTrue(torch.equal(before, self.encoder.embedding.weight.detach()))
        self.assertGreater(self.encoder.embedding.weight.grad[5].norm().item(), 0)
        for name in ("k_down", "k_up", "v_down", "v_up"):
            self.assertGreater(getattr(self.unet, name).weight.grad.norm().item(), 0)
        self.assertIsNone(self.unet.base.weight.grad)
        self.assertIsNone(self.vae.anchor.grad)
        self.assertIsNone(self.encoder.scale.grad)
        self.assertIsNone(self.unet.attn_probs)
        teacher, student = self.unet.calls[-2:]
        self.assertFalse(bool(teacher["gate"].any()))
        self.assertEqual(int(student["gate"].sum()), 1)
        self.assertTrue(teacher["cache_empty"] and student["cache_empty"])
        self.assertTrue(torch.equal(teacher["noisy"], student["noisy"]))
        self.assertTrue(torch.equal(teacher["t"], student["t"]))
        self.assertTrue(self.isolation.verify_teacher(self.encoder)["ordinary_embedding_unchanged"])
        ordinary = torch.ones(256, dtype=torch.bool)
        ordinary[5] = False
        self.encoder.embedding.weight.grad[ordinary] = 0
        self.assertEqual(self.encoder.embedding.weight.grad[ordinary].norm().item(), 0)

    def test_teacher_verification_rejects_ordinary_drift_but_allows_learned_modifier_binding(self):
        with torch.no_grad():
            self.encoder.embedding.weight[5].add_(1)
        self.isolation.bind_text_encoder(self.encoder)
        self.assertFalse(self.isolation.audit()["teacher_reference"]["modifier_initially_equals_initializer"])
        self.isolation.verify_teacher(self.encoder)
        with torch.no_grad():
            self.encoder.embedding.weight[6].add_(1)
        with self.assertRaisesRegex(ValueError, "ordinary encoder"):
            self.isolation.verify_teacher(self.encoder)

    def test_replay_and_exception_restore_rng_and_clear_attention(self):
        self.unet.eval()
        self.encoder.eval()
        def run():
            with self.context() as (loss, metrics):
                return loss.item(), metrics
        first = run()
        torch.rand(13)
        random.random()
        np.random.random()
        self.assertEqual(first, run())
        before = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "test exception"):
            with self.context():
                raise RuntimeError("test exception")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertIsNone(self.unet.attn_probs)

    def test_77_tokens_are_allowed_but_truncation_and_repeated_modifier_are_rejected(self):
        base = len(self.tokenizer.encode(combine_prompt(self.prompt, self.manifest["records"][0]["caption"])))
        exact = self.prompt + " word" * (77 - base)
        accepted = ConceptIsolation(self.path, self.tokenizer, 5, 8, [exact], 150)
        self.assertTrue(all(value["token_count"] == 77 for value in accepted.encodings.values()))
        for invalid in (exact + " word", self.prompt + " <T*>"):
            with self.assertRaises(ValueError):
                ConceptIsolation(self.path, self.tokenizer, 5, 8, [invalid], 150)

    def test_caption_field_is_opt_in_and_old_collation_remains_tensor_only(self):
        node = next(node for node in ast.parse(TRAINER.read_text(encoding="utf8")).body
                    if isinstance(node, ast.FunctionDef) and node.name == "collate_fn")
        scope = {"torch": torch}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(TRAINER), "exec"), scope)
        example = {"instance_prompt_ids": torch.ones(1, 77, dtype=torch.long),
                   "instance_images": self.pixels[0], "mask": torch.ones(64, 64), "instance_prompt": self.prompt}
        old = scope["collate_fn"]([example], False)
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in old.values()))
        new = scope["collate_fn"]([example], False, retain_instance_prompt=True)
        self.assertEqual(new["instance_prompt"], [self.prompt])
        self.assertTrue(all(torch.equal(old[key], new[key]) for key in old))


if __name__ == "__main__":
    unittest.main()
