import ast
import math
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "src/train/texture_background_preservation.py"
TRAINER = ROOT / "src/train/train_colorpeel.py"


class ConfigurationTests(unittest.TestCase):
    def validate(self, **changes):
        # Exercise the actual pure validation function even on hosts without torch.
        node = next(node for node in ast.parse(HELPER.read_text(encoding="utf8")).body
                    if isinstance(node, ast.FunctionDef) and node.name == "validate_texture_background_preservation")
        scope = {"math": math}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(HELPER), "exec"), scope)
        values = dict(texture_background_preservation_weight=1.0, modifier_token="<T*>",
                      initializer_token="texture", token_local_kv=False, subject_lora_mode="token_local_kv",
                      freeze_model="crossattn_kv", with_prior_preservation=False,
                      joint_two_object_binding=False, perfusion_subject=False)
        values.update(changes)
        return scope[node.name](SimpleNamespace(**values))

    def test_zero_weight_does_not_restrict_old_branches(self):
        self.assertFalse(self.validate(texture_background_preservation_weight=0, modifier_token="<S*>"))

    def test_valid_texture_fullrank_and_lora(self):
        self.assertTrue(self.validate())
        self.assertTrue(self.validate(token_local_kv=True, subject_lora_mode=None))

    def test_invalid_weight_or_training_scope(self):
        for changes in ({"texture_background_preservation_weight": -1},
                        {"texture_background_preservation_weight": float("nan")},
                        {"texture_background_preservation_weight": float("inf")},
                        {"modifier_token": "<S*>"}, {"initializer_token": "wood"},
                        {"with_prior_preservation": True}, {"joint_two_object_binding": True},
                        {"subject_lora_mode": "full_kv"}, {"token_local_kv": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.validate(**changes)

    def test_reference_cache_precedes_resume_and_default_is_zero(self):
        tree = ast.parse(TRAINER.read_text(encoding="utf8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        cache = next(node for node in calls if isinstance(node.func, ast.Name)
                     and node.func.id == "cache_reference_conditioning")
        restore = next(node for node in calls if isinstance(node.func, ast.Attribute)
                       and node.func.attr == "load_state")
        self.assertLess(cache.lineno, restore.lineno)
        flag = next(node for node in calls if node.args and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "--texture_background_preservation_weight")
        self.assertEqual(next(item.value.value for item in flag.keywords if item.arg == "default"), 0.0)


if torch is not None:
    from src.train.texture_background_preservation import (
        audit_background_preservation_gradients, background_mask, background_preservation_loss,
        cache_reference_conditioning, reference_prediction,
    )

    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(6, 3)
            self.dropout = torch.nn.Dropout(0.4)
            with torch.no_grad():
                self.embedding.weight[5].copy_(self.embedding.weight[2])

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, ids):
            # Deliberately consume all three RNG streams to test the guard.
            random.random()
            np.random.random()
            torch.rand(1)
            hidden = self.embedding(ids)
            return (hidden + 0.1 * hidden.cumsum(1),)

    class TinyUnet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(3, 4, bias=False).requires_grad_(False)
            self.k_down = torch.nn.Linear(3, 2, bias=False)
            self.k_up = torch.nn.Linear(2, 4, bias=False)
            self.v_down = torch.nn.Linear(3, 2, bias=False)
            self.v_up = torch.nn.Linear(2, 4, bias=False)
            self.dropout = torch.nn.Dropout(0.4)

        def forward(self, noisy, timesteps, encoder_hidden_states, cross_attention_kwargs):
            gate = cross_attention_kwargs["modifier_token_mask"][..., None]
            hidden = encoder_hidden_states
            projected = self.base(hidden) + gate * (self.k_up(self.k_down(hidden)) + self.v_up(self.v_down(hidden)))
            output = projected.mean(1)[:, :, None, None].expand_as(noisy) + noisy * 0.1
            return SimpleNamespace(sample=output)

        def groups(self):
            return {name: list(getattr(self, name).parameters()) for name in ("k_down", "k_up", "v_down", "v_up")}


@unittest.skipIf(torch is None, "PyTorch is not installed on this host")
class TensorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.encoder, self.unet = TinyEncoder(), TinyUnet()
        self.ids = torch.tensor([[0, 5, 1]])
        self.reference = cache_reference_conditioning(self.encoder, self.ids, 5, 2)
        self.noisy = torch.randn(1, 4, 7, 7)
        self.t = torch.tensor([500])
        self.mask = torch.zeros(1, 1, 7, 7)
        self.mask[:, :, 3, 3] = 1

    def prediction(self, ids, gate=None):
        return self.unet(self.noisy, self.t, self.encoder(ids)[0],
                         {"modifier_token_mask": ids == 5 if gate is None else gate}).sample

    def audit(self, student, teacher, *, wrong_control=False, require_nonzero=True):
        class_ids, empty_ids = torch.tensor([[0, 2, 1]]), torch.tensor([[0, 1, 1]])
        gate = torch.ones_like(class_ids, dtype=torch.bool) if wrong_control else None
        return audit_background_preservation_gradients(
            student_prediction=student, teacher_prediction=teacher, foreground_mask=self.mask,
            parameter_groups=self.unet.groups(), embedding_parameter=self.encoder.embedding.weight,
            modifier_token_id=5, class_only_loss=self.prediction(class_ids, gate).square().mean(),
            class_only_input_ids=class_ids, unconditional_loss=self.prediction(empty_ids).square().mean(),
            unconditional_input_ids=empty_ids, require_nonzero=require_nonzero)

    def test_mask_excludes_boundary_and_dilated_foreground(self):
        mask = background_mask(self.mask)
        self.assertEqual(mask.sum().item(), 16)
        self.assertEqual(mask[:, :, 2:5, 2:5].sum().item(), 0)
        self.assertEqual(mask[:, :, 0].sum().item(), 0)

    def test_loss_retains_channel_sum_and_ignores_excluded_pixels(self):
        student = torch.ones_like(self.noisy)
        student[:, :, 0] = 900
        student[:, :, 2:5, 2:5] = 900
        loss, _ = background_preservation_loss(student, torch.zeros_like(student), self.mask)
        self.assertEqual(loss.item(), 4.0)

    def test_invalid_or_empty_regions_and_connected_teacher_rejected(self):
        with self.assertRaises(ValueError):
            background_mask(torch.ones_like(self.mask))
        with self.assertRaises(ValueError):
            background_mask(self.mask + 0.1)
        with self.assertRaises(ValueError):
            background_preservation_loss(self.noisy, self.noisy.clone().requires_grad_(), self.mask)

    def test_cache_and_teacher_preserve_rng_modes_and_initial_context(self):
        self.encoder.dropout.eval()
        self.unet.dropout.eval()
        torch_state, python_state, numpy_state = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
        modes = [module.training for module in self.encoder.modules()]
        unet_modes = [module.training for module in self.unet.modules()]
        reference = cache_reference_conditioning(self.encoder, self.ids, 5, 2)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        self.assertTrue(np.array_equal(numpy_state[1], np.random.get_state()[1]))
        self.assertEqual(numpy_state[2:], np.random.get_state()[2:])
        self.assertEqual(modes, [module.training for module in self.encoder.modules()])
        before = reference_prediction(self.unet, self.noisy, self.t, self.ids, reference)
        with torch.no_grad():
            self.encoder.embedding.weight[5].add_(1)
            self.unet.k_up.weight.add_(1)
        after = reference_prediction(self.unet, self.noisy, self.t, self.ids, reference)
        self.assertTrue(torch.equal(before, after))
        self.assertFalse(after.requires_grad)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        self.assertTrue(np.array_equal(numpy_state[1], np.random.get_state()[1]))
        self.assertEqual(numpy_state[2:], np.random.get_state()[2:])
        self.assertEqual(unet_modes, [module.training for module in self.unet.modules()])
        with self.assertRaises(ValueError):
            cache_reference_conditioning(self.encoder, self.ids, 5, 2)

    def test_background_gradients_reach_lora_and_token_but_controls_are_zero(self):
        with torch.no_grad():
            self.encoder.embedding.weight[5].add_(0.1)
        teacher = reference_prediction(self.unet, self.noisy, self.t, self.ids, self.reference)
        report = self.audit(self.prediction(self.ids), teacher)
        self.assertTrue(report["passed"], report)
        self.assertTrue(all(value > 0 for value in report["gradients"]["background"]["adapter_group_norms"].values()))
        self.assertTrue(all(parameter.grad is None for parameter in self.unet.parameters()))
        self.assertIsNone(self.encoder.embedding.weight.grad)

    def test_wrong_negative_control_is_detected(self):
        teacher = reference_prediction(self.unet, self.noisy, self.t, self.ids, self.reference)
        report = self.audit(self.prediction(self.ids), teacher, wrong_control=True)
        self.assertFalse(report["passed"])
        self.assertFalse(report["negative_controls_zero"])

    def test_equal_initial_student_teacher_is_not_misreported_as_working(self):
        with torch.no_grad():
            self.unet.k_up.weight.zero_()
            self.unet.v_up.weight.zero_()
        teacher = reference_prediction(self.unet, self.noisy, self.t, self.ids, self.reference)
        student = self.prediction(self.ids)
        report = self.audit(student, teacher, require_nonzero=False)
        self.assertTrue(report["passed"])
        self.assertFalse(report["background_gradients_nonzero"])
        self.assertFalse(self.audit(student, teacher)["passed"])


if __name__ == "__main__":
    unittest.main()
