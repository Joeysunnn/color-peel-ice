"""CPU regressions for official ZeST conditioning and immutable shared assets."""

import contextlib
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from experiments.zest_material_counterfactual_v1 import assets, official


class OfficialTest(unittest.TestCase):
    def setUp(self):
        self.target = Image.new("RGB", (1024, 1024), (210, 100, 30))
        self.mask = Image.new("L", self.target.size, 0)
        self.mask.paste(255, (0, 0, 512, 1024))
        self.mask.paste(0, (100, 100, 120, 120))  # Protected content inside subject.
        self.depth = Image.new("RGB", self.target.size, (60, 60, 60))
        self.edge = Image.new("RGB", self.target.size, (0, 255, 0))
        self.reference = Image.new("RGB", self.target.size, (190, 90, 20))
        self.generated = Image.new("RGB", self.target.size, (20, 170, 230))
        self.adapter = Mock()
        self.adapter.pipe = SimpleNamespace(controlnet=object())
        self.adapter.generate.return_value = [self.generated]
        fake_torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        self.torch_patch = patch.dict(sys.modules, torch=fake_torch)
        self.torch_patch.start()
        self.addCleanup(self.torch_patch.stop)

    def generate(self, **kwargs):
        return official.generate(self.adapter, self.target, self.mask, self.depth,
                                 self.reference, **kwargs)

    def test_baseline_pins_zest_defaults_instead_of_marble_defaults(self):
        self.generate()
        call = self.adapter.generate.call_args.kwargs
        self.assertEqual((call["strength"], call["guidance_scale"], call["controlnet_conditioning_scale"]),
                         (.9999, 5.0, .9))
        self.assertEqual((call["scale"], call["num_inference_steps"], call["seed"], call["num_samples"]),
                         (1.0, 30, 42, 1))
        self.assertEqual(call["prompt"], "best quality, high quality")
        self.assertNotIn("edit_mlps", call)

    def test_effective_mask_grays_only_editable_surface_and_keeps_protected_rgb(self):
        before = self.target.tobytes(), self.mask.tobytes()
        self.generate()
        init = self.adapter.generate.call_args.kwargs["image"]
        self.assertEqual(init.getpixel((10, 10)), self.target.convert("L").convert("RGB").getpixel((10, 10)))
        for point in ((900, 10), (110, 110)):
            self.assertEqual(init.getpixel(point), self.target.getpixel(point))
        self.assertEqual((self.target.tobytes(), self.mask.tobytes()), before)

    def test_generated_subject_background_and_protection_are_not_rgb_overwritten(self):
        result = self.generate()
        self.assertIs(result, self.generated)
        for point in ((10, 10), (900, 10), (110, 110)):
            self.assertEqual(result.getpixel(point), (20, 170, 230))

    def test_single_and_multi_controlnets_receive_distinct_frozen_conditions(self):
        self.generate()
        single = self.adapter.generate.call_args.kwargs["control_image"]
        self.assertIsInstance(single, Image.Image)
        self.assertEqual(single.getpixel((0, 0)), (60, 60, 60))
        self.adapter.pipe.controlnet = SimpleNamespace(nets=[object(), object()])
        self.generate(edge_image=self.edge, edge_scale=.4, init_image=self.target)
        call = self.adapter.generate.call_args.kwargs
        self.assertEqual([image.getpixel((0, 0)) for image in call["control_image"]],
                         [(60, 60, 60), (0, 255, 0)])
        self.assertEqual(call["controlnet_conditioning_scale"], [.9, .4])
        self.assertEqual(call["image"].getpixel((10, 10)), self.target.getpixel((10, 10)))
        with self.assertRaises(ValueError):
            self.generate()  # Two networks cannot silently run with one condition.
        self.adapter.pipe.controlnet = object()
        with self.assertRaises(ValueError):
            self.generate(edge_image=self.edge)

    def test_unaligned_inputs_and_invalid_strength_fail_before_generation(self):
        with self.assertRaises(ValueError):
            self.generate(init_image=Image.new("RGB", (512, 512)))
        with self.assertRaises(ValueError):
            self.generate(strength=0)
        self.adapter.generate.assert_not_called()


class FrozenAssetsTest(unittest.TestCase):
    def test_changed_shared_hash_or_model_revision_is_rejected(self):
        receipt = {"official_commit": assets.OFFICIAL_COMMIT,
                   "ip_adapter_commit": assets.IP_ADAPTER_COMMIT,
                   "models": dict(assets.SHARED_MODELS), "with_canny": False,
                   "official": "zest", "ip_adapter_official": "tencent",
                   "sha256": {"read-only-shared-model": "expected"}}
        with patch.object(assets, "source_state"), patch.object(assets, "sha256", return_value="expected"):
            self.assertIs(assets.verify(receipt), receipt)
        with patch.object(assets, "source_state"), patch.object(assets, "sha256", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "frozen asset changed"):
                assets.verify(receipt)
        receipt["models"].pop("base")
        with patch.object(assets, "source_state") as source:
            with self.assertRaisesRegex(ValueError, "revisions differ"):
                assets.verify(receipt)
            source.assert_not_called()


if __name__ == "__main__":
    unittest.main()
