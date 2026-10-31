"""CPU checks for the pinned SD2.1 linked asset and immutable receipt."""

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.sd21_wood_branches_v1 import prepare_model as model


class SD21WoodModelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="sd21-model-", dir=os.environ.get("TMP"))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "mirror" / model.REVISION
        self.checker = self.root / "checker" / model.CHECKER_REVISION
        self.output = self.root / "assets" / "sd21"
        self.ids = {"spoon": 10, "material": 11, "texture": 12}
        for component in model.COMPONENTS:
            (self.origin(component) / component).mkdir(parents=True, exist_ok=True)
        self.write(self.source / "model_index.json", {
            "_class_name": "StableDiffusionPipeline", "_diffusers_version": "0.10.0",
            "unet": ["diffusers", "UNet2DConditionModel"], "vae": ["diffusers", "AutoencoderKL"],
            "text_encoder": ["transformers", "CLIPTextModel"], "tokenizer": ["transformers", "CLIPTokenizer"],
            "scheduler": ["diffusers", "PNDMScheduler"], "safety_checker": [None, None],
            "feature_extractor": [None, None], "requires_safety_checker": False})
        self.write(self.source / "unet/config.json", {
            "cross_attention_dim": 1024, "sample_size": 64, "use_linear_projection": True,
            "attention_head_dim": [5, 10, 20, 20], "block_out_channels": [320, 640, 1280, 1280],
            "in_channels": 4, "out_channels": 4})
        self.write(self.source / "text_encoder/config.json", {
            "architectures": ["CLIPTextModel"], "hidden_size": 1024,
            "num_hidden_layers": 23, "max_position_embeddings": 77})
        self.write(self.source / "scheduler/scheduler_config.json", {
            "_class_name": "PNDMScheduler", "prediction_type": "epsilon"})
        for component, name in (("vae", "config.json"), ("safety_checker", "config.json"),
                ("feature_extractor", "preprocessor_config.json"), ("tokenizer", "vocab.json"),
                ("tokenizer", "tokenizer_config.json"), ("tokenizer", "special_tokens_map.json")):
            self.write(self.origin(component) / component / name, {})
        (self.source / "tokenizer/merges.txt").write_text("#version: 0.2\n", encoding="utf8")
        for component, name in (("unet", "diffusion_pytorch_model.bin"),
                ("vae", "diffusion_pytorch_model.bin"), ("text_encoder", "pytorch_model.bin"),
                ("safety_checker", "pytorch_model.bin")):
            (self.origin(component) / component / name).write_bytes((component + " fixture weights").encode())
        self.tokens = patch.object(model, "_initializer_ids", return_value=self.ids)
        self.tokens.start()
        self.addCleanup(self.tokens.stop)
        if os.name == "nt":
            # This sandbox cannot create Windows links; retain real file/hash IO.
            aliases = {}
            resolve = Path.resolve

            def linked_copy(path, target, target_is_directory=False):
                shutil.copytree(target, path)
                aliases[resolve(path)] = resolve(target)

            def linked_resolve(path, strict=False):
                result = resolve(path, strict=strict)
                for link, target in aliases.items():
                    if result.is_relative_to(link):
                        return target / result.relative_to(link)
                return result

            links = patch.object(Path, "symlink_to", linked_copy)
            links.start()
            self.addCleanup(links.stop)
            resolutions = patch.object(Path, "resolve", linked_resolve)
            resolutions.start()
            self.addCleanup(resolutions.stop)

    def origin(self, component):
        return self.checker if component in ("safety_checker", "feature_extractor") else self.source

    @staticmethod
    def write(path, value):
        Path(path).write_text(json.dumps(value), encoding="utf8")

    def prepare(self):
        return model.prepare(self.output, self.source, checker=self.checker)

    def test_new_bundle_links_only_components_and_enables_common_checker(self):
        original = (self.source / "model_index.json").read_bytes()
        receipt = self.prepare()
        self.assertEqual(model.verify_model(self.output), receipt)
        self.assertEqual((self.source / "model_index.json").read_bytes(), original)
        index = json.loads((self.output / "model_index.json").read_text())
        self.assertTrue(index["requires_safety_checker"])
        self.assertEqual(index["safety_checker"], ["stable_diffusion", "StableDiffusionSafetyChecker"])
        self.assertEqual(index["feature_extractor"], ["transformers", "CLIPFeatureExtractor"])
        self.assertEqual(receipt["model"]["repo"], model.REPO)
        self.assertEqual(receipt["model"]["revision"], model.REVISION)
        self.assertEqual(receipt["architecture"], model.ARCHITECTURE)
        self.assertEqual(receipt["initializer_ids"], self.ids)
        self.assertEqual(receipt["safety_checker_source"], str(self.checker))
        for component in model.COMPONENTS:
            self.assertEqual((self.output / component).resolve(), (self.origin(component) / component).resolve())
        for relative, digest in receipt["files"].items():
            self.assertEqual(model._sha256(self.output / relative), digest)
        for path, digest in receipt["source_locks"].items():
            self.assertTrue(Path(path).is_absolute())
            self.assertEqual(model._sha256(path), digest)

    def test_existing_exact_bundle_is_reused_without_any_write(self):
        receipt = self.prepare()
        before = {name: (self.output / name).stat().st_mtime_ns for name in ("model_index.json", "model_source.json")}
        with (patch.object(Path, "write_text", side_effect=AssertionError("must not overwrite")),
              patch.object(Path, "write_bytes", side_effect=AssertionError("must not overwrite"))):
            self.assertEqual(self.prepare(), receipt)
        self.assertEqual(before, {name: (self.output / name).stat().st_mtime_ns for name in before})

    def test_existing_unknown_output_is_not_replaced(self):
        self.output.mkdir(parents=True)
        sentinel = self.output / "old-result.txt"
        sentinel.write_text("keep", encoding="utf8")
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_config_revision_and_required_weight_fail_before_creating_output(self):
        path = self.source / "unet/config.json"
        original = json.loads(path.read_text())
        for key, value in (("cross_attention_dim", 768), ("sample_size", 96),
                ("use_linear_projection", False), ("attention_head_dim", 8)):
            with self.subTest(key=key):
                self.write(path, {**original, key: value})
                with self.assertRaisesRegex(ValueError, "architecture"):
                    self.prepare()
                self.assertFalse(self.output.exists())
        self.write(path, original)
        weights = self.source / "unet/diffusion_pytorch_model.bin"
        weights.unlink()
        (self.source / "unet/diffusion_pytorch_model.fp16.bin").write_bytes(b"variant only")
        with self.assertRaisesRegex(FileNotFoundError, "non-variant weights"):
            self.prepare()
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, "revision"):
            model.prepare(self.output, self.source.parent, checker=self.checker)

    def test_receipt_source_and_bundle_tampering_are_rejected(self):
        receipt = self.prepare()
        index = self.output / "model_index.json"
        original = index.read_bytes()
        self.write(index, {**json.loads(original), "requires_safety_checker": False})
        with self.assertRaisesRegex(ValueError, "index changed"):
            model.verify_model(self.output)
        index.write_bytes(original)
        weights = self.source / "text_encoder/pytorch_model.bin"
        weights.write_bytes(b"changed weights")
        with self.assertRaisesRegex(ValueError, "receipt or source"):
            self.prepare()
        self.assertEqual(json.loads((self.output / "model_source.json").read_text()), receipt)

    def test_wrong_text_layer_or_prediction_target_is_rejected(self):
        for relative, key, value in (("text_encoder/config.json", "num_hidden_layers", 24),
                ("text_encoder/config.json", "hidden_size", 768),
                ("scheduler/scheduler_config.json", "prediction_type", "v_prediction")):
            with self.subTest(field=key):
                path = self.source / relative
                original = json.loads(path.read_text())
                self.write(path, {**original, key: value})
                with self.assertRaisesRegex(ValueError, "architecture"):
                    self.prepare()
                self.assertFalse(self.output.exists())
                self.write(path, original)

    def test_unexpected_bundle_entries_are_rejected(self):
        self.prepare()
        (self.output / "untracked.bin").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "bundle entries"):
            model.verify_model(self.output)

    def test_output_cannot_overlap_source_cache(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            model.prepare(self.source / "new", self.source, checker=self.checker)
        self.assertFalse((self.source / "new").exists())

    def test_initializer_check_uses_local_tokenizer_without_special_tokens(self):
        self.tokens.stop()
        tokenizer = SimpleNamespace(model_max_length=77, all_special_ids=[49406, 49407],
                                    encode=Mock(side_effect=lambda word, **kwargs: [self.ids[word]]))
        loader = Mock(return_value=tokenizer)
        modules = {"transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=loader))}
        with patch.dict(sys.modules, modules):
            self.assertEqual(model._initializer_ids(self.source), self.ids)
            loader.assert_called_once_with(str(self.source), subfolder="tokenizer",
                                           local_files_only=True, use_fast=False)
            tokenizer.encode.assert_any_call("material", add_special_tokens=False)
            tokenizer.encode.side_effect = lambda word, **kwargs: [1, 2]
            with self.assertRaisesRegex(ValueError, "one ordinary token"):
                model._initializer_ids(self.source)
            tokenizer.encode.side_effect = lambda word, **kwargs: [49407]
            with self.assertRaisesRegex(ValueError, "one ordinary token"):
                model._initializer_ids(self.source)

    def test_runtime_check_loads_cpu_configs_without_model_weights(self):
        self.prepare()
        loader = Mock(return_value=SimpleNamespace(hidden_size=1024, num_hidden_layers=23))
        scheduler = Mock(return_value=SimpleNamespace(config=SimpleNamespace(prediction_type="epsilon")))

        class UNet:
            def __init__(self, cross_attention_dim, sample_size, use_linear_projection, attention_head_dim):
                raise AssertionError("model allocation must not run")

            load_config = Mock(return_value=json.loads((self.source / "unet/config.json").read_text()))

        modules = {"transformers": SimpleNamespace(CLIPTextConfig=SimpleNamespace(from_pretrained=loader)),
                   "diffusers": SimpleNamespace(PNDMScheduler=SimpleNamespace(from_pretrained=scheduler)),
                   "custom_attention.unet_2d_condition_custom": SimpleNamespace(UNet2DConditionModel=UNet)}
        with patch.dict(sys.modules, modules):
            result = model.runtime_check(self.output)
        self.assertEqual(result["device"], "cpu")
        self.assertFalse(result["weights_loaded"])
        self.assertEqual(result["architecture"], model.ARCHITECTURE)
        loader.assert_called_once_with(str(self.output), subfolder="text_encoder", local_files_only=True)
        scheduler.assert_called_once_with(str(self.output), subfolder="scheduler", local_files_only=True)
        UNet.load_config.assert_called_once_with(str(self.output), subfolder="unet", local_files_only=True)


if __name__ == "__main__":
    unittest.main()
