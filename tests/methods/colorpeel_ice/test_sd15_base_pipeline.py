import sys
import types
import unittest
from unittest.mock import Mock, patch

from experiments.sd15_token_local_kv_v1 import evaluate


class BasePipelineTests(unittest.TestCase):
    def fixture(self):
        class PNDMScheduler:
            pass

        unet = Mock()
        pipe = Mock(scheduler=PNDMScheduler(), safety_checker=object())
        pipe.to.return_value = pipe
        unet_loader = Mock(return_value=unet)
        pipe_loader = Mock(return_value=pipe)
        modules = {
            "torch": types.SimpleNamespace(float16="float16"),
            "diffusers": types.SimpleNamespace(
                DiffusionPipeline=types.SimpleNamespace(from_pretrained=pipe_loader),
                PNDMScheduler=PNDMScheduler),
            "custom_attention.unet_2d_condition_custom": types.SimpleNamespace(
                UNet2DConditionModel=types.SimpleNamespace(from_pretrained=unet_loader)),
        }
        model = {"root": "/frozen/sd15", "revision": "locked"}

        def read_json(path):
            if path == evaluate.PROTOCOL:
                return {"model": model}
            return {"model": model, "files": {"unet/weights.bin": "same"}}

        return modules, read_json, unet, pipe, unet_loader, pipe_loader

    def test_base_has_no_adapter_or_extra_embedding(self):
        modules, read_json, unet, pipe, unet_loader, pipe_loader = self.fixture()
        with (patch.dict(sys.modules, modules), patch.object(evaluate, "read_json", side_effect=read_json),
              patch.object(evaluate, "sha256", return_value="same")):
            self.assertIs(evaluate.load_pipeline([], "cuda:2"), pipe)
        unet_loader.assert_called_once_with(
            "/frozen/sd15", subfolder="unet", local_files_only=True, torch_dtype="float16")
        pipe_loader.assert_called_once_with(
            "/frozen/sd15", unet=unet, low_cpu_mem_usage=False,
            local_files_only=True, torch_dtype="float16")
        pipe.to.assert_called_once_with("cuda:2")
        unet.load_attn_procs.assert_not_called()
        unet.set_attn_processor.assert_not_called()
        pipe.load_textual_inversion.assert_not_called()

    def test_base_rejects_changed_model_receipt(self):
        modules, read_json, _, _, unet_loader, pipe_loader = self.fixture()
        with (patch.dict(sys.modules, modules), patch.object(evaluate, "read_json", side_effect=read_json),
              patch.object(evaluate, "sha256", return_value="changed")):
            with self.assertRaisesRegex(ValueError, "receipt changed"):
                evaluate.load_pipeline([], "cuda:0")
        unet_loader.assert_not_called()
        pipe_loader.assert_not_called()

    def test_base_keeps_safety_and_scheduler_guards(self):
        for invalid in ("safety_checker", "scheduler"):
            with self.subTest(invalid=invalid):
                modules, read_json, _, pipe, _, _ = self.fixture()
                setattr(pipe, invalid, None)
                with (patch.dict(sys.modules, modules),
                      patch.object(evaluate, "read_json", side_effect=read_json),
                      patch.object(evaluate, "sha256", return_value="same")):
                    with self.assertRaisesRegex(ValueError, "safety checker and PNDM"):
                        evaluate.load_pipeline([], "cuda:0")


if __name__ == "__main__":
    unittest.main()
