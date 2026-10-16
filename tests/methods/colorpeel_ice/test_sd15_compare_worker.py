import contextlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image
from experiments.sd15_token_local_kv_v1 import compare_worker as worker
from experiments.sd15_token_local_kv_v1.contract import sha256


class WorkerTests(unittest.TestCase):
    def test_receipt_allows_audit_metadata_but_rejects_changed_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "checkpoint-300").mkdir()
            names = ["derived.bin", "final.bin", "checkpoint-300/pytorch_model.bin",
                     "checkpoint-300/pytorch_model_1.bin", "old.png", "metadata.json"]
            for name in names:
                (root / name).write_bytes(name.encode())
            selection = {"checkpoint": str(root), "source_checkpoint": str(root), "step": 300,
                         "hashes": {str(root / "derived.bin"): sha256(root / "derived.bin")},
                         "source_hashes": {str(root / "final.bin"): sha256(root / "final.bin")},
                         "source_adapter_sha256": sha256(root / names[2]),
                         "source_text_encoder_sha256": sha256(root / names[3])}
            (root / "receipt.json").write_text(json.dumps(selection), encoding="utf-8")
            job = {"selections": [{**selection, "tensor_count": 32}],
                   "reference": {"directory": ".", "files": {"metadata.json": sha256(root / "metadata.json")}},
                   "rows": [{"id": "row", "reference_image": str(root / "old.png"),
                             "reference_sha256": sha256(root / "old.png")}]}
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({"reference_root": str(root)}), encoding="utf-8")
            with patch.object(worker, "INFERENCE_PROTOCOL", protocol):
                worker.verify_inputs(job)
                (root / names[2]).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "original training snapshot"):
                    worker.verify_inputs(job)

    def test_generation_replays_seed_sampling_and_marks_filtered_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old.png"
            Image.new("RGB", (2, 2)).save(old)
            rows = [{"id": f"row{index}", "prompt": "a photo of <S*> mailbox", "seed": seed,
                     "image_path": f"images/{index:04d}.png", "reference_image": str(old),
                     "reference_sha256": sha256(old), "reference_filtered": False}
                    for index, seed in enumerate([42, 43])]
            job = {"id": "job", "output": str(root / "new"), "rows": rows,
                   "selections": [{"method": "fullrank", "token": "<S*>"}]}
            plan = root / "plan.json"
            plan.write_text(json.dumps({"git": {"commit": "same"}, "sampling": worker.SAMPLING,
                                        "jobs": [job]}), encoding="utf-8")
            generators, calls = [], []

            class Generator:
                def __init__(self, device):
                    self.device = device
                    generators.append(self)

                def manual_seed(self, seed):
                    self.seed = seed
                    return self

            def pipeline(prompt, **kwargs):
                calls.append((prompt, kwargs))
                return types.SimpleNamespace(images=[Image.new("RGB", (2, 2))],
                                             nsfw_content_detected=[len(calls) == 2])

            torch = types.SimpleNamespace(Generator=Generator, no_grad=contextlib.nullcontext)
            with (patch.dict(sys.modules, {"torch": torch}),
                  patch.object(worker, "git_info", return_value={"commit": "same"}),
                  patch.object(worker, "verify_inputs"),
                  patch.object(worker, "resolve_run_dir", side_effect=lambda path: path),
                  patch.object(worker, "package_versions", return_value={}),
                  patch.object(worker, "load_pipeline", return_value=pipeline),
                  patch.object(worker, "attention_masks", return_value={"modifier_token_mask": "mask"})):
                worker.run(plan, "job", "cuda:0")
            self.assertEqual([generator.seed for generator in generators], [42, 43])
            self.assertTrue(all(generator.device == "cuda:0" for generator in generators))
            self.assertIsNot(generators[0], generators[1])
            for row, (prompt, kwargs) in zip(rows, calls):
                self.assertEqual(prompt, row["prompt"])
                self.assertEqual({key: kwargs[key] for key in ("num_inference_steps", "guidance_scale", "width", "height")},
                                 {"num_inference_steps": 100, "guidance_scale": 3.5, "width": 512, "height": 512})
            statuses = [json.loads(line) for line in (root / "new/generation_status.jsonl").read_text().splitlines()]
            self.assertEqual([row["status"] for row in statuses], ["ok", "safety_filtered"])
            self.assertEqual(json.loads((root / "new/comparison_bundle.json").read_text())["pair_count"], 2)


if __name__ == "__main__":
    unittest.main()
