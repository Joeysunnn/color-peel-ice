import contextlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image
from experiments.texture_material_disentangled_v1 import infer


class Tensor:
    def __init__(self, values):
        self.values = values

    def to(self, device):
        return self

    def __eq__(self, value):
        return Tensor([[item == value for item in row] for row in self.values])

    def sum(self):
        return sum(sum(row) for row in self.values)


class InferenceTests(unittest.TestCase):
    def test_factorial_matrix_and_literal_token_swaps(self):
        protocol = infer.read_json(infer.PROTOCOL)
        rows = infer.build_rows(protocol)
        self.assertEqual(len(rows), 256)
        for method in infer.METHODS:
            self.assertEqual(sum(row["method"] == method for row in rows), 128)
        for row in rows:
            self.assertEqual(row["prompt"].count("<T*>"), int(row["condition"] in {"texture_only", "both"}))
            self.assertEqual(row["prompt"].count("<M*>"), int(row["condition"] in {"material_only", "both"}))
        example = next(row for row in rows if row["condition"] == "texture_only")
        self.assertIn(protocol["materials"][example["material"]]["literal"], example["prompt"])
        for condition in protocol["inference"]["conditions"]:
            self.assertEqual(sum(row["condition"] == condition for row in rows), 64)

    def test_gates_map_texture_to_primary_and_cfg_unconditional_to_zero(self):
        class Tokenizer:
            model_max_length = 6
            unk_token_id = 99

            def convert_tokens_to_ids(self, token):
                return {"<T*>": 4, "<M*>": 5}[token]

            def __call__(self, prompt, **kwargs):
                return types.SimpleNamespace(input_ids=Tensor([[1, 4, 5, 2, 2, 2]]))

        pipe = types.SimpleNamespace(tokenizer=Tokenizer(), unet=types.SimpleNamespace(device="cuda:0"))
        torch = types.SimpleNamespace(
            zeros_like=lambda tensor: Tensor([[False for item in row] for row in tensor.values]),
            cat=lambda tensors: Tensor([row for tensor in tensors for row in tensor.values]))
        with patch.dict(sys.modules, {"torch": torch}):
            masks = infer.attention_kwargs(pipe, "<T*> and <M*>", 3.5)["modifier_token_mask"]
            self.assertEqual(set(masks), {"subject", "material"})
            self.assertEqual(masks["subject"].values, [[False] * 6, [False, True, False, False, False, False]])
            self.assertEqual(masks["material"].values, [[False] * 6, [False, False, True, False, False, False]])
            with self.assertRaisesRegex(ValueError, "repeated or truncated"):
                infer.attention_kwargs(pipe, "<T*> <T*> <M*>", 3.5)

    def test_index_rejects_missing_concepts_and_changed_weights(self):
        protocol = infer.read_json(infer.PROTOCOL)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary).resolve()
            runs = []
            for method in infer.METHODS:
                for branch in infer.TOKENS:
                    for concept in infer.target_ids(protocol, branch):
                        token = infer.TOKENS[branch]
                        runs.append({"method": method, "branch": branch, "concept_id": concept,
                                     "checkpoint": str(checkpoint), "run": str(checkpoint), "status": "verified",
                                     "steps": protocol["training"]["steps"],
                                     "weight": infer.WEIGHTS[method], "token": token,
                                     "hashes": {str(checkpoint / name): "same" for name in
                                                (infer.WEIGHTS[method], token + ".bin", "adaptation_config.json")}})
            index = {"mode": "pilot", "protocol_sha256": infer.protocol_hash(protocol), "runs": runs}
            with patch.object(infer, "sha256", return_value="same"):
                infer.verify_inputs(index, protocol)
                with self.assertRaisesRegex(ValueError, "eight distinct"):
                    infer.verify_inputs({**index, "runs": runs[:-1]}, protocol)
                with self.assertRaisesRegex(ValueError, "not smoke"):
                    infer.verify_inputs({**index, "mode": "smoke"}, protocol)
                with self.assertRaisesRegex(ValueError, "pilot steps"):
                    infer.verify_inputs({**index, "runs": [{**runs[0], "steps": 2}, *runs[1:]]}, protocol)
            with patch.object(infer, "sha256", return_value="changed"):
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    infer.verify_inputs(index, protocol)

    def test_replays_sampling_and_seed_and_marks_safety_filter(self):
        protocol = infer.read_json(infer.PROTOCOL)
        rows = infer.build_rows(protocol)[:2]
        calls, generators = [], []

        class Generator:
            def __init__(self, device):
                self.device = device
                generators.append(self)

            def manual_seed(self, seed):
                self.seed = seed
                return self

        def pipeline(prompt, **kwargs):
            calls.append((prompt, kwargs))
            return types.SimpleNamespace(images=[Image.new("RGB", (2, 2))], nsfw_content_detected=[len(calls) == 2])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path = root / "training.json"
            runs = [{"method": method, "branch": branch, "concept_id": concept} for method in infer.METHODS
                    for branch in infer.TOKENS for concept in infer.target_ids(protocol, branch)]
            index_path.write_text(json.dumps({"git": {"commit": "same"}, "runs": runs}), encoding="utf-8")
            torch = types.SimpleNamespace(Generator=Generator, no_grad=contextlib.nullcontext,
                                          cuda=types.SimpleNamespace(empty_cache=lambda: None))
            with (patch.dict(sys.modules, {"torch": torch}),
                  patch.object(infer, "git_info", return_value={"commit": "same"}),
                  patch.object(infer, "verify_inputs"), patch.object(infer, "build_rows", return_value=rows),
                  patch.object(infer, "resolve_run_dir", side_effect=lambda path: path),
                  patch.object(infer, "package_versions", return_value={}),
                  patch.object(infer, "load_pipeline", return_value=pipeline) as loader,
                  patch.object(infer, "attention_kwargs", return_value={"modifier_token_mask": "gates"}),
                  patch.object(infer, "review_bundle", return_value=[])):
                infer.run(index_path, root / "new", "cuda:2")
            self.assertEqual(loader.call_count, 8)
            self.assertEqual([generator.seed for generator in generators], [42, 43])
            self.assertTrue(all(generator.device == "cuda:2" for generator in generators))
            for row, (prompt, kwargs) in zip(rows, calls):
                self.assertEqual(prompt, row["prompt"])
                self.assertEqual({key: kwargs[key] for key in ("num_inference_steps", "guidance_scale", "width", "height")},
                                 {"num_inference_steps": 100, "guidance_scale": 3.5, "width": 512, "height": 512})
                self.assertEqual(kwargs["cross_attention_kwargs"], {"modifier_token_mask": "gates"})
            statuses = [json.loads(line) for line in (root / "new/generation_status.jsonl").read_text().splitlines()]
            self.assertEqual([row["status"] for row in statuses], ["ok", "safety_filtered"])
            self.assertEqual(json.loads((root / "new/provenance.json").read_text())["status"], "succeeded")

    def test_review_excludes_filtered_images_from_blind_ratings(self):
        protocol = infer.read_json(infer.PROTOCOL)
        rows = [{"id": "first", "image_path": "first.png", "status": "ok"},
                {"id": "second", "image_path": "second.png", "status": "safety_filtered"}]
        with tempfile.TemporaryDirectory() as temporary, patch.object(infer, "contact_sheets", return_value=[]):
            output = Path(temporary)
            infer.review_bundle(output, rows, protocol)
            self.assertEqual(len((output / "blinded_ratings.csv").read_text().splitlines()), 2)
            self.assertIn("first.png", (output / "blinded_review.html").read_text())
            self.assertNotIn("second.png", (output / "blinded_review.html").read_text())

    def test_contact_sheets_cover_every_context_method_and_seed(self):
        protocol = infer.read_json(infer.PROTOCOL)
        rows = [{**row, "image_path": "sample.png", "status": "ok"} for row in infer.build_rows(protocol)]
        rows[0]["status"] = "safety_filtered"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            Image.new("RGB", (2, 2)).save(output / "sample.png")
            sheets = infer.contact_sheets(output, rows, protocol)
            self.assertEqual(len(sheets), 16)
            self.assertEqual(len(set(sheets)), 16)
            for path in sheets:
                with Image.open(output / path) as sheet:
                    self.assertEqual(sheet.size, (1088, 1264))


if __name__ == "__main__":
    unittest.main()
