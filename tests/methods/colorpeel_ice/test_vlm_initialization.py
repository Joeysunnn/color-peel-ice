import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from src.methods.colorpeel_ice import vlm_initialization as vlm
from src.train.initializer_token_utils import load_initializer_manifest


class Tokenizer:
    model_max_length = 77
    unk_token_id = 99
    all_special_ids = [98, 99]

    def get_vocab(self):
        return {"mailbox": 10, "orange": 11, "glossy": 12, "plain": 13}

    def encode(self, word, add_special_tokens=False):
        return {"mailbox": [10], "orange": [11], "glossy": [12], "plain": [13],
                "speckled": [20, 21]}.get(word, [99])


def response():
    return {
        "subject": {"category": "mailbox", "description": "Curved lid above a rectangular wall mounted box",
                    "confidence": "high", "candidates": ["mailbox"]},
        "color": {"description": "Dominant orange foreground", "confidence": "high", "candidates": ["orange"]},
        "material": {"substrate": "uncertain metal", "visible_finish": "glossy painted coating",
                     "description": "Narrow bright highlights across the coating", "confidence": "medium", "candidates": ["glossy"]},
        "texture": {"description": "No reliable visible spatial markings", "confidence": "low", "candidates": ["plain"]},
    }


class VlmInitializationTests(unittest.TestCase):
    def test_prompt_does_not_encode_mailbox_or_orange_answer(self):
        self.assertNotIn("mailbox", vlm.PROMPT.lower())
        self.assertNotIn("orange", vlm.PROMPT.lower())
        self.assertIn("Do not estimate numerical roughness", vlm.PROMPT)
        self.assertIn("plain or uniform", vlm.PROMPT)

    def test_strict_json_preserves_semantics_and_rejects_wrapping_duplicates_or_fields(self):
        value = response()
        self.assertEqual(vlm.parse_response(json.dumps(value)), value)
        invalid = ["```json\n" + json.dumps(value) + "\n```",
                   '{"subject":{},"subject":{}}', json.dumps({**value, "extra": 1})]
        value["material"]["roughness"] = 0.25
        invalid.append(json.dumps(value))
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                vlm.parse_response(raw)

    def test_semantic_separation_rejects_reflection_texture_and_substrate_initializer(self):
        for branch, word in [("texture", "glossy"), ("texture", "rough"), ("texture", "smooth"),
                             ("material", "wood"), ("material", "metal")]:
            value = response()
            value[branch]["candidates"] = [word]
            with self.subTest(branch=branch, word=word), self.assertRaises(ValueError):
                vlm.parse_response(json.dumps(value))
        for word in ("plain", "uniform", "textureless"):
            value = response()
            value["texture"]["candidates"] = [word]
            self.assertEqual(vlm.parse_response(json.dumps(value))["texture"]["candidates"], [word])

    def test_unknown_and_invalid_candidates_never_produce_ready_decision(self):
        value = response()
        value["texture"].update(confidence="unknown", candidates=[])
        parsed = vlm.parse_response(json.dumps(value))
        branches, failures = vlm.build_branches(parsed, Tokenizer())
        self.assertIsNone(branches["texture"]["initializer"])
        self.assertTrue(any("unknown" in error for error in failures))
        for candidates in (["fine grain"], ["<T*>"], ["speckled"], ["unknown"]):
            value = response()
            value["texture"]["candidates"] = candidates
            branches, failures = vlm.build_branches(vlm.parse_response(json.dumps(value)), Tokenizer())
            self.assertTrue(failures)
            self.assertIsNone(branches["texture"]["initializer"])

    def test_ranked_selection_records_bpe_rejection_without_truncation(self):
        value = response()
        value["texture"]["candidates"] = ["speckled", "plain"]
        branches, failures = vlm.build_branches(vlm.parse_response(json.dumps(value)), Tokenizer())
        texture = branches["texture"]
        self.assertEqual(failures, [])
        self.assertEqual((texture["initializer"], texture["initializer_token_id"]), ("plain", 13))
        self.assertEqual(texture["candidate_audit"][0]["token_ids"], [20, 21])
        self.assertIn("multiple_bpe", texture["candidate_audit"][0]["reasons"])

    def test_schema_rejects_bad_category_confidence_and_candidate_count(self):
        mutations = [("subject", "category", "<S*> mailbox"), ("texture", "confidence", []),
                     ("color", "confidence", "certain"), ("color", "candidates", ["orange"] * 5),
                     ("color", "candidates", []), ("color", "candidates", [2])]
        for branch, key, replacement in mutations:
            value = response()
            value[branch][key] = replacement
            with self.subTest(branch=branch, key=key), self.assertRaises(ValueError):
                vlm.parse_response(json.dumps(value))

    def test_model_receipt_hashes_weights_but_labels_disk_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / vlm.REVISION
            folder.mkdir()
            (folder / "weights.safetensors").write_bytes(b"fake-weight-fixture")
            (folder / "config.json").write_text('{"model_type":"qwen3_vl"}')
            (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "weights.safetensors"}}))
            receipt = vlm.model_receipt(folder)
            self.assertEqual(receipt["weights_total_bytes"], len(b"fake-weight-fixture"))
            self.assertEqual(receipt["weights"][0]["sha256"], vlm.sha256(folder / "weights.safetensors"))
            self.assertNotIn(str((folder / "weights.safetensors").resolve()), receipt["metadata_files"])
            self.assertIn("not GPU", receipt["size_note"])

    def _run_fixture(self, root, raw):
        image = root / "input.png"
        Image.new("RGB", (32, 24), (12, 34, 56)).save(image)
        sd = root / "sd"
        (sd / "tokenizer").mkdir(parents=True)
        (sd / "tokenizer/vocab.json").write_text("{}")
        fake = SimpleNamespace(runtime={"dtype": "float16", "do_sample": False})
        predictor = type("Predictor", (), {"runtime": fake.runtime, "__call__": lambda self, image: raw})()
        transformers = SimpleNamespace(CLIPTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()))
        with patch.object(vlm, "model_receipt", return_value={"metadata_files": {}, "weights": []}), \
                patch.object(vlm, "QwenPredictor", return_value=predictor), \
                patch.dict(sys.modules, {"transformers": transformers}):
            return vlm.run(image, root / "qwen", sd, root / "output", "cpu")

    def test_success_roundtrips_shared_loader_and_detects_source_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self._run_fixture(root, json.dumps(response()))
            self.assertEqual(path.name, "initializers.json")
            manifest = load_initializer_manifest(path, Tokenizer(), root / "sd")
            self.assertEqual(manifest["subject_category"], "mailbox")
            self.assertEqual(set(manifest["branches"]), set(vlm.TOKENS))
            self.assertEqual(manifest["response"], response())
            (root / "input.png").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                load_initializer_manifest(path, Tokenizer(), root / "sd")

    def test_failed_parse_preserves_raw_receipts_and_never_claims_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                self._run_fixture(root, "incomplete JSON {")
            self.assertEqual((root / "output/raw_response.txt").read_text(), "incomplete JSON {")
            self.assertTrue((root / "output/model_receipt.json").is_file())
            self.assertTrue((root / "output/generation_receipt.json").is_file())
            self.assertFalse((root / "output/initializers.json").exists())
            self.assertEqual(json.loads((root / "output/status.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
