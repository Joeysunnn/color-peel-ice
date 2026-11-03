"""V2 manifests bind controlled decisions to original two-axis VLM evidence."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.methods.colorpeel_ice import vlm_initialization as vlm
from src.train.initializer_token_utils import load_initializer_manifest
from tests.methods.colorpeel_ice import test_vlm_initialization as fixtures


class ControlledManifestTests(unittest.TestCase):
    def fixture(self, root, response=None):
        helper = fixtures.VlmInitializationTests()
        path = helper._run_fixture(root, json.dumps(response or fixtures.response()))
        return path, json.loads(path.read_text(encoding="utf8"))

    def test_tampered_semantics_word_token_audit_and_policy_are_rejected(self):
        changes = (
            lambda m: m["branches"]["material"]["metallicity"].update(label="metallic"),
            lambda m: m["branches"]["material"]["roughness"].update(confidence="low"),
            lambda m: m["branches"]["material"]["roughness"].update(evidence="Invented broad highlights"),
            lambda m: m["branches"]["material"].update(initializer="painted"),
            lambda m: m["branches"]["material"].update(initializer_token_id=15),
            lambda m: m["branches"]["material"].update(candidate_audit=[]),
            lambda m: m["branches"]["material"].update(selection_candidates=["material"]),
            lambda m: m["branches"]["material"]["selection_policy"].update(policy_version="unreviewed/v9"),
            lambda m: m["branches"]["material"]["selection_policy"].update(fallback=True),
            lambda m: m["response"]["material"]["roughness"].update(label="high"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, manifest = self.fixture(root)
            for index, change in enumerate(changes):
                altered = copy.deepcopy(manifest)
                change(altered)
                path.write_text(json.dumps(altered), encoding="utf8")
                with self.subTest(index=index), self.assertRaises(ValueError):
                    load_initializer_manifest(path, fixtures.Tokenizer(), root / "sd")

    def test_uncertain_fallback_is_explicit_and_original_raw_is_preserved(self):
        response = fixtures.response()
        for axis in ("metallicity", "roughness"):
            response["material"][axis].update(label="uncertain", confidence="low", evidence="Insufficient reflection evidence")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, manifest = self.fixture(root, response)
            loaded = load_initializer_manifest(path, fixtures.Tokenizer(), root / "sd")
            self.assertEqual(loaded, manifest)
            material = manifest["branches"]["material"]
            self.assertEqual((material["initializer"], material["initializer_token_id"]), ("material", 15))
            self.assertTrue(material["selection_policy"]["fallback"])
            self.assertEqual(material["selection_policy"]["selection_reason"], "uncertain_or_low_confidence")
            self.assertEqual(manifest["response"], response)
            self.assertEqual(json.loads((path.parent / "raw_response.txt").read_text()), response)
            self.assertEqual(json.loads((path.parent / "response.json").read_text()), response)
            self.assertNotIn("candidates", response["material"])

    def test_selected_multibpe_word_fails_without_generic_substitution(self):
        class SplitTokenizer(fixtures.Tokenizer):
            def encode(self, word, add_special_tokens=False):
                return [12, 88] if word == "glossy" else super().encode(word, add_special_tokens)
        response = vlm.parse_response(json.dumps(fixtures.response()))
        branches, failures = vlm.build_branches(response, SplitTokenizer())
        self.assertTrue(failures)
        self.assertIsNone(branches["material"]["initializer"])
        self.assertEqual(branches["material"]["selection_candidates"], ["glossy"])
        self.assertEqual(branches["material"]["selection_policy"]["initializer"], "glossy")
        self.assertEqual(branches["material"]["candidate_audit"][0]["reasons"], ["multiple_bpe"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _ = self.fixture(root)
            with self.assertRaises(ValueError):
                load_initializer_manifest(path, SplitTokenizer(), root / "sd")


if __name__ == "__main__":
    unittest.main()
