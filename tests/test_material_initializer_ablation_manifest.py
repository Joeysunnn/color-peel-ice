"""Controlled initializer choices must retain genuine, unchanged Qwen evidence."""

import copy
import json
from pathlib import Path
import unittest

from src.train.initializer_token_utils import (
    MATERIAL_ABLATION_INITIALIZERS, MATERIAL_ABLATION_SCHEMA,
    build_material_ablation_manifest, load_initializer_manifest, validate_initializer_training,
)
from tests import test_vlm_initializer_manifest as fixtures


class AblationTokenizer(fixtures.Tokenizer):
    def get_vocab(self):
        return {**super().get_vocab(), "painted": 16, "material": 17}

    def encode(self, word, add_special_tokens=False):
        if word in ("painted", "material"):
            if add_special_tokens:
                raise AssertionError("initializer encoding must exclude BOS/EOS")
            return [16 if word == "painted" else 17]
        return super().encode(word, add_special_tokens)


class MaterialAblationManifestTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ManifestTests("test_roundtrip_includes_low_confidence_without_inventing_a_threshold")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.tokenizer = AblationTokenizer()
        self.parent = self.fixture.load(tokenizer=self.tokenizer)
        self.output = self.fixture.root / "controlled" / "initializers.json"
        self.output.parent.mkdir()

    def build(self, initializer="material"):
        return build_material_ablation_manifest(
            self.fixture.path, self.tokenizer, self.fixture.model, initializer,
        )

    def load(self, manifest):
        self.output.write_text(json.dumps(manifest), encoding="utf8")
        return load_initializer_manifest(self.output, self.tokenizer, self.fixture.model)

    def test_all_three_choices_roundtrip_without_rewriting_qwen_evidence(self):
        locked_bytes = {name: Path(name).read_bytes() for name in self.parent["source_locks"]}
        parent_bytes = self.fixture.path.read_bytes()
        for word in MATERIAL_ABLATION_INITIALIZERS:
            with self.subTest(word=word):
                manifest = self.build(word)
                self.assertEqual(self.load(manifest), manifest)
                self.assertEqual(manifest["schema"], MATERIAL_ABLATION_SCHEMA)
                self.assertEqual(manifest["selection_origin"], "controlled_ablation")
                self.assertEqual(manifest["response"], self.parent["response"])
                self.assertEqual(manifest["vlm"], self.parent["vlm"])
                self.assertEqual(manifest["tokenizer"], self.parent["tokenizer"])
                self.assertEqual(manifest["source"], self.parent["source"])
                for branch in ("subject", "color", "texture"):
                    self.assertEqual(manifest["branches"][branch], self.parent["branches"][branch])
                entry = manifest["branches"]["material"]
                for field in ("description", "confidence", "candidates", "substrate", "visible_finish"):
                    self.assertEqual(entry[field], self.parent["branches"]["material"][field])
                self.assertEqual(entry["initializer"], word)
                self.assertEqual(entry["initializer_token_id"], self.tokenizer.encode(word)[0])
                self.assertEqual(entry["selection_candidates"], [word])
                self.assertEqual(manifest["source_locks"][str(self.fixture.path)], self.fixture.digest(self.fixture.path))
        self.assertEqual(self.fixture.path.read_bytes(), parent_bytes)
        self.assertEqual({name: Path(name).read_bytes() for name in locked_bytes}, locked_bytes)

    def test_only_allowed_ordinary_single_tokens_can_be_selected(self):
        for word in ("metal", "painted metal", "unknown", None):
            with self.subTest(word=word), self.assertRaises(ValueError):
                self.build(word)
        class SplitTokenizer(AblationTokenizer):
            def encode(self, word, add_special_tokens=False):
                return [16, 17] if word == "material" else super().encode(word, add_special_tokens)
        with self.assertRaisesRegex(ValueError, "ordinary tokenizer token"):
            build_material_ablation_manifest(self.fixture.path, SplitTokenizer(), self.fixture.model, "material")

    def test_parent_must_be_original_verified_vlm_manifest_and_absolute(self):
        with self.assertRaisesRegex(ValueError, "absolute parent"):
            build_material_ablation_manifest("relative.json", self.tokenizer, self.fixture.model, "material")
        self.load(self.build())
        with self.assertRaisesRegex(ValueError, "original VLM-v1"):
            build_material_ablation_manifest(self.output, self.tokenizer, self.fixture.model, "glossy")
        changed = copy.deepcopy(self.parent)
        changed["status"] = "failed"
        self.fixture.path.write_text(json.dumps(changed), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "verified"):
            self.build()

    def test_selection_hashes_other_branches_and_raw_response_cannot_be_tampered(self):
        original = self.build()
        modifications = (
            lambda m: m.update(parent_manifest_sha256="0" * 64),
            lambda m: m.update(selection_origin="qwen"),
            lambda m: m["source_locks"].pop(str(self.fixture.path)),
            lambda m: m["branches"]["material"].update(initializer_token_id=16),
            lambda m: m["branches"]["material"].update(candidate_audit=[]),
            lambda m: m["branches"]["material"].update(candidates=["material"]),
            lambda m: m["branches"]["subject"].update(initializer="box"),
            lambda m: m["response"]["material"].update(candidates=["material"]),
            lambda m: m["ablation"].update(branch="subject"),
        )
        for index, change in enumerate(modifications):
            altered = copy.deepcopy(original)
            change(altered)
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "differs"):
                self.load(altered)
        self.fixture.path.write_bytes(self.fixture.path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "differs"):
            self.load(original)

    def test_parent_raw_evidence_is_rechecked_from_its_original_directory(self):
        manifest = self.build()
        (self.fixture.root / "raw_response.txt").write_text("{}", encoding="utf8")
        with self.assertRaisesRegex(ValueError, "source lock"):
            self.load(manifest)

    def test_manual_ablation_cannot_enable_other_training_branches_or_objectives(self):
        fixture = fixtures.TrainingScopeTests()
        manifest = self.build()
        self.assertEqual(validate_initializer_training(fixture.args(initializer_token="material"), manifest), "material")
        with self.assertRaisesRegex(ValueError, "Material branch"):
            validate_initializer_training(fixture.args(modifier_token="<S*>", initializer_token="mailbox"), manifest)
        with self.assertRaises(ValueError):
            validate_initializer_training(fixture.args(initializer_token="material", concept_isolation_weight=1), manifest)
        with self.assertRaises(ValueError):
            validate_initializer_training(fixture.args(initializer_token="glossy"), manifest)


if __name__ == "__main__":
    unittest.main()
