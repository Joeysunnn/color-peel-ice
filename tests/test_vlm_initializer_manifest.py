import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from src.train.initializer_token_utils import (
    BRANCH_TOKENS, initializer_vector_audit, load_initializer_manifest,
    select_initializer, validate_initializer_training,
)

try:
    import torch
except ImportError:
    torch = None


class Tokenizer:
    model_max_length = 77
    all_special_ids = [0, 1, 2]
    unk_token_id = 2

    def get_vocab(self):
        return {"mailbox": 11, "orange": 12, "glossy": 13, "smooth": 14, "<S*>": 100}

    def encode(self, word, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("initializer encoding must exclude BOS/EOS")
        return {"mailbox": [11], "orange": [12], "glossy": [13], "smooth": [14],
                "cyan": [20, 21], "bos": [0], "alien": [2], "learned": [100]}.get(word, [2])


class CandidateTests(unittest.TestCase):
    def test_first_valid_word_is_selected_and_all_rejections_are_recorded(self):
        candidates = ["painted metal", "", "wood-grain", "cyan", "alien", "bos", "learned", " glossy ", "smooth"]
        word, token_id, audit = select_initializer(Tokenizer(), candidates)
        self.assertEqual((word, token_id), ("glossy", 13))
        self.assertEqual(len(audit), len(candidates))
        self.assertEqual([row["candidate"] for row in audit], candidates)
        self.assertEqual(audit[0]["reasons"], ["not_single_word"])
        self.assertEqual(audit[1]["reasons"], ["empty"])
        self.assertEqual(audit[2]["reasons"], ["not_single_word"])
        self.assertEqual(audit[3]["token_ids"], [20, 21])
        self.assertEqual(audit[3]["reasons"], ["multiple_bpe"])
        self.assertIn("unknown_token", audit[4]["reasons"])
        self.assertIn("special_token", audit[5]["reasons"])
        self.assertEqual(audit[6]["reasons"], ["modifier_token"])
        self.assertTrue(audit[7]["accepted"] and audit[8]["accepted"])

    def test_unknown_and_invalid_candidates_never_fall_back(self):
        word, token_id, audit = select_initializer(Tokenizer(), [None, "unknown", "null", "cyan", "<S*>"])
        self.assertIsNone(word)
        self.assertIsNone(token_id)
        self.assertEqual(audit[0]["reasons"], ["invalid_type"])
        self.assertEqual(audit[1]["reasons"], ["unknown_value"])
        self.assertEqual(select_initializer(Tokenizer(), [])[:2], (None, None))
        with self.assertRaises(ValueError):
            select_initializer(Tokenizer(), "orange")


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        (self.model / "tokenizer").mkdir(parents=True)
        self.image = self.root / "input.png"
        self.image.write_bytes(b"unchanged source image")
        self.files = {}
        for name in ("vocab.json", "merges.txt", "tokenizer_config.json"):
            path = self.model / "tokenizer" / name
            path.write_text(name, encoding="utf8")
            self.files[str(path)] = self.digest(path)
        self.manifest = {
            "schema": "colorpeel_vlm_initializers/v1", "status": "verified",
            "source": {"image": str(self.image), "image_sha256": self.digest(self.image)},
            "vlm": {"model_id": "Qwen/test", "revision": "locked-revision"},
            "tokenizer": {"model_root": str(self.model), "files": self.files.copy(), "model_max_length": 77},
            "source_locks": {str(self.image): self.digest(self.image), **self.files}, "branches": {},
        }
        for (branch, token), candidate in zip(BRANCH_TOKENS.items(), ("mailbox", "orange", "glossy", "smooth")):
            word, token_id, audit = select_initializer(Tokenizer(), [candidate])
            self.manifest["branches"][branch] = {
                "token": token, "description": candidate, "initializer": word, "initializer_token_id": token_id,
                "confidence": "low", "candidates": [candidate], "candidate_audit": audit,
            }
        self.manifest["branches"]["subject"]["category"] = "mailbox"
        self.manifest["branches"]["material"].update(substrate="painted metal", visible_finish="glossy")
        self.manifest["subject_category"] = "mailbox"
        self.manifest["response"] = {
            branch: {key: value for key, value in entry.items()
                     if key not in ("token", "initializer", "initializer_token_id", "candidate_audit")}
            for branch, entry in self.manifest["branches"].items()
        }
        for name in ("raw_response.txt", "response.json"):
            path = self.root / name
            path.write_text(json.dumps(self.manifest["response"]), encoding="utf8")
            self.manifest["source_locks"][str(path)] = self.digest(path)
        self.path = self.root / "initializers.json"

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def load(self, manifest=None, tokenizer=None, model=None):
        self.path.write_text(json.dumps(self.manifest if manifest is None else manifest), encoding="utf8")
        return load_initializer_manifest(self.path, tokenizer or Tokenizer(), model or self.model)

    def test_roundtrip_includes_low_confidence_without_inventing_a_threshold(self):
        self.assertEqual(self.load(), self.manifest)

    def test_source_and_tokenizer_bytes_must_match(self):
        self.image.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source lock"):
            self.load()
        self.image.write_bytes(b"unchanged source image")
        Path(next(iter(self.files))).write_bytes(b"changed vocabulary")
        with self.assertRaisesRegex(ValueError, "source lock"):
            self.load()

    def test_all_actual_tokenizer_files_and_correct_root_are_required(self):
        changed = copy.deepcopy(self.manifest)
        changed["tokenizer"]["files"].pop(next(iter(self.files)))
        with self.assertRaisesRegex(ValueError, "all actual tokenizer"):
            self.load(changed)
        (self.model / "tokenizer" / "added_tokens.json").write_text("{}", encoding="utf8")
        with self.assertRaisesRegex(ValueError, "all actual tokenizer"):
            self.load()
        with self.assertRaisesRegex(ValueError, "tokenizer root"):
            self.load(model=self.root / "different_base")

    def test_source_image_cannot_be_omitted_from_locks(self):
        changed = copy.deepcopy(self.manifest)
        changed["source_locks"].pop(str(self.image))
        changed["source"].pop("image_sha256")
        with self.assertRaisesRegex(ValueError, "source image"):
            self.load(changed)

    def test_incomplete_failed_or_unknown_decisions_are_not_verified(self):
        for field, value in (("status", "failed"), ("schema", "other")):
            changed = copy.deepcopy(self.manifest)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load(changed)
        for field, value in (("confidence", "unknown"), ("candidates", []), ("description", ""),
                             ("token", "<M*>"), ("initializer_token_id", True)):
            changed = copy.deepcopy(self.manifest)
            changed["branches"]["texture"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load(changed)
        changed = copy.deepcopy(self.manifest)
        changed["branches"].pop("material")
        with self.assertRaisesRegex(ValueError, "four named branches"):
            self.load(changed)

    def test_candidate_choice_audit_and_actual_tokenization_are_recomputed(self):
        for field, value in (("initializer", "orange"), ("initializer_token_id", 12), ("candidate_audit", [])):
            changed = copy.deepcopy(self.manifest)
            changed["branches"]["material"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "candidate audit"):
                self.load(changed)
        class DifferentTokenizer(Tokenizer):
            def encode(self, word, add_special_tokens=False):
                return [77] if word == "glossy" else super().encode(word, add_special_tokens)
        with self.assertRaisesRegex(ValueError, "candidate audit"):
            self.load(tokenizer=DifferentTokenizer())

    def test_weights_are_not_part_of_the_lightweight_intake_locks(self):
        changed = copy.deepcopy(self.manifest)
        changed["source_locks"][str(self.model / "pytorch_model.bin")] = "0" * 64
        with self.assertRaisesRegex(ValueError, "excluding model weights"):
            self.load(changed)

    def test_raw_response_parsed_response_and_semantic_fields_are_bound(self):
        for branch, field, value in (("subject", "category", "spoon"), ("material", "substrate", "wood"),
                                     ("material", "visible_finish", "matte"), ("texture", "description", "wood grain")):
            changed = copy.deepcopy(self.manifest)
            changed["branches"][branch][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "semantic fields"):
                self.load(changed)
        changed = copy.deepcopy(self.manifest)
        changed["subject_category"] = "spoon"
        with self.assertRaisesRegex(ValueError, "subject_category"):
            self.load(changed)
        changed = copy.deepcopy(self.manifest)
        changed["response"]["subject"]["category"] = "spoon"
        with self.assertRaisesRegex(ValueError, "manifest response"):
            self.load(changed)
        changed = copy.deepcopy(self.manifest)
        changed["source_locks"].pop(str(self.root / "raw_response.txt"))
        with self.assertRaisesRegex(ValueError, "locked beside"):
            self.load(changed)
        changed = copy.deepcopy(self.manifest)
        parsed = self.root / "response.json"
        parsed.write_text("{}", encoding="utf8")
        changed["source_locks"][str(parsed)] = self.digest(parsed)
        with self.assertRaisesRegex(ValueError, "parsed response"):
            self.load(changed)


class TrainingScopeTests(unittest.TestCase):
    def args(self, **changes):
        values = dict(modifier_token="<M*>", initializer_token="glossy", subject_lora_mode="token_local_kv",
            subject_lora_rank=64, subject_lora_alpha=64, freeze_model="crossattn_kv", token_local_kv=False,
            perfusion_subject=False, enable_xformers_memory_efficient_attention=False, k_learning_rate=None,
            v_learning_rate=None, adam_weight_decay=0, use_8bit_adam=False, resume_from_checkpoint=None,
            with_prior_preservation=False, joint_two_object_binding=False, strict_unpaired_modifier_updates=False,
            cos_weight=0, lambda_attention=0, texture_background_preservation_weight=0,
            texture_external_preservation_weight=0, concept_isolation_weight=0)
        values.update(changes)
        return SimpleNamespace(**values)

    def test_manifest_must_match_one_branch_and_existing_training_semantics(self):
        manifest = {"branches": {"material": {"initializer": "glossy"}}}
        self.assertEqual(validate_initializer_training(self.args(), manifest), "material")
        self.assertEqual(validate_initializer_training(self.args(max_train_steps=2), manifest), "material")
        for changes in ({"initializer_token": "metal"}, {"modifier_token": "<S*>+<M*>"},
                        {"subject_lora_rank": 4}, {"subject_lora_alpha": 4}, {"subject_lora_mode": "full_kv"},
                        {"freeze_model": "crossattn"}, {"adam_weight_decay": 0.01}, {"with_prior_preservation": True},
                        {"resume_from_checkpoint": "checkpoint-500"}, {"concept_isolation_weight": 1},
                        {"texture_external_preservation_weight": 100}, {"texture_background_preservation_weight": 1},
                        {"cos_weight": 0.2}, {"lambda_attention": 0.1}, {"k_learning_rate": 1e-5},
                        {"token_local_kv": True}, {"use_8bit_adam": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_initializer_training(self.args(**changes), manifest)

    def test_actual_lora_guard_keeps_legacy_default_and_requires_loaded_opt_in(self):
        path = Path(__file__).resolve().parents[1] / "src/train/train_colorpeel.py"
        tree = ast.parse(path.read_text(encoding="utf8"))
        block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
            and isinstance(node.test, ast.Attribute) and node.test.attr == "subject_lora_mode"
            and any(isinstance(child, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == "valid_concept" for target in child.targets) for child in node.body))
        stop = next(index for index, node in enumerate(block.body) if isinstance(node, ast.If)
                    and any(isinstance(child, ast.Raise) for child in node.body))
        guard = compile(ast.Module(body=block.body[:stop + 1], type_ignores=[]), str(path), "exec")
        for manifest_path, loaded, passes in ((None, None, False), ("initializers.json", None, False),
                                              ("initializers.json", {"status": "verified"}, True)):
            args = self.args(modifier_token=["<M*>"], initializer_token=["glossy"], initializer_manifest=manifest_path)
            scope = {"args": args, "initializer_manifest_data": loaded, "initializer_branch": "material"}
            if passes:
                exec(guard, scope)
                self.assertTrue(scope["valid_concept"])
            else:
                with self.assertRaisesRegex(ValueError, "LoRA requires"):
                    exec(guard, scope)

    def test_trainer_reads_manifest_only_when_opted_in_and_checks_before_copy(self):
        path = Path(__file__).resolve().parents[1] / "src/train/train_colorpeel.py"
        tree = ast.parse(path.read_text(encoding="utf8"))
        gate = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "args.initializer_manifest is not None")
        calls = [node for node in ast.walk(gate) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertIn("load_initializer_manifest", [node.func.id for node in calls])
        scope = next(node for node in calls if node.func.id == "validate_initializer_training")
        copy_audit = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                          and isinstance(node.func, ast.Name) and node.func.id == "initializer_vector_audit")
        self.assertLess(scope.lineno, copy_audit.lineno)
        self.assertIn('"initializer_manifest_sha256"', path.read_text(encoding="utf8"))


@unittest.skipIf(torch is None, "PyTorch is not installed on this host")
class VectorAuditTests(unittest.TestCase):
    def test_copy_hashes_actual_base_and_learned_vector_without_rng_change(self):
        base, before = torch.arange(1024, dtype=torch.float32), torch.zeros(1024)
        after, initializer_after = base.clone(), base.clone()
        rng = torch.get_rng_state().clone()
        audit = initializer_vector_audit(base, initializer_after, before, after)
        self.assertTrue(audit["initialized_equal_base"] and audit["initializer_unchanged"])
        self.assertEqual(audit["base_initializer_sha256"], audit["modifier_after_copy_sha256"])
        self.assertNotEqual(audit["modifier_before_copy_sha256"], audit["modifier_after_copy_sha256"])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(after, base))

    def test_mismatched_initializer_or_copy_and_wrong_model_width_are_rejected(self):
        base = torch.ones(1024)
        for values in ((base, base + 1, base, base), (base, base, base, base + 1),
                       tuple(torch.ones(768) for _ in range(4)), tuple(torch.ones(1024).half() for _ in range(4))):
            with self.assertRaises(ValueError):
                initializer_vector_audit(*values)


if __name__ == "__main__":
    unittest.main()
