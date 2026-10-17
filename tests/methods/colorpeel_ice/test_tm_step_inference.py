import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.texture_material_disentangled_v1 import data, step_ablation, step_inference, train


class StepInferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        self.source, self.batch = root / "original", root / "completed"
        self.receipt = {"source": str(self.source), "source_index_sha256": "original-index"}
        self.training_git = {"branch": "codex/texture-material-v1", "commit": "a" * 40}
        self.audits = self.batch / "training/audits"
        self.audits.mkdir(parents=True)
        self.write(self.batch / "status.json", {
            "status": "failed", "git": {**self.training_git, "clean_worktree": True},
            "source": self.receipt, "protocol_sha256": data.protocol_sha256()})
        self.selections = []
        for index, recipe in enumerate(train.recipes()):
            run = self.batch / "training" / f"run{index}"
            run.mkdir()
            self.write(run / "config.yaml", step_ablation.build_config(recipe, self.source / "staging"))
            self.write(run / "manifest.json", {"status": "succeeded", "returncode": 0, "git": self.training_git})
            selection = {**recipe, "status": "verified", "steps": 1000, "run": str(run),
                         "checkpoint": str(run / "checkpoints")}
            self.write(self.audits / f"audit{index}.json", selection)
            self.selections.append(selection)
        verifier = patch.object(step_inference, "_verify")
        self.verifier = verifier.start()
        self.addCleanup(verifier.stop)
        config_reader = patch.object(step_inference, "read_config", side_effect=self.read)
        config_reader.start()
        self.addCleanup(config_reader.stop)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def collect(self):
        return step_inference.completed_selections(self.batch, self.source, self.receipt)

    def test_reconstructs_eight_using_training_revision_without_current_revision_check(self):
        with patch.object(step_inference, "git_info", return_value={"commit": "b" * 40}) as current:
            selections, revision, locks = self.collect()
        self.assertEqual(selections, self.selections)
        self.assertEqual(revision, self.training_git)
        self.assertEqual(self.verifier.call_count, 8)
        current.assert_not_called()
        self.assertEqual(set(locks), {str(self.batch / "status.json"),
                                     *(str(self.audits / f"audit{index}.json") for index in range(8))})
        self.assertTrue(all(step_inference.sha256(Path(path)) == digest for path, digest in locks.items()))
        self.assertEqual(self.read(self.batch / "status.json")["status"], "failed")

    def test_rejects_missing_audit(self):
        (self.audits / "audit7.json").unlink()
        with self.assertRaisesRegex(ValueError, "exactly eight"):
            self.collect()

    def test_rejects_duplicate_concept_method_with_eight_audits(self):
        self.write(self.audits / "audit7.json", self.selections[0])
        with self.assertRaisesRegex(ValueError, "concepts or methods"):
            self.collect()

    def test_rejects_changed_config(self):
        path = Path(self.selections[0]["run"]) / "config.yaml"
        config = self.read(path)
        config["args"]["learning_rate"] *= 2
        self.write(path, config)
        with self.assertRaisesRegex(ValueError, "config, path, steps or provenance"):
            self.collect()

    def test_rejects_failed_manifest_and_mismatched_training_revision(self):
        path = Path(self.selections[0]["run"]) / "manifest.json"
        original = self.read(path)
        for changed in ({"status": "failed"}, {"returncode": 1},
                        {"git": {**self.training_git, "commit": "b" * 40}}):
            with self.subTest(changed=changed):
                self.write(path, {**original, **changed})
                with self.assertRaisesRegex(ValueError, "config, path, steps or provenance"):
                    self.collect()


if __name__ == "__main__":
    unittest.main()
