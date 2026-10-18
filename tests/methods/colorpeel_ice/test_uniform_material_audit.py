import json
from pathlib import Path
import unittest
from unittest.mock import patch

import yaml

from experiments.material_uniform_albedo_v1 import audit, data, run
from experiments.sd15_token_local_kv_v1.snapshots import LOSS_FIELDS, STATE_FILES
from experiments.texture_material_disentangled_v1 import step_snapshots
from tests.methods.colorpeel_ice import test_sd15_snapshots as doubles
from tests.methods.colorpeel_ice import test_tm_step_snapshots as snapshot_tests


class UniformMaterialAuditTests(unittest.TestCase):
    def setUp(self):
        self.helper = snapshot_tests.StepSnapshotTests()
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.root = self.helper.root
        self.torch = self.helper.torch
        patcher = patch.object(audit, "sha256", side_effect=lambda path:
                               doubles.sha256(self.helper.physical(path)))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.dataset = {"status": "complete", "protocol_sha256": data.protocol_sha256()}
        patcher = patch.object(data, "verify_staging", return_value=self.dataset)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def fixture(self):
        source = self.root / "source"
        staging = source / "staging"
        self.write(staging / "audit.json", self.dataset)
        revision = {"branch": "codex/fixture", "commit": "training_revision"}
        self.write(source / "status.json", {"status": "execution_verified", "git": revision,
                   "completed_training_runs": 4, "completed_steps": 4000,
                   "protocol_sha256": data.protocol_sha256(), "dataset": self.dataset})
        selections = []
        for recipe in run.recipes():
            item = self.helper.fixture(recipe["method"], "material",
                name=f"source/training/{recipe['method']}__{recipe['concept_id']}")
            directory, checkpoint = Path(item["run"]), Path(item["checkpoint"])
            (directory / "config.yaml").write_text(yaml.safe_dump(run.build_config(recipe, staging)), encoding="utf-8")
            self.write(directory / "manifest.json", {"status": "succeeded", "returncode": 0, "git": revision})
            self.write(checkpoint / "embedding_update_audit.json", {
                "observed_optimization_steps": 1000, "modifier_tokens": [{"token": "<M*>", "token_id": 2,
                "exposure_steps": 1000, "nonzero_gradient_steps": 1000, "initial_final_l2_delta": 0.1}],
                "non_modifier_embedding_drift": {"changed_rows": 0}})
            metrics = checkpoint / "training_metrics.jsonl"
            metrics.write_text("".join(json.dumps({"step": step, **dict.fromkeys(LOSS_FIELDS, 0.5),
                "losses_finite": {"all": True}, "present_modifier_tokens": ["<M*>"]}) + "\n"
                for step in range(1, 1001)), encoding="utf-8")
            # The generic fixture also has a300-step checkpoint, absent in this revision.
            extra = checkpoint / "checkpoint-300"
            for path in extra.iterdir():
                path.unlink()
            extra.rmdir()
            for step in run.STEPS:
                for name in STATE_FILES[2:]:
                    (checkpoint / f"checkpoint-{step}" / name).write_bytes(b"training state")
            item.update(recipe)
            item["hashes"] = {path: audit.sha256(Path(path)) for path in item["hashes"]}
            item["hashes"][str(metrics)] = audit.sha256(metrics)
            selections.append(item)
        index = {"git": revision, "study": "material_uniform_albedo_v1",
                 "protocol_sha256": data.protocol_sha256(), "dataset": self.dataset, "runs": selections}
        self.write(source / "training/training_index.json", index)
        for step in run.STEPS:
            derived = [step_snapshots.export_step(item, step, source / f"step{step}/checkpoints/"
                f"{item['method']}__{item['concept_id']}") for item in selections]
            self.write(source / f"step{step}/training_index.json", {"git": revision,
                "study": "material_uniform_albedo_v1", "protocol_sha256": data.protocol_sha256(),
                "training_steps": step, "source_training_steps": 1000,
                "dataset_audit_sha256": audit.sha256(staging / "audit.json"), "runs": derived})
        return source

    def test_all_four_trajectories_and_twelve_paired_exports_are_read_only(self):
        source = self.fixture()
        before = {str(p): doubles.sha256(p) for p in source.rglob("*") if p.is_file()}
        report = audit.audit_training(source)
        self.assertEqual(report["completed_steps"], 4000)
        self.assertEqual(report["paired_snapshots"], 12)
        self.assertEqual(len(report["runs"]), 4)
        self.assertTrue(all(r["metrics"]["token_present_every_step"] for r in report["runs"]))
        index, selections, locks = audit.verify_source(source, 750)
        self.assertEqual(index["training_git"]["commit"], "training_revision")
        self.assertEqual(len(selections), 4)
        self.assertIn(str(source / "status.json"), locks)
        self.assertEqual(before, {str(p): doubles.sha256(p) for p in source.rglob("*") if p.is_file()})

    def test_incomplete_training_and_changed_config_are_rejected(self):
        source = self.fixture()
        status_path = source / "status.json"
        status = json.loads(status_path.read_text())
        self.write(status_path, {**status, "status": "training"})
        with self.assertRaisesRegex(ValueError, "completed uniform"):
            audit.verify_source(source, 500)
        self.write(status_path, status)
        item = json.loads((source / "training/training_index.json").read_text())["runs"][0]
        path = Path(item["run"]) / "config.yaml"
        config = yaml.safe_load(path.read_text())
        config["args"]["seed"] = 43
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "training config"):
            audit.audit_training(source)

    def test_selected_step_membership_and_token_are_strict(self):
        source = self.fixture()
        path = source / "step500/training_index.json"
        original = json.loads(path.read_text())
        for change in ({"training_steps": 750}, {"runs": original["runs"][:-1]}):
            self.write(path, {**original, **change})
            with self.assertRaises(ValueError):
                audit.verify_source(source, 500)
        changed = json.loads(json.dumps(original))
        changed["runs"][0]["token"] = "<T*>"
        self.write(path, changed)
        with self.assertRaisesRegex(ValueError, "four verified Material"):
            audit.verify_source(source, 500)
        with self.assertRaisesRegex(ValueError, "existing500"):
            audit.verify_source(source, 600)

    def test_hash_drift_and_rehashed_but_unpaired_export_are_rejected(self):
        source = self.fixture()
        path = source / "step500/training_index.json"
        index = json.loads(path.read_text())
        item = index["runs"][0]
        token_path = Path(item["checkpoint"]) / "<M*>.bin"
        doubles.write_state({"<M*>": doubles.FakeTensor((768,), 2)}, token_path)
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            audit.verify_source(source, 500)
        item["hashes"][str(token_path)] = audit.sha256(token_path)
        self.write(Path(item["checkpoint"]) / "receipt.json", item)
        self.write(path, index)
        with self.assertRaisesRegex(ValueError, "differs from its paired"):
            audit.audit_training(source)


if __name__ == "__main__":
    unittest.main()
