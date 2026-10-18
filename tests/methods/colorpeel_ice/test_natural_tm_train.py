from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from experiments.natural_texture_material_v1 import train
from experiments.material_uniform_albedo_v1 import run as uniform_material


class NaturalTrainingOrchestrationTests(unittest.TestCase):
    def test_natural_material_preserves_current_uniform_material_training_args(self):
        protocol = json.loads(train.schema.PARENT.read_text(encoding="utf8"))
        staging = Path("natural_staging").resolve()
        for recipe in train.recipes():
            if recipe["branch"] != "material":
                continue
            baseline_recipe = next(item for item in uniform_material.recipes()
                if item["method"] == recipe["method"] and item["concept_id"] == "polished_metal")
            baseline = uniform_material.build_config(baseline_recipe, staging)
            actual = train.build_config(recipe, staging, protocol)
            expected_args = {**baseline["args"], "concepts_list": str(staging / recipe["concept_id"] / "concepts.json")}
            self.assertEqual(actual["args"], expected_args)
            self.assertEqual(actual["environment"], baseline["environment"])

    @contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            source = Path(temporary).resolve() / "accepted_dataset"
            source.mkdir()
            (source / "staging").mkdir()
            revision = {"commit": "a" * 40, "branch": "codex/texture-material-v1"}
            (source / "status.json").write_text(json.dumps({"git": revision}), encoding="utf8")
            references = {}
            for recipe in train.recipes():
                concept = recipe["concept_id"]
                references[concept] = {"source_id": recipe["source_id"], "branch": recipe["branch"],
                    "source": "natural.jpg", "image_count": 16 if recipe["branch"] == "texture" else 8,
                    "training_assets_manifest": str(source / "staging" / concept / "training_assets_manifest.jsonl"),
                    "images": [{"image": str(source / "staging" / concept / "images/actual_rgb.png"),
                                "image_sha256": "immutable-rgb", "prompt": "actual training caption"}]}
            (source / "staging/training_references.json").write_text(json.dumps(references), encoding="utf8")
            locks = {str(path): train.schema.sha256(path) for path in source.rglob("*") if path.is_file()}
            protocol = json.loads(train.schema.PARENT.read_text(encoding="utf8"))
            protocol.update(study="natural_texture_material_v1", phase=2)
            output = Path(temporary).resolve() / "new_training"
            verify = stack.enter_context(patch.object(train, "verify_dataset",
                return_value=(protocol, {"train_images": 72}, locks)))
            git = stack.enter_context(patch.object(train, "git_info", return_value=revision))
            stack.enter_context(patch.object(train, "resolve_run_dir", side_effect=lambda path: path.resolve()))
            stack.enter_context(patch.object(train.shared, "verify_model", return_value={"frozen_model": True}))
            stack.enter_context(patch.object(train.shutil, "disk_usage", return_value=SimpleNamespace(free=80 << 30)))
            events, submitted, exports = [], [], []

            def dry_run(command, **kwargs):
                self.assertEqual(command[-1], "--dry-run")
                config = Path(command[command.index("--config") + 1])
                directory = Path(command[command.index("--run-dir") + 1])
                self.assertEqual(directory.parent, output / "preflight")
                self.assertFalse(directory.exists())
                directory.mkdir()
                shutil.copyfile(config, directory / "config.yaml")
                (directory / "manifest.json").write_text(json.dumps({"status": "dry-run", "git": revision}))
                events.append(("dry-run", str(directory)))
                return SimpleNamespace(returncode=0)

            def queue(gpu, jobs, batch):
                self.assertEqual(len([item for item in events if item[0] == "dry-run"]), 12)
                self.assertEqual(len(jobs), 4)
                events.append(("queue", gpu))
                submitted.extend(jobs)
                return [{"status": "verified", "audit": {**job["recipe"], "steps": 1000,
                    "run": job["run"], "final_snapshot_matches_export": True}} for job in jobs]

            def export(selection, step, destination):
                self.assertEqual(len(submitted), 12)
                self.assertEqual(destination.parts[-3], f"step{step}")
                destination.mkdir(parents=True)
                exports.append((selection["method"], selection["concept_id"], step))
                return {**selection, "steps": step, "training_steps": 1000}

            command = stack.enter_context(patch.object(train.subprocess, "run", side_effect=dry_run))
            queue_mock = stack.enter_context(patch.object(train.shared, "run_queue", side_effect=queue))
            export_mock = stack.enter_context(patch.object(train, "export_step", side_effect=export))
            yield SimpleNamespace(source=source, output=output, revision=revision, locks=locks,
                references=references, verify=verify, git=git, command=command, queue=queue_mock,
                export=export_mock, submitted=submitted, exports=exports, events=events)

    def test_twelve_preflights_precede_three_queues_and_thirty_six_paired_exports(self):
        with self.fixture() as f:
            state = train.run(f.source, f.output, [1, 2, 3])
            self.assertEqual(state["status"], "execution_verified")
            self.assertEqual((state["completed_training_runs"], state["completed_steps"]), (12, 12000))
            self.assertFalse(state["resume"])
            self.assertFalse(state["inference_launched"])
            self.assertEqual(f.command.call_count, 12)
            self.assertEqual({call.args[0] for call in f.queue.call_args_list}, {1, 2, 3})
            self.assertEqual(len({event[1] for event in f.events[:12]}), 12)
            self.assertEqual({(job["recipe"]["concept_id"], job["recipe"]["method"])
                for job in f.submitted}, {(recipe["concept_id"], recipe["method"]) for recipe in train.recipes()})
            for job in f.submitted:
                self.assertEqual(Path(job["run"]).parent, f.output / "training")
                self.assertNotIn("--dry-run", job["command"])
                config_path = Path(job["command"][job["command"].index("--config") + 1])
                self.assertEqual(train.read_config(config_path), job["config"])
                self.assertEqual(job["config"]["args"]["checkpoint_steps"], [500, 750, 1000])
            self.assertEqual(len(f.exports), 36)
            for step in (500, 750, 1000):
                index = json.loads((f.output / f"step{step}/training_index.json").read_text())
                self.assertEqual(index["training_steps"], step)
                self.assertEqual(index["source_training_steps"], 1000)
                self.assertEqual(len(index["runs"]), 12)
                self.assertEqual({row["steps"] for row in index["runs"]}, {step})
                self.assertFalse((f.output / f"step{step}/inference").exists())
            refs = json.loads((f.output / "training_references.json").read_text())
            self.assertEqual(refs["groups"], f.references)
            self.assertTrue(refs["same_dataset_for_original_and_rank64"])
            self.assertEqual(f.verify.call_count, 2)
            self.assertTrue(all(train.schema.sha256(path) == digest for path, digest in f.locks.items()))

    def test_failed_dry_run_prevents_every_actual_training_and_export(self):
        with self.fixture() as f:
            original = f.command.side_effect
            def fail_second(command, **kwargs):
                result = original(command, **kwargs)
                if len(f.events) == 2:
                    result.returncode = 7
                return result
            f.command.side_effect = fail_second
            with self.assertRaisesRegex(RuntimeError, "dry-run failed"):
                train.run(f.source, f.output, [1, 2, 3])
            f.queue.assert_not_called()
            f.export.assert_not_called()
            self.assertEqual(f.command.call_count, 2)
            for path in (f.output / "status.json", f.output / "train_outputs/status.json"):
                state = json.loads(path.read_text())
                self.assertEqual(state["status"], "failed")
                self.assertEqual(state["completed_training_runs"], 0)

    def test_dataset_change_after_preflight_prevents_training(self):
        with self.fixture() as f:
            original = f.command.side_effect
            def mutate_source(command, **kwargs):
                result = original(command, **kwargs)
                if len(f.events) == 12:
                    (f.source / "status.json").write_text("changed")
                return result
            f.command.side_effect = mutate_source
            with self.assertRaisesRegex(ValueError, "dataset changed"):
                train.run(f.source, f.output, [1, 2, 3])
            f.queue.assert_not_called()
            f.export.assert_not_called()

    def test_failed_training_audit_prevents_all_exports(self):
        with self.fixture() as f:
            f.queue.side_effect = lambda gpu, jobs, batch: [{"status": "failed"}]
            with self.assertRaisesRegex(RuntimeError, "checkpoint audit failed"):
                train.run(f.source, f.output, [1, 2, 3])
            f.export.assert_not_called()
            self.assertEqual(json.loads((f.output / "status.json").read_text())["status"], "failed")

    def test_existing_output_and_repeated_gpus_are_rejected_before_source_reads(self):
        with self.fixture() as f:
            f.output.mkdir()
            marker = f.output / "preserve.txt"
            marker.write_text("old result")
            for output, gpus in ((f.output, [1, 2, 3]), (f.output.parent / "new", [1, 1]),
                                 (f.output.parent / "new", []), (f.output.parent / "new", [-1])):
                with self.subTest(gpus=gpus), self.assertRaisesRegex(ValueError, "new external"):
                    train.run(f.source, output, gpus)
            f.verify.assert_not_called()
            f.command.assert_not_called()
            self.assertEqual(marker.read_text(), "old result")


if __name__ == "__main__":
    unittest.main()
