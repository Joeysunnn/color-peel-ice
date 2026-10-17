import copy
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from experiments.material_uniform_albedo_v1 import run
from experiments.texture_material_disentangled_v1 import step_ablation, train


class UniformMaterialTrainingTests(unittest.TestCase):
    def test_exact_four_material_recipes_and_frozen_training_arguments(self):
        recipes = run.recipes()
        self.assertEqual(len(recipes), 4)
        self.assertEqual({(recipe["method"], recipe["concept_id"]) for recipe in recipes}, {
            (method, concept) for method in ("fullrank", "lora_r64")
            for concept in ("polished_metal", "rough_dielectric")})
        self.assertEqual(run.STEPS, (500, 750, 1000))
        for recipe in recipes:
            self.assertEqual((recipe["branch"], recipe["token"]), ("material", "<M*>"))
            baseline = step_ablation.build_config(recipe, Path("fixed_staging"))
            expected = copy.deepcopy(baseline)
            expected["run"].update(study="material_uniform_albedo_v1",
                variant=f"{recipe['concept_id']}_{recipe['method']}_uniform_steps1000")
            expected["texture_material"]["protocol_sha256"] = run.data.protocol_sha256()
            expected["args"]["checkpoint_steps"] = [500, 750, 1000]
            self.assertEqual(run.build_config(recipe, Path("fixed_staging")), expected)
            self.assertEqual(expected["args"]["max_train_steps"], 1000)
            self.assertNotIn("resume_from_checkpoint", expected["args"])

    def test_texture_recipe_cannot_be_trained(self):
        texture = next(recipe for recipe in train.recipes() if recipe["branch"] == "texture")
        with self.assertRaisesRegex(ValueError, "only the four Material"):
            run.build_config(texture, Path("fixed_staging"))

    def test_existing_output_is_rejected_without_touching_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            marker = output / "old.txt"
            marker.write_text("preserve", encoding="utf8")
            with patch.object(run, "git_info", return_value={"commit": "a" * 40}), \
                 patch.object(run, "resolve_run_dir", return_value=output), \
                 patch.object(run.train, "verify_model") as verify_model, \
                 patch.object(run.subprocess, "run") as command:
                with self.assertRaisesRegex(ValueError, "new output directory"):
                    run.run(output, 1, [1, 2])
                verify_model.assert_not_called()
                command.assert_not_called()
            self.assertEqual(list(output.iterdir()), [marker])
            self.assertEqual(marker.read_text(encoding="utf8"), "preserve")

    def test_failed_stage_never_starts_training_or_exports(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            output = Path(temporary).resolve() / "new"
            stack.enter_context(patch.object(run, "git_info", return_value={"commit": "a" * 40}))
            stack.enter_context(patch.object(run, "resolve_run_dir", side_effect=lambda path: path.resolve()))
            stack.enter_context(patch.object(run.train, "verify_model", return_value={"verified": True}))
            stack.enter_context(patch.object(run.shutil, "disk_usage", return_value=SimpleNamespace(free=40 << 30)))
            command = stack.enter_context(patch.object(run.subprocess, "run",
                side_effect=[SimpleNamespace(returncode=0), SimpleNamespace(returncode=7)]))
            queue = stack.enter_context(patch.object(run.train, "run_queue"))
            export = stack.enter_context(patch.object(run, "export_step"))
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                run.run(output, 1, [1, 2])
            queue.assert_not_called()
            export.assert_not_called()
            self.assertEqual(command.call_count, 2)
            state = json.loads((output / "status.json").read_text(encoding="utf8"))
            self.assertEqual(state["status"], "failed")
            self.assertEqual([item["stage"] for item in state["steps"]], ["plan", "render"])
            self.assertEqual(state["steps"][-1]["returncode"], 7)

    def test_completed_four_runs_export_each_paired_step_without_inference(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            output = Path(temporary).resolve() / "new"
            revision = {"commit": "a" * 40, "branch": "codex/material-uniform"}
            stack.enter_context(patch.object(run, "git_info", return_value=revision))
            stack.enter_context(patch.object(run.train, "git_info", return_value=revision))
            stack.enter_context(patch.object(run, "resolve_run_dir", side_effect=lambda path: path.resolve()))
            stack.enter_context(patch.object(run.train, "verify_model", return_value={"verified": True}))
            stack.enter_context(patch.object(run.shutil, "disk_usage", return_value=SimpleNamespace(free=40 << 30)))
            command = stack.enter_context(patch.object(run.subprocess, "run", return_value=SimpleNamespace(returncode=0)))
            audit = {"status": "verified", "train_images": 16}
            stack.enter_context(patch.object(run.data, "verify_staging", return_value=audit))
            stack.enter_context(patch.object(run, "sha256", return_value="audit-sha"))
            submitted = []

            def complete_queue(gpu, jobs, batch):
                submitted.extend(jobs)
                return [{"status": "verified", "audit": {**job["recipe"], "steps": 1000}}
                        for job in jobs]

            def export(selection, step, destination):
                destination.mkdir(parents=True)
                return {**selection, "steps": step, "training_steps": 1000}

            stack.enter_context(patch.object(run.train, "run_queue", side_effect=complete_queue))
            export_mock = stack.enter_context(patch.object(run, "export_step", side_effect=export))
            run.run(output, 1, [1, 2])
            self.assertEqual(command.call_count, 3)
            self.assertEqual(len(submitted), 4)
            self.assertEqual({job["recipe"]["branch"] for job in submitted}, {"material"})
            self.assertEqual(export_mock.call_count, 12)
            self.assertEqual([call.args[1] for call in export_mock.call_args_list],
                             [500] * 4 + [750] * 4 + [1000] * 4)
            for step in run.STEPS:
                index = json.loads((output / f"step{step}/training_index.json").read_text(encoding="utf8"))
                self.assertEqual(index["training_steps"], step)
                self.assertEqual(index["source_training_steps"], 1000)
                self.assertEqual(len(index["runs"]), 4)
                self.assertEqual({item["steps"] for item in index["runs"]}, {step})
                self.assertFalse((output / f"step{step}/inference").exists())
            state = json.loads((output / "status.json").read_text(encoding="utf8"))
            self.assertEqual(state["status"], "execution_verified")
            self.assertEqual(state["completed_steps"], 4000)
            self.assertEqual(state["completed_training_runs"], 4)
            training_index = json.loads((output / "training/training_index.json").read_text(encoding="utf8"))
            self.assertEqual(training_index["texture_branch"], "unchanged")


if __name__ == "__main__":
    unittest.main()
