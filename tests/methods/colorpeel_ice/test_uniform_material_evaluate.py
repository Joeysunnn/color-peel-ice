from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.material_uniform_albedo_v1 import evaluate
from experiments.texture_material_disentangled_v1 import material_probe as probe


class UniformMaterialEvaluationTests(unittest.TestCase):
    def coordinator_mocks(self, stack, source, old):
        revision = {"branch": "codex/texture-material-v1", "commit": "inference"}
        stack.enter_context(patch.object(evaluate, "git_info", return_value=revision))
        stack.enter_context(patch.object(evaluate, "resolve_run_dir", side_effect=lambda path: path.resolve()))
        events = []
        training = {"status": "verified", "completed_training_runs": 4,
                    "completed_steps": 4000, "paired_snapshots": 12}

        def audit(path):
            self.assertEqual(path, source)
            events.append("audit")
            return training

        audit_mock = stack.enter_context(patch.object(evaluate.audit, "audit_training", side_effect=audit))
        old.mkdir()
        locked = old / "preserved.json"
        locked.write_text("old results", encoding="utf-8")
        locks = {str(locked): probe.sha256(locked)}

        def validate(path):
            self.assertEqual(events[0], "audit")
            events.append(path.name)
            return None, None, None, locks

        validation = stack.enter_context(patch.object(evaluate.compare, "validate_probe", side_effect=validate))

        def run_step(actual_source, actual_old, directory, step, gpu):
            self.assertEqual(actual_source, source)
            self.assertEqual(actual_old, old)
            self.assertEqual(directory.name, f"step{step}")
            self.assertEqual(len(events), 4)
            return {"step": step, "gpu": gpu, "status": "succeeded"}

        generation = stack.enter_context(patch.object(evaluate, "run_step", side_effect=run_step))
        return audit_mock, validation, generation, locked, training

    def test_existing_output_is_rejected_before_audit(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary).resolve()
            output = root / "existing"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("preserve", encoding="utf-8")
            stack.enter_context(patch.object(evaluate, "git_info", return_value={"commit": "new"}))
            stack.enter_context(patch.object(evaluate, "resolve_run_dir", side_effect=lambda path: path))
            audit = stack.enter_context(patch.object(evaluate.audit, "audit_training"))
            generation = stack.enter_context(patch.object(evaluate, "run_step"))
            with self.assertRaisesRegex(ValueError, "new external directory"):
                evaluate.run(root / "source", root / "old", output, [1, 2, 3])
            audit.assert_not_called()
            generation.assert_not_called()
            self.assertEqual(list(output.iterdir()), [marker])
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_failed_training_audit_prevents_probe(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary).resolve()
            output = root / "new"
            audit, validation, generation, _, _ = self.coordinator_mocks(stack, root / "source", root / "old")
            audit.side_effect = ValueError("paired snapshot mismatch")
            with self.assertRaisesRegex(ValueError, "paired snapshot mismatch"):
                evaluate.run(root / "source", root / "old", output, [1, 2, 3])
            validation.assert_not_called()
            generation.assert_not_called()
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")
            self.assertFalse(status["retraining"])
            self.assertFalse((output / "index.html").exists())

    def test_audited_four_runs_and_twelve_snapshots_dispatch_three_steps(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary).resolve()
            output = root / "new"
            audit, validation, generation, locked, training = self.coordinator_mocks(
                stack, root / "source", root / "old")
            evaluate.run(root / "source", root / "old", output, [1, 2, 3])
            audit.assert_called_once_with(root / "source")
            self.assertEqual([call.args[0].name for call in validation.call_args_list],
                             ["step500", "step750", "step1000"])
            self.assertEqual({call.args[3:] for call in generation.call_args_list},
                             {(500, 1), (750, 2), (1000, 3)})
            self.assertEqual(json.loads((output / "training_audit.json").read_text(encoding="utf-8")), training)
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "execution_verified")
            self.assertEqual(status["generated_images"], 288)
            self.assertFalse(status["retraining"])
            self.assertEqual(len(status["steps"]), 3)
            self.assertEqual(locked.read_text(encoding="utf-8"), "old results")
            html = (output / "index.html").read_text(encoding="utf-8")
            for step in probe.STEPS:
                self.assertIn(f'step{step}/comparison/index.html', html)
                self.assertIn(f'step{step}/probe/index.html', html)

    def test_old_result_hash_drift_prevents_verified_status(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary).resolve()
            output = root / "new"
            _, _, generation, locked, _ = self.coordinator_mocks(stack, root / "source", root / "old")
            original = generation.side_effect

            def drift(*args):
                result = original(*args)
                if args[3] == 500:
                    locked.write_text("changed during generation", encoding="utf-8")
                return result

            generation.side_effect = drift
            with self.assertRaisesRegex(ValueError, "old results changed"):
                evaluate.run(root / "source", root / "old", output, [1, 2, 3])
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")
            self.assertFalse((output / "index.html").exists())

    def test_uniform_probe_uses_new_source_audit_and_preserves_three_arms(self):
        protocol = probe.data.read_protocol()
        rows = probe.build_rows(protocol, 750)[:3]
        selection = {"method": rows[0]["method"], "branch": "material", "concept_id": rows[0]["material"]}
        index = {"training_git": {"commit": "uniform_training"}, "git": {"commit": "uniform_export"},
                 "protocol_sha256": "uniform_protocol"}
        calls = []

        def generate(pipe, selected_rows, output, sampling, device, learned):
            calls.append((pipe, selected_rows, sampling, device, learned))
            with (output / "generation_status.jsonl").open("a", encoding="utf-8") as handle:
                for row in selected_rows:
                    path = output / row["image_path"]
                    Image.new("RGB", (2, 2)).save(path)
                    record = {**row, "status": "safety_filtered" if learned else "ok",
                              "image_sha256": probe.sha256(path), "material_adapter_loaded": learned}
                    handle.write(json.dumps(record) + "\n")

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            source, output = Path(temporary).resolve() / "source", Path(temporary).resolve() / "new"
            stack.enter_context(patch.object(probe, "resolve_run_dir", side_effect=lambda path: path))
            stack.enter_context(patch.object(probe, "git_info", return_value={"commit": "new_probe"}))
            stack.enter_context(patch.object(probe, "package_versions", return_value={}))
            stack.enter_context(patch.object(probe, "build_rows", return_value=rows))
            stack.enter_context(patch.object(probe, "load_pipeline", side_effect=lambda items, device: items))
            stack.enter_context(patch.object(probe, "generate", side_effect=generate))
            stack.enter_context(patch.object(probe, "review_bundle", return_value=[]))
            stack.enter_context(patch.dict(sys.modules, {"torch": SimpleNamespace(
                cuda=SimpleNamespace(empty_cache=lambda: None))}))
            verifier = stack.enter_context(patch.object(evaluate.audit, "verify_source",
                return_value=(index, [selection], {})))
            old_verifier = stack.enter_context(patch.object(probe, "verify_source"))
            old_inputs = stack.enter_context(patch.object(probe.infer, "verify_inputs"))
            probe.run(source, output, 750, "cuda:2", uniform_material=True)
            self.assertEqual(verifier.call_count, 2)
            self.assertTrue(all(call.args == (source, 750) for call in verifier.call_args_list))
            old_verifier.assert_not_called()
            old_inputs.assert_not_called()
            self.assertEqual([call[4] for call in calls], [True, False])
            self.assertEqual([row["condition"] for row in calls[0][1]], ["learned_material"])
            self.assertEqual([row["condition"] for row in calls[1][1]], ["base", "literal"])
            for _, _, sampling, device, _ in calls:
                self.assertEqual(sampling, {**protocol["inference"], "conditions": list(probe.CONDITIONS)})
                self.assertEqual(device, "cuda:2")
            metadata = probe.read_json(output / "provenance.json")
            self.assertEqual(metadata["status"], "succeeded")
            self.assertEqual(metadata["training_git"], index["training_git"])
            self.assertEqual(metadata["source_export_git"], index["git"])
            self.assertIsNone(metadata["source_inference_git"])
            self.assertEqual(metadata["source_kind"], "uniform_material")
            self.assertFalse(metadata["texture_adapter_loaded"])
            self.assertFalse(metadata["retraining"])
            summary = probe.read_json(output / "summary.json")
            self.assertEqual(summary["generated_images"], 3)
            self.assertEqual(summary["safety_filtered"], 1)
            self.assertEqual(summary["by_condition"], {name: 1 for name in probe.CONDITIONS})


if __name__ == "__main__":
    unittest.main()
