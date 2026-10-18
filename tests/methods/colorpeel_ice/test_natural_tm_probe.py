"""Matched natural-image probes preserve sampler and branch isolation."""

import contextlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from experiments.natural_texture_material_v1 import probe
from experiments.natural_texture_material_v1 import run_probe as runner


LITERALS = {
    "texture": {"mailbox": "fine speckled", "metal_spoon": "smooth", "wood_spoon": "vertical grain"},
    "material": {"mailbox": "painted metal", "metal_spoon": "polished metal", "wood_spoon": "wood"},
}


def protocol():
    return json.loads(probe.schema.PARENT.read_text(encoding="utf8"))


class NaturalProbeTests(unittest.TestCase):
    def test_three_arm_matrix_keeps_approved_literals_contexts_and_step_pairing(self):
        settings = protocol()
        rows = probe.build_rows(settings, 500)
        self.assertEqual(len(rows), 288)
        self.assertEqual(len({row["id"] for row in rows}), 288)
        groups, base_prompts = {}, {}
        for row in rows:
            context = next(item for item in settings["dataset"]["heldout_contexts"]
                           if item["id"] == row["context"])
            branch, condition, prompt = row["branch"], row["condition"], row["prompt"]
            expected = f"a photo of a {context['color']} {context['shape']}"
            if condition != "base":
                value = LITERALS[branch][row["source_id"]] if condition == "literal" else (
                    "<T*>" if branch == "texture" else "<M*>")
                expected += f" with {value} texture" if branch == "texture" else f" made of {value}"
            expected += f", {settings['dataset']['light_phrases'][context['light']]}"
            self.assertEqual(prompt, expected)
            forbidden = ("<M*>", "made of", "metal", "wood", "plastic", "polished") if branch == "texture" else (
                "<T*>", "texture", "speckled", "striped", "grain", "checkerboard")
            for cue in forbidden:
                self.assertNotIn(cue, prompt)
            self.assertEqual(row["concept_id"], row["source_id"] + "_" + branch)
            self.assertIn(row["seed"], (42, 43))
            key = tuple(row[field] for field in ("method", "branch", "source_id", "context", "seed"))
            groups.setdefault(key, []).append(row)
            if condition == "base":
                base_prompts.setdefault((row["context"], row["seed"]), set()).add(prompt)
        self.assertEqual(len(groups), 96)
        self.assertTrue(all(len(prompts) == 1 for prompts in base_prompts.values()))
        for group in groups.values():
            self.assertEqual({row["condition"] for row in group},
                             {"base", "literal", "learned_" + group[0]["branch"]})
            for field in ("shape", "color", "light", "seed"):
                self.assertEqual(len({row[field] for row in group}), 1)
        for step in (750, 1000):
            self.assertEqual([{**row, "training_steps": 500}
                              for row in probe.build_rows(settings, step)], rows)
        with self.assertRaises(ValueError):
            probe.build_rows(settings, 300)

    def mocked_run(self, root, mutate_source=False):
        settings = protocol()
        selections = [{"method": method, "branch": branch, "concept_id": source + "_" + branch,
                       "token": "<T*>" if branch == "texture" else "<M*>"}
                      for method in probe.METHODS for branch in ("texture", "material")
                      for source in probe.schema.SOURCES]
        source = root / "old_training"
        source.mkdir()
        (source / "status.json").write_text(json.dumps({"git": {"commit": "old-training"}}), encoding="utf8")
        marker = source / "preserved.txt"
        marker.write_text("old source", encoding="utf8")
        locks = {str(marker): probe.schema.sha256(marker)}
        references = {(item["method"], item["concept_id"]): {
            "branch": item["branch"], "images": [f"actual RGB {i}" for i in range(
                16 if item["branch"] == "texture" else 8)]} for item in selections}
        calls, loaded, generators = [], [], []
        mask = object()

        class Generator:
            def __init__(self, device):
                self.device = device
                generators.append(self)

            def manual_seed(self, seed):
                self.seed = seed
                return self

        def load_pipeline(selection, device):
            loaded.append(selection)

            def pipeline(prompt, **kwargs):
                calls.append((selection, prompt, kwargs))
                if mutate_source and len(calls) == 288:
                    marker.write_text("changed externally", encoding="utf8")
                return types.SimpleNamespace(images=[Image.new("RGB", (2, 2))],
                                             nsfw_content_detected=[False])
            return pipeline

        torch = types.SimpleNamespace(Generator=Generator, no_grad=contextlib.nullcontext,
                                      cuda=types.SimpleNamespace(empty_cache=Mock()))
        output = root / "new_probe"
        with (patch.dict(sys.modules, {"torch": torch}),
              patch.object(probe, "git_info", return_value={"commit": "new-inference"}),
              patch.object(probe, "resolve_run_dir", side_effect=lambda path: path),
              patch.object(probe, "verify_source", return_value=(settings, selections, locks, references)) as verify,
              patch.object(probe, "package_versions", return_value={}),
              patch.object(probe, "load_pipeline", side_effect=load_pipeline),
              patch.object(probe, "attention_masks", return_value={"modifier_token_mask": mask}) as masks,
              patch.object(probe, "review_bundle", return_value=["review.png"]) as review):
            if mutate_source:
                with self.assertRaisesRegex(ValueError, "Source checkpoint, actual training images"):
                    probe.run(source, output, 500, "cuda:2")
                review.assert_not_called()
            else:
                probe.run(source, output, 500, "cuda:2")
                self.assertEqual(review.call_args.args[3], references)
                self.assertEqual(marker.read_text(encoding="utf8"), "old source")
            self.assertEqual(verify.call_count, 2)
        return output, selections, calls, loaded, generators, masks, mask

    def test_run_isolates_twelve_single_branch_adapters_and_keeps_sampler(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, selections, calls, loaded, generators, masks, mask = self.mocked_run(Path(temporary))
            self.assertEqual(loaded.count([]), 1)
            self.assertEqual([item[0] for item in loaded if item], selections)
            self.assertEqual(len(calls), 288)
            self.assertEqual(len(generators), 288)
            self.assertEqual(len({id(generator) for generator in generators}), 288)
            self.assertEqual(masks.call_count, 96)
            for selected, prompt, kwargs in calls:
                self.assertEqual(kwargs["num_inference_steps"], 100)
                self.assertEqual(kwargs["guidance_scale"], 3.5)
                self.assertEqual((kwargs["width"], kwargs["height"]), (512, 512))
                self.assertEqual(kwargs["generator"].device, "cuda:2")
                self.assertIn(kwargs["generator"].seed, (42, 43))
                self.assertEqual(bool(selected), "<T*>" in prompt or "<M*>" in prompt)
                if selected:
                    self.assertEqual(len(selected), 1)
                    self.assertIn(selected[0]["token"], prompt)
                    self.assertIs(kwargs["cross_attention_kwargs"]["modifier_token_mask"], mask)
                else:
                    self.assertNotIn("cross_attention_kwargs", kwargs)
            for call in masks.call_args_list:
                expected_token = "<T*>" if "<T*>" in call.args[1] else "<M*>"
                self.assertEqual(call.args[2:], ([expected_token], 3.5))
            ledger = [json.loads(line) for line in (output / "generation_status.jsonl").read_text().splitlines()]
            for row in ledger:
                learned = row["condition"].startswith("learned_")
                self.assertEqual(row["texture_adapter_loaded"], learned and row["branch"] == "texture")
                self.assertEqual(row["material_adapter_loaded"], learned and row["branch"] == "material")
            summary = json.loads((output / "summary.json").read_text(encoding="utf8"))
            self.assertEqual(summary["by_condition"], {"base": 96, "literal": 96,
                                                      "learned_texture": 48, "learned_material": 48})
            self.assertEqual(summary["by_method"], {method: 144 for method in probe.METHODS})
            self.assertEqual(summary["by_branch"], {"texture": 144, "material": 144})
            self.assertEqual(summary["human_ratings"], "pending")
            self.assertIsNone(summary["disentanglement_score"])
            self.assertEqual(summary["contact_sheets"], ["review.png"])
            provenance = json.loads((output / "provenance.json").read_text(encoding="utf8"))
            self.assertEqual(provenance["status"], "succeeded")
            self.assertEqual(provenance["git"], {"commit": "new-inference"})
            self.assertEqual(provenance["training_git"], {"commit": "old-training"})
            self.assertEqual(provenance["sampling"]["scheduler"], "PNDM")
            self.assertEqual(provenance["sampling"]["dtype"], "float16")
            self.assertTrue(provenance["sampling"]["safety_checker"])
            self.assertFalse(provenance["retraining"])

    def test_source_change_fails_before_publishing_review_or_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, *_ = self.mocked_run(Path(temporary), mutate_source=True)
            self.assertFalse((output / "summary.json").exists())
            provenance = json.loads((output / "provenance.json").read_text(encoding="utf8"))
            self.assertEqual(provenance["status"], "failed")

    def test_existing_output_and_training_descendants_are_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "existing"
            existing.mkdir()
            marker = existing / "old.txt"
            marker.write_text("preserve", encoding="utf8")
            with (patch.object(probe, "resolve_run_dir", side_effect=lambda path: path),
                  patch.object(probe, "verify_source") as verify,
                  patch.object(probe, "load_pipeline") as load):
                for output in (existing, root / "source" / "nested_new"):
                    with self.assertRaisesRegex(ValueError, "new external probe directory"):
                        probe.run(root / "source", output, 500)
            verify.assert_not_called()
            load.assert_not_called()
            self.assertEqual(marker.read_text(encoding="utf8"), "preserve")
            self.assertEqual(list(existing.iterdir()), [marker])
            self.assertFalse((root / "source").exists())


class NaturalProbeBatchTests(unittest.TestCase):
    def mocked_batch(self, root, failure=None):
        source = root / "old_training"
        source.mkdir()
        marker = source / "training_receipt.json"
        marker.write_text('{"preserved": true}', encoding="utf8")
        old_source = {str(path): path.read_bytes() for path in source.rglob("*") if path.is_file()}
        git = {"commit": "new-inference", "branch": "codex/texture-material-v1"}
        audit = {"training_git": {"commit": "old-training", "branch": git["branch"]},
                 "completed_training_runs": 12, "completed_steps": 12000, "paired_exports": 36,
                 "source_locks": {str(marker): probe.schema.sha256(marker)},
                 "runs": [{"concept_id": name + "_" + branch, "method": method, "steps": 1000,
                           "metrics": {"total_loss": {"first": 0.3, "last": 0.1}}}
                          for method in probe.METHODS for branch in ("texture", "material")
                          for name in probe.schema.SOURCES]}
        output = root / "new_batch"
        calls = []

        def popen(command, **kwargs):
            calls.append((command, kwargs))
            folder = Path(command[command.index("--output") + 1])
            step = int(command[command.index("--step") + 1])
            folder.mkdir()
            sheets = [f"contact_sheets/group_{i}.png" for i in range(
                23 if failure == "missing_sheet" and step == 750 else 24)]
            (folder / "contact_sheets").mkdir()
            for sheet in sheets:
                (folder / sheet).write_bytes(b"verified child sheet")
            (folder / "summary.json").write_text(json.dumps({
                "status": "succeeded", "training_steps": step, "generated_images": 288, "missing_images": 0,
                "contact_sheets": sheets}), encoding="utf8")

            def wait():
                self.assertEqual(len(calls), 3, "Launch all three GPUs before waiting")
                return 1 if failure == "failed_child" and step == 750 else 0

            return types.SimpleNamespace(pid=100 + step, wait=wait)

        git_calls = 0

        def read_git():
            nonlocal git_calls
            git_calls += 1
            return {**git, "branch": "codex/other"} if failure == "branch_drift" and git_calls == 5 else git

        with (patch.object(runner, "resolve_run_dir", side_effect=lambda path: path),
              patch.object(runner, "git_info", side_effect=read_git),
              patch.object(runner, "audit_training", return_value=audit) as training_audit,
              patch.object(runner.subprocess, "Popen", side_effect=popen)):
            if failure:
                with self.assertRaises((RuntimeError, ValueError)):
                    runner.run(source, output, [1, 2, 3])
            else:
                state = runner.run(source, output, [1, 2, 3])
                self.assertEqual(state["status"], "execution_verified")
                self.assertEqual(state["generated_images"], 864)
            training_audit.assert_called_once_with(source)
        self.assertEqual({str(path): path.read_bytes() for path in source.rglob("*") if path.is_file()},
                         old_source)
        return output, source, calls, audit

    def test_batch_audits_once_and_passes_same_serializable_cache_to_three_gpus(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, source, calls, audit = self.mocked_batch(Path(temporary))
            cache = output / "training_audit.json"
            self.assertEqual(json.loads(cache.read_text(encoding="utf8")), audit)
            self.assertEqual(len(calls), 3)
            for (command, kwargs), step, gpu in zip(calls, (500, 750, 1000), (1, 2, 3)):
                self.assertEqual(command[command.index("--step") + 1], str(step))
                self.assertEqual(command[command.index("--output") + 1], str(output / f"step{step}"))
                self.assertEqual(command[command.index("--source-run") + 1], str(source))
                self.assertEqual(command[command.index("--training-audit") + 1], str(cache))
                self.assertEqual(command[command.index("--device") + 1], "cuda:0")
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], str(gpu))
                self.assertEqual(kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
                self.assertEqual(kwargs["cwd"], runner.ROOT)
                self.assertEqual(kwargs["stdin"], runner.subprocess.DEVNULL)
            state = json.loads((output / "status.json").read_text(encoding="utf8"))
            self.assertEqual(state["git"]["commit"], "new-inference")
            self.assertEqual(state["training_git"]["commit"], "old-training")
            self.assertEqual(state["verified_paired_exports"], 36)
            self.assertTrue(all(len(job["summary"]["contact_sheets"]) == 24 for job in state["steps"]))

    def test_failed_child_or_missing_sheet_never_reports_execution_verified(self):
        for failure in ("failed_child", "missing_sheet"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                output, *_ = self.mocked_batch(Path(temporary), failure=failure)
                state = json.loads((output / "status.json").read_text(encoding="utf8"))
                self.assertEqual(state["status"], "failed")
                self.assertNotIn("generated_images", state)
                self.assertEqual(state["steps"][1]["status"], "failed")
                self.assertEqual(json.loads((output / "repro_outputs/status.json").read_text()), state)

    def test_checkout_branch_drift_fails_final_batch_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, *_ = self.mocked_batch(Path(temporary), failure="branch_drift")
            state = json.loads((output / "status.json").read_text(encoding="utf8"))
            self.assertEqual(state["status"], "failed")
            self.assertIn("checkout changed", state["error"])
            self.assertNotIn("generated_images", state)


if __name__ == "__main__":
    unittest.main()
