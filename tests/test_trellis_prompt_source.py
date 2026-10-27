"""CPU checks for frozen-source audits and cross-revision prompt inference."""

import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from experiments.subject_marble_color30_v1 import inference as common
from experiments.subject_trellis_dataset_v1 import prompt_forms, run_prompt_forms as runner
from experiments.sd15_token_local_kv_v1.contract import sha256


class TrellisPromptSourceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="trellis-prompt-source-", dir=os.environ.get("TMP"))
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source, self.checkout, self.output = (self.root / name for name in ("source", "checkout", "output"))
        for path in (self.source, self.checkout, self.output):
            path.mkdir()
        source_patch = patch.multiple(runner, SOURCE=self.source, SOURCE_CHECKOUT=self.checkout)
        source_patch.start()
        self.addCleanup(source_patch.stop)
        self.dataset = self.source / "training_dataset"
        self.dataset.mkdir()
        self.rgb, self.mask = self.root / "actual_rgb.png", self.root / "actual_mask.png"
        Image.new("RGB", (512, 512), (50, 140, 170)).save(self.rgb)
        Image.new("L", (512, 512), 255).save(self.mask)
        refs = [{"id": f"actual_{i:03d}", "kind": "synthetic_material_counterfactual",
                 "image": str(self.rgb), "mask": str(self.mask),
                 "image_sha256": sha256(self.rgb), "mask_sha256": sha256(self.mask),
                 "caption": f"a photo of <S*> mailbox in blue color with a rough metallic surface {i}"}
                for i in range(72)]
        (self.dataset / "training_assets_manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in refs), encoding="utf8")
        for name, value in (("concepts.json", []), ("source_locks.json", {}), ("approval.json", {}),
                            ("dataset_audit.json", {"rows": 72, "source_mode": "synthetic_native_geometry",
                                                   "selected_colors": ["green", "blue"]})):
            runner.write(self.dataset / name, value)
        self.runs = []
        for method in prompt_forms.METHODS:
            run = self.source / "training" / method / "runs" / "completed"
            checkpoint = run / "checkpoints"
            checkpoint.mkdir(parents=True)
            final = checkpoint / "final.bin"
            final.write_bytes((method + " final").encode())
            selection = {"method": method, "run": str(run), "checkpoint": str(checkpoint),
                         "status": "verified", "token": "<S*>", "branch": "subject", "steps": 1000,
                         "hashes": {str(final): sha256(final)}}
            self.runs.append(selection)
            for step in prompt_forms.STEPS:
                folder = checkpoint / f"checkpoint-{step}"
                folder.mkdir()
                for name in runner.snapshots.STATE_FILES:
                    (folder / name).write_bytes(f"{method}:{step}:{name}".encode())
                pair = {str(folder / name): sha256(folder / name)
                        for name in ("pytorch_model.bin", "pytorch_model_1.bin")}
                receipt = self.source / f"step{step}" / "checkpoints" / method / "receipt.json"
                receipt.parent.mkdir(parents=True)
                runner.write(receipt, {"method": method, "step": step, "steps": step, "training_steps": 1000,
                    "source_checkpoint": str(checkpoint), "source_snapshot_hashes": pair,
                    "source_hashes": selection["hashes"], "paired_embedding_finite": True,
                    "source_adapter_sha256": pair[str(folder / "pytorch_model.bin")],
                    "source_text_encoder_sha256": pair[str(folder / "pytorch_model_1.bin")]})
        runner.write(self.source / "training_index.json", {"git": runner.TRAINING_GIT, "runs": self.runs})
        runner.write(self.source / "status.json", {"status": "execution_verified", "git": runner.TRAINING_GIT,
            "completed_steps": {method: 1000 for method in prompt_forms.METHODS},
            "training_images": 72, "inference_steps": [500, 750, 1000]})
        for step in prompt_forms.STEPS:
            runner.write(self.source / f"step{step}" / "provenance.json", {
                "status": "succeeded", "git": runner.TRAINING_GIT, "step": step,
                "generated_images": 36, "training": self.runs})

    def hashes(self):
        return {str(path): sha256(path) for path in self.source.rglob("*") if path.is_file()}

    def test_source_revision_mismatch_is_rejected(self):
        with patch.object(runner.subprocess, "check_output", side_effect=["", runner.TRAINING_GIT["branch"], "wrong-commit"]):
            with self.assertRaisesRegex(ValueError, "revision changed"):
                runner.source_revision()

    def test_audit_script_rejects_imports_outside_frozen_checkout(self):
        with patch.object(Path, "cwd", return_value=self.checkout), \
             patch.object(runner.training, "verify_dataset") as verify, \
             self.assertRaisesRegex(ValueError, "imports must come from"):
            exec(runner.AUDIT_SCRIPT, {"__name__": "__main__"})
        verify.assert_not_called()

    def test_changed_500_adapter_is_rejected_against_historical_receipt(self):
        before = (self.source / "training_index.json").read_bytes()
        self.assertEqual(len(runner.recorded_snapshot_locks(self.runs[0])), 9)
        changed = Path(self.runs[0]["checkpoint"]) / "checkpoint-500" / "pytorch_model.bin"
        changed.write_bytes(b"changed after original paired export")
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            runner.recorded_snapshot_locks(self.runs[0])
        self.assertEqual((self.source / "training_index.json").read_bytes(), before)

    def test_nonzero_original_audit_fails_with_clean_environment_and_no_relabeling(self):
        before = self.hashes()
        result = types.SimpleNamespace(returncode=7, stdout="not valid JSON", stderr="original strict audit failed")
        overrides = {key: "untrusted inherited override" for key in (
            "PYTHONPATH", "PYTHONHOME", "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR")}
        with patch.object(runner, "source_revision", return_value=runner.TRAINING_GIT), \
             patch.dict(os.environ, overrides), patch.object(runner.subprocess, "run", return_value=result) as child:
            with self.assertRaisesRegex(RuntimeError, "strict original-revision"):
                runner.source_audit(self.output)
        command, = child.call_args.args
        self.assertEqual(command[:3], [sys.executable, "-B", "-c"])
        self.assertEqual(command[-1], str(self.source))
        self.assertEqual(child.call_args.kwargs["cwd"], self.checkout)
        self.assertFalse(set(overrides) & set(child.call_args.kwargs["env"]))
        self.assertEqual(child.call_args.kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(self.hashes(), before)
        self.assertFalse((self.output / "source_training_audit.json").exists())

    def test_successful_source_audit_locks_original_receipts_and_all_states(self):
        before = self.hashes()
        result = types.SimpleNamespace(returncode=0, stdout=json.dumps({"git": runner.TRAINING_GIT, "runs": self.runs}), stderr="")
        with patch.object(runner, "source_revision", return_value=runner.TRAINING_GIT), \
             patch.object(runner.subprocess, "run", return_value=result):
            runs, locks = runner.source_audit(self.output)
        self.assertEqual(runs, self.runs)
        for selection in self.runs:
            for step in prompt_forms.STEPS:
                receipt = self.source / f"step{step}" / "checkpoints" / selection["method"] / "receipt.json"
                self.assertEqual(locks[str(receipt)], sha256(receipt))
                for name in runner.snapshots.STATE_FILES:
                    path = Path(selection["checkpoint"]) / f"checkpoint-{step}" / name
                    self.assertEqual(locks[str(path)], sha256(path))
        self.assertEqual(self.hashes(), before)
        saved = json.loads((self.output / "source_training_audit.json").read_text())
        self.assertEqual(saved["git"], runner.TRAINING_GIT)

    def test_wrong_source_index_git_is_rejected_without_rewriting_it(self):
        index = self.source / "training_index.json"
        value = json.loads(index.read_text())
        value["git"] = {"branch": "new-inference", "commit": "wrong"}
        runner.write(index, value)
        before = index.read_bytes()
        with patch.object(runner, "source_revision", return_value=runner.TRAINING_GIT), \
             patch.object(runner.subprocess, "run") as child, self.assertRaisesRegex(ValueError, "completed specified"):
            runner.source_audit(self.output)
        child.assert_not_called()
        self.assertEqual(index.read_bytes(), before)

    def test_no_loader_keeps_default_strict_selection_path(self):
        module = types.SimpleNamespace(ROOT=Path(__file__).resolve().parents[1], verify_dataset=Mock())
        with patch.object(common, "_selections", side_effect=ValueError("default strict selection")) as strict:
            with self.assertRaisesRegex(ValueError, "default strict selection"):
                common.run(self.dataset, self.output / "default", 500,
                    self.source / "training_index.json", True, training_module=module, reference_count=72)
        strict.assert_called_once_with((self.source / "training_index.json").resolve(),
                                       self.dataset.resolve(), training_module=module)

    def test_selection_loader_executes_sixty_images_and_three_sheets(self):
        before = (self.source / "training_index.json").read_bytes()
        selected = Mock(return_value=self.runs)
        module = types.SimpleNamespace(ROOT=Path(__file__).resolve().parents[1], verify_dataset=Mock())
        note = "Three literal material forms; frozen training revision; unchanged sampling."
        prompts = []

        class Generator:
            def __init__(self, device):
                self.device = device

            def manual_seed(self, seed):
                self.seed = seed
                return self

        def pipe(prompt, **kwargs):
            prompts.append((prompt, kwargs["generator"].seed))
            self.assertEqual((kwargs["num_inference_steps"], kwargs["guidance_scale"]), (100, 3.5))
            return types.SimpleNamespace(images=[Image.new("RGB", (512, 512), "gray")], nsfw_content_detected=[False])

        fake_torch = types.SimpleNamespace(Generator=Generator, cuda=types.SimpleNamespace(empty_cache=lambda: None))
        with patch.object(common, "git_info", return_value={"branch": "new-inference", "commit": "new"}), \
             patch.object(common, "_selections", side_effect=AssertionError("default current-Git selection path must not run")), \
             patch.object(common, "export_subject_step", return_value={"hashes": {str(self.rgb): sha256(self.rgb)}}) as export, \
             patch.object(common.evaluate, "load_pipeline", return_value=pipe), \
             patch.object(common.evaluate, "attention_masks", return_value={}), patch.dict("sys.modules", {"torch": fake_torch}):
            result = common.run(self.dataset, self.output / "step750", 750,
                self.source / "training_index.json", True, training_module=module,
                reference_count=72, reference_label="TRELLIS.2 prompt forms", probe_rows=prompt_forms.prompt_rows,
                probe_labels=prompt_forms.LABELS, selection_loader=selected, baseline_note=note)
        selected.assert_called_once_with((self.source / "training_index.json").resolve(), self.dataset.resolve())
        self.assertEqual((result["status"], result["expected_images"], result["generated_images"]), ("succeeded", 60, 60))
        self.assertEqual(result["baseline"], note)
        self.assertEqual(result["training"], self.runs)
        self.assertEqual(len(prompts), 60)
        self.assertEqual(len(result["contact_sheets"]), 3)
        self.assertEqual([call.args[1] for call in export.call_args_list], [750, 750])
        self.assertEqual(module.verify_dataset.call_count, 2)
        self.assertEqual((self.source / "training_index.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
