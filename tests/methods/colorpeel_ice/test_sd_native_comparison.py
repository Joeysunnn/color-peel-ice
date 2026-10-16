"""Operational tests for the paired native comparison; no Torch import needed."""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments/sd_native_comparison_v1/generate.py"
SPEC = importlib.util.spec_from_file_location("sd_native_comparison", SOURCE)
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


class NativeComparisonTests(unittest.TestCase):
    def test_pair_matrix_uses_same_prompt_seed_and_path(self):
        protocol = native.read_protocol()
        first = native.rows(protocol, "sd14")
        second = native.rows(protocol, "sd15")
        self.assertEqual(len(first), 20)
        self.assertEqual({row["object"] for row in first}, {"chair", "mug", "vase", "sofa"})
        for left, right in zip(first, second):
            self.assertEqual({key: left[key] for key in ("pair_id", "prompt", "seed", "image_path")},
                             {key: right[key] for key in ("pair_id", "prompt", "seed", "image_path")})
            self.assertNotEqual(left["id"], right["id"])
        self.assertEqual(len({row["pair_id"] for row in first}), 20)

    def test_protocol_rejects_different_safety_revision_or_sampling(self):
        protocol = native.read_protocol()
        mutations = [
            ("safety", "enabled", False),
            ("safety", "source_model", "sd15"),
            ("sampling", "latent_policy", "cuda_randn"),
            ("sampling", "num_inference_steps", 50),
        ]
        for section, key, value in mutations:
            changed = copy.deepcopy(protocol)
            changed[section][key] = value
            with patch.object(native.PROTOCOL.__class__, "read_text", return_value=json.dumps(changed)):
                with self.assertRaises(ValueError):
                    native.read_protocol()
        self.assertEqual(len(protocol["models"]["sd15"]["revision"]), 40)

    def test_outputs_are_new_and_outside_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            with patch.dict(os.environ, {"COLORPEEL_RUN_ROOT": str(run_root)}):
                self.assertEqual(native.output_directory(run_root / "fresh"), run_root / "fresh")
                with self.assertRaises(ValueError):
                    native.output_directory(run_root)
                with self.assertRaises(ValueError):
                    native.output_directory(ROOT / "forbidden")
                (run_root / "exists").mkdir()
                with self.assertRaises(FileExistsError):
                    native.output_directory(run_root / "exists")
            with patch.dict(os.environ, {"COLORPEEL_RUN_ROOT": str(ROOT / "runs")}):
                with self.assertRaises(ValueError):
                    native.output_directory(ROOT / "runs/new")

    def test_dirty_git_blocks_execution(self):
        with patch.object(native.subprocess, "check_output", return_value=" M README.md\n"):
            with self.assertRaisesRegex(RuntimeError, "dirty worktree"):
                native.git_info()

    def run_with_fake_environment(self, output, *, dry_run, execute=None):
        args = argparse.Namespace(model="sd15", output=output, device="cuda:0", dry_run=dry_run)
        with patch.dict(os.environ, {"COLORPEEL_RUN_ROOT": str(output.parent)}), \
                patch.object(native, "git_info", return_value={"clean_worktree": True}), \
                patch.object(native, "model_inventory", return_value={"hashes_complete": not dry_run}), \
                patch.object(native.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "test==1\n", "")), \
                patch.object(native, "execute", side_effect=execute) as execution:
            result = native.run(args)
        return result, execution

    def test_dry_run_writes_manifest_and_plan_without_model_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dry"
            result, execution = self.run_with_fake_environment(output, dry_run=True)
            self.assertEqual(result, output)
            execution.assert_not_called()
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "dry_run")
            self.assertEqual(manifest["generated_rows"], 0)
            self.assertEqual(manifest["safety_source"]["repo"], "CompVis/stable-diffusion-v1-4")
            ledger = [json.loads(line) for line in (output / "ledger.jsonl").read_text().splitlines()]
            self.assertEqual(len(ledger), 20)
            self.assertTrue(all(row["status"] == "planned" for row in ledger))
            self.assertTrue((output / "model_inventory.json").is_file())

    def test_setup_failure_is_recorded_and_propagated(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "failed"
            with self.assertRaisesRegex(RuntimeError, "simulated CUDA failure"):
                self.run_with_fake_environment(output, dry_run=False,
                                               execute=RuntimeError("simulated CUDA failure"))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["generated_rows"], 0)
            self.assertIn("simulated CUDA failure", manifest["error"])
            self.assertTrue(manifest["finished_at"])
            ledger = [json.loads(line) for line in (output / "ledger.jsonl").read_text().splitlines()]
            self.assertEqual(len(ledger), 20)
            self.assertTrue(all(row["status"] == "not_run" for row in ledger))

    def test_empty_execution_cannot_claim_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "empty"
            with self.assertRaisesRegex(RuntimeError, "Incomplete generation ledger"):
                self.run_with_fake_environment(output, dry_run=False)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "failed")

    def test_partial_failure_preserves_completed_rows(self):
        def fail_after_first(protocol, model, device, output, entries, manifest):
            native.append_ledger(output, {**entries[0], "status": "succeeded",
                                         "latent_sha256": "noise", "image_sha256": "png",
                                         "safety_filtered": False, "error": None})
            raise RuntimeError("second row failed")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "partial"
            with self.assertRaisesRegex(RuntimeError, "second row failed"):
                self.run_with_fake_environment(output, dry_run=False, execute=fail_after_first)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["returncode"], 1)
            self.assertEqual(manifest["generated_rows"], 1)
            ledger = [json.loads(line) for line in (output / "ledger.jsonl").read_text().splitlines()]
            self.assertEqual(len(ledger), 20)
            self.assertEqual(ledger[0]["status"], "succeeded")
            self.assertTrue(all(row["status"] == "not_run" for row in ledger[1:]))

    def test_inventory_checks_download_revision_and_weight_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            protocol = native.read_protocol()
            protocol["models"]["sd15"]["path"] = str(base)
            protocol["models"]["sd14"]["path"] = str(base)
            (base / "model_index.json").write_text("{}")
            for name in (*native.MODEL_COMPONENTS, "safety_checker", "feature_extractor"):
                (base / name).mkdir()
                (base / name / "test.bin").write_bytes(b"weight")
            download = {key: protocol["models"]["sd15"][key] for key in ("repo", "revision")}
            (base / "download_manifest.json").write_text(json.dumps(download))
            inventory = native.model_inventory(protocol, "sd15", full_hash=True)
            original_hash = inventory["components"]["unet"]["test.bin"]["sha256"]
            (base / "unet/test.bin").write_bytes(b"changed weight")
            changed = native.model_inventory(protocol, "sd15", full_hash=True)
            self.assertNotEqual(original_hash, changed["components"]["unet"]["test.bin"]["sha256"])
            download["revision"] = "incorrect"
            (base / "download_manifest.json").write_text(json.dumps(download))
            with self.assertRaisesRegex(ValueError, "download manifest repo/revision"):
                native.model_inventory(protocol, "sd15", full_hash=True)


if __name__ == "__main__":
    unittest.main()
