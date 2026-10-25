"""CPU regression for real official ImageChops semantics and immutable inputs."""

import ast
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageChops, ImageEnhance

from experiments.subject_marble_diagnostics_v1 import identity, run


class InputRegression(unittest.TestCase):
    def test_identity_controls_separate_empty_and_zero_networks_and_limit_extra_seeds(self):
        rows = identity.rows()
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row["id"] for row in rows}), 12)
        by_id = {row["id"]: row for row in rows}
        for seed in (42, 43, 44):
            self.assertEqual(by_id[f"seed{seed}__none"]["controls"], {})
            self.assertEqual(by_id[f"seed{seed}__joint_zero"]["controls"], {"metallic": 0, "roughness": 0})
            if seed != 42:
                self.assertEqual({row["control_name"] for row in rows if row["seed"] == seed}, {"none", "joint_zero"})
        self.assertEqual(by_id["seed42__M+0"]["controls"], {"metallic": 0})
        self.assertEqual(by_id["seed42__R+0"]["controls"], {"roughness": 0})

    def test_large_RGB_difference_does_not_become_albedo_leakage_or_identity_verdict(self):
        original = np.full((10, 10, 3), 20, dtype=np.uint8)
        changed = np.full_like(original, 220)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 255
        result = identity.evaluate(original, changed, mask)
        self.assertEqual(result["appearance_diagnostics"]["foreground_RGB_MAE_255"], 200)
        self.assertEqual(result["albedo_preservation"]["status"], "not_identifiable_from_RGB_alone")
        self.assertEqual(result["flags"], ["manual_identity_review_required"])
        self.assertEqual(result["content_review"]["texture_patterns"], "manual_review_required")

    def test_joint_grids_keep_both_networks_even_at_zero_controls(self):
        for stage, reference in (("mailbox_bowl", "bowl"), ("mailbox_self", "self")):
            rows = run.joint_mailbox_rows(stage)
            self.assertEqual(len(rows), 9)
            self.assertEqual(len({row["id"] for row in rows}), 9)
            self.assertEqual({(row["controls"]["metallic"], row["controls"]["roughness"]) for row in rows},
                             {(m, r) for m in (-20, 0, 20) for r in (-1, 0, 1)})
            self.assertTrue(all(set(row["controls"]) == {"metallic", "roughness"} for row in rows))
            self.assertTrue(all(row["reference"] == reference and row["target_size"] == 1024 for row in rows))

    @unittest.skipUnless(os.environ.get("MARBLE_OFFICIAL_ROOT"), "set pinned official source path")
    def test_official_preprocessing_reproduces_bug_and_verifies_alignment_fix(self):
        source = Path(os.environ["MARBLE_OFFICIAL_ROOT"]) / "marble.py"
        module = ast.parse(source.read_text(encoding="utf8"))
        function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                        and node.name == "prepare_init_image")
        namespace = {"Image": Image, "ImageChops": ImageChops, "ImageEnhance": ImageEnhance}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        official_init = namespace["prepare_init_image"]
        original = Image.new("RGB", (512, 512), (210, 91, 32))
        mask = Image.new("L", original.size)
        mask.paste(255, (142, 291, 441, 512))  # real mailbox bbox, entirely below y256
        before_rgb, before_mask = original.tobytes(), mask.tobytes()
        target1024, mask1024 = run.aligned_inputs(original, mask)
        broken = np.asarray(official_init(original, mask1024))
        self.assertTrue(np.array_equal(broken, np.asarray(target1024)))
        fixed = np.asarray(official_init(target1024, mask1024))
        foreground, background = np.asarray(mask1024)[..., 0] == 255, np.asarray(mask1024)[..., 0] == 0
        self.assertTrue(np.all(fixed[foreground, 0] == fixed[foreground, 1]))
        self.assertTrue(np.all(fixed[foreground, 1] == fixed[foreground, 2]))
        self.assertTrue(np.array_equal(fixed[background], np.asarray(target1024)[background]))
        self.assertFalse(np.array_equal(fixed, broken))
        self.assertEqual(original.tobytes(), before_rgb)
        self.assertEqual(mask.tobytes(), before_mask)

    def test_reference_changes_only_background_without_changing_canvas_or_object_scale(self):
        original = Image.new("RGB", (512, 512), (210, 91, 32))
        mask = Image.new("L", original.size)
        mask.paste(255, (142, 291, 441, 512))
        result = run.masked_reference(original, mask)
        inside = np.asarray(mask) > 0
        self.assertEqual(result.size, original.size)
        self.assertTrue(np.array_equal(np.asarray(result)[inside], np.asarray(original)[inside]))
        self.assertTrue(np.all(np.asarray(result)[~inside] == 127))
        self.assertTrue(np.all(np.asarray(original) == np.array([210, 91, 32])))

    def test_existing_results_are_refused_before_model_or_git_operations(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "old_result"
            output.mkdir()
            marker = output / "old.png"
            marker.write_bytes(b"preserve existing result")
            with patch.object(run, "resolve_run_dir", side_effect=lambda path: path), patch.object(run, "git_info") as git:
                with self.assertRaisesRegex(ValueError, "fresh"):
                    run.run(Path(directory) / "no_assets", output, 2, "mailbox")
                git.assert_not_called()
            self.assertEqual(marker.read_bytes(), b"preserve existing result")


if __name__ == "__main__":
    unittest.main()
