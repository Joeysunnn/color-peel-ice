"""CPU checks for intrinsic QC, invalid inputs and conservative acceptance."""

import copy
import json
import unittest

import numpy as np

from experiments.zest_material_counterfactual_v1.quality import evaluate


class QualityTest(unittest.TestCase):
    def setUp(self):
        y, x = np.indices((512, 512))
        gray = (110 + 35 * np.sin(x / 6) + 25 * np.sin(y / 11)).astype(np.uint8)
        albedo = np.stack((gray, gray, gray), axis=2)
        obj = (x >= 32) & (x < 480) & (y >= 32) & (y < 480)
        protected = (x >= 80) & (x < 100) & (y >= 80) & (y < 100)
        region = (x >= 64) & (x < 448) & (y >= 64) & (y < 448)
        maps = {"albedo": albedo, "metallic": np.full((512, 512), .1),
                "roughness": np.full((512, 512), .4)}
        depth = x.astype(float) / 512 + y.astype(float) / 1024
        self.arguments = dict(original_rgb=albedo.copy(), generated_rgb=albedo.copy(),
                              object_mask=obj, edit_mask=obj & ~protected, protected_mask=protected,
                              valid_mask=obj.copy(), region_mask=region,
                              original_maps=maps, generated_maps=copy.deepcopy(maps),
                              original_depth=depth, generated_depth=depth * 2 + 7,
                              thresholds={})
        self.calibrated = {
            "max_albedo_deltaE76_p95": 1e-8, "max_albedo_delta_ab_p95": 1e-8,
            "min_albedo_edge_precision": .9, "min_albedo_edge_recall": .9,
            "min_albedo_high_frequency_correlation": .9,
            "max_aligned_depth_error": 1e-8, "max_background_mae255": 0,
            "max_protected_mae255": 0, "require_manual_review": True,
            "manual_review_passed": True,
        }

    def evaluate(self):
        result = evaluate(**self.arguments)
        json.dumps(result, allow_nan=False)
        return result

    def test_default_thresholds_and_review_never_approve(self):
        result = self.evaluate()
        self.assertEqual(result["status"], "pending_manual_review")
        self.assertFalse(result["training_eligible"])
        self.assertIn("manual_review_required", result["reasons"])
        self.assertTrue(any(reason.startswith("uncalibrated:") for reason in result["reasons"]))

    def test_calibrated_identity_and_affine_depth_can_pass_after_review(self):
        self.arguments["thresholds"] = self.calibrated
        result = self.evaluate()
        self.assertEqual(result["status"], "accepted", result)
        self.assertTrue(result["training_eligible"])
        self.assertAlmostEqual(result["metrics"]["depth_alignment_scale"], .5)
        self.assertLess(result["metrics"]["aligned_depth_error"], 1e-12)

    def test_rgb_appearance_change_is_not_albedo_or_texture_failure(self):
        self.arguments["thresholds"] = self.calibrated
        self.arguments["generated_rgb"][self.arguments["edit_mask"]] = [0, 200, 255]
        result = self.evaluate()
        self.assertEqual(result["status"], "accepted", result)
        self.assertEqual(result["metrics"]["albedo_deltaE76_p95"], 0)
        self.assertGreater(result["metrics"]["albedo_high_frequency_correlation"], .99)

    def test_zero_texture_is_unassessable_instead_of_perfect(self):
        self.arguments["thresholds"] = self.calibrated
        self.arguments["original_maps"]["albedo"][:] = 100
        self.arguments["generated_maps"]["albedo"][:] = 100
        result = self.evaluate()
        self.assertEqual(result["status"], "pending_manual_review")
        self.assertIsNone(result["metrics"]["albedo_edge_recall"])
        self.assertFalse(result["training_eligible"])
        self.assertIn("source_albedo_has_no_assessable_texture", result["reasons"])

    def test_missing_maps_and_empty_masks_are_invalid(self):
        del self.arguments["generated_maps"]["albedo"]
        self.assertEqual(self.evaluate()["status"], "invalid")
        self.arguments["region_mask"][:] = False
        self.assertEqual(self.evaluate()["status"], "invalid")

    def test_nan_map_and_infinite_depth_are_invalid_json(self):
        self.arguments["generated_maps"]["metallic"][0, 0] = np.nan
        self.assertEqual(self.evaluate()["status"], "invalid")
        self.arguments["generated_maps"]["metallic"][0, 0] = .1
        self.arguments["generated_depth"][0, 0] = np.inf
        self.assertEqual(self.evaluate()["status"], "invalid")

    def test_nan_threshold_is_invalid_json(self):
        self.arguments["thresholds"]["max_background_mae255"] = np.nan
        self.assertEqual(self.evaluate()["status"], "invalid")

    def test_background_and_protected_changes_fail_separately(self):
        self.arguments["thresholds"] = self.calibrated
        self.arguments["generated_rgb"][0, 0] = 255
        self.arguments["generated_rgb"][self.arguments["protected_mask"]] = 255
        result = self.evaluate()
        self.assertEqual(result["status"], "rejected")
        for name in ("max_background_mae255", "max_protected_mae255"):
            self.assertIn("threshold_failed:" + name, result["reasons"])

    def test_material_direction_uses_estimates_and_reference_target(self):
        self.arguments["thresholds"] = dict(self.calibrated, target_metallic=.8,
                                             target_roughness=.4, min_material_direction_delta=.05)
        result = self.evaluate()
        self.assertEqual(result["status"], "rejected")
        self.assertIn("material_direction_failed:metallic", result["reasons"])
        self.arguments["generated_maps"]["metallic"][:] = .6
        result = self.evaluate()
        self.assertEqual(result["status"], "accepted", result)
        self.assertAlmostEqual(result["metrics"]["estimated_material"]["metallic"]["estimated_delta"], .5)

    def test_distinct_depth_distortion_fails_after_full_field_alignment(self):
        self.arguments["thresholds"] = self.calibrated
        self.arguments["generated_depth"][120:320, 120:320] += .8
        result = self.evaluate()
        self.assertEqual(result["status"], "rejected")
        self.assertIn("threshold_failed:max_aligned_depth_error", result["reasons"])

    def test_zero_direction_margin_does_not_accept_no_material_change(self):
        self.arguments["thresholds"] = dict(self.calibrated, target_metallic=.8,
                                             min_material_direction_delta=0.)
        result = self.evaluate()
        self.assertEqual(result["status"], "rejected")
        self.assertIn("material_direction_failed:metallic", result["reasons"])

    def test_protected_edit_overlap_is_invalid(self):
        self.arguments["edit_mask"] |= self.arguments["protected_mask"]
        self.assertEqual(self.evaluate()["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
