from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
import json

import numpy as np
from PIL import Image

from src.methods.colorpeel_ice import neutral_lch_injection as injection
from scripts.methods.colorpeel_ice import inject_d1_neutral_lch as runner


class NeutralLchInjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((8, 8, 3), 128, dtype=np.uint8)
        self.image[0, 0] = [1, 2, 3]
        self.mask = np.zeros((8, 8), dtype=np.uint8)
        self.mask[2:6, 2:6] = 255

    def test_lab_linear_round_trip(self) -> None:
        linear = np.array([[0.18, 0.18, 0.18], [0.2, 0.4, 0.6]])
        np.testing.assert_allclose(injection.lab_to_linear_rgb(injection.linear_rgb_to_lab(linear)), linear, atol=2e-6)

    def test_injection_preserves_l_and_background(self) -> None:
        output, evidence = injection.inject_rgb(self.image, self.mask, 0.0, 0.0)
        self.assertEqual(evidence["status"], "injected")
        self.assertTrue(np.array_equal(output[self.mask == 0], self.image[self.mask == 0]))
        before = injection.linear_rgb_to_lab(injection._srgb_to_linear(self.image[self.mask == 255]))
        after = injection.linear_rgb_to_lab(injection._srgb_to_linear(output[self.mask == 255]))
        np.testing.assert_allclose(after[:, 0], before[:, 0], atol=0.8)
        np.testing.assert_allclose(after[:, 1:], 0.0, atol=0.8)

    def test_strict_gamut_rejects_without_output(self) -> None:
        output, evidence = injection.inject_rgb(self.image, self.mask, 120.0, 120.0)
        self.assertIsNone(output)
        self.assertEqual(evidence["status"], "gamut_rejected")
        self.assertGreater(evidence["gamut_rejected_pixel_count"], 0)

    def test_mask_must_be_binary(self) -> None:
        mask = self.mask.copy(); mask[2, 2] = 128
        with self.assertRaises(injection.NeutralLchInjectionError):
            injection.inject_rgb(self.image, mask, 0.0, 0.0)

    def test_pilot_identity_and_rejection_summary(self) -> None:
        requests = injection.pilot_requests()
        self.assertEqual(len(requests), 18)
        self.assertEqual(len({row["request_id"] for row in requests}), 18)
        self.assertEqual(len(injection.future_full_requests()), 90)
        self.assertEqual(injection.future_full_requests()[5], requests[0])
        rows = [{**row, "status": "injected", "e_ch": 0.0} for row in requests]
        self.assertTrue(injection.summarize_measurements(rows)["overall_pass"])
        rows[0]["status"] = "gamut_rejected"
        self.assertEqual(injection.summarize_measurements(rows)["status"], "gamut_rejected")

    def test_runner_plan_and_inject_records_immutable_input_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root, neutral = Path(temp) / "run", Path(temp) / "neutral"
            neutral.mkdir()
            runner.plan(root)
            self.assertTrue((root / runner.PLAN).is_file())
            self.assertTrue((root / runner.CONTRACT).is_file())
            image = np.full((512, 512, 3), 128, dtype=np.uint8)
            mask = np.zeros((512, 512), dtype=np.uint8)
            mask[128:384, 128:384] = 255
            for base in injection.neutral_requests():
                directory = neutral / "renders" / base["request_id"]
                directory.mkdir(parents=True)
                Image.fromarray(image).save(directory / "image.png")
                Image.fromarray(mask).save(directory / "mask.png")
            rows = runner.inject(root, neutral)["rows"]
            self.assertEqual(len(rows), 18)
            self.assertTrue(all("neutral_image_sha256" in row and "neutral_mask_sha256" in row for row in rows))
            self.assertTrue(all(row["status"] in {"injected", "gamut_rejected"} for row in rows))
            with self.assertRaises(injection.NeutralLchInjectionError):
                runner.inject(root, neutral)

    def test_runner_rejects_tampered_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            runner.plan(root)
            plan = json.loads((root / runner.PLAN).read_text(encoding="utf-8"))
            plan["injection_requests"][0]["target_a"] = 999.0
            (root / runner.PLAN).write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaises(injection.NeutralLchInjectionError):
                runner._load_plan(root)


if __name__ == "__main__":
    unittest.main()
