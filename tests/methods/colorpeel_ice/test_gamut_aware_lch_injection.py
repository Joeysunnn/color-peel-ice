from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.methods.colorpeel_ice import gamut_aware_lch_injection as injection
from scripts.methods.colorpeel_ice import run_d1_gamut_aware_lch_injection as runner


class GamutAwareLchInjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((16, 16, 3), 128, dtype=np.uint8)
        self.image[0, 0] = [1, 2, 3]
        self.mask = np.zeros((16, 16), dtype=np.uint8)
        self.mask[3:13, 3:13] = 255

    def test_lab_linear_round_trip(self) -> None:
        linear = np.array([[0.18, 0.18, 0.18], [0.2, 0.4, 0.6]])
        np.testing.assert_allclose(injection.lab_to_linear_rgb(injection.linear_rgb_to_lab(linear)), linear, atol=2e-6)

    def test_attenuation_is_fixed_and_shading_aware(self) -> None:
        values = injection.shading_attenuation(np.array([10.0, 40.0, 55.0, 70.0, 95.0]))
        np.testing.assert_allclose(values, [0.0, 1.0, 1.0, 1.0, 0.0])

    def test_max_chroma_is_in_gamut_and_boundary_is_maximal(self) -> None:
        chroma = injection.max_srgb_chroma(np.array([50.0]), np.deg2rad(45.0))[0]
        good = injection.lab_to_linear_rgb(np.array([[50.0, chroma * np.cos(np.deg2rad(45.0)), chroma * np.sin(np.deg2rad(45.0))]]))
        bad = injection.lab_to_linear_rgb(np.array([[50.0, (chroma + 0.02) * np.cos(np.deg2rad(45.0)), (chroma + 0.02) * np.sin(np.deg2rad(45.0))]]))
        self.assertTrue(((good >= 0) & (good <= 1)).all())
        self.assertTrue((bad < 0).any() or (bad > 1).any())

    def test_injection_preserves_background_l_and_hue(self) -> None:
        output, evidence, maps = injection.inject_rgb(self.image, self.mask, 30.0, 30.0)
        self.assertTrue(np.array_equal(output[self.mask == 0], self.image[self.mask == 0]))
        self.assertEqual(evidence["rgb_clipping_pixel_count"], 0)
        before = injection.linear_rgb_to_lab(injection._srgb_to_linear(self.image[self.mask == 255]))
        after = injection.linear_rgb_to_lab(injection._srgb_to_linear(output[self.mask == 255]))
        np.testing.assert_allclose(after[:, 0], before[:, 0], atol=0.8)
        hue = np.arctan2(after[:, 2], after[:, 1])
        np.testing.assert_allclose(hue, np.deg2rad(45.0), atol=np.deg2rad(2.0))
        self.assertEqual(maps["r"].shape, self.mask.shape)

    def test_gamut_limiting_reduces_chroma_without_clipping(self) -> None:
        output, evidence, maps = injection.inject_rgb(self.image, self.mask, 90.0, 90.0)
        self.assertEqual(output.dtype, np.uint8)
        self.assertGreater(evidence["gamut_limited_pixel_count"], 0)
        inside = self.mask == 255
        self.assertTrue((maps["C_final"][inside] <= maps["C_base"][inside] + 1e-6).all())

    def test_request_identity_and_summary(self) -> None:
        requests = injection.pilot_requests()
        self.assertEqual(len(requests), 18)
        self.assertEqual(len(injection.neutral_requests()), 9)
        from src.methods.colorpeel_ice import natural_image_target_selection as selection
        selected = {row["source"]["stable_id"]: row["target"] for row in selection.pipeline_development_records()}
        for request in requests:
            target = selected[request["stable_id"]]
            self.assertEqual(request["target_a"], target["a"])
            self.assertEqual(request["target_b"], target["b"])
        rows = [{**row, "e_ch": 0.0, "gamut_limited_pixel_ratio": 0.0, "total_reduction_median": 0.0,
                 "total_reduction_p95": 0.0, "outside_mask_changed_pixel_count": 0, "rgb_clipping_pixel_count": 0} for row in requests]
        self.assertTrue(injection.summarize_measurements(rows)["overall_pass"])
        rows[0]["e_ch"] = 6.0
        self.assertFalse(injection.summarize_measurements(rows)["overall_pass"])

    def _make_neutral_fixture(self, root: Path) -> None:
        image = np.full((512, 512, 3), 128, dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[128:384, 128:384] = 255
        records = []
        for request in injection.neutral_requests():
            directory = root / "neutral" / "renders" / request["request_id"]
            directory.mkdir(parents=True)
            image_path, mask_path, metadata_path = directory / "image.png", directory / "mask.png", directory / "metadata.json"
            Image.fromarray(image).save(image_path)
            Image.fromarray(mask).save(mask_path)
            metadata_path.write_text("{}\n", encoding="utf-8")
            records.append({"request_id": request["request_id"], "shape": request["shape"], "view_index": request["view_index"],
                            "image_relative_path": str(image_path.relative_to(root)), "image_sha256": runner.file_sha256(image_path),
                            "mask_relative_path": str(mask_path.relative_to(root)), "mask_sha256": runner.file_sha256(mask_path),
                            "metadata_relative_path": str(metadata_path.relative_to(root)), "metadata_sha256": runner.file_sha256(metadata_path)})
        contract = json.loads((root / runner.CONTRACT_NAME).read_text(encoding="utf-8"))
        (root / runner.NEUTRAL_MANIFEST_NAME).write_text(json.dumps({"schema": "d1_gamut_aware_lch_neutral_manifest/v1", "contract_sha256": injection.canonical_sha256(contract), "request_count": 9, "records": records}), encoding="utf-8")

    def test_runner_records_hashes_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root, assets = Path(temp) / "run", Path(temp) / "assets"
            for relative in runner.ASSETS.values():
                path = assets / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(relative.encode("utf-8"))
            runner.plan(root, assets)
            self._make_neutral_fixture(root)
            rows = runner.inject(root)["rows"]
            self.assertEqual(len(rows), 18)
            self.assertTrue(all("chroma_map_sha256" in row for row in rows))
            self.assertIn("overall_pass", runner.analyze(root))

        with tempfile.TemporaryDirectory() as temp:
            root, assets = Path(temp) / "run", Path(temp) / "assets"
            for relative in runner.ASSETS.values():
                path = assets / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(relative.encode("utf-8"))
            runner.plan(root, assets)
            plan = json.loads((root / runner.PLAN_NAME).read_text(encoding="utf-8"))
            plan["injection_requests"][0]["target_a"] = 999.0
            (root / runner.PLAN_NAME).write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaises(injection.GamutAwareLchError):
                runner._load_plan_contract(root)

        with tempfile.TemporaryDirectory() as temp:
            root, assets = Path(temp) / "run", Path(temp) / "assets"
            for relative in runner.ASSETS.values():
                path = assets / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(relative.encode("utf-8"))
            runner.plan(root, assets)
            self._make_neutral_fixture(root)
            metadata = next((root / "neutral" / "renders").glob("*/metadata.json"))
            metadata.write_text('{"tampered": true}\n', encoding="utf-8")
            with self.assertRaises(injection.GamutAwareLchError):
                runner.inject(root)


if __name__ == "__main__":
    unittest.main()
