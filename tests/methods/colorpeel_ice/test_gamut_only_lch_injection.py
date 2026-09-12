from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.methods.colorpeel_ice import run_d1_gamut_only_lch_injection as runner
from src.methods.colorpeel_ice import gamut_aware_lch_injection as source_batch
from src.methods.colorpeel_ice import gamut_only_lch_injection as injection


class GamutOnlyLchInjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((16, 16, 3), 128, dtype=np.uint8)
        self.image[0, 0] = [1, 2, 3]
        self.mask = np.zeros((16, 16), dtype=np.uint8)
        self.mask[3:13, 3:13] = 255

    def test_full_target_chroma_is_requested_without_shading_attenuation(self) -> None:
        output, evidence, maps = injection.inject_rgb(self.image, self.mask, 20.0, 20.0)
        inside = self.mask == 255
        target_C, _ = injection.target_lch(20.0, 20.0)
        np.testing.assert_allclose(maps["C_desired"][inside], target_C)
        np.testing.assert_allclose(maps["C_final"][inside], target_C, atol=1e-6)
        self.assertEqual(evidence["gamut_limited_pixel_count"], 0)
        self.assertTrue(np.array_equal(output[self.mask == 0], self.image[self.mask == 0]))

    def test_only_gamut_reduces_chroma_without_rgb_clipping(self) -> None:
        _, evidence, maps = injection.inject_rgb(self.image, self.mask, 90.0, 90.0)
        inside = self.mask == 255
        self.assertGreater(evidence["gamut_limited_pixel_count"], 0)
        self.assertEqual(evidence["rgb_clipping_pixel_count"], 0)
        self.assertTrue((maps["C_final"][inside] <= maps["C_desired"][inside] + 1e-6).all())

    def test_requests_and_summary_are_exact(self) -> None:
        requests = injection.pilot_requests()
        self.assertEqual(len(requests), 18)
        self.assertTrue(all(row["request_id"].startswith("gamut_only__") for row in requests))
        rows = [{**row, "e_ch": 0.0, "gamut_limited_pixel_ratio": 0.0, "gamut_reduction_median": 0.0,
                 "gamut_reduction_p95": 0.0, "outside_mask_changed_pixel_count": 0, "rgb_clipping_pixel_count": 0} for row in requests]
        self.assertTrue(injection.summarize_measurements(rows)["overall_pass"])
        rows[0]["gamut_reduction_p95"] = 0.6
        self.assertFalse(injection.summarize_measurements(rows)["overall_pass"])

    def _source_fixture(self, root: Path) -> tuple[str, str]:
        image = np.full((512, 512, 3), 128, dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[128:384, 128:384] = 255
        contract = {"source": "synthetic"}
        (root / runner.SOURCE_CONTRACT_NAME).write_text(json.dumps(contract), encoding="utf-8")
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
        manifest = {"schema": "d1_gamut_aware_lch_neutral_manifest/v1", "contract_sha256": source_batch.canonical_sha256(contract), "request_count": 9, "records": records}
        manifest_path = root / runner.SOURCE_MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return runner.file_sha256(manifest_path), runner.file_sha256(root / runner.SOURCE_CONTRACT_NAME)

    def test_runner_binds_reused_sources_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root, neutral = Path(temp) / "run", Path(temp) / "source"
            neutral.mkdir()
            manifest_hash, contract_hash = self._source_fixture(neutral)
            with patch.object(injection, "SOURCE_NEUTRAL_MANIFEST_SHA256", manifest_hash), patch.object(injection, "SOURCE_NEUTRAL_CONTRACT_SHA256", contract_hash):
                runner.plan(root, neutral)
                self.assertEqual(len(runner.inject(root, neutral)["rows"]), 18)
                self.assertIn("overall_pass", runner.analyze(root, neutral))

        with tempfile.TemporaryDirectory() as temp:
            root, neutral = Path(temp) / "run", Path(temp) / "source"
            neutral.mkdir()
            manifest_hash, contract_hash = self._source_fixture(neutral)
            with patch.object(injection, "SOURCE_NEUTRAL_MANIFEST_SHA256", manifest_hash), patch.object(injection, "SOURCE_NEUTRAL_CONTRACT_SHA256", contract_hash):
                runner.plan(root, neutral)
                metadata = next((neutral / "neutral" / "renders").glob("*/metadata.json"))
                metadata.write_text('{"tampered": true}\n', encoding="utf-8")
                with self.assertRaises(injection.GamutOnlyLchError):
                    runner.inject(root, neutral)


if __name__ == "__main__":
    unittest.main()
