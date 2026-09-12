from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from scripts.methods.colorpeel_ice import stage_d1_emission_color_transfer as staging


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EmissionColorTransferStagingTests(unittest.TestCase):
    def _protocol(self, analysis_sha256: str) -> dict:
        return {
            "schema": "emission_color_transfer_protocol/v1",
            "source_pilot": {"analysis_sha256": analysis_sha256, "required_status": "passed"},
            "target": {"stable_id": "D1GT:81/130.png"},
            "training": {
                "modifier_token": "<C*>", "initializer_token": "orange",
                "subjects": ["cube", "sphere", "cylinder"], "view_indices": [0, 8, 16],
                "image_count": 9, "prompt_template": "a photo of {subject} shape in <C*> color",
            },
            "approval_state": {"short_transfer_training_approved": True},
        }

    def _source_root(self, root: Path) -> tuple[Path, str]:
        source = root / "source"
        source.mkdir()
        records = []
        for request_id, request in staging.expected_requests(self._protocol("unused")).items():
            image_relative = f"images/{request_id}.png"
            metadata_relative = f"metadata/{request_id}.json"
            image, metadata = source / image_relative, source / metadata_relative
            image.parent.mkdir(parents=True, exist_ok=True)
            metadata.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (2, 2), (10, 20, 30)).save(image)
            metadata.write_text(json.dumps({"request": {
                "stable_id": "D1GT:81/130.png", "shape": request["shape"],
                "view_index": request["view_index"], "material": "Emission",
                "emission_strength": 1.0,
            }}), encoding="utf-8")
            records.append({"request_id": request_id, "image_relative_path": image_relative,
                            "metadata_relative_path": metadata_relative,
                            "image_sha256": _sha256(image), "metadata_sha256": _sha256(metadata)})
        for index in range(9):
            records.append({"request_id": f"irrelevant-{index}"})
        (source / staging.MANIFEST_NAME).write_text(json.dumps({
            "schema": "d1_emission_color_branch_pilot_manifest/v1", "request_count": 18, "records": records,
        }), encoding="utf-8")
        analysis = source / staging.ANALYSIS_NAME
        analysis.write_text(json.dumps({"status": "passed", "overall_pass": True, "request_count": 18}), encoding="utf-8")
        return source, _sha256(analysis)

    def test_stage_selects_the_verified_nine_images_and_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, analysis_sha256 = self._source_root(root)
            with mock.patch.object(staging, "protocol", return_value=self._protocol(analysis_sha256)):
                result = staging.stage(source, root / "staged")
            self.assertEqual(result["record_count"], 9)
            concepts = json.loads(Path(result["concepts"]).read_text(encoding="utf-8"))
            self.assertEqual(len(concepts), 9)
            self.assertEqual({row["instance_prompt"][0] for row in concepts}, {
                "a photo of cube shape in <C*> color", "a photo of sphere shape in <C*> color",
                "a photo of cylinder shape in <C*> color",
            })
            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["record_count"], 9)
            self.assertTrue(all((Path(row["instance_data_dir"]) / "img.png").is_file() for row in concepts))

    def test_stage_rejects_tampered_source_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, analysis_sha256 = self._source_root(root)
            image = next((source / "images").glob("*.png"))
            image.write_bytes(b"tampered")
            with mock.patch.object(staging, "protocol", return_value=self._protocol(analysis_sha256)):
                with self.assertRaisesRegex(staging.StagingError, "artifact hash"):
                    staging.stage(source, root / "staged")


if __name__ == "__main__":
    unittest.main()
