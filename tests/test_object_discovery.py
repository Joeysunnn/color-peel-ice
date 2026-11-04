"""Category discovery preserves input/provenance before per-instance segmentation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageOps

from src.methods.colorpeel_ice import object_discovery as discovery


class ResponseTests(unittest.TestCase):
    def test_multiple_categories_and_complete_json_fence_preserve_raw_categories(self):
        response = {"categories": ["Chair", "coffee mug", "apple"]}
        for raw in (json.dumps(response), "```json\n" + json.dumps(response) + "\n```",
                    "```\n" + json.dumps(response) + "\n```"):
            with self.subTest(raw=raw):
                self.assertEqual(discovery.parse_response(raw), response)

    def test_invalid_shape_duplicate_or_attribute_syntax_is_rejected(self):
        invalid = ('[]', '{}', '{"categories":"chair"}', '{"categories":[true]}',
                   '{"categories":["chair"],"boxes":[]}', '{"categories":["chair","Chair"]}',
                   '{"categories":["chair"],"categories":["mug"]}',
                   '{"categories":["<S*>"]}', '{"categories":["coffee-mug"]}',
                   '{"categories":[""]}', '{"categories":[" chair"]}',
                   '{"categories":["chair  leg"]}', '{"categories":[NaN]}',
                   'Found {"categories":["chair"]}', '```json\n{"categories":["chair"]}')
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                discovery.parse_response(raw)


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.image = self.root / "photo.png"
        Image.new("RGB", (12, 8), "orange").save(self.image)
        self.model = self.root / "model"
        self.model.mkdir()
        self.metadata = self.model / "config.json"
        self.metadata.write_text('{}', encoding="utf8")
        self.output = self.root / "discovery"

    def invoke(self, raw, *, mutate=None, token_limit=False):
        received = []
        receipt = {"metadata_files": {str(self.metadata): discovery.sha256(self.metadata)},
                   "weights": [], "model_dir": str(self.model)}

        class Predictor:
            def __init__(self, model_dir, device, prompt):
                self.runtime = {"device": device, "reached_token_limit": token_limit}
                received.append((model_dir, device, prompt))

            def __call__(self, image):
                received.append(image.copy())
                if mutate:
                    mutate()
                return raw

        with patch.object(discovery, "model_receipt", return_value=receipt), \
                patch.object(discovery, "QwenPredictor", Predictor):
            path = discovery.run(self.image, self.model, self.output, device="cpu")
        return path, received

    def test_manifest_binds_original_canonical_response_and_model_receipts(self):
        raw = '```json\n{"categories":["Chair","coffee mug"]}\n```'
        original_hash = discovery.sha256(self.image)
        path, received = self.invoke(raw)
        manifest = json.loads(path.read_text(encoding="utf8"))
        self.assertEqual(path, self.output / "categories.json")
        self.assertEqual(manifest["schema"], discovery.SCHEMA)
        self.assertEqual(manifest["status"], "verified")
        self.assertEqual(manifest["categories"], ["chair", "coffee mug"])
        self.assertEqual(manifest["response"], {"categories": ["Chair", "coffee mug"]})
        self.assertEqual(received[0], (self.model, "cpu", discovery.PROMPT))
        self.assertEqual((self.output / "raw_response.txt").read_text(encoding="utf8"), raw)
        self.assertEqual(manifest["source"]["image_sha256"], original_hash)
        self.assertEqual(discovery.sha256(self.image), original_hash)
        canonical = Path(manifest["source"]["canonical_image"])
        expected_locks = {self.image, canonical, self.metadata} | {
            self.output / name for name in ("prompt.txt", "raw_response.txt", "response.json",
                                            "model_receipt.json", "generation_receipt.json")}
        self.assertEqual(set(manifest["source_locks"]), {str(path) for path in expected_locks})
        for locked, digest in manifest["source_locks"].items():
            self.assertTrue(Path(locked).is_absolute())
            self.assertEqual(discovery.sha256(locked), digest)
        self.assertEqual(manifest["source"]["canonical_image_sha256"], discovery.sha256(canonical))
        self.assertEqual(json.loads((self.output / "status.json").read_text())["status"], "verified")

    def test_exif_orientation_is_applied_once_to_canonical_and_predictor_image(self):
        self.image = self.root / "rotated.jpg"
        original = Image.new("RGB", (12, 8), "red")
        original.paste("blue", (0, 0, 6, 8))
        exif = original.getexif()
        exif[274] = 6
        original.save(self.image, exif=exif)
        with Image.open(self.image) as opened:
            expected = ImageOps.exif_transpose(opened).convert("RGB")
        path, received = self.invoke('{"categories":["mug"]}')
        source = json.loads(path.read_text())["source"]
        self.assertEqual(source["original_size"], [12, 8])
        self.assertEqual(source["input_size"], [8, 12])
        self.assertEqual(received[1].tobytes(), expected.tobytes())
        with Image.open(source["canonical_image"]) as canonical:
            self.assertEqual(canonical.mode, "RGB")
            self.assertEqual(canonical.size, expected.size)
            self.assertEqual(ImageOps.exif_transpose(canonical).tobytes(), expected.tobytes())

    def test_empty_invalid_and_truncated_responses_preserve_failure_evidence(self):
        for index, (raw, limited) in enumerate((('{"categories":[]}', False),
                                               ('not JSON', False), ('{"categories":["chair"]}', True))):
            self.output = self.root / f"failed{index}"
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.invoke(raw, token_limit=limited)
            self.assertEqual((self.output / "raw_response.txt").read_text(), raw)
            status = json.loads((self.output / "status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertTrue(status["error"])
            self.assertFalse((self.output / "categories.json").exists())
            self.assertTrue((self.output / "generation_receipt.json").is_file())

    def test_original_or_model_metadata_mutation_is_rejected(self):
        for index, changed in enumerate((self.image, self.metadata)):
            self.output = self.root / f"changed{index}"
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "provenance changed"):
                self.invoke('{"categories":["chair"]}', mutate=lambda: changed.write_bytes(b"changed"))
            self.assertFalse((self.output / "categories.json").exists())
            self.assertEqual(json.loads((self.output / "status.json").read_text())["status"], "failed")
            Image.new("RGB", (12, 8), "orange").save(self.image)

    def test_output_must_be_fresh_and_outside_repo_and_model(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("original")
        repo = Path(discovery.__file__).resolve().parents[3]
        for output in (self.output, self.model / "new", repo / "new-object-discovery"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "fresh external"):
                discovery.run(self.image, self.model, output)
        self.assertEqual(marker.read_text(), "original")
        self.assertFalse((self.model / "new").exists())
        self.assertFalse((repo / "new-object-discovery").exists())


if __name__ == "__main__":
    unittest.main()
