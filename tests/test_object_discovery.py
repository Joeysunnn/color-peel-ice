"""Category discovery preserves input/provenance before per-instance segmentation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageOps

from src.methods.colorpeel_ice import object_discovery as discovery


def entity(category, kind="object", evidence="A separate physical boundary is visible"):
    return {"category": category, "kind": kind, "evidence": evidence}


def raw_objects(*categories):
    return json.dumps({"entities": [entity(category) for category in categories]})


class ResponseTests(unittest.TestCase):
    def test_multiple_categories_and_complete_json_fence_preserve_raw_categories(self):
        response = {"entities": [entity(category) for category in ("Chair", "coffee mug", "apple")]}
        for raw in (json.dumps(response), "```json\n" + json.dumps(response) + "\n```",
                    "```\n" + json.dumps(response) + "\n```"):
            with self.subTest(raw=raw):
                self.assertEqual(discovery.parse_response(raw), response)

    def test_invalid_shape_duplicate_or_attribute_syntax_is_rejected(self):
        invalid = ['[]', '{}', '{"entities":"chair"}', '{"entities":[true]}',
                   '{"entities":[],"boxes":[]}', '{"entities":[],"entities":[]}',
                   '{"categories":["chair"]}', '{"entities":[NaN]}',
                   'Found ' + raw_objects("chair"), '```json\n' + raw_objects("chair")]
        for value in ("<S*>", "coffee-mug", "", " chair", "chair  leg"):
            invalid.append(raw_objects(value))
        for change in ({"kind": "unknown"}, {"kind": []}, {"evidence": " "},
                       {"evidence": 1}, {"bbox": [1, 2, 3, 4]}):
            invalid.append(json.dumps({"entities": [{**entity("chair"), **change}]}))
        invalid.append('{"entities":[{"category":"chair","kind":"object"}]}')
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                discovery.parse_response(raw)

    def test_selection_excludes_background_parts_text_and_keeps_complete_objects(self):
        response = {"entities": [entity("Chair"), entity("brick wall", "background", "Continuous enclosing scene surface"),
            entity("chair leg", "part", "Attached component of the chair"),
            entity("printed numbers", "text", "Markings painted onto a surface"),
            entity("number plate", "object", "A bounded physical plate carries the markings"),
            entity("door", "object", "A whole door has visible outer boundaries"), entity("chair")]}
        original = json.dumps(response)
        selection = discovery.select_categories(discovery.parse_response(original))
        self.assertEqual(selection["categories"], ["chair", "number plate", "door"])
        self.assertEqual([row["reason"] for row in selection["decisions"]],
                         ["whole_object", "excluded_background", "excluded_part", "excluded_text",
                          "whole_object", "whole_object", "duplicate_object_category"])
        self.assertEqual(json.dumps(response), original)
        self.assertFalse(selection["classification_quality_verified"])


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
        raw = '```json\n' + raw_objects("Chair", "coffee mug") + '\n```'
        original_hash = discovery.sha256(self.image)
        path, received = self.invoke(raw)
        manifest = json.loads(path.read_text(encoding="utf8"))
        self.assertEqual(path, self.output / "categories.json")
        self.assertEqual(manifest["schema"], discovery.SCHEMA)
        self.assertEqual(manifest["status"], "verified")
        self.assertEqual(manifest["categories"], ["chair", "coffee mug"])
        self.assertEqual(manifest["response"], json.loads(raw_objects("Chair", "coffee mug")))
        self.assertEqual(manifest["selection"], json.loads((self.output / "selection.json").read_text()))
        self.assertEqual(manifest["selection"]["categories"], manifest["categories"])
        self.assertEqual(received[0], (self.model, "cpu", discovery.PROMPT))
        self.assertEqual((self.output / "raw_response.txt").read_text(encoding="utf8"), raw)
        self.assertEqual(manifest["source"]["image_sha256"], original_hash)
        self.assertEqual(discovery.sha256(self.image), original_hash)
        canonical = Path(manifest["source"]["canonical_image"])
        expected_locks = {self.image, canonical, self.metadata} | {
            self.output / name for name in ("prompt.txt", "raw_response.txt", "response.json", "selection.json",
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
        path, received = self.invoke(raw_objects("mug"))
        source = json.loads(path.read_text())["source"]
        self.assertEqual(source["original_size"], [12, 8])
        self.assertEqual(source["input_size"], [8, 12])
        self.assertEqual(received[1].tobytes(), expected.tobytes())
        with Image.open(source["canonical_image"]) as canonical:
            self.assertEqual(canonical.mode, "RGB")
            self.assertEqual(canonical.size, expected.size)
            self.assertEqual(ImageOps.exif_transpose(canonical).tobytes(), expected.tobytes())

    def test_empty_invalid_and_truncated_responses_preserve_failure_evidence(self):
        for index, (raw, limited) in enumerate(((raw_objects(), False),
                                               ('not JSON', False), (raw_objects("chair"), True))):
            self.output = self.root / f"failed{index}"
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.invoke(raw, token_limit=limited)
            self.assertEqual((self.output / "raw_response.txt").read_text(), raw)
            status = json.loads((self.output / "status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertTrue(status["error"])
            self.assertFalse((self.output / "categories.json").exists())
            self.assertTrue((self.output / "generation_receipt.json").is_file())

    def test_only_excluded_entities_fail_with_selection_receipt_preserved(self):
        response = {"entities": [entity("wall", "background", "Continuous scene surface"),
                                  entity("printed letters", "text", "Ink marks without separate physical extent")]}
        raw = json.dumps(response)
        with self.assertRaisesRegex(ValueError, "No identifiable object"):
            self.invoke(raw)
        self.assertEqual((self.output / "raw_response.txt").read_text(), raw)
        self.assertEqual(json.loads((self.output / "response.json").read_text()), response)
        selection = json.loads((self.output / "selection.json").read_text())
        self.assertEqual(selection["categories"], [])
        self.assertEqual(len(selection["decisions"]), 2)
        self.assertTrue(all(not row["included"] for row in selection["decisions"]))
        self.assertFalse((self.output / "categories.json").exists())

    def test_original_or_model_metadata_mutation_is_rejected(self):
        for index, changed in enumerate((self.image, self.metadata)):
            self.output = self.root / f"changed{index}"
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "provenance changed"):
                self.invoke(raw_objects("chair"), mutate=lambda: changed.write_bytes(b"changed"))
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
