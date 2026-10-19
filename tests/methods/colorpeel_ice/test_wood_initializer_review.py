import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image

from experiments.natural_material_initializer_v1 import review
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256


class WoodInitializerReviewTests(unittest.TestCase):
    def fixture(self, root, source_id="wood_spoon"):
        concept = source_id + "_material"
        literal = json.loads(review.TRANSFER_PROMPTS.read_text(encoding="utf8"))["literal_material"][source_id]
        (root / "images").mkdir()
        source = root / "source.png"
        Image.new("RGB", (128, 128), (200, 30, 50)).save(source)
        images, colors = [], []
        for index in range(8):
            path = root / f"actual_training_{index}.png"
            color = (30 + 20 * index, 160, 90)
            colors.append(color)
            Image.new("RGB", (128, 128), color).save(path)
            images.append({"id": f"training_{index}", "image": str(path), "image_sha256": sha256(path),
                           "prompt": f"actual RGB training caption {index}"})
        references = {method: {"method": method, "branch": "material", "source_id": source_id,
            "concept_id": concept, "step": 750, "run": str(root / "material_init_run" / method),
            "checkpoint": str(root / "material_init_step750" / method),
            "baseline_run": str(root / "metal_init_run" / method),
            "baseline_checkpoint": str(root / "metal_init_step750" / method),
            "dataset_manifest": str(root / "original_manifest.jsonl"), "images": images,
            "source": {"input": str(source), "roughness": 0.2941176, "metallic": 0.0078431}}
            for method in review.METHODS}
        rows = []
        for method, context, seed, condition in itertools.product(
                review.METHODS, review.CONTEXTS, (42, 43), review.CONDITIONS):
            identity = f"{method}_{context}_{seed}_{condition}"
            path = f"images/{identity}.png"
            Image.new("RGB", (16, 16), (10, 20, 30)).save(root / path)
            attribute = literal if condition == "literal" else "<M*>"
            prompt = f"a photo of a {context}" + ("" if condition == "base" else f" made of {attribute}")
            rows.append({"id": identity, "method": method, "concept_id": concept,
                         "source_id": source_id, "branch": "material",
                         "context": context, "seed": seed, "condition": condition, "prompt": prompt,
                         "image_path": path, "image_sha256": sha256(root / path), "status": "ok"})
        return rows, references, colors

    def test_four_sheets_show_all_actual_rgb_and_both_initializers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, references, colors = self.fixture(root)
            rows[0]["status"] = "safety_filtered"
            sheets = review.review_bundle(root, rows, references)
            self.assertEqual(len(sheets), 4)
            for path in sheets:
                with Image.open(root / path) as image:
                    self.assertEqual(image.width, 1536)
                    # Every distinct real RGB thumb is visible, not a texture or source-photo proxy.
                    for index, color in enumerate(colors):
                        matches = sum(all(abs(a - b) < 6 for a, b in zip(
                            image.getpixel((154 + index * 140 + 40, y)), color)) for y in range(150, 1000))
                        self.assertGreater(matches, 100)
            receipt = read_json(root / "training_references.json")
            self.assertEqual(len(receipt["groups"]), 2)
            for group in receipt["groups"]:
                self.assertEqual(group["initializer"], "material")
                self.assertEqual(group["images"], references[group["method"]]["images"])
                self.assertEqual(group["conditions"], {"base": None, "literal": None,
                    "learned_metal_init": "images", "learned_material_init": "images"})
                self.assertIn("metal_init", group["baseline_checkpoint"])
                self.assertIn("material_init", group["checkpoint"])
            index = (root / "index.html").read_text(encoding="utf8")
            self.assertEqual(index.count("<figure>"), 16)
            for number in range(8):
                self.assertIn(f"actual_training_{number}.png", index)
            self.assertIn("Source photo is separate", index)
            self.assertEqual(index.count("contact_sheets/"), 8)

    def test_wood_initializer_titles_columns_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, references, _ = self.fixture(root)
            for reference in references.values():
                reference["initializer"] = "wood"
            original_text = review.ImageDraw.ImageDraw.text
            with mock.patch.object(review.ImageDraw.ImageDraw, "text", autospec=True,
                                   side_effect=original_text) as drawn_text:
                sheets = review.review_bundle(root, rows, references)
            self.assertEqual(len(sheets), 4)
            labels = [call.args[2] for call in drawn_text.call_args_list]
            self.assertEqual(labels.count("Learned: wood init"), 4)
            self.assertFalse(any("material-init" in label or "Learned: material init" in label
                                 for label in labels))
            self.assertTrue(any("Material initializer transfer: metal vs wood" in label for label in labels))
            self.assertTrue(any("initialized with wood." in label for label in labels))
            self.assertTrue(any("New wood-init checkpoint:" in label for label in labels))
            index = (root / "index.html").read_text(encoding="utf8")
            self.assertIn("Material initializer transfer: metal vs wood</title>", index)
            self.assertIn("metal initialization, wood initialization", index)
            self.assertNotIn("material initialization", index)
            groups = read_json(root / "training_references.json")["groups"]
            self.assertEqual({group["initializer"] for group in groups}, {"wood"})
            self.assertTrue(all(group["images"] == references[group["method"]]["images"]
                                for group in groups))

    def test_mixed_initializers_and_wood_for_other_sources_are_rejected(self):
        for source_id, initializers in (("wood_spoon", ("wood", "material")),
                                        ("mailbox", ("wood", "wood")),
                                        ("metal_spoon", ("wood", "wood")),
                                        ("wood_spoon", ("metal", "metal"))):
            with self.subTest(source=source_id, initializers=initializers), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                rows, references, _ = self.fixture(root, source_id)
                for method, initializer in zip(review.METHODS, initializers):
                    references[method]["initializer"] = initializer
                with self.assertRaisesRegex(ValueError, "initializer must match"):
                    review.review_bundle(root, rows, references)
                self.assertFalse((root / "contact_sheets").exists())

    def test_changed_training_rgb_is_rejected_before_publishing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, references, _ = self.fixture(root)
            path = references["fullrank"]["images"][0]["image"]
            Image.new("RGB", (128, 128), "white").save(path)
            with self.assertRaisesRegex(ValueError, "training RGB changed"):
                review.review_bundle(root, rows, references)
            self.assertFalse((root / "contact_sheets").exists())
            self.assertFalse((root / "training_references.json").exists())

    def test_duplicate_transfer_row_is_rejected_before_publishing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, references, _ = self.fixture(root)
            rows[-1] = dict(rows[0])
            with self.assertRaisesRegex(ValueError, "64 distinct"):
                review.review_bundle(root, rows, references)
            self.assertFalse((root / "contact_sheets").exists())

    def test_mailbox_and_metal_spoon_use_frozen_literals_and_source_paths(self):
        for source_id, literal in (("mailbox", "painted metal"), ("metal_spoon", "polished metal")):
            with self.subTest(source=source_id), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                rows, references, _ = self.fixture(root, source_id)
                sheets = review.review_bundle(root, rows, references)
                self.assertEqual(len(sheets), 4)
                self.assertTrue(all(f"__{source_id}_material__" in path for path in sheets))
                index = (root / "index.html").read_text(encoding="utf8")
                self.assertIn(f"{source_id} Material initializer transfer", index)
                self.assertIn(f"literal {literal}", index)
                self.assertNotIn("literal wood", index)
                groups = read_json(root / "training_references.json")["groups"]
                self.assertEqual({group["source_id"] for group in groups}, {source_id})
                self.assertEqual({group["concept_id"] for group in groups}, {source_id + "_material"})

    def test_mixed_reference_sources_are_rejected_before_publishing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, references, _ = self.fixture(root, "mailbox")
            references["lora_r64"]["source_id"] = "metal_spoon"
            references["lora_r64"]["concept_id"] = "metal_spoon_material"
            with self.assertRaisesRegex(ValueError, "reference identity"):
                review.review_bundle(root, rows, references)
            self.assertFalse((root / "contact_sheets").exists())

    def test_mixed_row_sources_are_rejected_before_publishing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, references, _ = self.fixture(root, "mailbox")
            rows[0]["source_id"] = "metal_spoon"
            with self.assertRaisesRegex(ValueError, "64 distinct single-source"):
                review.review_bundle(root, rows, references)
            self.assertFalse((root / "contact_sheets").exists())


if __name__ == "__main__":
    unittest.main()
