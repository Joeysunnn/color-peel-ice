import itertools
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from experiments.natural_material_initializer_v1 import review
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256


class WoodInitializerReviewTests(unittest.TestCase):
    def fixture(self, root):
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
        references = {method: {"method": method, "branch": "material", "source_id": "wood_spoon",
            "concept_id": review.CONCEPT, "step": 750, "run": str(root / "material_init_run" / method),
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
            attribute = "wood" if condition == "literal" else "<M*>"
            prompt = f"a photo of a {context}" + ("" if condition == "base" else f" made of {attribute}")
            rows.append({"id": identity, "method": method, "concept_id": review.CONCEPT,
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


if __name__ == "__main__":
    unittest.main()
