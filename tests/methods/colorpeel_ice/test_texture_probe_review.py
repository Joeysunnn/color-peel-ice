import csv
import itertools
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from experiments.texture_material_disentangled_v1 import data, texture_probe_review as review
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256


class TextureProbeReviewTests(unittest.TestCase):
    def fixture(self, root):
        protocol = data.read_protocol()
        (root / "images").mkdir()
        references, colors = {}, {}
        for group, (method, texture) in enumerate(itertools.product(
                review.infer.METHODS, review.infer.target_ids(protocol, "texture"))):
            images = []
            colors[method, texture] = (25 + group * 45, 110, 205 - group * 35)
            for index in range(16):
                path = root / f"actual_training_{group}_{index}.png"
                Image.new("RGB", (16, 16), colors[method, texture]).save(path)
                images.append({"id": f"texture_rgb_{group}_{index}", "image": str(path),
                               "image_sha256": sha256(path), "prompt": f"actual training caption {index}"})
            references[method, texture] = {"method": method, "texture": texture, "step": 750,
                "run": str(root / "training_run"), "checkpoint": str(root / "checkpoint750"),
                "dataset_manifest": str(root / "dataset_manifest.json"),
                "dataset_manifest_sha256": "a" * 64, "images": images}
        rows = []
        for method, texture, context, seed, condition in itertools.product(
                review.infer.METHODS, review.infer.target_ids(protocol, "texture"),
                protocol["dataset"]["heldout_contexts"], protocol["inference"]["seeds"], review.CONDITIONS):
            identity = f"{method}_{texture}_{context['id']}_{seed}_{condition}"
            row = {"id": identity, "method": method, "texture": texture, "context": context["id"],
                   "seed": seed, "condition": condition, "training_steps": 750,
                   "status": "ok", "image_path": f"images/{identity}.png"}
            Image.new("RGB", (16, 16), (10, 20, 30)).save(root / row["image_path"])
            rows.append(row)
        return protocol, rows, references, colors

    def test_actual_training_rgb_on_eight_three_column_sheets_and_complete_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, rows, references, colors = self.fixture(root)
            rows[0]["status"] = "safety_filtered"
            sheets = review.review_bundle(root, rows, protocol, references)
            self.assertEqual(len(sheets), 8)
            for sheet in sheets:
                method, texture, _ = Path(sheet).stem.split("__")
                expected = colors[method, texture]
                with Image.open(root / sheet) as image:
                    self.assertEqual(image.width, 3 * 272)
                    # Every training thumb contains the actual, distinctive RGB source pixels.
                    for column in range(8):
                        matches = [y for y in range(100, 500) if all(abs(a - b) < 5 for a, b in
                                   zip(image.getpixel((8 + column * 102 + 40, y)), expected))]
                        self.assertGreater(len(matches), 150)
            provenance = read_json(root / "training_references.json")
            self.assertEqual(len(provenance["groups"]), 4)
            for group in provenance["groups"]:
                self.assertEqual(group["conditions"], {"base": None, "literal": None, "learned_texture": "images"})
                self.assertEqual(group["images"], references[group["method"], group["texture"]]["images"])
                self.assertEqual(len(group["images"]), 16)
            page = (root / "index.html").read_text(encoding="utf-8")
            self.assertEqual(page.count("<figure>"), 64)
            self.assertIn("actual_training_0_15.png", page)
            with (root / "blinded_ratings.csv").open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 95)
            blind = (root / "blinded_review.html").read_text(encoding="utf-8")
            self.assertEqual(blind.count("<figure>"), 95)
            self.assertNotIn(rows[0]["image_path"], blind)
            self.assertNotIn("actual_training", blind)

    def test_rejects_modified_source_rgb(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, rows, references, _ = self.fixture(root)
            first = next(iter(references.values()))["images"][0]
            Image.new("RGB", (16, 16), "white").save(first["image"])
            with self.assertRaisesRegex(ValueError, "training RGB changed"):
                review.review_bundle(root, rows, protocol, references)


if __name__ == "__main__":
    unittest.main()
