import csv
import itertools
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from experiments.natural_texture_material_v1 import probe_review as review
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.texture_material_disentangled_v1 import data


class NaturalProbeReviewTests(unittest.TestCase):
    def fixture(self, root):
        protocol = data.read_protocol()
        (root / "images").mkdir()
        source = root / "source_photo.png"
        Image.new("RGB", (128, 128), (230, 60, 20)).save(source)
        references, rows, colors = {}, [], {}
        for number, (method, branch, source_id) in enumerate(itertools.product(
                review.METHODS, review.BRANCHES, review.SOURCES)):
            concept = source_id + "_" + branch
            colors[method, concept] = (30 + number * 16, 150, 90)
            images = []
            for index in range(16 if branch == "texture" else 8):
                path = root / f"actual_training_{number}_{index}.png"
                Image.new("RGB", (128, 128), colors[method, concept]).save(path)
                images.append({"id": f"training_{number}_{index}", "image": str(path),
                               "image_sha256": sha256(path), "prompt": f"actual caption {index}"})
            references[method, concept] = {"method": method, "branch": branch, "source_id": source_id,
                "concept_id": concept, "step": 750, "run": str(root / "training_run"),
                "checkpoint": str(root / "checkpoint750"), "dataset_manifest": str(root / "manifest.jsonl"),
                "images": images, "source": {"input": str(source), "texture": str(root / "not_training.png"),
                "roughness": 0.25, "metallic": 0.8}}
            for context, seed, condition in itertools.product(protocol["dataset"]["heldout_contexts"],
                    protocol["inference"]["seeds"], ("base", "literal", "learned_" + branch)):
                identity = f"{method}_{concept}_{context['id']}_{seed}_{condition}"
                row = {"id": identity, "method": method, "branch": branch, "source_id": source_id,
                       "concept_id": concept, "context": context["id"], "seed": seed, "condition": condition,
                       "training_steps": 750, "status": "ok", "image_path": f"images/{identity}.png",
                       "prompt": "a photo of a red sphere with <T*> texture, soft studio lighting"
                                  if branch == "texture" else "a photo of a red sphere made of <M*>, soft studio lighting"}
                Image.new("RGB", (16, 16), (10, 20, 30)).save(root / row["image_path"])
                rows.append(row)
        return protocol, rows, references, colors

    def test_twenty_four_sheets_show_all_actual_rgb_and_complete_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, rows, references, colors = self.fixture(root)
            rows[0]["status"] = "safety_filtered"
            sheets = review.review_bundle(root, rows, protocol, references)
            self.assertEqual(len(sheets), 24)
            for name in sheets:
                method, concept, _ = Path(name).stem.split("__")
                expected = colors[method, concept]
                with Image.open(root / name) as image:
                    self.assertEqual(image.width, 1152)
                    # Distinctive RGB pixels from every training thumb, not a texture-map proxy.
                    for column in range(8):
                        matches = sum(all(abs(a - b) < 6 for a, b in zip(
                            image.getpixel((144 + column * 126 + 40, y)), expected)) for y in range(100, 900))
                        self.assertGreater(matches, 190 if concept.endswith("texture") else 95)
            receipt = read_json(root / "training_references.json")
            self.assertEqual(len(receipt["groups"]), 12)
            for group in receipt["groups"]:
                self.assertEqual(group["images"], references[group["method"], group["concept_id"]]["images"])
                self.assertEqual(group["conditions"], {"base": None, "literal": None,
                                                     "learned_" + group["branch"]: "images"})
            index = (root / "index.html").read_text(encoding="utf8")
            self.assertEqual(index.count("<figure>"), 144)
            self.assertIn("actual_training_0_15.png", index)
            self.assertNotIn("not_training.png", index)
            with (root / "blinded_ratings.csv").open(encoding="utf8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 287)
            blind = (root / "blinded_review.html").read_text(encoding="utf8")
            self.assertEqual(blind.count("<figure>"), 287)
            self.assertNotIn(rows[0]["image_path"], blind)
            self.assertNotIn("actual_training", blind)
            self.assertNotIn("training_run", blind)
            self.assertEqual(len(read_json(root / "blinded_key.json")["samples"]), 288)

    def test_changed_training_rgb_is_rejected_before_writing_sheets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, rows, references, _ = self.fixture(root)
            item = next(iter(references.values()))["images"][0]
            Image.new("RGB", (128, 128), "white").save(item["image"])
            with self.assertRaisesRegex(ValueError, "training RGB changed"):
                review.review_bundle(root, rows, protocol, references)
            self.assertFalse((root / "contact_sheets").exists())


if __name__ == "__main__":
    unittest.main()
