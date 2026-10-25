"""CPU checks for five-color MARBLE inputs and immutable preparation."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from experiments.subject_marble_color30_v1 import data


class MarbleColor30DataTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.original = self.root / "old"
        self.mask = self.root / "mask.png"
        mask = Image.new("L", (512, 512))
        mask.paste(255, (100, 100, 400, 400))
        mask.save(self.mask)
        colors = []
        for index, (name, source_color) in enumerate(data.COLOR_MAPPING):
            path = self.original / "category" / source_color / "images/image.png"
            path.parent.mkdir(parents=True)
            Image.new("RGB", (512, 512), (index * 30, 100, 200)).save(path)
            colors.append({"name": name, "source_color": source_color, "image_sha256": data.sha256(path)})
        self.config = {"original_root": str(self.original), "mask": str(self.mask),
                       "mask_sha256": data.sha256(self.mask), "colors": colors,
                       "metallic": [-20, 20], "roughness": [-1, 0, 1], "inference": {"seed": 42}}
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf8")
        self.output = self.root / "prepared"

    def prepare(self):
        return data.prepare(self.config_path, self.output)

    def test_thirty_unique_cases_corrected_labels_and_signed_controls(self):
        rows = data.rows(self.config)
        self.assertEqual(len(rows), 30)
        self.assertEqual(len({row["id"] for row in rows}), 30)
        self.assertEqual({row["color"] for row in rows}, {"pink", "green", "cyan", "blue", "purple"})
        self.assertTrue(all(not row["training_eligible"] for row in rows))
        self.assertEqual(rows[0]["id"], "pink__M-20__R-1")
        self.assertEqual(rows[0]["material_state"], "high")
        self.assertEqual(rows[0]["caption"], "a photo of <S*> mailbox in pink color with a smooth metallic surface")
        self.config["colors"][0]["name"] = "red"
        with self.assertRaises(ValueError):
            data.rows(self.config)

    def test_byte_copies_alignment_and_read_only_verification(self):
        plan = self.prepare()
        for color in self.config["colors"]:
            source = plan["sources"][color["name"]]
            self.assertEqual(data.sha256(self.output / source["original"]), color["image_sha256"])
            self.assertEqual(data.sha256(self.output / source["target"]), data.sha256(self.output / source["reference"]))
            with Image.open(self.output / source["target"]) as handle:
                self.assertEqual(handle.size, (1024, 1024))
            self.assertEqual(source["mask"], "source/mask.png")
        self.assertEqual(data.sha256(self.output / "source/object_mask512.png"), self.config["mask_sha256"])
        receipt = (self.output / "prepare_receipt.json").read_bytes()
        self.assertEqual(data.verify_prepared(self.output), plan)
        self.assertEqual((self.output / "prepare_receipt.json").read_bytes(), receipt)

    def test_changed_prepared_image_is_rejected_without_repair(self):
        self.prepare()
        path = self.output / "source/pink/target.png"
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            data.verify_prepared(self.output)
        self.assertEqual(path.read_bytes(), b"changed")

    def test_changed_source_after_prepare_is_rejected(self):
        self.prepare()
        self.mask.write_bytes(b"changed source")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            data.verify_prepared(self.output)

    def test_bad_source_hash_is_rejected_before_output_creation(self):
        self.mask.write_bytes(b"changed source")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_existing_output_is_not_overwritten(self):
        self.prepare()
        before = (self.output / "prepare_receipt.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual((self.output / "prepare_receipt.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
