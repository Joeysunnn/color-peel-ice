"""CPU checks for ZeST preparation, protected pixels and immutable resume."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from experiments.zest_material_counterfactual_v1 import prepare as module


class PrepareTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.assets = self.root / "assets"
        self.assets.mkdir()
        self.original = np.full((512, 512, 3), [240, 190, 120], dtype=np.uint8)
        valid = np.zeros((512, 512), dtype=np.uint8)
        valid[100:400, 100:400] = 255
        object_mask = valid.copy()
        object_mask[400:410, 100:400] = 255
        region = np.zeros_like(valid)
        region[120:180, 120:180] = 255
        protected = np.zeros_like(valid)
        protected[200:250, 200:250] = 255
        images = {"image": Image.fromarray(self.original), "object_mask": Image.fromarray(object_mask),
                  "valid_mask": Image.fromarray(valid), "material_region": Image.fromarray(region),
                  "protected_mask": Image.fromarray(protected),
                  "albedo": Image.new("RGB", (512, 512), (200, 60, 20)),
                  "roughness": Image.new("L", (512, 512), 90), "metallic": Image.new("L", (512, 512), 10),
                  "control_depth": Image.new("RGB", (1024, 1024), (150, 150, 150))}
        rgba = Image.fromarray(self.original).convert("RGBA")
        rgba.putalpha(Image.fromarray(valid))
        images["supermat_input"] = rgba
        self.paths = {}
        for key, image in images.items():
            path = self.assets / f"{key}.png"
            image.save(path)
            self.paths[key] = str(path)
        subject = {"id": "mailbox", **{key: self.paths[key] for key in (
            "image", "object_mask", "valid_mask", "material_region", "protected_mask", "supermat_input", "control_depth")},
            "original_maps": {key: self.paths[key] for key in ("albedo", "roughness", "metallic")},
            "expected_sha256": {path: module.sha256(path) for path in self.paths.values()}}
        self.config = {"subjects": [subject], "bank_source": "mailbox",
                       "renderer": {"profile": "experiments/natural_material_fullmaps_v1/renderer_profile_v2.json",
                                    "base_scene": "/scene.blend", "sphere_path": "/sphere.blend",
                                    "resolution": 1024, "cycles_samples": 64, "seed": 42, "light": "soft_front"},
                       "inference": {"seed": 42}, "variants": ["baseline", "color", "texture", "combined"],
                       "thresholds": {}, "supermat": {}}
        self.config_path = self.root / "config.json"
        self.output = self.root / "prepared"

    def run_prepare(self, resume=False):
        self.config_path.write_text(json.dumps(self.config), encoding="utf8")
        return module.prepare(self.config_path, self.output, resume)

    def test_prepares_relative_inputs_nine_grid_plus_anchor_and_reuses_receipt(self):
        plan = self.run_prepare()
        subject = plan["subjects"][0]
        self.assertAlmostEqual(subject["estimated_metallic"], 10 / 255)
        self.assertAlmostEqual(subject["estimated_roughness"], 90 / 255)
        for relative in subject["source_files"].values():
            self.assertFalse(Path(relative).is_absolute())
            self.assertTrue((self.output / relative).is_file())
        bank = json.loads((self.output / "bank_plan.json").read_text())
        self.assertEqual(len(bank["records"]), 10)
        self.assertEqual(bank["records"][-1]["id"], "estimated_original")
        receipt_before = (self.output / "prepare_receipt.json").read_bytes()
        self.assertEqual(self.run_prepare(True), plan)
        self.assertEqual((self.output / "prepare_receipt.json").read_bytes(), receipt_before)
        self.assertIn("adapted ZeST", subject["control_depth_policy"])

    def test_color_initialization_preserves_protected_and_invalid_pixels(self):
        subject = self.run_prepare()["subjects"][0]
        files = subject["source_files"]
        gray = np.asarray(Image.open(self.output / files["initialization_gray"]))
        colored = np.asarray(Image.open(self.output / files["initialization_color"]))
        target = np.asarray(Image.open(self.output / files["target"]))
        edit = np.asarray(Image.open(self.output / files["edit_mask"]))
        valid = np.asarray(Image.open(self.paths["valid_mask"]).resize((1024, 1024), Image.Resampling.NEAREST)) > 0
        protected = np.asarray(Image.open(self.paths["protected_mask"]).resize((1024, 1024), Image.Resampling.NEAREST)) > 0
        self.assertTrue(np.array_equal(colored[~valid | protected], gray[~valid | protected]))
        self.assertTrue(np.array_equal(colored[protected], target[protected]))
        self.assertTrue(np.all(edit[protected] == 0))
        self.assertGreater(subject["color_init_stats"]["clipped_pixels"], 0)
        self.assertEqual(subject["color_init_stats"]["outside_valid_edit_changed_pixels"], 0)
        edges = np.asarray(Image.open(self.output / files["albedo_edges"]))
        self.assertTrue(np.all(edges[~valid | protected] == 0))

    def test_changed_prepared_artifact_is_rejected_without_overwrite(self):
        plan = self.run_prepare()
        path = self.output / plan["subjects"][0]["source_files"]["original_albedo"]
        path.write_bytes(b"corruption")
        with self.assertRaisesRegex(ValueError, "Prepared artifact changed"):
            self.run_prepare(True)
        self.assertEqual(path.read_bytes(), b"corruption")

    def test_supermat_rgba_alpha_cannot_be_repaired_edit_mask(self):
        rgba = Image.fromarray(self.original).convert("RGBA")
        rgba.putalpha(Image.open(self.paths["object_mask"]))
        rgba.save(self.paths["supermat_input"])
        self.config["subjects"][0]["expected_sha256"].pop(self.paths["supermat_input"])
        with self.assertRaisesRegex(ValueError, "RGBA must exactly match"):
            self.run_prepare()
        self.assertFalse(self.output.exists())

    def test_expected_source_hash_change_is_rejected(self):
        self.config["subjects"][0]["expected_sha256"][self.paths["albedo"]] = "wrong"
        with self.assertRaisesRegex(ValueError, "immutable source hash differs"):
            self.run_prepare()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
