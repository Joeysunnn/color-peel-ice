"""Renderer input guards run without importing Blender or touching a GPU."""

import copy
import tempfile
from pathlib import Path
import unittest

from experiments.texture_material_disentangled_v1 import render


class TextureMaterialRendererTests(unittest.TestCase):
    def fixture(self, directory):
        texture = Path(directory) / "texture.png"
        texture.write_bytes(b"immutable texture fixture")
        protocol = {"dataset": {"expected_train_images": 1, "heldout_contexts": [],
                                "colors_linear_rgb": {"red": [0.6, 0.06, 0.04]},
                                "light_phrases": {"soft_front": "soft studio lighting"}},
                    "textures": {"stripes": {}}, "materials": {"metal": {}}}
        row = {"id": "train_one", "split": "train", "shape": "sphere", "color": "red",
               "light": "soft_front", "texture_id": "stripes", "material_id": "metal",
               "seed": 42, "texture_path": str(texture.resolve()),
               "texture_sha256": render.sha256(texture)}
        plan = {"status": "complete", "protocol_sha256": render.protocol_sha256(protocol),
                "records": [row]}
        return protocol, plan, texture

    def test_changed_texture_is_rejected_before_blender(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol, plan, texture = self.fixture(directory)
            self.assertEqual(render.validate_plan(plan, protocol), plan["records"])
            texture.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Texture hash mismatch"):
                render.validate_plan(plan, protocol)

    def test_untrusted_identifiers_and_duplicate_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol, original, _ = self.fixture(directory)
            for identifier in ("../old_results", "/absolute", "..", "foo/bar", "foo\\bar"):
                plan = copy.deepcopy(original)
                plan["records"][0]["id"] = identifier
                with self.subTest(identifier=identifier), self.assertRaisesRegex(ValueError, "Unsafe render ID"):
                    render.validate_plan(plan, protocol)
            protocol["dataset"]["expected_train_images"] = 2
            plan = copy.deepcopy(original)
            plan["protocol_sha256"] = render.protocol_sha256(protocol)
            plan["records"].append(copy.deepcopy(plan["records"][0]))
            with self.assertRaisesRegex(ValueError, "Duplicate render ID"):
                render.validate_plan(plan, protocol)

    def test_protocol_or_incomplete_plan_cannot_be_rendered(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol, plan, _ = self.fixture(directory)
            for field, value in (("status", "smoke"), ("protocol_sha256", "0" * 64)):
                changed = {**plan, field: value}
                with self.assertRaisesRegex(ValueError, "different protocol"):
                    render.validate_plan(changed, protocol)

    def test_existing_output_is_rejected_without_blender_or_plan_read(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "old_result"
            marker.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                render.main(["--plan", "missing.json", "--output", directory, "--gpu", "3"])
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
