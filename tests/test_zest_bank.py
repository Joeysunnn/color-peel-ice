"""CPU checks for immutable inputs and non-destructive PBR bank resume."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, PngImagePlugin

from experiments.zest_material_counterfactual_v1 import render_bank as bank


class ReferenceBankTest(unittest.TestCase):
    def test_render_timestamp_is_not_a_mask_pixel_change(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("first.png", "second.png")]
            image = Image.new("L", (8, 8), 0)
            image.paste(255, (2, 2, 6, 6))
            for path, date in zip(paths, ("19:56:24", "19:56:26")):
                metadata = PngImagePlugin.PngInfo()
                metadata.add_text("Date", date)
                image.save(path, pnginfo=metadata)
            self.assertNotEqual(bank.sha256(paths[0]), bank.sha256(paths[1]))
            digests = []
            for path in paths:
                with Image.open(path) as loaded:
                    digests.append(bank.mask_pixels_sha256([value / 255 for pixel in loaded.convert("RGBA").getdata() for value in pixel]))
            self.assertEqual(*digests)
            self.assertNotEqual(bank.mask_pixels_sha256([0., 0., 0., 1.]), bank.mask_pixels_sha256([1., 1., 1., 1.]))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("scene.blend", "sphere.blend"):
            (self.root / name).write_bytes(name.encode())
        self.plan = {"profile": str(bank.ROOT / "experiments/natural_material_fullmaps_v1/renderer_profile_v2.json"),
                     "renderer": {"base_scene": str(self.root / "scene.blend"),
                                  "sphere_path": str(self.root / "sphere.blend"),
                                  "resolution": 1024, "cycles_samples": 64, "seed": 42, "light": "soft_front"},
                     "albedo_linear_rgb": [.49, .014, .0015],
                     "records": [{"id": f"M{m}_R{r}".replace(".", "p"), "metallic": m, "roughness": r}
                                 for m in (0, .5, 1) for r in (.2, .5, .8)]}
        self.path = self.root / "plan.json"

    def load(self, plan=None):
        self.path.write_text(json.dumps(self.plan if plan is None else plan), encoding="utf8")
        return bank.load_plan(self.path)

    def test_bank_locks_plan_profile_and_both_scene_assets(self):
        plan, profile, paths, locks = self.load()
        self.assertEqual(len(plan["records"]), 9)
        self.assertEqual(profile["object"]["asset_name"], "Sphere")
        self.assertEqual(set(paths), {"profile", "base_scene", "sphere_path"})
        self.assertIn(str(self.path.resolve()), locks)
        self.assertEqual(len(locks), 5)

    def test_reject_duplicate_unsafe_and_nonphysical_parameters(self):
        candidates = []
        for key, value in (("id", "../old"), ("roughness", float("nan")), ("metallic", 1.1), ("roughness", True)):
            plan = copy.deepcopy(self.plan)
            plan["records"][0][key] = value
            candidates.append(plan)
        plan = copy.deepcopy(self.plan)
        plan["records"][1]["id"] = plan["records"][0]["id"]
        candidates.append(plan)
        for plan in candidates:
            with self.subTest(plan=plan):
                with self.assertRaises(ValueError):
                    self.load(plan)

    def test_resume_requires_identical_inputs_and_explicit_flag(self):
        locks = self.load()[3]
        output = self.root / "bank"
        saved = bank.open_output(output, locks, False)
        with self.assertRaises(FileExistsError):
            bank.open_output(output, locks, False)
        self.assertEqual(bank.open_output(output, locks, True), saved)
        changed = dict(locks)
        changed[str(self.path.resolve())] = "changed"
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            bank.open_output(output, changed, True)

    def test_completed_reference_must_have_valid_local_output_hashes(self):
        output = self.root / "bank"
        output.mkdir()
        expected = self.plan["records"][0]
        record = dict(expected)
        for name in ("image", "mask", "input_rgba", "metadata"):
            path = output / name
            path.write_bytes(name.encode())
            record.update({name: str(path), name + "_sha256": bank.sha256(path)})
        self.assertEqual(bank.checked_record(record, output, expected), record)
        Path(record["image"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "changed or missing"):
            bank.checked_record(record, output, expected)


if __name__ == "__main__":
    unittest.main()
