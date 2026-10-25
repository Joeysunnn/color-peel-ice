"""Protect the chosen MARBLE protocol and future one-caption-per-image dataset."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from PIL import Image

from experiments.subject_marble_color30_v1 import data, run


class MarbleColor30RunTest(unittest.TestCase):
    def test_official_call_keeps_self_reference_both_mlps_and_independent_target(self):
        target = Image.new("RGB", (1024, 1024), (200, 80, 40))
        depth, mask = Image.new("L", target.size, 100), Image.new("RGB", target.size, "white")
        edited = Image.new("RGB", target.size, (20, 200, 250))
        marble = SimpleNamespace(run_parametric_control=Mock(return_value=edited))
        mlps = {"metallic": object(), "roughness": object()}
        for m, r in ((-20, 0), (20, 1)):
            run.generate_edit(marble, object(), mlps, {"controls": {"metallic": m, "roughness": r}}, target, depth, mask)
        for call in marble.run_parametric_control.call_args_list:
            self.assertEqual(call.args[1].tobytes(), target.tobytes())
            self.assertIsNone(call.kwargs["texture_image"])
            self.assertEqual(call.kwargs["num_inference_steps"], 30)
            self.assertEqual(call.kwargs["seed"], 42)
            self.assertEqual(set(call.args[2]), set(mlps.values()))
        self.assertEqual(marble.run_parametric_control.call_args_list[0].args[2][mlps["roughness"]], 0)
        with self.assertRaises(ValueError):
            run.generate_edit(marble, object(), mlps, {"controls": {}}, target.resize((512, 512)), depth, mask)

    def test_completed_output_and_conditions_are_verified_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, condition = root / "raw.png", root / "depth.png"
            Image.new("RGB", (1024, 1024)).save(raw)
            Image.new("L", (1024, 1024), 50).save(condition)
            row = {"id": "pink__M-20__R+0", "color": "pink", "controls": {"metallic": -20, "roughness": 0}}
            saved = {**row, "raw_image": "raw.png", "raw_sha256": data.sha256(raw),
                     "condition_hashes": {"depth.png": data.sha256(condition)}}
            self.assertEqual(run.validate_records(root, {"rows": [row]}, [saved]), {row["id"]})
            condition.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "condition changed"):
                run.validate_records(root, {"rows": [row]}, [saved])
            self.assertEqual(condition.read_bytes(), b"changed")
            raw.write_bytes(b"changed raw")
            with self.assertRaisesRegex(ValueError, "raw output changed"):
                run.validate_records(root, {"rows": [row]}, [saved])

    def test_package_has_thirty_plus_five_and_unique_caption_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source").mkdir()
            mask = Image.new("L", (512, 512), 255)
            mask.save(root / "source/object_mask512.png")
            config = {"colors": [{"name": c, "source_color": s} for c, s in data.COLOR_MAPPING],
                      "metallic": [-20, 20], "roughness": [-1, 0, 1]}
            plan = {"rows": data.rows(config), "sources": {}}
            for color, _ in data.COLOR_MAPPING:
                path = root / "source" / (color + ".png")
                Image.new("RGB", (512, 512), (120, 80, 160)).save(path)
                plan["sources"][color] = {"original": "source/" + path.name}
            raw = root / "raw.png"
            Image.new("RGB", (1024, 1024), (90, 100, 180)).save(raw)
            generated = [{**row, "raw_image": "raw.png", "raw_sha256": data.sha256(raw), "condition_hashes": {}} for row in plan["rows"]]
            manifest = run.package(root, plan, generated)
            self.assertEqual(len(manifest["records"]), 35)
            concepts = data.read(root / "dataset/concepts.json")
            self.assertEqual(len({c["instance_data_dir"] for c in concepts}), 35)
            for concept in concepts:
                self.assertEqual(len(list(Path(concept["instance_data_dir"]).glob("*.png"))), 1)
            for row in manifest["records"][:5]:
                self.assertEqual(data.sha256(root / "dataset" / row["image"]), data.sha256(root / row["source_image"]))
            before = data.sha256(root / "dataset" / manifest["records"][0]["image"])
            run.package(root, plan, generated)
            self.assertEqual(before, data.sha256(root / "dataset" / manifest["records"][0]["image"]))


if __name__ == "__main__":
    unittest.main()
