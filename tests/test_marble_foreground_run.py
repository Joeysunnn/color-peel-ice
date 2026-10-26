"""Verify explicit clean model inputs and no original-foreground replacement."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from experiments.subject_marble_color30_v1 import data
from experiments.subject_marble_foreground_v1 import run


class ForegroundRunTest(unittest.TestCase):
    def test_official_call_uses_explicit_isolated_reference_and_frozen_sampling(self):
        target = Image.new("RGB", (1024, 1024), (127, 127, 127))
        reference = Image.new("RGB", target.size, (210, 80, 20))
        depth, mask = Image.new("RGB", target.size), Image.new("L", target.size, 255)
        marble = SimpleNamespace(run_parametric_control=Mock(return_value=target))
        mlps = {name: object() for name in ("metallic", "roughness")}
        for controls in ({}, {"metallic": -20, "roughness": 0}):
            run.generate_edit(marble, object(), mlps, {"controls": controls}, target, reference, depth, mask)
        for call in marble.run_parametric_control.call_args_list:
            self.assertEqual(call.args[1].tobytes(), target.tobytes())
            self.assertEqual(call.kwargs["texture_image"].tobytes(), reference.tobytes())
            self.assertEqual(call.kwargs["seed"], 42)
            self.assertEqual(call.kwargs["num_inference_steps"], 30)
            self.assertEqual(call.kwargs["mask"].mode, "RGB")
        self.assertEqual(marble.run_parametric_control.call_args_list[0].args[2], {})
        self.assertEqual(marble.run_parametric_control.call_args_list[1].args[2][mlps["roughness"]], 0)

    def test_candidates_keep_generated_foreground_and_neutralize_every_exterior(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source").mkdir()
            mask = Image.new("L", (512, 512))
            mask.paste(255, (100, 120, 420, 400))
            mask.save(root / "source/object_mask512.png")
            config = {"colors": [{"name": c, "source_color": s} for c, s in data.COLOR_MAPPING],
                      "metallic": [-20, 20], "roughness": [-1, 0, 1]}
            sources = {}
            for color, _ in data.COLOR_MAPPING:
                path = root / "source" / (color + ".png")
                Image.new("RGB", (512, 512), (20, 210, 40)).save(path)
                sources[color] = {"original": "source/" + path.name}
            raw = root / "raw.png"
            Image.new("RGB", (1024, 1024), (210, 30, 80)).save(raw)
            rows = data.rows(config)
            records = [{**row, "raw_image": "raw.png", "raw_sha256": data.sha256(raw)} for row in rows]
            raw_before = raw.read_bytes()
            run.package(root, {"rows": rows, "sources": sources}, records)
            manifest = data.read(root / "dataset/manifest.json")["records"]
            self.assertEqual(len(manifest), 35)
            inside = np.asarray(mask) != 0
            for record in manifest:
                path = root / "dataset" / record["image"]
                self.assertEqual(set(path.parent.iterdir()), {path})
                value = np.asarray(Image.open(path))
                self.assertTrue((value[~inside] == 127).all())
                expected = (210, 30, 80) if record["kind"] == "marble_counterfactual" else (20, 210, 40)
                self.assertTrue((value[inside] == expected).all())
                rgba = np.asarray(Image.open(root / record["foreground_rgba"]))
                self.assertTrue((rgba[~inside] == 0).all())
            self.assertEqual(raw.read_bytes(), raw_before)


if __name__ == "__main__":
    unittest.main()
