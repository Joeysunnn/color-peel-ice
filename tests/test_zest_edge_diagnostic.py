"""Necessary checks for usable internal Albedo edges and immutable source copies."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from experiments.zest_material_counterfactual_v1 import edge_diagnostic as edge


class EdgeDiagnosticTest(unittest.TestCase):
    def fixture(self, patterned):
        support = np.zeros((512, 512), dtype=np.uint8)
        support[80:440, 90:420] = 255
        pixels = np.zeros((512, 512), dtype=np.uint8)
        x = np.arange(512)
        signal = np.rint(128 + 2 * np.sin(2 * np.pi * x / 24)).astype(np.uint8)
        pixels[support > 0] = np.broadcast_to(signal if patterned else np.full(512, 128, dtype=np.uint8), pixels.shape)[support > 0]
        return Image.fromarray(pixels).convert("RGB"), Image.fromarray(support)

    def test_boundary_alone_is_not_texture_and_low_contrast_internal_texture_survives(self):
        old = Image.new("RGB", (1024, 1024))
        plain, support = self.fixture(False)
        _, plain_stats = edge.edges(plain, support, support.resize((1024, 1024)), old)
        self.assertEqual(plain_stats["new_edge_count"], 0)
        striped, support = self.fixture(True)
        pictures, stats = edge.edges(striped, support, support.resize((1024, 1024)), old)
        self.assertEqual(stats["gray_span_255"], 4.)
        self.assertGreater(stats["new_interior25_edge_count"], 1000)
        self.assertEqual(stats["boundary_edge_count_new"], 0)
        edges = np.asarray(pictures["edges"].convert("L")) > 0
        interior = np.asarray(pictures["support_interior3"]) > 0
        self.assertFalse(np.any(edges & ~interior))

    def test_new_prepared_run_preserves_all_originals_and_standard_plan_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "original", root / "edge_diagnostic"
            folder = source / "source/mailbox"
            folder.mkdir(parents=True)
            albedo, support = self.fixture(True)
            images = {"original_albedo": albedo, "valid_mask": support,
                      "edit_mask": support.resize((1024, 1024)), "albedo_edges": Image.new("RGB", (1024, 1024)),
                      "target": Image.new("RGB", (1024, 1024), (220, 150, 50)),
                      "initialization_gray": Image.new("RGB", (1024, 1024), (100, 100, 100))}
            files = {}
            for name, image in images.items():
                path = folder / (name + ".png")
                image.save(path)
                files[name] = path.relative_to(source).as_posix()
            plan = {"input_hashes": {}, "subjects": [{"id": "mailbox", "source_files": files, "edge_policy": "old"}],
                    "inference": {"seed": 42, "strength": .9999}, "thresholds": {"max_protected_mae255": 1},
                    "variants": ["baseline", "color", "texture", "combined"]}
            edge.write(source / "plan.json", plan)
            edge.write(source / "config.json", {"frozen": True})
            prepared_hashes = {p.relative_to(source).as_posix(): edge.sha256(p) for p in source.rglob("*") if p.is_file()}
            edge.write(source / "prepare_receipt.json", {"input_hashes": {}, "output_hashes": prepared_hashes})
            (source / "bank").mkdir()
            edge.write(source / "bank/manifest.json", {"status": "complete", "records": [{"image": "/immutable/reference.png"}]})
            before = {str(p): edge.sha256(p) for p in source.rglob("*") if p.is_file()}
            self.assertEqual(edge.run(source, output), output)
            after = {str(p): edge.sha256(p) for p in source.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            new = edge.load_plan(output)
            self.assertEqual(new["inference"], plan["inference"])
            self.assertEqual(new["thresholds"], plan["thresholds"])
            self.assertEqual(edge.sha256(source / "bank/manifest.json"), edge.sha256(output / "bank/manifest.json"))
            for key in files:
                if key != "albedo_edges":
                    self.assertEqual(edge.sha256(source / files[key]), edge.sha256(output / files[key]))
            self.assertNotEqual(edge.sha256(source / files["albedo_edges"]), edge.sha256(output / files["albedo_edges"]))
            with self.assertRaises(FileExistsError):
                edge.run(source, output)
            self.assertEqual(before, {str(p): edge.sha256(p) for p in source.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
