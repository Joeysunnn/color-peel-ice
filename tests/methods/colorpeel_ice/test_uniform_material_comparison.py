import csv
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from experiments.material_uniform_albedo_v1 import compare


class UniformMaterialComparisonTests(unittest.TestCase):
    def fixture(self, root, step=500, filtered=False):
        root.mkdir()
        (root / "images").mkdir()
        protocol = compare.probe.data.read_protocol()
        rows = compare.probe.build_rows(protocol, step)
        (root / "generation_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        generated = []
        for row in rows:
            path = root / row["image_path"]
            Image.new("RGB", (512, 512)).save(path)
            generated.append({**row, "status": "safety_filtered" if filtered and row["condition"] == "learned_material"
                              else "ok", "image_sha256": compare.probe.sha256(path),
                              "material_adapter_loaded": row["condition"] == "learned_material"})
        (root / "generation_status.jsonl").write_text("".join(json.dumps(row) + "\n" for row in generated))
        compare.probe.write_json(root / "provenance.json", {"status": "succeeded", "training_steps": step,
            "sampling": {**protocol["inference"], "conditions": list(compare.probe.CONDITIONS)},
            "texture_adapter_loaded": False})
        compare.probe.write_json(root / "summary.json", {"status": "succeeded", "training_steps": step,
            "generated_images": 96, "expected_images": 96, "missing_images": 0})
        return generated

    def test_exact_matched_sheets_and_blind_pairs_preserve_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root / "old")
            self.fixture(root / "new")
            report = compare.run(root / "old", root / "new", root / "comparison")
            self.assertEqual(report["matched_pairs"], 96)
            self.assertEqual(report["learned_material_pairs"], 32)
            self.assertEqual(report["blind_review_pairs"], 32)
            self.assertEqual(report["control_image_hashes_equal"], 64)
            self.assertEqual(len(report["contact_sheets"]), 8)
            with Image.open(root / "comparison" / report["contact_sheets"][0]) as image:
                self.assertEqual(image.width, 6 * 272)
            for path, digest in report["source_locks"].items():
                self.assertEqual(compare.probe.sha256(Path(path)), digest)
            with self.assertRaisesRegex(ValueError, "new and outside"):
                compare.run(root / "old", root / "new", root / "old" / "comparison")

    def test_rejects_prompt_seed_sampling_hash_and_step_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = self.fixture(root / "source")
            protocol = compare.probe.data.read_protocol()
            ledger = root / "source/generation_status.jsonl"
            original = ledger.read_text()
            for field, replacement in (("prompt", "changed"), ("seed", 999), ("image_sha256", "bad")):
                changed = [{**rows[0], field: replacement}, *rows[1:]]
                ledger.write_text("".join(json.dumps(row) + "\n" for row in changed))
                with self.assertRaises(ValueError):
                    compare.verify_probe(root / "source", protocol)
            ledger.write_text(original)
            provenance_path = root / "source/provenance.json"
            provenance = compare.probe.read_json(provenance_path)
            compare.probe.write_json(provenance_path, {**provenance, "sampling": {}})
            with self.assertRaisesRegex(ValueError, "step/sampling"):
                compare.verify_probe(root / "source", protocol)
            compare.probe.write_json(provenance_path, provenance)
            self.fixture(root / "new", step=750)
            with self.assertRaisesRegex(ValueError, "training steps differ"):
                compare.run(root / "source", root / "new", root / "comparison")

    def test_safety_filtered_pairs_excluded_and_control_difference_is_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root / "old")
            rows = self.fixture(root / "new", filtered=True)
            path = root / "new" / rows[0]["image_path"]
            Image.new("RGB", (512, 512), "red").save(path)
            rows[0]["image_sha256"] = compare.probe.sha256(path)
            (root / "new/generation_status.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            report = compare.run(root / "old", root / "new", root / "comparison")
            self.assertEqual(report["control_image_hashes_equal"], 63)
            self.assertEqual(report["blind_review_pairs"], 0)
            self.assertEqual(report["safety_filtered_pairs"], 32)
            with (root / "comparison/paired_blind_ratings.csv").open(newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])


if __name__ == "__main__":
    unittest.main()
