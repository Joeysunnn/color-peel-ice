import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from experiments.sd15_token_local_kv_v1.comparison_bundle import (
    bundle_all, bundle_job, verified_bundle,
)
from experiments.sd15_token_local_kv_v1.contract import sha256


class ComparisonBundleTests(unittest.TestCase):
    def make_job(self, root, name="job", count=2):
        directory = root / name
        (directory / "images").mkdir(parents=True)
        references = root / (name + "_old")
        references.mkdir()
        rows, statuses = [], []
        for index in range(count):
            reference = references / f"{index:04d}.png"
            generated = directory / "images" / f"{index:04d}.png"
            Image.new("RGB", (8, 8), (index, 20, 30)).save(reference)
            Image.new("RGB", (8, 8), (40, 50, index)).save(generated)
            row = {"id": f"pair-{index}", "prompt": "a photo of <S*> mailbox " * 12,
                   "seed": 42 + index, "subject_step": 300, "material_step": 5000,
                   "image_path": f"images/{index:04d}.png", "reference_image": str(reference),
                   "reference_sha256": sha256(reference), "reference_filtered": index == 0}
            rows.append(row)
            statuses.append({**row, "status": "safety_filtered" if index == 1 else "ok",
                             "image_sha256": sha256(generated)})
        self.json(directory / "provenance.json", {"status": "succeeded"})
        self.jsonl(directory / "generation_manifest.jsonl", rows)
        self.jsonl(directory / "generation_status.jsonl", statuses)
        return directory, rows, statuses

    def json(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def jsonl(self, path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_complete_bundle_records_hashes_filtering_and_new_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory, rows, _ = self.make_job(root, "job with spaces")
            original_hashes = [sha256(Path(row["reference_image"])) for row in rows]
            summary = bundle_job(directory)
            self.assertEqual(summary["pair_count"], 2)
            self.assertEqual(summary["filtered_counts"], {"sd14": 1, "sd15": 1})
            self.assertEqual(summary, verified_bundle(directory))
            sheet = directory / summary["sheets"][0]["path"]
            self.assertEqual(sha256(sheet), summary["sheets"][0]["sha256"])
            with Image.open(sheet) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertGreater(image.height, 700)
            self.json(root / "comparison_plan.json", {"jobs": [{"id": "S <300>",
                                                                  "output": str(directory)}]})
            result = bundle_all(root)
            self.assertEqual((result["status"], result["pair_count"]), ("complete", 2))
            index = (root / "index.html").read_text(encoding="utf-8")
            self.assertIn("job%20with%20spaces/comparison_sheets/page_001.jpg", index)
            self.assertIn("S &lt;300&gt;", index)
            self.assertEqual(original_hashes, [sha256(Path(row["reference_image"])) for row in rows])
            with self.assertRaises(FileExistsError):
                bundle_job(directory)
            with self.assertRaises(FileExistsError):
                bundle_all(root)

    def test_partial_duplicate_and_mismatched_rows_rejected_before_sheets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory, _, statuses = self.make_job(root)
            for invalid in (statuses[:1], [statuses[0], statuses[0]],
                            [statuses[0], {**statuses[1], "seed": 999}]):
                self.jsonl(directory / "generation_status.jsonl", invalid)
                with self.assertRaises(ValueError):
                    bundle_job(directory)
                self.assertFalse((directory / "comparison_sheets").exists())
            self.jsonl(directory / "generation_status.jsonl", statuses)
            self.json(directory / "provenance.json", {"status": "running"})
            with self.assertRaises(ValueError):
                bundle_job(directory)

    def test_source_and_output_hash_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory, rows, statuses = self.make_job(root)
            Image.new("RGB", (8, 8), "red").save(Path(rows[0]["reference_image"]))
            with self.assertRaises(ValueError):
                bundle_job(directory)
            rows[0]["reference_sha256"] = sha256(Path(rows[0]["reference_image"]))
            statuses[0]["reference_sha256"] = rows[0]["reference_sha256"]
            self.jsonl(directory / "generation_manifest.jsonl", rows)
            self.jsonl(directory / "generation_status.jsonl", statuses)
            Image.new("RGB", (8, 8), "blue").save(directory / rows[1]["image_path"])
            with self.assertRaises(ValueError):
                bundle_job(directory)

    def test_bundle_all_checks_all_jobs_and_existing_sheet_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            first, _, _ = self.make_job(root, "first")
            second, _, statuses = self.make_job(root, "second")
            self.json(root / "comparison_plan.json", {"jobs": [
                {"id": name, "output": str(path)} for name, path in (("first", first), ("second", second))]})
            self.jsonl(second / "generation_status.jsonl", statuses[:1])
            with self.assertRaises(ValueError):
                bundle_all(root)
            self.assertFalse((first / "comparison_sheets").exists())
            self.jsonl(second / "generation_status.jsonl", statuses)
            summary = bundle_job(first)
            (first / summary["sheets"][0]["path"]).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                bundle_all(root)
            self.assertFalse((root / "index.html").exists())

    def test_pagination_preserves_all_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory, _, _ = self.make_job(root, count=11)
            summary = bundle_job(directory)
            self.assertEqual(summary["pair_count"], 11)
            self.assertEqual([sheet["path"] for sheet in summary["sheets"]], [
                "comparison_sheets/page_001.jpg", "comparison_sheets/page_002.jpg"])
            self.assertEqual(summary, verified_bundle(directory))


if __name__ == "__main__":
    unittest.main()
