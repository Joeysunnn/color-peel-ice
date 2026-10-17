"""CPU fixtures exercise complete evaluation, missing data and integrity guards."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from experiments.texture_material_disentangled_v1 import data, evaluate, infer


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class TextureMaterialEvaluationTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        inference, renders = root / "inference", root / "renders"
        (inference / "images").mkdir(parents=True)
        renders.mkdir()
        protocol = data.read_protocol()
        rows = infer.build_rows(protocol)
        generated = []
        for index, row in enumerate(rows):
            image = inference / row["image_path"]
            Image.new("RGB", (8, 8), (index % 255, 120, 160)).save(image)
            generated.append({**row, "image_sha256": data.sha256(image),
                              "status": "safety_filtered" if index == 0 else "ok"})
        write_rows(inference / "generation_manifest.jsonl", rows)
        write_rows(inference / "generation_status.jsonl", generated)
        data.write_json(inference / "provenance.json", {
            "status": "succeeded", "protocol_sha256": data.protocol_sha256(),
            "sampling": protocol["inference"], "row_count": 256, "git": {"commit": "a" * 40}})
        data.write_json(inference / "summary.json", {
            "status": "succeeded", "expected_images": 256, "generated_images": 256,
            "missing_images": 0, "safety_filtered": 1, "by_method": {"fullrank": 128, "lora_r64": 128}})
        oracle = []
        for row in data._records(protocol):
            folder = renders / row["id"]
            folder.mkdir()
            image, mask = folder / "image.png", folder / "mask.png"
            Image.new("RGB", (8, 8), (120, 80, 40)).save(image)
            Image.new("L", (8, 8), 255).save(mask)
            material = protocol["materials"][row["material_id"]]
            oracle.append({**row, "image": str(image), "mask": str(mask),
                           "image_sha256": data.sha256(image), "mask_sha256": data.sha256(mask),
                           "roughness": material["roughness"], "metallic": material["metallic"],
                           "texture_projection": "UV", "texture_colorspace": "Non-Color"})
        data.write_json(renders / "manifest.json", {"status": "complete",
                        "protocol_sha256": data.protocol_sha256(), "records": oracle})
        return inference, renders, generated

    def test_complete_matrix_creates_sixteen_oracle_pages_without_quality_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            inference, renders, _ = self.fixture(directory)
            output = Path(directory) / "evaluation"
            report = evaluate.evaluate(inference, renders, output)
            self.assertEqual(report["scientific_status"], "pending_blinded_review")
            self.assertEqual(report["safety_filtered"], 1)
            self.assertEqual(report["generated_images"], 256)
            self.assertIsNone(report["brdf_accuracy"])
            self.assertEqual(len(report["comparison_pages"]), 16)
            self.assertEqual(len(list((output / "comparison_pages").glob("*.jpg"))), 16)
            self.assertEqual(sum(group["generated"] for group in report["by_method_context_condition"]), 256)
            self.assertEqual(sum(group["valid_observations"] for group in report["by_method_context_condition"]), 255)
            with self.assertRaises(FileExistsError):
                evaluate.evaluate(inference, renders, output)

    def test_partial_duplicated_or_changed_condition_ledgers_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            inference, renders, generated = self.fixture(directory)
            altered = [*generated[:-1], {**generated[-1], "prompt": "changed prompt"}]
            for rows in (generated[:-1], [*generated[:-1], generated[0]], altered):
                write_rows(inference / "generation_status.jsonl", rows)
                with self.assertRaisesRegex(ValueError, "Generation ledger"):
                    evaluate.validate(inference, renders)

    def test_safety_count_and_generated_or_oracle_hash_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            inference, renders, generated = self.fixture(directory)
            summary_path = inference / "summary.json"
            summary = evaluate.read_json(summary_path)
            data.write_json(summary_path, {**summary, "safety_filtered": 0})
            with self.assertRaisesRegex(ValueError, "summary counts"):
                evaluate.validate(inference, renders)
            data.write_json(summary_path, summary)
            image = inference / generated[0]["image_path"]
            original = image.read_bytes()
            image.write_bytes(b"changed PNG")
            with self.assertRaisesRegex(ValueError, "image hash"):
                evaluate.validate(inference, renders)
            image.write_bytes(original)
            oracle = evaluate.read_json(renders / "manifest.json")["records"]
            Path(oracle[0]["mask"]).write_bytes(b"changed mask")
            with self.assertRaisesRegex(ValueError, "Oracle mask hash"):
                evaluate.validate(inference, renders)


if __name__ == "__main__":
    unittest.main()
