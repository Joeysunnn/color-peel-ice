"""Check approval staging, frozen Subject recipes and the deferred inference matrix."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.subject_marble_color30_v1 import data, inference, run, stage_training, training, train_pipeline
from experiments.subject_multicolor_multimaterial_b import training as baseline


class Marble35TrainingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        old = self.root / "old"
        mask_path = self.root / "mask.png"
        mask = Image.new("L", (512, 512))
        mask.paste(255, (100, 100, 400, 400))
        mask.save(mask_path)
        colors = []
        for number, (name, source) in enumerate(data.COLOR_MAPPING):
            image = old / "category" / source / "images/image.png"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (512, 512), (number * 30, 80, 120)).save(image)
            colors.append({"name": name, "source_color": source, "image_sha256": data.sha256(image)})
        config = {"original_root": str(old), "mask": str(mask_path), "mask_sha256": data.sha256(mask_path),
            "colors": colors, "metallic": [-20, 20], "roughness": [-1, 0, 1]}
        config_path = self.root / "config.json"
        data.write(config_path, config)
        prepared = self.root / "generated"
        plan = data.prepare(config_path, prepared)
        raw = prepared / "raw.png"
        Image.new("RGB", (1024, 1024), (120, 80, 60)).save(raw)
        records = [{**row, "raw_image": "raw.png", "raw_sha256": data.sha256(raw), "condition_hashes": {}}
                   for row in plan["rows"]]
        run.package(prepared, plan, records)
        data.write(prepared / "generation_manifest.json", {"status": "generated35_pending_user_review", "records": records})
        self.candidates, self.dataset = prepared / "dataset", self.root / "approved"

    def test_staging_preserves_all_bytes_and_does_not_rewrite_candidates(self):
        before = data.sha256(self.candidates / "manifest.json")
        self.assertEqual(stage_training.stage(self.candidates, self.dataset)["rows"], 35)
        rows = [json.loads(line) for line in (self.dataset / "training_assets_manifest.jsonl").read_text().splitlines()]
        self.assertEqual({color: sum(row["color"] == color for row in rows) for color, _ in data.COLOR_MAPPING},
                         {color: 7 for color, _ in data.COLOR_MAPPING})
        for row in rows:
            self.assertTrue(row["training_eligible"])
            for key in ("image", "mask"):
                self.assertEqual(Path(row[key]).read_bytes(), Path(row["source_" + key]).read_bytes())
        self.assertEqual(before, data.sha256(self.candidates / "manifest.json"))
        with self.assertRaises(ValueError):
            stage_training.stage(self.candidates, self.dataset)
        Path(rows[0]["image"]).write_bytes(b"changed training image")
        with self.assertRaises(ValueError):
            stage_training.verify_dataset(self.dataset)

    def test_wrong_requested_caption_is_rejected_before_staging(self):
        manifest = data.read(self.candidates / "manifest.json")
        manifest["records"][5]["caption"] = "a photo of <S*> mailbox with a wood surface"
        data.write(self.candidates / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "caption"):
            stage_training.stage(self.candidates, self.dataset)
        self.assertFalse(self.dataset.exists())

    def test_recipes_equal_experiment_b_and_jobs_keep_both_methods_and_three_steps(self):
        for method in training.METHODS:
            expected = baseline.build_config(method, self.dataset)
            actual = training.build_config(method, self.dataset)
            self.assertEqual(actual["args"], expected["args"])
            self.assertEqual(actual["args"]["checkpoint_steps"], [500, 750, 1000])
            self.assertEqual(actual["args"]["initializer_token"], "mailbox")
            self.assertEqual(actual["args"]["max_train_steps"], 1000)
            if method == "lora_r64":
                self.assertEqual((actual["args"]["subject_lora_rank"], actual["args"]["subject_lora_alpha"]), (64, 64))
        jobs = train_pipeline.commands(self.dataset, self.root / "training", 2)
        self.assertEqual([job["name"] for job in jobs], ["fullrank", "lora_r64", "step500", "step750", "step1000"])
        self.assertTrue(all(job["command"][3].startswith("experiments.subject_marble_color30_v1.") for job in jobs))

    def test_inference_dry_run_keeps_all35_actual_references_and_twenty_prompts(self):
        stage_training.stage(self.candidates, self.dataset)
        output = self.root / "inference"
        with patch.object(inference, "git_info", return_value={"commit": "test"}):
            result = inference.run(self.dataset, output, 750)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["generated_images"], 0)
        self.assertEqual(result["training_images"], 35)
        rows = [json.loads(line) for line in (output / "generation_manifest.jsonl").read_text().splitlines()]
        self.assertEqual(rows, inference.prompt_rows(750))
        self.assertEqual(len(rows), 20)
        self.assertEqual({row["seed"] for row in rows}, {42, 43})
        with Image.open(output / "training_contact_sheet.jpg") as sheet:
            self.assertGreater(sheet.height, 2000)


if __name__ == "__main__":
    unittest.main()
