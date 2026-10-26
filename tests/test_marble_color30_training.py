"""Check approval staging, frozen Subject recipes and the deferred inference matrix."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.subject_marble_color30_v1 import data, inference, run, stage_training, training, train_pipeline
from experiments.subject_multicolor_multimaterial_b import training as baseline
from experiments.subject_marble_foreground_v1 import run as foreground_run


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

    def foreground_candidates(self):
        generated = self.root / "foreground_generated"
        shutil.copytree(self.candidates.parent, generated, ignore=shutil.ignore_patterns("dataset"))
        self.candidates = generated / "dataset"
        prepared = self.candidates.parent
        plan = data.read(prepared / "plan.json")
        plan["config"].update(schema="marble_foreground_only/v1", neutral_rgb=[127, 127, 127])
        with Image.open(prepared / "source/object_mask512.png") as mask:
            for source in plan["sources"].values():
                location = prepared / source["original"]
                with Image.open(location) as image:
                    neutral = Image.composite(image, Image.new("RGB", image.size, (127, 127, 127)), mask)
                neutral.save(location)
        data.write(prepared / "config.json", plan["config"])
        data.write(prepared / "plan.json", plan)
        receipt = data.read(prepared / "prepare_receipt.json")
        receipt["output_hashes"] = {relative: data.sha256(prepared / relative) for relative in receipt["output_hashes"]}
        data.write(prepared / "prepare_receipt.json", receipt)
        generation = data.read(prepared / "generation_manifest.json")
        generation.update(status="generated35_candidates_and5_diagnostics_pending_review",
            identity_records=[{"id": color + "__no_attribute", "color": color, "controls": {}} for color in plan["sources"]])
        data.write(prepared / "generation_manifest.json", generation)
        # Fixture package into a new directory, preserving the old scene candidates.
        foreground_run.package(prepared, plan, generation["records"])

    def test_foreground_staging_excludes_diagnostics_and_keeps_previous_training_arguments(self):
        self.foreground_candidates()
        before = data.sha256(self.candidates / "manifest.json")
        audit = stage_training.stage(self.candidates, self.dataset)
        self.assertEqual(audit["source_mode"], "foreground_only")
        concepts = data.read(self.dataset / "concepts.json")
        self.assertEqual(len(concepts), 35)
        self.assertFalse(any("no_attribute" in row["instance_data_dir"] for row in concepts))
        for method in training.METHODS:
            config = training.build_config(method, self.dataset)
            self.assertEqual(config["args"], baseline.build_config(method, self.dataset)["args"])
            self.assertEqual(config["run"]["study"], "subject_marble_foreground_training_v1")
        self.assertEqual(before, data.sha256(self.candidates / "manifest.json"))
        audit["source_mode"] = "original_scene"
        data.write(self.dataset / "dataset_audit.json", audit)
        with self.assertRaisesRegex(ValueError, "source mode"):
            stage_training.verify_dataset(self.dataset)

    def test_foreground_staging_rejects_scene_pixels_even_when_candidate_hash_is_consistent(self):
        self.foreground_candidates()
        manifest = data.read(self.candidates / "manifest.json")
        row = manifest["records"][5]
        path = self.candidates / row["image"]
        with Image.open(path) as handle:
            altered = handle.copy()
        altered.putpixel((0, 0), (20, 80, 200))
        altered.save(path)
        row["image_sha256"] = data.sha256(path)
        data.write(self.candidates / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "outside its mask"):
            stage_training.stage(self.candidates, self.dataset)

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
