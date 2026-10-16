import copy
import json
from pathlib import Path
import unittest
import tempfile

from experiments.sd15_token_local_kv_v1.contract import (
    PROTOCOL, ROOT, build_config, read_json, validate_config, verify_directory_members,
)
from scripts.launch.colorpeel_run import read_config


class SD15MigrationTests(unittest.TestCase):
    def test_full_recipes_change_only_the_base_from_frozen_sd14(self):
        protocol = read_json(PROTOCOL)
        self.assertEqual(len(protocol["recipes"]), 6)
        for name, recipe in protocol["recipes"].items():
            with self.subTest(recipe=name):
                config = read_json(ROOT / "configs" / (name + ".json"))
                validate_config(config, {}, verify_assets=False)
                restored = copy.deepcopy(config["args"])
                restored["pretrained_model_name_or_path"] = "CompVis/stable-diffusion-v1-4"
                self.assertEqual(restored, recipe["source_config"]["args"])
                self.assertEqual(config["data_manifest"], recipe["source_config"]["data_manifest"])
                # The server launcher parses JSON through PyYAML when installed.
                self.assertEqual(read_config(ROOT / "configs" / (name + ".json")), config)

    def test_methods_remain_distinct_and_rank64_keeps_selected_dose(self):
        original = build_config("subject_fullrank")["args"]
        lora = build_config("subject_lora_r64")["args"]
        self.assertTrue(original["token_local_kv"])
        self.assertEqual(original["max_train_steps"], 2000)
        self.assertNotIn("subject_lora_rank", original)
        self.assertEqual((lora["subject_lora_rank"], lora["subject_lora_alpha"]), (64, 64))
        self.assertEqual(lora["kv_learning_rate"], 5e-5)
        self.assertEqual(lora["max_train_steps"], 3000)

    def test_smoke_changes_only_training_dose_and_checkpoint_plan(self):
        for name in read_json(PROTOCOL)["recipes"]:
            full = build_config(name)["args"]
            smoke = read_json(ROOT / "configs" / (name + "_smoke2.json"))
            validate_config(smoke, {}, verify_assets=False)
            expected = {**full, "max_train_steps": 2, "checkpoint_steps": [2]}
            self.assertEqual(smoke["args"], expected)

    def test_rejects_base_recipe_rank_data_and_protocol_drift(self):
        base = build_config("subject_lora_r64")
        for field, value in (("pretrained_model_name_or_path", "CompVis/stable-diffusion-v1-4"),
                             ("subject_lora_rank", 4), ("concepts_list", "/wrong"),
                             ("kv_learning_rate", 1e-5), ("initializer_token", "gorilla")):
            changed = copy.deepcopy(base)
            changed["args"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_config(changed, {}, verify_assets=False)
        changed = copy.deepcopy(base)
        changed["sd15_migration"]["protocol_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_config(changed, {}, verify_assets=False)

    def test_existing_model_and_experiment_paths_are_not_outputs(self):
        model = read_json(PROTOCOL)["model"]
        self.assertNotEqual(model["root"], model["source_root"])
        self.assertNotEqual(model["root"], model["checker_root"])
        self.assertIn("sd15_token_local_kv_v1/assets", model["root"])

    def test_unmanifested_images_cannot_silently_expand_training_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, masks = root / "images", root / "masks"
            images.mkdir()
            masks.mkdir()
            image, mask = images / "one.png", masks / "one.png"
            image.touch()
            mask.touch()
            concepts = [{"instance_data_dir": str(images), "instance_mask_dir": str(masks)}]
            records = [{"image": str(image), "mask": str(mask)}]
            verify_directory_members(concepts, records)
            (images / "extra.png").touch()
            (masks / "extra.png").touch()
            with self.assertRaisesRegex(ValueError, "unmanifested"):
                verify_directory_members(concepts, records)

    def test_evaluator_rejects_a_successful_legacy_sd14_run(self):
        from experiments.sd15_token_local_kv_v1.evaluate import verify_run
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            source = read_json(PROTOCOL)["recipes"]["subject_fullrank"]["source_config"]
            (run / "config.yaml").write_text(json.dumps(source), encoding="utf-8")
            (run / "manifest.json").write_text(json.dumps({
                "status": "succeeded", "returncode": 0, "stage": "train", "run": source["run"]}),
                encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_run(run, "<S*>")


if __name__ == "__main__":
    unittest.main()
