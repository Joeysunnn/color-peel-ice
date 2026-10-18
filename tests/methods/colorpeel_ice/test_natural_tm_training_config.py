import copy
import json
from pathlib import Path
import unittest

from experiments.natural_texture_material_v1 import dataset_protocol as schema, training_config
from experiments.texture_material_disentangled_v1 import step_ablation


class NaturalTrainingConfigTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(schema.PARENT.read_text(encoding="utf8"))
        self.protocol.update(study="natural_texture_material_v1", phase=2)
        self.protocol["training"].update(steps=1000, checkpoint_steps=[500, 750, 1000])
        self.staging = Path("natural_staging").resolve()

    def test_twelve_independent_recipes_share_only_the_corresponding_data(self):
        recipes = training_config.recipes()
        self.assertEqual(len(recipes), 12)
        self.assertEqual({(item["source_id"], item["branch"], item["method"]) for item in recipes},
            {(source, branch, method) for source in schema.SOURCES
             for branch in ("texture", "material") for method in ("fullrank", "lora_r64")})
        for item in recipes:
            with self.subTest(recipe=item):
                self.assertEqual(item["concept_id"], item["source_id"] + "_" + item["branch"])
                config = training_config.build_config(item, self.staging, self.protocol)
                comparison = training_config.compare_config(item, config, self.staging, self.protocol)
                baseline = step_ablation.build_config(comparison["phase1_template_recipe"], self.staging)
                expected_args = copy.deepcopy(baseline["args"])
                expected_args.update(concepts_list=str(self.staging / item["concept_id"] / "concepts.json"),
                                     checkpoint_steps=[500, 750, 1000])
                self.assertEqual(config["args"], expected_args)
                self.assertEqual(set(comparison["argument_differences_at_same_staging"]),
                                 {"concepts_list", "checkpoint_steps"})
                self.assertEqual(config["environment"], baseline["environment"])
                self.assertEqual(config["run"], {"study": "natural_texture_material_v1",
                    "variant": f"{item['concept_id']}_{item['method']}_steps1000", "seed": 42})
                self.assertEqual(config["data_manifest"], str(self.staging / item["concept_id"] /
                    "training_assets_manifest.jsonl"))
                self.assertEqual(config["texture_material"], {**item, "mode": "pilot",
                    "protocol_sha256": schema.protocol_sha256(self.protocol)})

    def test_original_training_and_adapter_settings_are_preserved(self):
        for item in training_config.recipes():
            args = training_config.build_config(item, self.staging, self.protocol)["args"]
            self.assertEqual(args["modifier_token"], "<T*>" if item["branch"] == "texture" else "<M*>")
            self.assertEqual(args["initializer_token"], "texture" if item["branch"] == "texture" else "metal")
            for name, value in {"seed": 42, "resolution": 512, "train_batch_size": 1,
                    "gradient_accumulation_steps": 1, "learning_rate": 1e-5,
                    "max_train_steps": 1000, "checkpointing_steps": 300,
                    "mixed_precision": "no", "hflip": False, "cos_weight": 0.0,
                    "freeze_model": "crossattn_kv", "lr_scheduler": "constant"}.items():
                self.assertEqual(args[name], value)
            self.assertNotIn("resume_from_checkpoint", args)
            if item["method"] == "fullrank":
                self.assertTrue(args["token_local_kv"])
                self.assertEqual(args["kv_learning_rate"], 1e-5)
                self.assertNotIn("subject_lora_mode", args)
            else:
                self.assertNotIn("token_local_kv", args)
                self.assertEqual(args["subject_lora_mode"], "token_local_kv")
                self.assertEqual(args["kv_learning_rate"], 5e-5)
                self.assertEqual((args["subject_lora_rank"], args["subject_lora_alpha"]), (64, 64))

    def test_unknown_or_misassigned_recipe_is_rejected(self):
        recipe = training_config.recipes()[0]
        for update in ({"source_id": "unknown"}, {"branch": "subject"}, {"token": "<M*>"},
                       {"method": "lora_r32"}, {"concept_id": "vertical_stripes"}):
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, "twelve natural"):
                training_config.build_config({**recipe, **update}, self.staging, self.protocol)

    def test_argument_and_provenance_drift_is_rejected(self):
        recipe = training_config.recipes()[0]
        config = training_config.build_config(recipe, self.staging, self.protocol)
        for name, value in {"learning_rate": 2e-5, "seed": 43, "checkpoint_steps": [500, 1000],
                            "concepts_list": "old/concepts.json", "cos_weight": 0.1}.items():
            modified = copy.deepcopy(config)
            modified["args"][name] = value
            with self.subTest(argument=name), self.assertRaisesRegex(ValueError, "frozen recipe"):
                training_config.compare_config(recipe, modified, self.staging, self.protocol)
        modified = copy.deepcopy(config)
        modified["texture_material"]["concept_id"] = "vertical_stripes"
        with self.assertRaisesRegex(ValueError, "frozen recipe"):
            training_config.compare_config(recipe, modified, self.staging, self.protocol)


if __name__ == "__main__":
    unittest.main()
