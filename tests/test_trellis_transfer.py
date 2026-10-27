"""CPU checks for the blue/green probes and legacy material-only behavior."""

import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.subject_marble_color30_v1 import inference as shared
from experiments.subject_trellis_dataset_v1 import inference, training
from experiments.sd15_token_local_kv_v1.contract import sha256


class TrellisTransferTest(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parents[1]
        self.temporary = tempfile.TemporaryDirectory(prefix="test-trellis-transfer-", dir=workspace)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        if not self.root.resolve().is_relative_to(workspace):
            raise ValueError("test cleanup must remain in the workspace")
        root_patch = patch.object(training, "ROOT", self.root / "project")
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.rgb, self.mask = self.root / "actual_rgb.png", self.root / "actual_mask.png"
        Image.new("RGB", (512, 512), (70, 150, 180)).save(self.rgb)
        Image.new("L", (512, 512), 255).save(self.mask)

    def prepare(self, count):
        audit = {"rows": count, "source_mode": "synthetic_native_geometry",
                 "schema": "trellis_subject_blue_green72_training/v1" if count == 72 else "trellis_subject180_training/v1",
                 "selected_colors": ["blue", "green"] if count == 72 else ["pink", "green", "cyan", "blue", "purple"]}
        references = [{"id": f"actual_{i:03d}", "kind": "synthetic_material_counterfactual",
            "image": str(self.rgb), "mask": str(self.mask), "image_sha256": sha256(self.rgb),
            "mask_sha256": sha256(self.mask),
            "caption": f"a photo of <S*> mailbox in blue color with a rough metallic surface {i}"}
            for i in range(count)]
        (self.dataset / "training_assets_manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in references), encoding="utf8")
        for name, value in (("concepts.json", []), ("source_locks.json", {}), ("dataset_audit.json", audit)):
            (self.dataset / name).write_text(json.dumps(value), encoding="utf8")
        return audit, references

    def test_four_independent_prompts_keep_legacy_material_rows(self):
        suffixes = {"in the snow": "context_transfer", "surrounded by flowers": "context_transfer",
                    "in white color": "color_transfer", "in orange color": "color_transfer"}
        for step in inference.STEPS:
            rows = inference.blue_green_prompt_rows(step)
            self.assertEqual(rows[:20], shared.prompt_rows(step))
            self.assertEqual(len(rows), 36)
            self.assertEqual(len({row["id"] for row in rows}), 36)
            self.assertEqual(len({row["image_path"] for row in rows}), 36)
            self.assertEqual({row["step"] for row in rows}, {step})
            extra = rows[20:]
            self.assertTrue(all("material" not in row for row in extra))
            for condition, test_type in suffixes.items():
                selected = [row for row in extra if row["condition"] == condition]
                self.assertEqual({row["prompt"] for row in selected}, {f"a photo of <S*> mailbox {condition}"})
                self.assertEqual({row["test_type"] for row in selected}, {test_type})
                self.assertEqual({(row["method"], row["seed"]) for row in selected},
                                 {(method, seed) for method in inference.METHODS for seed in (42, 43)})

    def test_legacy_180_remains_twenty_images(self):
        audit, _ = self.prepare(180)
        self.assertEqual(inference.expected_images(self.dataset), 20)
        for step in inference.STEPS:
            self.assertEqual(inference.prompt_rows(step), shared.prompt_rows(step))
            self.assertEqual(len(inference.prompt_rows(step)), 20)
        with patch.object(training, "verify_dataset", return_value=audit), patch.object(shared, "git_info", return_value={"commit": "cpu-test"}):
            result = inference.run(self.dataset, self.root / "legacy", 500)
        self.assertEqual(result["expected_images"], 20)
        self.assertEqual(result["training_images"], 180)
        with Image.open(self.root / "legacy" / result["contact_sheets"][0]) as sheet:
            self.assertEqual(sheet.width, 1280)

    def test_subset_changes_training_metadata_only(self):
        self.prepare(180)
        previous = {method: training.build_config(method, self.dataset) for method in training.METHODS}
        self.prepare(72)
        for method in training.METHODS:
            config = training.build_config(method, self.dataset)
            self.assertEqual(config["run"]["study"], "subject_trellis_blue_green72_training_v1")
            self.assertEqual(config["status"], "subject_trellis_blue_green72_prepared")
            before, after = previous[method]["args"].copy(), config["args"].copy()
            before.pop("concepts_list")
            after.pop("concepts_list")
            self.assertEqual(before, after)
            self.assertEqual(config["args"]["max_train_steps"], 1000)
            self.assertEqual(config["args"]["checkpoint_steps"], [500, 750, 1000])

    def test_blue_green_dry_run_has_72_references_and_36_probes(self):
        audit, references = self.prepare(72)
        self.assertEqual(inference.expected_images(self.dataset), 36)
        with patch.object(training, "verify_dataset", return_value=audit) as verify, patch.object(shared, "git_info", return_value={"commit": "cpu-test"}):
            result = inference.run(self.dataset, self.root / "subset", 750)
        self.assertEqual(verify.call_count, 2)
        self.assertEqual((result["status"], result["expected_images"], result["generated_images"]), ("dry_run", 36, 0))
        self.assertEqual(result["training_images"], 72)
        self.assertEqual(result["experiment"], "subject_trellis_blue_green72_training_v1")
        self.assertIn("four declared independent context/color probes", result["baseline"])
        self.assertEqual(len(result["contact_sheets"]), 1)
        root = self.root / "subset"
        self.assertEqual(json.loads((root / "training_references.json").read_text())["images"], references)
        self.assertEqual(len((root / "generation_manifest.jsonl").read_text().splitlines()), 36)
        self.assertIn("TRELLIS.2 blue/green72", (root / "index.html").read_text())
        with Image.open(root / result["contact_sheets"][0]) as sheet:
            self.assertEqual(sheet.width, 9 * 256)

    def test_executed_hook_requires_36_outputs_and_three_sheets(self):
        audit, _ = self.prepare(72)
        index = self.root / "training_index.json"
        index.write_text("{}")
        selections = [{"method": method, "run": str(self.root / "training" / method)}
                      for method in inference.METHODS]
        generated_prompts = []

        class Generator:
            def __init__(self, device):
                self.device = device

            def manual_seed(self, seed):
                self.seed = seed
                return self

        def pipe(prompt, **kwargs):
            generated_prompts.append((prompt, kwargs["generator"].seed))
            self.assertEqual((kwargs["num_inference_steps"], kwargs["guidance_scale"]), (100, 3.5))
            return types.SimpleNamespace(images=[Image.new("RGB", (512, 512), "gray")], nsfw_content_detected=[False])

        fake_torch = types.SimpleNamespace(Generator=Generator, cuda=types.SimpleNamespace(empty_cache=lambda: None))
        with patch.object(training, "verify_dataset", return_value=audit), patch.object(shared, "git_info", return_value={"commit": "cpu-test"}), \
             patch.object(shared, "_selections", return_value=selections), \
             patch.object(shared, "export_subject_step", return_value={"hashes": {str(self.rgb): sha256(self.rgb)}}) as export, \
             patch.object(shared.evaluate, "load_pipeline", return_value=pipe), patch.object(shared.evaluate, "attention_masks", return_value={}), \
             patch.dict("sys.modules", {"torch": fake_torch}):
            result = inference.run(self.dataset, self.root / "executed", 1000, index, True)
        self.assertEqual((result["status"], result["expected_images"], result["generated_images"]), ("succeeded", 36, 36))
        self.assertEqual(len(generated_prompts), 36)
        self.assertEqual(len(result["contact_sheets"]), 3)
        self.assertEqual([call.args[1] for call in export.call_args_list], [1000, 1000])


if __name__ == "__main__":
    unittest.main()
