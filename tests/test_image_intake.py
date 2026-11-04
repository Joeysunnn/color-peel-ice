"""Exercise per-instance orchestration without models, GPUs or training datasets."""

import inspect
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from src.methods.colorpeel_ice import image_intake as intake
from src.methods.colorpeel_ice import vlm_initialization as vlm


class IntakeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.image = self.root / "photo.png"
        Image.new("RGB", (16, 12), "white").save(self.image)
        self.output = self.root / "intake"
        self.args = SimpleNamespace(image=self.image, output=self.output, gpu=2,
            qwen_model_dir=self.root / "qwen", sd_model_dir=self.root / "sd21",
            vlm_python=self.root / "vlm-python", sam_python=self.root / "sam-python")
        self.calls = []
        self.segmentation_status = "succeeded"
        self.segment_exit = None
        self.failed_initializers = set()
        self.discovery_hook = self.segmentation_hook = self.initializer_hook = None
        self.words = {
            "object_001": {"subject": "chair", "color": "red", "material": "matte", "texture": "striped"},
            "object_002": {"subject": "chair", "color": "blue", "material": "glossy", "texture": "uniform"},
            "object_003": {"subject": "mug", "color": "green", "material": "material", "texture": "spotted"},
        }

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        intake.write_json(path, value)

    def fake_stage(self, command, log, gpu):
        self.calls.append(command)
        self.assertEqual(gpu, 2)
        destination = Path(command[command.index("--output") + 1])
        input_image = Path(command[command.index("--image") + 1])
        destination.mkdir(parents=True)
        Path(log).write_text("mock stage\n", encoding="utf8")
        if "--stage" in command:
            stage = command[command.index("--stage") + 1]
            self.assertIn(stage, ("discover", "segment"))
            self.assertEqual(input_image, self.image)
            source = {"image": str(self.image), "image_sha256": intake.sha256(self.image)}
            canonical = destination / "source.png"
            Image.open(self.image).save(canonical)
            source.update(canonical_image=str(canonical), canonical_image_sha256=intake.sha256(canonical))
            if stage == "discover":
                manifest = {"schema": "colorpeel_object_categories/v1", "status": "verified",
                    "source": source, "categories": ["chair", "mug"],
                    "source_locks": {str(self.image): source["image_sha256"],
                                     str(canonical): source["canonical_image_sha256"]}}
                if self.discovery_hook:
                    self.discovery_hook(manifest)
                self.write(destination / "categories.json", manifest)
                return
            objects = []
            for index, (object_id, words) in enumerate(self.words.items()):
                folder = destination / "objects" / object_id
                folder.mkdir(parents=True)
                row = {"id": object_id, "category": words["subject"], "source_image": str(canonical)}
                for name, mode, size, color in (
                    ("object_mask", "L", (16, 12), 255),
                    ("cutout", "RGBA", (16, 12), (10 + index, 20, 30, 255)),
                    ("overlay", "RGB", (16, 12), (10 + index, 20, 30)),
                    ("vlm_input", "RGB", (4 + index, 6), (60 * index, 40, 80)),
                ):
                    path = folder / (name + ".png")
                    Image.new(mode, size, color).save(path)
                    row[name], row[name + "_sha256"] = str(path), intake.sha256(path)
                objects.append(row)
            manifest = {"schema": "colorpeel_object_segmentation/v1", "status": self.segmentation_status,
                        "source": source, "objects": objects, "categories": ["chair", "mug"],
                        "failures": [], "rejected": []}
            if self.segmentation_hook:
                self.segmentation_hook(manifest)
            self.write(destination / "objects.json", manifest)
            self.write(destination / "status.json", {"status": manifest["status"]})
            if self.segment_exit:
                raise subprocess.CalledProcessError(self.segment_exit, command)
            return
        self.assertEqual(Path(command[2]).name, "initialize_from_image.py")
        object_id = destination.name
        if object_id in self.failed_initializers:
            raise subprocess.CalledProcessError(1, command)
        manifest = {"schema": "colorpeel_vlm_initializers/v2", "status": "verified",
                    "source": {"image": str(input_image), "image_sha256": intake.sha256(input_image)},
                    "source_locks": {str(input_image): intake.sha256(input_image)},
                    "branches": {branch: {"initializer": word} for branch, word in self.words[object_id].items()}}
        if self.initializer_hook:
            self.initializer_hook(object_id, manifest)
        self.write(destination / "initializers.json", manifest)

    def invoke(self):
        with patch.object(intake, "run_stage", side_effect=self.fake_stage), \
                patch("scripts.launch.colorpeel_run.git_info", return_value={"commit": "a" * 40, "dirty": False}), \
                patch("scripts.launch.colorpeel_run.resolve_run_dir", side_effect=lambda path: Path(path).resolve()):
            path = intake.run(self.args)
        return json.loads(path.read_text(encoding="utf8"))

    def test_every_instance_uses_its_own_crop_and_four_initializers_without_training(self):
        result = self.invoke()
        self.assertEqual(result["status"], "ready_for_derived_data")
        self.assertFalse(result["training_started"])
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(len(result["objects"]), 3)
        seen_inputs = []
        for row, command in zip(result["objects"], self.calls[2:]):
            source = command[command.index("--image") + 1]
            seen_inputs.append(source)
            self.assertEqual(source, row["segmentation"]["vlm_input"])
            self.assertNotEqual(source, str(self.image))
            self.assertEqual(row["initializers"], self.words[row["object_id"]])
            self.assertEqual(row["status"], "ready_for_derived_data")
            for artifact in ("object_mask", "cutout", "vlm_input"):
                path = row["segmentation"][artifact]
                self.assertEqual(result["source_locks"][path], intake.sha256(path))
        self.assertEqual(len(set(seen_inputs)), 3)
        self.assertNotEqual(result["objects"][0]["initializers"], result["objects"][1]["initializers"])

    def test_partial_segmentation_retains_ready_instances_but_never_reports_all_ready(self):
        self.segmentation_status, self.segment_exit = "partial", 1
        result = self.invoke()
        self.assertEqual(result["status"], "partial")
        self.assertTrue(all(row["status"] == "ready_for_derived_data" for row in result["objects"]))
        self.assertEqual(len(self.calls), 5)

    def test_segmentation_failure_is_not_recovered_as_partial(self):
        self.segmentation_status, self.segment_exit = "failed", 1
        with self.assertRaises((ValueError, subprocess.CalledProcessError)):
            self.invoke()
        state = intake.read_json(self.output / "intake.json")
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["objects"], [])
        self.assertEqual(len(self.calls), 2)

    def test_one_failed_initializer_does_not_reuse_another_or_skip_remaining_instances(self):
        self.failed_initializers = {"object_002"}
        result = self.invoke()
        self.assertEqual(result["status"], "partial")
        self.assertEqual([row["status"] for row in result["objects"]],
                         ["ready_for_derived_data", "failed", "ready_for_derived_data"])
        self.assertNotIn("initializers", result["objects"][1])
        self.assertEqual(result["objects"][2]["initializers"], self.words["object_003"])

    def test_no_successful_initializer_yields_failed_not_partial(self):
        self.failed_initializers = set(self.words)
        result = self.invoke()
        self.assertEqual(result["status"], "failed")
        self.assertTrue(all(row["status"] == "failed" for row in result["objects"]))

    def test_discovery_or_segmentation_from_another_image_is_rejected(self):
        for stage in ("discovery", "segmentation"):
            self.output = self.root / stage
            self.args.output = self.output
            self.calls.clear()
            self.discovery_hook = self.segmentation_hook = None
            setattr(self, stage + "_hook", lambda manifest: manifest["source"].update(image_sha256="0" * 64))
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                self.invoke()
            self.assertEqual(intake.read_json(self.output / "intake.json")["status"], "failed")
            self.assertEqual(len(self.calls), 1 if stage == "discovery" else 2)

    def test_corrupt_cutout_cannot_be_ready_for_derived_data(self):
        self.segmentation_hook = lambda manifest: Path(manifest["objects"][0]["cutout"]).write_bytes(b"changed")
        result = self.invoke()
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["objects"][0]["status"], "failed")

    def test_missing_branch_cannot_become_a_four_branch_ready_instance(self):
        def missing_branch(object_id, manifest):
            if object_id == "object_001":
                del manifest["branches"]["texture"]
        self.initializer_hook = missing_branch
        result = self.invoke()
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["objects"][0]["status"], "failed")

    def test_later_receipt_cannot_replace_original_hash_with_mutated_source_hash(self):
        def replace_source_lock(manifest):
            Image.new("RGB", (16, 12), "black").save(self.image)
            manifest["source_locks"][str(self.image)] = intake.sha256(self.image)
        self.discovery_hook = replace_source_lock
        with self.assertRaises(ValueError):
            self.invoke()
        self.assertEqual(intake.read_json(self.output / "intake.json")["status"], "failed")

    def test_source_mutation_during_initialization_fails_final_lock_check(self):
        def mutate_original(object_id, manifest):
            if object_id == "object_003":
                Image.new("RGB", (16, 12), "black").save(self.image)
        self.initializer_hook = mutate_original
        with self.assertRaisesRegex(ValueError, "Source changed"):
            self.invoke()
        self.assertEqual(intake.read_json(self.output / "intake.json")["status"], "failed")

    def test_cli_exit_code_distinguishes_partial_and_failed_from_ready(self):
        path = self.root / "result.json"
        for status, expected in (("partial", 1), ("failed", 1), ("ready_for_derived_data", 0)):
            self.write(path, {"status": status})
            with self.subTest(status=status), patch.object(intake, "run", return_value=path):
                self.assertEqual(intake.main(["--image", str(self.image), "--output", str(self.output)]), expected)

    def test_old_single_object_qwen_prompt_remains_the_default(self):
        parameter = inspect.signature(vlm.QwenPredictor.__init__).parameters["prompt"]
        self.assertIs(parameter.default, vlm.PROMPT)
        self.assertIn("Identify the main foreground object", parameter.default)
        self.assertIn('"metallicity"', parameter.default)
        self.assertNotIn('"categories"', parameter.default)


if __name__ == "__main__":
    unittest.main()
