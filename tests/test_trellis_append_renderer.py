"""CPU checks for append-only image placement and explicit five-seed reports."""

import copy
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from experiments.subject_marble_color30_v1 import inference as renderer
from experiments.subject_trellis_dataset_v1 import prompt_forms
from experiments.sd15_token_local_kv_v1.contract import sha256

SEEDS = (44, 45, 46, 47, 48)


class AppendRendererTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="trellis-append-renderer-", dir=os.environ.get("TMP"))
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.dataset = self.root / "training_dataset"
        self.dataset.mkdir()
        self.rgb, self.mask = self.root / "actual_rgb.png", self.root / "actual_mask.png"
        Image.new("RGB", (512, 512), (50, 150, 170)).save(self.rgb)
        Image.new("L", (512, 512), 255).save(self.mask)
        self.references = [{"id": f"actual_{i:03d}", "kind": "synthetic_material_counterfactual",
            "image": str(self.rgb), "mask": str(self.mask), "image_sha256": sha256(self.rgb),
            "mask_sha256": sha256(self.mask), "caption": f"actual training mailbox caption {i}"}
            for i in range(72)]
        (self.dataset / "training_assets_manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.references), encoding="utf8")
        for name, value in (("concepts.json", []), ("source_locks.json", {}),
                            ("dataset_audit.json", {"rows": 72, "selected_colors": ["green", "blue"]})):
            (self.dataset / name).write_text(json.dumps(value), encoding="utf8")
        self.index = self.root / "training_index.json"
        self.index.write_text("{}")
        self.old_step = self.root / "experiment" / "step500"
        self.old_step.mkdir(parents=True)
        self.old_png = self.old_step / "images/fullrank/wood__adjective__seed42.png"
        self.old_png.parent.mkdir(parents=True)
        Image.new("RGB", (512, 512), "orange").save(self.old_png)
        self.old_sheet = self.old_step / "fullrank_contact_sheet.jpg"
        self.old_sheet.write_bytes(b"old sheet sentinel")
        self.output = self.root / "experiment/extensions/new_append/step500"
        self.training = types.SimpleNamespace(ROOT=Path(__file__).resolve().parents[1], verify_dataset=Mock())
        self.selections = [{"method": method, "run": str(self.root / "old_training" / method)}
                           for method in prompt_forms.METHODS]

    @staticmethod
    def probes(step):
        prototypes = [row for row in prompt_forms.prompt_rows(step) if row["seed"] == 42]
        return [{**row, "id": row["id"].replace("__seed42", f"__seed{seed}"), "seed": seed,
                 "image_path": row["image_path"].replace("__seed42.png", f"__seed{seed}.png")}
                for row in prototypes for seed in SEEDS]

    def run_common(self, **kwargs):
        options = dict(training_module=self.training, reference_count=72,
            reference_label="TRELLIS.2 seeds44_48", probe_rows=self.probes,
            probe_labels=prompt_forms.LABELS, selection_loader=Mock(return_value=self.selections),
            sampling_seeds=SEEDS, image_root=self.old_step)
        options.update(kwargs)
        return renderer.run(self.dataset, self.output, 500, self.index,
                            options.pop("execute", True), **options)

    @staticmethod
    def fake_torch():
        class Generator:
            def __init__(self, device):
                self.device = device

            def manual_seed(self, seed):
                self.seed = seed
                return self

        return types.SimpleNamespace(Generator=Generator, cuda=types.SimpleNamespace(empty_cache=lambda: None))

    def test_default_two_seed_sheet_matches_explicit_two_seed_pixels(self):
        default, explicit = self.root / "default", self.root / "explicit"
        default.mkdir()
        explicit.mkdir()
        rows = [{**row, "status": "ok"} for row in renderer.prompt_rows(500) if row["method"] == "fullrank"]
        for output in (default, explicit):
            for row in rows:
                path = output / row["image_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (512, 512), (row["seed"], 80, 100)).save(path)
        name = renderer._contact_sheet(default, self.references[:14], rows, "fullrank")
        renderer._contact_sheet(explicit, self.references[:14], rows, "fullrank", seeds=(42, 43))
        self.assertEqual((default / name).read_bytes(), (explicit / name).read_bytes())
        with Image.open(default / name) as sheet:
            self.assertEqual(sheet.width, 1280)

    def test_five_seed_run_publishes_150_images_and_keeps_old_artifacts(self):
        global_sampling = copy.deepcopy(renderer.SAMPLING)
        old_hashes = {path: sha256(path) for path in (self.old_png, self.old_sheet)}
        generated = []

        def pipe(prompt, **kwargs):
            seed = kwargs["generator"].seed
            generated.append(seed)
            self.assertEqual((kwargs["num_inference_steps"], kwargs["guidance_scale"]), (100, 3.5))
            return types.SimpleNamespace(images=[Image.new("RGB", (512, 512), (seed, 80, 100))], nsfw_content_detected=[False])

        with patch.object(renderer, "git_info", return_value={"commit": "append-only"}), \
             patch.object(renderer, "_selections", side_effect=AssertionError("explicit source loader required")), \
             patch.object(renderer, "export_subject_step", return_value={"hashes": {str(self.rgb): sha256(self.rgb)}}), \
             patch.object(renderer.evaluate, "load_pipeline", return_value=pipe), \
             patch.object(renderer.evaluate, "attention_masks", return_value={}), \
             patch.dict("sys.modules", {"torch": self.fake_torch()}):
            result = self.run_common()
        self.assertEqual((result["status"], result["expected_images"], result["generated_images"]), ("succeeded", 150, 150))
        self.assertEqual(result["sampling"]["seeds"], list(SEEDS))
        self.assertEqual(renderer.SAMPLING, global_sampling)
        self.assertEqual(result["image_root"], str(self.old_step.resolve()))
        self.assertEqual(len(generated), 150)
        self.assertEqual(set(generated), set(SEEDS))
        self.assertEqual({path: sha256(path) for path in old_hashes}, old_hashes)
        self.assertEqual(len(result["contact_sheets"]), 3)
        self.assertTrue((self.output / "provenance.json").is_file())
        self.assertFalse((self.output / "images").exists())
        ledger = [json.loads(line) for line in (self.output / "generation_status.jsonl").read_text().splitlines()]
        self.assertEqual(len(ledger), 150)
        for row in ledger:
            self.assertFalse(Path(row["image_path"]).is_absolute())
            self.assertEqual(sha256(self.old_step / row["image_path"]), row["image_sha256"])
        with Image.open(self.output / "training_contact_sheet.jpg") as training_sheet, \
             Image.open(self.output / "fullrank_contact_sheet.jpg") as method_sheet:
            self.assertEqual(method_sheet.height - training_sheet.height, 82 + 315 * 5)
            for number, seed in enumerate(SEEDS):
                pixel = method_sheet.getpixel((128, training_sheet.height + 60 + number * 315 + 128))
                self.assertLessEqual(max(abs(a - b) for a, b in zip(pixel, (seed, 80, 100))), 3)

    def test_existing_seed44_is_rejected_before_creating_reports(self):
        path = self.old_step / self.probes(500)[0]["image_path"]
        path.write_bytes(b"existing seed44 sentinel")
        before = sha256(path)
        with self.assertRaises(FileExistsError):
            self.run_common(execute=False)
        self.assertEqual(sha256(path), before)
        self.assertFalse(self.output.exists())

    def test_relative_path_escape_is_rejected(self):
        row = {**self.probes(500)[0], "image_path": "../outside.png"}
        with self.assertRaisesRegex(ValueError, "inside image root"):
            self.run_common(execute=False, probe_rows=lambda step: [row], sampling_seeds=(44,))
        self.assertFalse((self.old_step.parent / "outside.png").exists())

    def test_dataset_cannot_be_the_image_root(self):
        with self.assertRaisesRegex(ValueError, "outside code and dataset"):
            self.run_common(execute=False, image_root=self.dataset)

    def test_training_run_cannot_be_the_image_root(self):
        root = Path(self.selections[0]["run"]) / "images"
        root.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "outside training runs"):
            self.run_common(image_root=root)
        self.assertFalse(self.output.exists())

    def test_xb_refuses_a_writer_that_wins_after_preflight(self):
        target = self.old_step / self.probes(500)[0]["image_path"]

        def pipe(prompt, **kwargs):
            target.write_bytes(b"concurrent writer sentinel")
            return types.SimpleNamespace(images=[Image.new("RGB", (512, 512), "gray")], nsfw_content_detected=[False])

        with patch.object(renderer, "git_info", return_value={"commit": "append-only"}), \
             patch.object(renderer, "export_subject_step", return_value={"hashes": {str(self.rgb): sha256(self.rgb)}}), \
             patch.object(renderer.evaluate, "load_pipeline", return_value=pipe), \
             patch.object(renderer.evaluate, "attention_masks", return_value={}), \
             patch.dict("sys.modules", {"torch": self.fake_torch()}), self.assertRaises(FileExistsError):
            self.run_common()
        self.assertEqual(target.read_bytes(), b"concurrent writer sentinel")
        self.assertEqual(json.loads((self.output / "provenance.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
