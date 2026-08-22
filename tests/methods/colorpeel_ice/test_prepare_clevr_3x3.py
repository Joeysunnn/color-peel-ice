import contextlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest

from PIL import Image


REPO_ROOT = Path(__file__).parents[3]
MODULE_PATH = REPO_ROOT / "src" / "methods" / "colorpeel_ice" / "prepare_clevr_3x3.py"
SPEC = importlib.util.spec_from_file_location("prepare_clevr_3x3", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
MANIFEST_PATH = (
    REPO_ROOT
    / "experiments"
    / "clevr_subject_color_3x3"
    / "manifests"
    / "clevr_3x3_manifest.json"
)


class Clevr3x3ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = MODULE.load_experiment_manifest(MANIFEST_PATH)

    def test_samples_are_exact_cartesian_product(self):
        expected = set(itertools.product(("cube", "sphere", "cylinder"), ("red", "cyan", "gray")))
        actual = {(sample["shape"], sample["color"]) for sample in self.manifest["samples"]}
        self.assertEqual(actual, expected)
        self.assertEqual(len(self.manifest["samples"]), 9)

    def test_each_prompt_contains_one_subject_and_one_color_modifier(self):
        subject_tokens = {entry["token"] for entry in self.manifest["shapes"]}
        color_tokens = {entry["token"] for entry in self.manifest["colors"]}
        for sample in self.manifest["samples"]:
            prompt = sample["instance_prompt"][0]
            modifiers = re.findall(r"<[^>]+>", prompt)
            self.assertEqual(len(modifiers), 2, sample["id"])
            self.assertEqual(sum(token in subject_tokens for token in modifiers), 1, sample["id"])
            self.assertEqual(sum(token in color_tokens for token in modifiers), 1, sample["id"])

    def test_locked_ids_tokens_rgb_and_material(self):
        expected_ids = {
            ("cube", "red"): "003_cube_red_metal",
            ("cube", "cyan"): "013_cube_cyan_metal",
            ("cube", "gray"): "001_cube_gray_metal",
            ("sphere", "red"): "019_sphere_red_metal",
            ("sphere", "cyan"): "029_sphere_cyan_metal",
            ("sphere", "gray"): "017_sphere_gray_metal",
            ("cylinder", "red"): "035_cylinder_red_metal",
            ("cylinder", "cyan"): "045_cylinder_cyan_metal",
            ("cylinder", "gray"): "033_cylinder_gray_metal",
        }
        shape_tokens = {"cube": "<s1*>", "sphere": "<s2*>", "cylinder": "<s3*>"}
        color_tokens = {"red": "<c1*>", "cyan": "<c2*>", "gray": "<c3*>"}
        color_rgb = {"red": [173, 35, 35], "cyan": [41, 208, 208], "gray": [87, 87, 87]}
        for sample in self.manifest["samples"]:
            key = (sample["shape"], sample["color"])
            self.assertEqual(sample["id"], expected_ids[key])
            self.assertEqual(sample["subject_token"], shape_tokens[sample["shape"]])
            self.assertEqual(sample["color_token"], color_tokens[sample["color"]])
            self.assertEqual(sample["rgb"], color_rgb[sample["color"]])
            self.assertEqual(sample["material"], "metal")


class Clevr3x3DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "dataset"
        self.root.mkdir()
        self.output = Path(self.temp_dir.name) / "staging"
        self._create_dataset()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_dataset(self):
        image_template = Path(self.temp_dir.name) / "rgb.jpg"
        Image.new("RGB", (512, 512), (128, 128, 128)).save(image_template)
        mask_templates = {}
        foreground = {"cube": 13900, "sphere": 14973, "cylinder": 22868}
        for shape, count in foreground.items():
            path = Path(self.temp_dir.name) / f"mask_{shape}.png"
            data = bytearray(512 * 512)
            data[:count] = b"\xff" * count
            mask = Image.frombytes("L", (512, 512), bytes(data))
            mask.save(path)
            mask_templates[shape] = path

        selected_rgb = {"gray": [87, 87, 87], "red": [173, 35, 35], "cyan": [41, 208, 208]}
        records = []
        for sample in MODULE.expected_dataset_samples():
            sample_dir = self.root / sample["id"]
            sample_dir.mkdir()
            shutil.copyfile(image_template, sample_dir / "img.jpg")
            shutil.copyfile(mask_templates[sample["shape"]], sample_dir / f"mask_{sample['shape']}_0.png")
            scene = {
                "image_index": sample["index"],
                "image_filename": "img.jpg",
                "objects": [
                    {
                        "shape": sample["shape"],
                        "color": sample["color"],
                        "material": sample["material"],
                    }
                ],
            }
            (sample_dir / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
            records.append(
                {
                    **sample,
                    "image_path": f"{sample['id']}/img.jpg",
                    "scene_path": f"{sample['id']}/scene.json",
                    "rgb": selected_rgb.get(sample["color"], [0, 0, 0]),
                }
            )
        (self.root / "metadata.json").write_text(
            json.dumps({"sample_count": 48, "resolution": [512, 512]}),
            encoding="utf-8",
        )
        (self.root / "manifest.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_dry_run_validates_without_writing(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = MODULE.main(
                [
                    "--dataset-root",
                    str(self.root),
                    "--output-dir",
                    str(self.output),
                    "--dry-run",
                ]
            )
        self.assertEqual(result["status"], "validated")
        self.assertEqual(result["sample_count"], 48)
        self.assertEqual(result["selected_sample_count"], 9)
        self.assertFalse(result["training_uses_gt_masks"])
        self.assertFalse(self.output.exists())

    def test_staging_contains_only_training_images(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = MODULE.main(
                [
                    "--dataset-root",
                    str(self.root),
                    "--output-dir",
                    str(self.output),
                ]
            )
        self.assertEqual(result["status"], "staged")
        concepts = json.loads((self.output / "concepts.json").read_text(encoding="utf-8"))
        self.assertEqual(len(concepts), 9)
        for concept in concepts:
            concept_dir = Path(concept["instance_data_dir"])
            self.assertEqual([path.name for path in concept_dir.iterdir()], ["img.jpg"])
        audit = json.loads((self.output / "dataset_audit.json").read_text(encoding="utf-8"))
        self.assertFalse(audit["training_uses_gt_masks"])
        self.assertEqual(audit["gt_mask_count"], 48)


if __name__ == "__main__":
    unittest.main()
