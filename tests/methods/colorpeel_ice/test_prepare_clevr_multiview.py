import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw, ImageOps


REPO_ROOT = Path(__file__).parents[3]
MODULE_PATH = REPO_ROOT / "src" / "methods" / "colorpeel_ice" / "prepare_clevr_multiview.py"
SPEC = importlib.util.spec_from_file_location("prepare_clevr_multiview", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
BASE_MANIFEST = (
    REPO_ROOT / "experiments" / "clevr_subject_color_3x3" / "manifests" / "clevr_3x3_manifest.json"
)
PROTOCOL_MANIFEST = (
    REPO_ROOT / "experiments" / "clevr_subject_color_3x3" / "manifests" / "clevr_multiview_protocol.json"
)


class MultiviewProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.protocol = MODULE.load_inputs(BASE_MANIFEST, PROTOCOL_MANIFEST)

    def test_requests_are_nine_cells_by_twenty_views(self):
        requests = MODULE.build_render_requests(self.base, self.protocol)
        self.assertEqual(len(requests), 180)
        self.assertEqual(len({(row["cell_id"], row["view_index"]) for row in requests}), 180)
        for cell_index, cell in enumerate(self.base["samples"]):
            rows = [row for row in requests if row["cell_id"] == cell["id"]]
            self.assertEqual(sum(row["split"] == "train" for row in rows), 16)
            self.assertEqual(sum(row["split"] == "audit" for row in rows), 4)
            for row in rows:
                self.assertEqual(row["render_seed"], 420000 + cell_index * 100 + row["view_index"])

    def test_direct_cli_imports_shared_contract_from_repo_root(self):
        result = subprocess.run(
            [sys.executable, str(MODULE_PATH), "--help"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("{plan,realize}", result.stdout)

    def test_protocol_does_not_fabricate_renderer_metadata(self):
        for row in MODULE.build_render_requests(self.base, self.protocol):
            for field in MODULE.RENDERER_FIELDS:
                self.assertIsNone(row[field])
            self.assertIsNone(row["empirical_rgb"])

    def test_vector_sum_allows_only_blender_float32_rounding(self):
        MODULE._validate_vector_sum(
            [7.481131076812744, -6.5076398849487305, 5.279051303863525],
            [-0.06422106498087743, -0.3582052753411801, 0.37571427349391007],
            [7.416910171508789, -6.865845203399658, 5.654765605926514],
            "Camera",
        )
        with self.assertRaisesRegex(MODULE.ProtocolError, "base plus jitter"):
            MODULE._validate_vector_sum([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], [1.10001, 2.2, 3.3], "Camera")
        expected_scale = 1.3 / math.sqrt(2.0)
        self.assertTrue(MODULE._vector_matches(
            [0.919238805770874, 0.919238805770874, 0.919238805770874],
            [expected_scale, expected_scale, expected_scale],
        ))
        self.assertFalse(MODULE._vector_matches(
            [0.919248805770874, 0.919238805770874, 0.919238805770874],
            [expected_scale, expected_scale, expected_scale],
        ))

    def test_folds_are_exact_matchings_with_two_partners_per_axis(self):
        actual = {
            fold["id"]: {tuple(pair) for pair in fold["held_out"]}
            for fold in self.protocol["folds"]
        }
        self.assertEqual(actual, MODULE.EXPECTED_FOLDS)
        grid = {(shape, color) for shape in ("cube", "sphere", "cylinder") for color in ("red", "cyan", "gray")}
        for held_out in actual.values():
            train = grid - held_out
            for shape in ("cube", "sphere", "cylinder"):
                self.assertEqual(len({color for candidate, color in train if candidate == shape}), 2)
            for color in ("red", "cyan", "gray"):
                self.assertEqual(len({shape for shape, candidate in train if candidate == color}), 2)

    def test_changed_fold_is_rejected(self):
        changed = json.loads(json.dumps(self.protocol))
        changed["folds"][0]["held_out"][0] = ["cube", "cyan"]
        with self.assertRaisesRegex(MODULE.ProtocolError, "Held-out folds"):
            MODULE.validate_protocol(self.base, changed)

    def test_protocol_identity_is_locked(self):
        changed = json.loads(json.dumps(self.protocol))
        changed["version"] = 2
        with self.assertRaisesRegex(MODULE.ProtocolError, "version"):
            MODULE.validate_protocol(self.base, changed)

    def test_plan_records_blocked_renderer_without_creating_images(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "protocol"
            with contextlib.redirect_stdout(io.StringIO()):
                result = MODULE.main(["plan", "--output-dir", str(output)])
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["blocked_reason"], "multiview_renderer_not_provided_or_missing")
            rows = [json.loads(line) for line in (output / "render_requests.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 180)
            self.assertFalse(any(path.suffix.lower() in {".jpg", ".png"} for path in output.rglob("*")))


class MultiviewRealizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.render_root = self.root / "rendered"
        self.render_root.mkdir()
        self.output = self.root / "prepared"
        self.base, self.protocol = MODULE.load_inputs(BASE_MANIFEST, PROTOCOL_MANIFEST)

    def tearDown(self):
        self.temp.cleanup()

    def _make_realization(self):
        mask_template = self.root / "mask.png"
        background_template = self.root / "background.png"
        mask = Image.new("L", (512, 512), 0)
        ImageDraw.Draw(mask).rectangle((156, 156, 356, 356), fill=255)
        mask.save(mask_template)
        ImageOps.invert(mask).save(background_template)

        requests = MODULE.build_render_requests(self.base, self.protocol)
        asset_hashes = {"base_scene_blendfile": "a" * 64, "properties_json": "b" * 64}
        contract = {
            "schema_version": 1,
            "profile_id": MODULE.EXPECTED_PROFILE["profile_id"],
            "profile_sha256": MODULE.canonical_sha256(MODULE.EXPECTED_PROFILE),
            "requests_sha256": MODULE.canonical_sha256(requests),
            "request_count": 180,
            "asset_sha256": asset_hashes,
        }
        (self.render_root / "render_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        contract_sha256 = MODULE.canonical_sha256(contract)
        records = []
        light_bases = {
            "Lamp_Key": [6.4, -2.9, 4.2],
            "Lamp_Back": [-1.1, 2.6, 5.8],
            "Lamp_Fill": [-4.6, -4.0, 3.0],
        }
        for request in requests:
            view_dir = self.render_root / request["cell_id"] / f"view_{request['view_index']:02d}"
            view_dir.mkdir(parents=True)
            image_path = view_dir / "img.jpg"
            mask_path = view_dir / f"mask_{request['shape']}_0.png"
            background_mask_path = view_dir / "background.png"
            scene_path = view_dir / "scene.json"
            pixel_value = 10 + request["view_index"] * 12
            Image.new("RGB", (512, 512), (pixel_value, 20 + request["cell_index"] * 10, 30)).save(image_path)
            shutil.copyfile(mask_template, mask_path)
            shutil.copyfile(background_template, background_mask_path)
            jitter = MODULE.official_jitter_metadata(request["render_seed"], MODULE.EXPECTED_PROFILE)
            camera_base = [7.4, -6.5, 5.3]
            camera = {
                "name": "Camera",
                "base_location": camera_base,
                "jitter_offset": jitter["camera_offset"],
                "final_location": [camera_base[i] + jitter["camera_offset"][i] for i in range(3)],
                "rotation_euler": [1.0, 0.0, 0.7],
                "rotation_policy": "preserve_base_scene",
                "lens": 35.0,
                "sensor_width": 32.0,
            }
            light = {
                "order": MODULE.EXPECTED_PROFILE["lights"]["order"],
                "lights": {},
                "fixed_lights": {
                    "Area": {
                        "base_location": [0.0, 0.0, 10.0],
                        "final_location": [0.0, 0.0, 10.0],
                        "rgb": [1.0, 1.0, 1.0],
                        "type": "SUN",
                        "energy": 0.45,
                    }
                },
            }
            for name in MODULE.EXPECTED_PROFILE["lights"]["order"]:
                offset = jitter["light_offsets"][name]
                base = light_bases[name]
                light["lights"][name] = {
                    "base_location": base,
                    "jitter_offset": offset,
                    "final_location": [base[i] + offset[i] for i in range(3)],
                    "rgb": [1.0, 1.0, 1.0],
                    "type": "AREA",
                    "energy": 1000.0,
                }
            background = MODULE.EXPECTED_PROFILE["background"]
            object_scale = 1.3 / math.sqrt(2.0) if request["shape"] == "cube" else 1.3
            scene = {
                "renderer_profile_id": MODULE.EXPECTED_PROFILE["profile_id"],
                "render_seed": request["render_seed"],
                "cycles_seed": request["render_seed"],
                "camera": camera,
                "light": light,
                "background": background,
                "renderer": {
                    "blender_version": "4.2.11",
                    "engine": "CYCLES",
                    "cycles_samples": 512,
                    "cycles_device": "CUDA",
                    "cuda_devices": [{"name": "Tesla V100-SXM2-32GB", "type": "CUDA", "id": "CUDA_0"}],
                },
                "asset_sha256": asset_hashes,
                "objects": [{
                    "shape": request["shape"],
                    "color": request["color"],
                    "material": request["material"],
                    "material_backend": "clevr_asset_node_group",
                    "rotation": 0.0,
                    "nominal_scale": 1.3,
                    "applied_scale": [object_scale, object_scale, object_scale],
                    "3d_coords": [0.0, 0.0, object_scale],
                }],
            }
            scene_path.write_text(json.dumps(scene), encoding="utf-8")
            record = {
                **request,
                "camera": camera,
                "light": light,
                "background": background,
                "scene_json": scene_path.relative_to(self.render_root).as_posix(),
                "image": image_path.relative_to(self.render_root).as_posix(),
                "mask": mask_path.relative_to(self.render_root).as_posix(),
                "background_mask": background_mask_path.relative_to(self.render_root).as_posix(),
                "renderer_profile_id": MODULE.EXPECTED_PROFILE["profile_id"],
                "render_contract_sha256": contract_sha256,
                "foreground_pixels": 201 * 201,
            }
            record["artifact_sha256"] = {
                field: MODULE._file_sha256(self.render_root / record[field])
                for field in ("scene_json", "image", "mask", "background_mask")
            }
            records.append(record)
        manifest = self.root / "renderer_realization.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        return records, manifest

    def test_realize_validates_views_and_builds_leak_free_fold_assets(self):
        records, manifest = self._make_realization()
        with contextlib.redirect_stdout(io.StringIO()):
            result = MODULE.main([
                "realize",
                "--render-root", str(self.render_root),
                "--render-manifest", str(manifest),
                "--output-dir", str(self.output),
            ])
        self.assertEqual(result["status"], "validated")
        self.assertEqual(result["realized_view_count"], 180)
        self.assertEqual(result["training_seeds"], [42, 43, 44])
        realized = [json.loads(line) for line in (self.output / "realized_views.jsonl").read_text().splitlines()]
        self.assertEqual(realized[0]["nominal_rgb"], [173, 35, 35])
        self.assertEqual(realized[0]["empirical_rgb"]["value"], [10.0, 20.0, 30.0])
        self.assertEqual(realized[0]["empirical_rgb"]["source"], "realized_view_gt_mask")
        self.assertEqual(realized[0]["empirical_rgb"]["foreground_pixels"], 201 * 201)
        review_rows = (self.output / "multiview_human_review.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(review_rows), 181)
        with Image.open(self.output / "multiview_contact_sheet.png") as contact_sheet:
            self.assertEqual(contact_sheet.size, (1000, 2016))

        for fold_id in ("a", "b", "c"):
            fold_dir = self.output / "folds" / f"fold_{fold_id}"
            protocol = json.loads((fold_dir / "fold_protocol.json").read_text())
            self.assertEqual(protocol["train_view_count"], 96)
            self.assertEqual(protocol["seen_audit_view_count"], 24)
            self.assertEqual(protocol["held_out_view_count"], 60)
            self.assertEqual(protocol["held_out_train_view_count"], 48)
            self.assertEqual(protocol["held_out_audit_view_count"], 12)
            self.assertEqual(protocol["training_seeds"], [42, 43, 44])
            self.assertFalse(protocol["training_uses_gt_masks"])
            concepts = json.loads((fold_dir / "concepts.json").read_text())
            self.assertEqual(len(concepts), 6)
            for concept in concepts:
                files = list(Path(concept["instance_data_dir"]).iterdir())
                self.assertEqual(len(files), 16)
                self.assertTrue(all(path.suffix == ".jpg" for path in files))
            config_paths = sorted(fold_dir.glob("train_config_seed*.json"))
            self.assertEqual(len(config_paths), 3)
            for seed, config_path in zip((42, 43, 44), config_paths):
                config = json.loads(config_path.read_text())
                self.assertEqual(config["run"]["variant"], f"multiview_fold_{fold_id}_seed{seed}")
                self.assertEqual(config["run"]["seed"], seed)
                self.assertEqual(config["args"]["seed"], seed)
                self.assertEqual(config["args"]["concepts_list"], str(fold_dir / "concepts.json"))
                self.assertEqual(
                    config["args"]["initializer_token"],
                    "cube+sphere+cylinder+red+turquoise+gray",
                )
        self.assertEqual(len(list((self.output / "folds").glob("fold_*/train_config_seed*.json"))), 9)

    def test_missing_renderer_file_is_rejected(self):
        records, _ = self._make_realization()
        records[0]["scene_json"] = "missing/scene.json"
        with self.assertRaisesRegex(MODULE.ProtocolError, "Missing realized scene_json"):
            MODULE.validate_realization(self.render_root, records, self.base, self.protocol)

    def test_duplicate_images_are_rejected_as_fake_multiview(self):
        records, _ = self._make_realization()
        first_cell = records[0]["cell_id"]
        first_image = self.render_root / records[0]["image"]
        for record in records[1:]:
            if record["cell_id"] == first_cell:
                shutil.copyfile(first_image, self.render_root / record["image"])
                record["artifact_sha256"]["image"] = MODULE._file_sha256(self.render_root / record["image"])
        with self.assertRaisesRegex(MODULE.ProtocolError, "20 distinct rendered images"):
            MODULE.validate_realization(self.render_root, records, self.base, self.protocol)

    def test_fixed_background_drift_is_rejected(self):
        records, _ = self._make_realization()
        records[0]["background"] = {**records[0]["background"], "varied": True}
        with self.assertRaisesRegex(MODULE.ProtocolError, "fixed background"):
            MODULE.validate_realization(self.render_root, records, self.base, self.protocol)

    def test_noncomplementary_background_mask_is_rejected(self):
        records, _ = self._make_realization()
        background_path = self.render_root / records[0]["background_mask"]
        shutil.copyfile(self.root / "mask.png", background_path)
        records[0]["artifact_sha256"]["background_mask"] = MODULE._file_sha256(background_path)
        with self.assertRaisesRegex(MODULE.ProtocolError, "not complements"):
            MODULE.validate_realization(self.render_root, records, self.base, self.protocol)

    def test_nonempty_output_directory_is_rejected(self):
        _, manifest = self._make_realization()
        self.output.mkdir()
        (self.output / "stale.txt").write_text("do not mix runs", encoding="utf-8")
        with self.assertRaisesRegex(MODULE.ProtocolError, "Output directory must be empty"):
            with contextlib.redirect_stdout(io.StringIO()):
                MODULE.main([
                    "realize",
                    "--render-root", str(self.render_root),
                    "--render-manifest", str(manifest),
                    "--output-dir", str(self.output),
                ])


if __name__ == "__main__":
    unittest.main()
