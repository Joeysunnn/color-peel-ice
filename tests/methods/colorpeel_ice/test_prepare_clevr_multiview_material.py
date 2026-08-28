import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, PngImagePlugin

from src.methods.colorpeel_ice.multiview_render_contract import (
    EXPECTED_PROFILE,
    EXPECTED_PROFILE_V2,
    EXPECTED_PROFILE_V3,
    canonical_sha256,
    orbit_jitter_metadata,
)
from src.methods.colorpeel_ice import prepare_clevr_multiview as old_prepare
from src.methods.colorpeel_ice import prepare_clevr_multiview_material as material_prepare


REPO_ROOT = Path(__file__).parents[3]
EXPERIMENT = REPO_ROOT / "experiments" / "clevr_subject_color_material_3x3x2"
PROFILE = EXPERIMENT / "configs" / "multiview_render_v3_material.json"
MANIFEST = EXPERIMENT / "manifests" / "clevr_material_manifest.json"
PROTOCOL = EXPERIMENT / "manifests" / "clevr_multiview_material_protocol.json"
RENDERER_PATH = REPO_ROOT / "scripts" / "methods" / "colorpeel_ice" / "render_clevr_multiview.py"
SPEC = importlib.util.spec_from_file_location("material_renderer", RENDERER_PATH)
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)


class MaterialProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.protocol, cls.cells = material_prepare.load_inputs(MANIFEST, PROTOCOL)
        cls.requests = material_prepare.build_render_requests(cls.manifest, cls.protocol)

    def test_profiles_are_versioned_and_historical_fingerprints_unchanged(self):
        self.assertEqual(json.loads(PROFILE.read_text(encoding="utf-8")), EXPECTED_PROFILE_V3)
        self.assertEqual(canonical_sha256(EXPECTED_PROFILE),
                         "246cb06778a74f994311c3e1e3a8a4aa973ce7a308d3cd2732dcdcc021bf8529")
        self.assertEqual(canonical_sha256(EXPECTED_PROFILE_V2),
                         "4890b5a481b7f903f383beeee97607c7f8fd410708925eea923c54cab2b3ece8")
        self.assertEqual(canonical_sha256(EXPECTED_PROFILE_V3),
                         "0db0afe3f5697c7c7a41b12b4b463331ff3d0e4e6ea248096f3145795eab076a")
        self.assertEqual(canonical_sha256(self.requests),
                         "3e9364600aba103163325f19e87530744acaf50f5ca71567339e41bc91d341be")
        old_base, old_protocol = old_prepare.load_inputs(
            old_prepare.DEFAULT_BASE_MANIFEST,
            REPO_ROOT / "experiments" / "clevr_subject_color_3x3" / "manifests" /
            "clevr_multiview_protocol_v2.json",
        )
        self.assertEqual(canonical_sha256(old_prepare.build_render_requests(old_base, old_protocol)),
                         "ebba607110a141f3854878e266d2a0fc99dde9b6cba70d7a7e31de731eee68f3")

    def test_v3_nested_values_do_not_alias_v2(self):
        changed = json.loads(json.dumps(EXPECTED_PROFILE_V3))
        changed["camera"]["azimuth_jitter_degrees"] = 99
        self.assertEqual(EXPECTED_PROFILE_V2["camera"]["azimuth_jitter_degrees"], 18.0)

    def test_360_requests_are_paired_and_renderer_valid(self):
        self.assertEqual(len(self.cells), 18)
        self.assertEqual(len(self.requests), 360)
        RENDERER.validate_requests(self.requests, EXPECTED_PROFILE_V3)
        first, second = self.requests[:2]
        self.assertEqual((first["material"], second["material"]), ("metal", "rubber"))
        self.assertEqual(first["render_seed"], second["render_seed"])
        self.assertEqual(first["shape_color_index"], second["shape_color_index"])
        self.assertEqual(orbit_jitter_metadata(first["render_seed"], EXPECTED_PROFILE_V3),
                         orbit_jitter_metadata(second["render_seed"], EXPECTED_PROFILE_V3))
        for shape in material_prepare.SHAPES:
            for color in material_prepare.COLORS:
                for view in range(20):
                    pair = [row for row in self.requests
                            if row["shape"] == shape and row["color"] == color and row["view_index"] == view]
                    self.assertEqual(len(pair), 2)
                    self.assertEqual({row["material"] for row in pair}, {"metal", "rubber"})
                    self.assertEqual(len({row["render_seed"] for row in pair}), 1)

    def test_invalid_material_and_pair_index_fail(self):
        changed = json.loads(json.dumps(self.requests))
        changed[0]["material"] = "glass"
        with self.assertRaisesRegex(RENDERER.RendererError, "material"):
            RENDERER.validate_requests(changed, EXPECTED_PROFILE_V3)

    def test_v3_resume_contract_checks_material_specific_fields(self):
        expected = self.requests[0]
        changed = {**expected, "shape_color_index": expected["shape_color_index"] + 1}
        contract = RENDERER.stable_contract(self.requests, EXPECTED_PROFILE_V3, {"asset": "a" * 64})
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RENDERER.RendererError, "shape_color_index"):
                RENDERER.verify_completed_record(Path(temporary), changed, expected, contract)
        changed = json.loads(json.dumps(self.requests))
        changed[0]["shape_color_index"] = 9
        with self.assertRaisesRegex(RENDERER.RendererError, "shape_color_index"):
            RENDERER.validate_requests(changed, EXPECTED_PROFILE_V3)

    def test_folds_partition_grid_with_locked_balance(self):
        held_out = [{tuple(cell) for cell in fold["held_out"]} for fold in self.protocol["folds"]]
        self.assertEqual(sum(len(fold) for fold in held_out), 18)
        self.assertEqual(len(set().union(*held_out)), 18)
        for fold in held_out:
            train = {(s, c, m) for s in material_prepare.SHAPES for c in material_prepare.COLORS
                     for m in material_prepare.MATERIALS} - fold
            self.assertEqual(len(train), 12)
            self.assertEqual({sum(row[0] == value for row in train) for value in material_prepare.SHAPES}, {4})
            self.assertEqual({sum(row[1] == value for row in train) for value in material_prepare.COLORS}, {4})
            self.assertEqual({sum(row[2] == value for row in train) for value in material_prepare.MATERIALS}, {6})

    def test_plan_writes_exact_request_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "plan"
            status = material_prepare.plan_protocol(
                self.manifest, self.protocol, output, RENDERER_PATH
            )
            rows = material_prepare._read_jsonl(output / "render_requests.jsonl")
        self.assertEqual(status["request_count"], 360)
        self.assertEqual(status["paired_smoke_prefix_count"], 2)
        self.assertEqual(rows, self.requests)

    def test_v3_contract_hashes_both_native_material_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape_dir, material_dir = root / "shapes", root / "materials"
            shape_dir.mkdir(); material_dir.mkdir()
            properties = {
                "shapes": {"cube": "Cube", "sphere": "Sphere", "cylinder": "Cylinder"},
                "colors": {"red": [173, 35, 35], "cyan": [41, 208, 208], "gray": [87, 87, 87]},
                "materials": {"metal": "MyMetal", "rubber": "Rubber"}, "sizes": {"large": 1.0},
            }
            properties_path, base = root / "properties.json", root / "base.blend"
            properties_path.write_text(json.dumps(properties), encoding="utf-8"); base.write_bytes(b"base")
            for name in ("Cube", "Sphere", "Cylinder"):
                (shape_dir / f"{name}.blend").write_bytes(name.encode())
            for name in ("MyMetal", "Rubber"):
                (material_dir / f"{name}.blend").write_bytes(name.encode())
            _, hashes = RENDERER.collect_asset_hashes(
                properties_path, base, shape_dir, material_dir, EXPECTED_PROFILE_V3
            )
        self.assertIn("material_metal", hashes)
        self.assertIn("material_rubber", hashes)

    def test_rgb_equivalence_gate_accepts_sparse_max_five_rounding_noise(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, candidate = root / "reference.png", root / "candidate.png"
            Image.new("RGB", (512, 512), (10, 20, 30)).save(reference)
            changed = Image.new("RGB", (512, 512), (10, 20, 30))
            changed.putpixel((0, 0), (15, 20, 30))
            changed.save(candidate)
            result = material_prepare._decoded_rgb_difference(candidate, reference)
        self.assertTrue(result["pixel_equivalent"])
        self.assertEqual(result["max_abs_difference"], 5)
        self.assertEqual(result["changed_channel_values"], 1)
        self.assertEqual(result["total_channel_values"], 512 * 512 * 3)
        self.assertLessEqual(result["changed_channel_fraction"], 0.001)
        self.assertLessEqual(result["mean_abs_difference"], 0.001)

    def test_rgb_equivalence_gate_rejects_excess_mean_or_changed_fraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.png"
            Image.new("RGB", (512, 512), (10, 20, 30)).save(reference)
            mean_candidate = Image.new("RGB", (512, 512), (10, 20, 30))
            for index in range(1000):
                mean_candidate.putpixel((index % 512, index // 512), (11, 20, 30))
            mean_path = root / "mean.png"; mean_candidate.save(mean_path)
            mean_result = material_prepare._decoded_rgb_difference(mean_path, reference)
        self.assertFalse(mean_result["pixel_equivalent"])
        self.assertGreater(mean_result["mean_abs_difference"], 0.001)
        self.assertGreater(mean_result["changed_channel_fraction"], 0.001)

    def test_mask_gate_uses_decoded_pixels_not_png_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first.png", root / "second.png"
            mask = Image.new("L", (32, 32), 0); mask.putpixel((3, 4), 255)
            mask.save(first, compress_level=0)
            metadata = PngImagePlugin.PngInfo(); metadata.add_text("audit", "same decoded pixels")
            mask.save(second, compress_level=9, pnginfo=metadata)
            hashes_differ = material_prepare._file_sha256(first) != material_prepare._file_sha256(second)
            pixels_equal = material_prepare._decoded_pixels_equal(first, second)
            mask.putpixel((5, 6), 255); mask.save(second)
            changed_pixels_equal = material_prepare._decoded_pixels_equal(first, second)
        self.assertTrue(hashes_differ)
        self.assertTrue(pixels_equal)
        self.assertFalse(changed_pixels_equal)

    def test_equivalence_gate_v1_is_preserved_but_protocol_requires_v2(self):
        self.assertEqual(material_prepare.V2_METAL_EQUIVALENCE_V1["id"],
                         "decoded_pixel_equivalence_v1")
        self.assertEqual(material_prepare.V2_METAL_EQUIVALENCE["id"],
                         "decoded_pixel_equivalence_v2")
        changed = json.loads(json.dumps(self.protocol))
        changed["realization_contract"]["v2_metal_reference_equivalence"] = (
            material_prepare.V2_METAL_EQUIVALENCE_V1
        )
        with self.assertRaisesRegex(material_prepare.ProtocolError, "equivalence gate"):
            material_prepare.validate_protocol(self.manifest, changed)

    @unittest.skipUnless(material_prepare.yaml is not None, "PyYAML is required for staging config tests")
    def test_training_staging_is_288_full_and_192_per_fold_image_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); render_root = root / "render"; output = root / "realized"
            realized = []
            for cell in self.cells:
                for view in range(16):
                    path = render_root / cell["id"] / f"view_{view:02d}" / "img.jpg"
                    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(f"{cell['id']}:{view}".encode())
                    realized.append({"cell_id": cell["id"], "shape": cell["shape"], "color": cell["color"],
                                     "material": cell["material"], "view_index": view, "split": "train",
                                     "image": path.relative_to(render_root).as_posix()})
            output.mkdir()
            summaries = material_prepare.build_training_outputs(
                render_root, realized, self.cells, self.protocol, output,
                EXPERIMENT / "configs" / "material_base.yaml",
            )
            full_files = list((output / "full_grid" / "train_assets").glob("*/*"))
            self.assertEqual(len(full_files), 288)
            self.assertTrue(all(path.suffix == ".jpg" for path in full_files))
            for fold in "abc":
                files = list((output / "folds" / f"fold_{fold}" / "train_assets").glob("*/*"))
                self.assertEqual(len(files), 192)
                self.assertTrue(all(path.suffix == ".jpg" for path in files))
            self.assertEqual(len(summaries), 3)
            self.assertEqual(len(list((output / "evaluation_configs").glob("generate_fold_*.json"))), 9)


if __name__ == "__main__":
    unittest.main()
