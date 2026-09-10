from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from src.methods.colorpeel_ice import renderer_color_calibration as calibration


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts/methods/colorpeel_ice/render_d1_color_calibration_preflight.py"
SPEC = importlib.util.spec_from_file_location("d1_preflight", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class D1EmissionPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.assets = self.root / "assets"
        (self.assets / "data/shapes").mkdir(parents=True)
        (self.assets / "data/base_scene.blend").write_bytes(b"base scene fixture")
        (self.assets / "data/shapes/Sphere.blend").write_bytes(b"sphere fixture")
        self.output = self.root / "output"
        self.plan, self.contract = preflight.make_plan(
            self.output, self.assets, "data/base_scene.blend", "data/shapes/Sphere.blend"
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_artifacts(self, color_delta: int = 0) -> None:
        records = []
        for request in self.plan["requests"]:
            relative = f"renders/{request['request_id']}"
            directory = self.output / relative
            directory.mkdir(parents=True)
            rgb = tuple(channel + color_delta for channel in request["expected_decoded_srgb_uint8"])
            image = Image.new("RGB", (512, 512), rgb)
            image.save(directory / "image.png", format="PNG")
            mask = Image.new("L", (512, 512), 0)
            for y in range(128, 384):
                for x in range(128, 384):
                    mask.putpixel((x, y), 255)
            mask.save(directory / "mask.png", format="PNG")
            camera_location = [5.0, 5.0, 5.0]
            camera_target = [0.0, 0.0, 1.3]
            delta = [actual - target for actual, target in zip(camera_location, camera_target)]
            radius = math.sqrt(sum(value * value for value in delta))
            metadata = {
                "request": request,
                "render_contract_sha256": preflight.canonical_sha256(self.contract),
                "blender_version": "4.2.11",
                "blender_build_identifier": "fixture-build",
                "renderer": {"engine": "CYCLES", "cuda_devices": [{"name": "fixture", "type": "CUDA", "id": "0"}],
                             "samples": 512, "resolution": [512, 512],
                             "image": {"format": "PNG", "mode": "RGB", "bits_per_channel": 8}},
                "base_scene_relative_path": self.contract["assets"]["base_scene"]["relative_path"],
                "base_scene_sha256": self.contract["assets"]["base_scene"]["sha256"],
                "shape_asset_relative_path": self.contract["assets"]["shape_asset"]["relative_path"],
                "shape_asset_sha256": self.contract["assets"]["shape_asset"]["sha256"],
                "emission_socket_rgba": request["socket_rgba"],
                "color_management": {"display_device": "sRGB", "view_transform": "Standard", "look": None,
                                     "exposure": 0.0, "gamma": 1.0},
                "camera": {"name": "Camera", "view_index": 0, "jitter": "none", "location": camera_location,
                           "look_at": camera_target, "lens_mm": 35.0, "sensor_width_mm": 32.0,
                           "sensor_height_mm": 18.0, "shift_x": 0.0, "shift_y": 0.0,
                           "base_scene_camera_azimuth_degrees": math.degrees(math.atan2(delta[1], delta[0])),
                           "base_scene_camera_radius": radius,
                           "base_scene_camera_elevation_degrees": math.degrees(math.atan2(delta[2], math.hypot(delta[0], delta[1])))},
                "lights": {"jitter": "none", "records": [
                    {"name": name, "position": [float(index), 1.0, 2.0], "energy": 100.0, "rgb": [1.0, 1.0, 1.0]}
                    for index, name in enumerate(["Lamp_Key", "Lamp_Back", "Lamp_Fill", "Area"])
                ]},
                "world": {"rgba": [0.05, 0.05, 0.05, 1.0]},
                "ground": {"rgba": [0.5, 0.5, 0.5, 1.0], "metallic": 0.0, "roughness": 1.0},
                "foreground_pixels": 256 * 256,
                "image_relative_path": f"{relative}/image.png",
                "image_sha256": _sha(directory / "image.png"),
                "mask_relative_path": f"{relative}/mask.png",
                "mask_sha256": _sha(directory / "mask.png"),
            }
            preflight.atomic_json(directory / "metadata.json", metadata)
            records.append({
                "request_id": request["request_id"],
                "image_relative_path": metadata["image_relative_path"],
                "image_sha256": metadata["image_sha256"],
                "mask_relative_path": metadata["mask_relative_path"],
                "mask_sha256": metadata["mask_sha256"],
                "metadata_relative_path": f"{relative}/metadata.json",
                "metadata_sha256": _sha(directory / "metadata.json"),
            })
        preflight.atomic_json(self.output / preflight.MANIFEST_NAME, {
            "schema": "d1_emission_encoding_preflight_manifest/v1",
            "contract_sha256": preflight.canonical_sha256(self.contract),
            "request_count": 4,
            "records": records,
        })

    def _refresh_record_hashes(self, request_id: str) -> None:
        manifest = preflight.load_json(self.output / preflight.MANIFEST_NAME)
        record = next(row for row in manifest["records"] if row["request_id"] == request_id)
        directory = self.output / "renders" / request_id
        metadata = preflight.load_json(directory / "metadata.json")
        metadata["image_sha256"] = _sha(directory / "image.png")
        metadata["mask_sha256"] = _sha(directory / "mask.png")
        preflight.atomic_json(directory / "metadata.json", metadata)
        record["image_sha256"] = metadata["image_sha256"]
        record["mask_sha256"] = metadata["mask_sha256"]
        record["metadata_sha256"] = _sha(directory / "metadata.json")
        preflight.atomic_json(self.output / preflight.MANIFEST_NAME, manifest)

    def test_plan_is_exact_four_requests_and_rejects_reused_or_escaping_roots(self) -> None:
        self.assertEqual(len(self.plan["requests"]), 4)
        self.assertEqual(self.plan["requests_sha256"], preflight.FROZEN_PREFLIGHT_REQUESTS_SHA256)
        self.assertEqual(self.contract["assets"]["base_scene"]["relative_path"], "data/base_scene.blend")
        with self.assertRaisesRegex(preflight.PreflightError, "new or empty"):
            preflight.make_plan(self.output, self.assets, "data/base_scene.blend", "data/shapes/Sphere.blend")
        with self.assertRaisesRegex(preflight.PreflightError, "escapes"):
            preflight.make_plan(self.root / "other", self.assets, "../outside.blend", "data/shapes/Sphere.blend")

    def test_plan_and_contract_payload_drift_rejects(self) -> None:
        plan_path = self.output / preflight.PLAN_NAME
        changed = preflight.load_json(plan_path)
        changed["requests"][0]["linear_rgb"] = [0.0, 0.0, 0.0]
        preflight.atomic_json(plan_path, changed)
        with self.assertRaisesRegex(preflight.PreflightError, "payload hash"):
            preflight._load_plan_contract(self.output)

    def test_current_head_and_adapter_hash_are_bound_to_contract(self) -> None:
        with patch.object(preflight, "_git_commit", return_value="f" * 40):
            with self.assertRaisesRegex(preflight.PreflightError, "Current repository HEAD"):
                preflight._load_plan_contract(self.output)
        with patch.object(preflight, "_adapter_script_sha256", return_value="0" * 64):
            with self.assertRaisesRegex(preflight.PreflightError, "Adapter script hash"):
                preflight._load_plan_contract(self.output)

    def test_normal_analysis_and_rc1_gate_boundaries_pass(self) -> None:
        self._write_artifacts()
        result = preflight.analyze(self.output)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["gate"]["overall_pass"])
        rows = [dict(request, decoded_png_rgb_uint8=[byte + delta for byte, delta in zip(
            request["expected_decoded_srgb_uint8"], [-1, 0, 1])], decoded_lab_delta=0.5)
                for request in self.plan["requests"]]
        self.assertTrue(calibration.summarize_preflight_measurements(rows)["overall_pass"])

    def test_analysis_threshold_failure_writes_failed_summary(self) -> None:
        self._write_artifacts(color_delta=5)
        result = preflight.analyze(self.output)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["gate"]["overall_pass"])

    def test_analysis_rejects_manifest_and_metadata_hash_or_payload_drift(self) -> None:
        self._write_artifacts()
        manifest = preflight.load_json(self.output / preflight.MANIFEST_NAME)
        manifest["records"][0]["image_sha256"] = "0" * 64
        preflight.atomic_json(self.output / preflight.MANIFEST_NAME, manifest)
        with self.assertRaisesRegex(preflight.PreflightError, "image hash differs"):
            preflight.analyze(self.output)

        self.tearDown()
        self.setUp()
        self._write_artifacts()
        manifest = preflight.load_json(self.output / preflight.MANIFEST_NAME)
        manifest["records"].append(dict(manifest["records"][0]))
        manifest["request_count"] = 5
        preflight.atomic_json(self.output / preflight.MANIFEST_NAME, manifest)
        with self.assertRaisesRegex(preflight.PreflightError, "Manifest contract differs"):
            preflight.analyze(self.output)

        # Restore fixtures and change only a metadata request payload.
        self.tearDown()
        self.setUp()
        self._write_artifacts()
        request_id = self.plan["requests"][0]["request_id"]
        metadata_path = self.output / "renders" / request_id / "metadata.json"
        metadata = preflight.load_json(metadata_path)
        metadata["request"]["render_seed"] = 0
        preflight.atomic_json(metadata_path, metadata)
        manifest = preflight.load_json(self.output / preflight.MANIFEST_NAME)
        manifest["records"][0]["metadata_sha256"] = _sha(metadata_path)
        preflight.atomic_json(self.output / preflight.MANIFEST_NAME, manifest)
        with self.assertRaisesRegex(preflight.PreflightError, "payload differs"):
            preflight.analyze(self.output)

        self.tearDown()
        self.setUp()
        self._write_artifacts()
        request_id = self.plan["requests"][0]["request_id"]
        metadata_path = self.output / "renders" / request_id / "metadata.json"
        metadata = preflight.load_json(metadata_path)
        metadata["renderer"]["samples"] = 1
        preflight.atomic_json(metadata_path, metadata)
        manifest = preflight.load_json(self.output / preflight.MANIFEST_NAME)
        manifest["records"][0]["metadata_sha256"] = _sha(metadata_path)
        preflight.atomic_json(self.output / preflight.MANIFEST_NAME, manifest)
        with self.assertRaisesRegex(preflight.PreflightError, "renderer evidence"):
            preflight.analyze(self.output)

    def test_analysis_rejects_wrong_image_or_mask_and_bad_masks(self) -> None:
        self._write_artifacts()
        request_id = self.plan["requests"][0]["request_id"]
        directory = self.output / "renders" / request_id
        Image.new("RGBA", (512, 512), (1, 2, 3, 255)).save(directory / "image.png", format="PNG")
        self._refresh_record_hashes(request_id)
        with self.assertRaisesRegex(preflight.PreflightError, "Image must"):
            preflight.analyze(self.output)

        self.tearDown()
        self.setUp()
        self._write_artifacts()
        request_id = self.plan["requests"][0]["request_id"]
        directory = self.output / "renders" / request_id
        Image.new("L", (512, 512), 127).save(directory / "mask.png", format="PNG")
        self._refresh_record_hashes(request_id)
        with self.assertRaisesRegex(preflight.PreflightError, "only 0 and 255"):
            preflight.analyze(self.output)

        self.tearDown()
        self.setUp()
        self._write_artifacts()
        request_id = self.plan["requests"][0]["request_id"]
        directory = self.output / "renders" / request_id
        Image.new("RGB", (512, 512), (255, 255, 255)).save(directory / "mask.png", format="PNG")
        self._refresh_record_hashes(request_id)
        with self.assertRaisesRegex(preflight.PreflightError, "Mask must"):
            preflight.analyze(self.output)

        self.tearDown()
        self.setUp()
        self._write_artifacts()
        request_id = self.plan["requests"][0]["request_id"]
        directory = self.output / "renders" / request_id
        Image.new("L", (512, 512), 255).save(directory / "mask.png", format="PNG")
        metadata = preflight.load_json(directory / "metadata.json")
        metadata["foreground_pixels"] = 512 * 512
        preflight.atomic_json(directory / "metadata.json", metadata)
        self._refresh_record_hashes(request_id)
        with self.assertRaisesRegex(preflight.PreflightError, "ratio"):
            preflight.analyze(self.output)

    def test_color_management_and_linear_socket_helpers_without_bpy(self) -> None:
        scene = SimpleNamespace(
            display_settings=SimpleNamespace(display_device="Display P3"),
            view_settings=SimpleNamespace(view_transform="AgX", look="Medium High Contrast", exposure=2.0, gamma=2.0),
        )
        self.assertEqual(preflight.configure_color_management(scene), {
            "display_device": "sRGB", "view_transform": "Standard", "look": None, "exposure": 0.0, "gamma": 1.0,
        })
        request = self.plan["requests"][2]
        self.assertEqual(preflight.emission_socket_rgba(request), tuple(request["linear_rgb"] + [1.0]))
        changed = dict(request, socket_rgba=[value / 255.0 for value in request["expected_decoded_srgb_uint8"]] + [1.0])
        with self.assertRaisesRegex(preflight.PreflightError, "differs"):
            preflight.emission_socket_rgba(changed)
        with self.assertRaisesRegex(preflight.PreflightError, "Blender"):
            preflight.render(self.output, self.assets)
        self.assertEqual(preflight.main(["render", "--output-root", str(self.output), "--asset-root", str(self.assets)]), 2)
        self.assertTrue((self.output / "preflight_render_failure.json").is_file())

    def test_blender_version_is_strict_tuple_with_canonical_metadata_form(self) -> None:
        self.assertEqual(preflight.canonical_blender_version((4, 2, 11)), "4.2.11")
        with self.assertRaisesRegex(preflight.PreflightError, "Blender must be 4.2.11"):
            preflight.canonical_blender_version((4, 2, 10))

    def test_derived_camera_scalars_allow_only_float32_round_trip_error(self) -> None:
        location = [7.481131553649902, -6.5076398849487305, 5.34366512298584]
        target = [0.0, 0.0, 1.2999999523162842]
        camera = {
            "base_scene_camera_radius": 10.708311464359104,
            "base_scene_camera_azimuth_degrees": math.degrees(math.atan2(location[1], location[0])),
            "base_scene_camera_elevation_degrees": 22.186293736756948,
        }
        self.assertTrue(preflight.derived_camera_scalars_match(camera, location, target))
        camera["base_scene_camera_elevation_degrees"] += 1e-4
        self.assertFalse(preflight.derived_camera_scalars_match(camera, location, target))

    def test_exact_edt_uses_tight_bbox_and_not_square_erosion(self) -> None:
        narrow = [False] * (512 * 512)
        for y in range(100, 130):
            for x in range(100, 180):
                narrow[y * 512 + x] = True
        _, narrow_contract = preflight._mask_interior(narrow, 512, 512)
        self.assertEqual((narrow_contract["bbox_w"], narrow_contract["bbox_h"], narrow_contract["s"], narrow_contract["radius"]),
                         (80, 30, 30, 1))

        diagonal_hole = [False] * (512 * 512)
        for y in range(100, 300):
            for x in range(100, 300):
                diagonal_hole[y * 512 + x] = True
        diagonal_hole[203 * 512 + 204] = False  # Distance five from (200, 200), but inside old square radius five.
        interior, contract = preflight._mask_interior(diagonal_hole, 512, 512)
        self.assertEqual(contract["radius"], 5)
        self.assertIn(200 * 512 + 200, interior)  # exact EDT >= 5 includes the Euclidean boundary.


if __name__ == "__main__":
    unittest.main()
