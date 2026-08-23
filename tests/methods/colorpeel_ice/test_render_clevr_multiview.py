import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

from src.methods.colorpeel_ice.multiview_render_contract import (
    EXPECTED_PROFILE,
    EXPECTED_PROFILE_V2,
    canonical_sha256,
    look_at_alignment,
    look_at_y_up_alignment,
    orbit_jitter_metadata,
    orbit_location,
    official_jitter_metadata,
    spherical_pose,
)
from src.methods.colorpeel_ice import prepare_clevr_multiview as prepare


REPO_ROOT = Path(__file__).parents[3]
RENDERER_PATH = REPO_ROOT / "scripts" / "methods" / "colorpeel_ice" / "render_clevr_multiview.py"
SPEC = importlib.util.spec_from_file_location("render_clevr_multiview", RENDERER_PATH)
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)
PROFILE_PATH = (
    REPO_ROOT / "experiments" / "clevr_subject_color_3x3" / "configs" / "multiview_render.json"
)
PROFILE_V2_PATH = (
    REPO_ROOT / "experiments" / "clevr_subject_color_3x3" / "configs" / "multiview_render_v2.json"
)
BASE_MANIFEST = (
    REPO_ROOT / "experiments" / "clevr_subject_color_3x3" / "manifests" / "clevr_3x3_manifest.json"
)
PROTOCOL_MANIFEST = (
    REPO_ROOT / "experiments" / "clevr_subject_color_3x3" / "manifests" / "clevr_multiview_protocol.json"
)
PROTOCOL_V2_MANIFEST = (
    REPO_ROOT / "experiments" / "clevr_subject_color_3x3" / "manifests" / "clevr_multiview_protocol_v2.json"
)


class RendererContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.protocol = prepare.load_inputs(BASE_MANIFEST, PROTOCOL_MANIFEST)
        cls.requests = prepare.build_render_requests(cls.base, cls.protocol)
        cls.base_v2, cls.protocol_v2 = prepare.load_inputs(BASE_MANIFEST, PROTOCOL_V2_MANIFEST)
        cls.requests_v2 = prepare.build_render_requests(cls.base_v2, cls.protocol_v2)

    def test_tracked_profile_is_exactly_locked(self):
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(profile, EXPECTED_PROFILE)
        self.assertEqual(RENDERER.validate_profile(profile), EXPECTED_PROFILE)
        changed = json.loads(json.dumps(profile))
        changed["camera"]["jitter_magnitude"] = 0.6
        with self.assertRaisesRegex(RENDERER.RendererError, "differs from locked"):
            RENDERER.validate_profile(changed)

    def test_v1_fingerprints_are_historical_golden_values(self):
        self.assertEqual(
            canonical_sha256(EXPECTED_PROFILE),
            "246cb06778a74f994311c3e1e3a8a4aa973ce7a308d3cd2732dcdcc021bf8529",
        )
        self.assertEqual(
            canonical_sha256(self.requests),
            "97162f88528794f03e553dee70bfacc83b20646587a5ed9a7617f2c23f818c2d",
        )

    def test_v2_profile_is_separate_and_exactly_locked(self):
        profile = json.loads(PROFILE_V2_PATH.read_text(encoding="utf-8"))
        self.assertEqual(profile, EXPECTED_PROFILE_V2)
        self.assertEqual(RENDERER.validate_profile(profile), EXPECTED_PROFILE_V2)
        self.assertNotEqual(canonical_sha256(profile), canonical_sha256(EXPECTED_PROFILE))
        self.assertNotEqual(canonical_sha256(self.requests_v2), canonical_sha256(self.requests))
        changed = json.loads(json.dumps(profile))
        changed["camera"]["azimuth_jitter_degrees"] = 19.0
        with self.assertRaisesRegex(RENDERER.RendererError, "differs from locked"):
            RENDERER.validate_profile(changed)

    def test_blender_cycles_arguments_are_locked_and_removed_before_argparse(self):
        argv = [
            "blender", "--background", "--", "--cycles-device", "CUDA",
            "--cycles-print-stats", "--requests", "requests.jsonl",
        ]
        self.assertEqual(
            RENDERER.extract_blender_args(argv),
            ["--requests", "requests.jsonl"],
        )
        with self.assertRaisesRegex(RENDERER.RendererError, "remain CUDA"):
            RENDERER.extract_blender_args([
                "blender", "--", "--cycles-device", "OPTIX", "--requests", "requests.jsonl",
            ])

    def test_seed_420000_exact_official_draw_order(self):
        offsets = official_jitter_metadata(420000, EXPECTED_PROFILE)
        self.assertEqual(offsets["camera_offset"], [
            -0.06422106498087743,
            -0.3582052753411801,
            0.37571427349391007,
        ])
        self.assertEqual(offsets["light_offsets"]["Lamp_Key"], [
            -0.0378744594480267,
            0.1430989399786884,
            0.011210578240198776,
        ])
        self.assertEqual(offsets["light_offsets"]["Lamp_Back"], [
            -0.5493293981591734,
            -0.7580448101549195,
            0.3912924353178602,
        ])
        self.assertEqual(offsets["light_offsets"]["Lamp_Fill"], [
            -0.2868291690090399,
            0.22022692467949256,
            0.07768975391329636,
        ])
        self.assertTrue(all(-0.5 <= value < 0.5 for value in offsets["camera_offset"]))
        self.assertTrue(all(
            -1.0 <= value < 1.0
            for values in offsets["light_offsets"].values()
            for value in values
        ))

    def test_seed_420000_exact_orbit_draw_order_and_ranges(self):
        offsets = orbit_jitter_metadata(420000, EXPECTED_PROFILE_V2)
        self.assertEqual(offsets["camera_orbit_jitter"], {
            "azimuth_degrees": -2.3119583393115875,
            "elevation_degrees": -7.164105506823601,
            "distance_fraction": 0.03757142734939101,
        })
        jitter = offsets["camera_orbit_jitter"]
        self.assertTrue(-18.0 <= jitter["azimuth_degrees"] < 18.0)
        self.assertTrue(-10.0 <= jitter["elevation_degrees"] < 10.0)
        self.assertTrue(-0.05 <= jitter["distance_fraction"] < 0.05)
        self.assertEqual(
            offsets["light_offsets"],
            official_jitter_metadata(420000, EXPECTED_PROFILE)["light_offsets"],
        )

    def test_orbit_pose_round_trip_and_look_at_axes(self):
        target = [0.0, 0.0, 1.3]
        location = orbit_location(target, 10.0, -45.0, 30.0)
        pose = spherical_pose(location, target)
        self.assertAlmostEqual(pose["radius"], 10.0)
        self.assertAlmostEqual(pose["azimuth_degrees"], -45.0)
        self.assertAlmostEqual(pose["elevation_degrees"], 30.0)
        quaternion = [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]
        self.assertAlmostEqual(look_at_alignment([0.0, -1.0, 0.0], [0.0, 0.0, 0.0], quaternion), 1.0)
        self.assertAlmostEqual(look_at_y_up_alignment(
            [0.0, -1.0, 0.0], [0.0, 0.0, 0.0], quaternion
        ), 1.0)

    def test_requests_include_profile_and_empty_renderer_fields(self):
        RENDERER.validate_requests(self.requests)
        for request in self.requests:
            self.assertEqual(request["renderer_profile_id"], "multiview_render_v1")
            self.assertEqual(request["renderer_profile_sha256"], canonical_sha256(EXPECTED_PROFILE))
            for field in RENDERER.RENDERER_OWNED_FIELDS:
                self.assertIsNone(request[field])

    def test_v2_requests_require_v2_profile(self):
        RENDERER.validate_requests(self.requests_v2, EXPECTED_PROFILE_V2)
        with self.assertRaisesRegex(RENDERER.RendererError, "profile"):
            RENDERER.validate_requests(self.requests_v2, EXPECTED_PROFILE)


class RendererResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "rendered"
        base, protocol = prepare.load_inputs(BASE_MANIFEST, PROTOCOL_MANIFEST)
        self.requests = prepare.build_render_requests(base, protocol)
        self.contract = RENDERER.stable_contract(
            self.requests,
            EXPECTED_PROFILE,
            {"base_scene_blendfile": "a" * 64},
        )
        RENDERER.prepare_output_root(self.output, self.contract, resume=False)

    def tearDown(self):
        self.temp.cleanup()

    def _write_completed(self, request_index=0):
        request = self.requests[request_index]
        view_dir = self.output / request["cell_id"] / f"view_{request['view_index']:02d}"
        view_dir.mkdir(parents=True)
        paths = {
            "image": view_dir / "img.jpg",
            "mask": view_dir / f"mask_{request['shape']}_0.png",
            "background_mask": view_dir / "background.png",
            "scene_json": view_dir / "scene.json",
        }
        for field, path in paths.items():
            path.write_bytes(field.encode("ascii"))
        record = {
            **request,
            **{field: path.relative_to(self.output).as_posix() for field, path in paths.items()},
            "renderer_profile_id": EXPECTED_PROFILE["profile_id"],
            "render_contract_sha256": canonical_sha256(self.contract),
            "artifact_sha256": {field: RENDERER.file_sha256(path) for field, path in paths.items()},
        }
        RENDERER.write_json(view_dir / ".record.json", record)
        RENDERER.append_jsonl(self.output / "renderer_realization.jsonl", record)
        return request, record, view_dir

    def test_matching_resume_skips_verified_record(self):
        request, record, _ = self._write_completed()
        RENDERER.prepare_output_root(self.output, self.contract, resume=True)
        completed = RENDERER.load_completed_records(
            self.output,
            {(item["cell_id"], item["view_index"]): item for item in self.requests},
            self.contract,
        )
        self.assertEqual(completed[(request["cell_id"], request["view_index"])], record)

    def test_resume_rejects_hash_tamper(self):
        _, record, _ = self._write_completed()
        (self.output / record["image"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(RENDERER.RendererError, "hash changed"):
            RENDERER.load_completed_records(
                self.output,
                {(item["cell_id"], item["view_index"]): item for item in self.requests},
                self.contract,
            )

    def test_resume_rejects_partial_directory(self):
        partial = self.output / ".partial"
        partial.mkdir()
        with self.assertRaisesRegex(RENDERER.RendererError, "partial directory"):
            RENDERER.prepare_output_root(self.output, self.contract, resume=True)

    def test_resume_rejects_orphan_final_directory(self):
        request = self.requests[0]
        orphan = self.output / request["cell_id"] / f"view_{request['view_index']:02d}"
        orphan.mkdir(parents=True)
        with self.assertRaisesRegex(RENDERER.RendererError, "orphan final"):
            RENDERER.load_completed_records(
                self.output,
                {(item["cell_id"], item["view_index"]): item for item in self.requests},
                self.contract,
            )

    def test_resume_rejects_duplicate_manifest_row(self):
        _, record, _ = self._write_completed()
        RENDERER.append_jsonl(self.output / "renderer_realization.jsonl", record)
        with self.assertRaisesRegex(RENDERER.RendererError, "Duplicate resume record"):
            RENDERER.load_completed_records(
                self.output,
                {(item["cell_id"], item["view_index"]): item for item in self.requests},
                self.contract,
            )

    def test_resume_rejects_v2_contract_in_v1_output(self):
        base, protocol_v2 = prepare.load_inputs(BASE_MANIFEST, PROTOCOL_V2_MANIFEST)
        requests_v2 = prepare.build_render_requests(base, protocol_v2)
        contract_v2 = RENDERER.stable_contract(
            requests_v2,
            EXPECTED_PROFILE_V2,
            {"base_scene_blendfile": "a" * 64},
        )
        with self.assertRaisesRegex(RENDERER.RendererError, "contract differs"):
            RENDERER.prepare_output_root(self.output, contract_v2, resume=True)


if __name__ == "__main__":
    unittest.main()
