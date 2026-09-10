from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.methods.colorpeel_ice import render_d1_color_calibration_reduced_direct as runner
from src.methods.colorpeel_ice import renderer_color_calibration as calibration
from src.methods.colorpeel_ice.natural_image_masks import exact_edt


class ReducedDirectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = self.root / "assets"
        for name, relative in runner.ASSETS.items():
            path = self.assets / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
        self.output = self.root / "output"
        self.plan, self.contract = runner.make_plan(self.output, self.assets)
        self.request = self.plan["requests"][0]

    def _base(self):
        target = runner._float32([0, 0, 1.3])
        location = [5., 5., 5.]
        delta = [a - b for a, b in zip(location, target)]
        camera = {"name": "Camera", "view_index": 0, "jitter": "none", "location": location,
                  "look_at": target, "lens_mm": 35., "sensor_width_mm": 32., "sensor_height_mm": 18.,
                  "shift_x": 0., "shift_y": 0., "base_scene_camera_azimuth_degrees": 45.,
                  "base_scene_camera_radius": math.sqrt(sum(v * v for v in delta)),
                  "base_scene_camera_elevation_degrees": math.degrees(math.atan2(delta[2], math.hypot(*delta[:2])))}
        lights = {"jitter": "none", "records": [{"name": name, "position": [float(i), 1., 2.], "energy": 100.,
                                                "rgb": [1., 1., 1.]}
                                               for i, name in enumerate(("Lamp_Key", "Lamp_Back", "Lamp_Fill", "Area"))]}
        return {"camera": camera, "lights": lights}

    def _metadata(self, request):
        base = self._base()
        location = runner._float32(runner.view_location(base["camera"], request["view_index"]))
        direction = [a - b for a, b in zip(base["camera"]["look_at"], location)]
        length = math.sqrt(sum(v * v for v in direction))
        camera = dict(base["camera"], view_index=request["view_index"], location=location,
                      forward=runner._float32([v / length for v in direction]))
        metadata = {"request": deepcopy(request), "render_contract_sha256": runner.canonical_sha256(self.contract),
                    "blender_version": "4.2.11", "blender_build_identifier": "fixture",
                    "renderer": {"engine": "CYCLES", "samples": 512, "resolution": [512, 512],
                                 "cuda_devices": [{"name": "fixture", "type": "CUDA", "id": "0"}],
                                 "image": {"format": "PNG", "mode": "RGB", "bits_per_channel": 8}},
                    "render_seed": request["render_seed"], "base_scene_state": base,
                    "material": {"node_group": "Rubber", "input": "Color", "output": "Shader",
                                 "socket_rgba": runner._float32(request["socket_rgba"]), "surface_linked": True},
                    "object": {"shape": "sphere", "location": runner._float32([0, 0, 1.3]),
                               "scale": runner._float32([1.3] * 3), "rotation_euler": [0, 0, 0]},
                    "camera": camera, "lights": deepcopy(base["lights"]),
                    "world": {"rgba": [0.05, 0.05, 0.05, 1.]},
                    "ground": {"rgba": [0.5, 0.5, 0.5, 1.], "metallic": 0., "roughness": 1.},
                    "color_management": {"display_device": "sRGB", "view_transform": "Standard", "look": None,
                                         "exposure": 0., "gamma": 1.}, "foreground_pixels": 128 * 128,
                    "image_relative_path": "unused", "image_sha256": "unused",
                    "mask_relative_path": "unused", "mask_sha256": "unused"}
        for name, asset in self.contract["assets"].items():
            metadata[f"{name}_relative_path"] = asset["relative_path"]
            metadata[f"{name}_sha256"] = asset["sha256"]
        return metadata

    def _artifacts(self, failing=False):
        records = []
        for request in self.plan["requests"]:
            relative = f"renders/{request['request_id']}"
            directory = self.output / relative
            directory.mkdir(parents=True)
            rgb = (128, 128, 128) if failing else tuple(request["preview_srgb_uint8"])
            Image.new("RGB", (512, 512), rgb).save(directory / "image.png")
            mask = Image.new("L", (512, 512), 0)
            mask.paste(255, (192, 192, 320, 320))
            mask.save(directory / "mask.png")
            metadata = self._metadata(request)
            record = {"request_id": request["request_id"]}
            for kind in ("image", "mask"):
                metadata[f"{kind}_relative_path"] = f"{relative}/{kind}.png"
                metadata[f"{kind}_sha256"] = runner.file_sha256(directory / f"{kind}.png")
                record.update({key: metadata[key] for key in (f"{kind}_relative_path", f"{kind}_sha256")})
            runner.atomic_json(directory / "metadata.json", metadata)
            record.update(metadata_relative_path=f"{relative}/metadata.json",
                          metadata_sha256=runner.file_sha256(directory / "metadata.json"))
            records.append(record)
        manifest = {"schema": f"{runner.PREFIX}_manifest/v1", "contract_sha256": runner.canonical_sha256(self.contract),
                    "request_count": 18, "records": records}
        runner.atomic_json(self.output / runner.MANIFEST_NAME, manifest)
        return manifest

    def test_frozen_matrix_and_socket(self):
        rows = self.plan["requests"]
        self.assertEqual(rows, calibration.build_reduced_direct_requests())
        self.assertEqual(len(rows), 18)
        self.assertEqual(runner.canonical_sha256(rows), runner.REQUEST_HASH)
        self.assertEqual(len({r["stable_id"] for r in rows}), 6)
        for stable_id in {r["stable_id"] for r in rows}:
            self.assertEqual([r["view_index"] for r in rows if r["stable_id"] == stable_id], [0, 8, 16])
        for row in rows:
            self.assertEqual(row["shape"], "sphere")
            self.assertEqual(list(runner.rubber_socket_rgba(row)), row["linear_rgb"] + [1.])
            self.assertNotEqual(row["socket_rgba"][:3], [x / 255 for x in row["preview_srgb_uint8"]])

    def test_plan_provenance_and_no_overwrite(self):
        self.assertEqual(self.contract["git_commit"], runner.shared._git_commit())
        self.assertEqual(self.contract["adapter_script_sha256"], runner.file_sha256(Path(runner.__file__)))
        self.assertEqual(self.contract["protocol_canonical_sha256"], calibration.FROZEN_PROTOCOL_CANONICAL_SHA256)
        self.assertEqual(runner._verify_runtime_assets(self.assets, self.contract),
                         [self.assets / relative for relative in runner.ASSETS.values()])
        with self.assertRaisesRegex(runner.ReducedDirectError, "new or empty"):
            runner.make_plan(self.output, self.assets)
        with patch.object(runner.shared, "_git_commit", return_value="f" * 40):
            with self.assertRaisesRegex(runner.ReducedDirectError, "HEAD"):
                runner._load_plan_contract(self.output)
        with patch.object(runner, "_script_hash", return_value="0" * 64):
            with self.assertRaisesRegex(runner.ReducedDirectError, "script hash"):
                runner._load_plan_contract(self.output)

    def test_plan_and_asset_tampering(self):
        changed = deepcopy(self.plan)
        changed["requests"][0]["render_seed"] = 0
        runner.atomic_json(self.output / runner.PLAN_NAME, changed)
        with self.assertRaisesRegex(runner.ReducedDirectError, "payload"):
            runner._load_plan_contract(self.output)
        runner.atomic_json(self.output / runner.PLAN_NAME, self.plan)
        path = self.assets / runner.ASSETS["material_asset"]
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(runner.ReducedDirectError, "material_asset hash"):
            runner._verify_runtime_assets(self.assets, self.contract)
        changed = deepcopy(self.contract)
        changed["assets"]["material_asset"]["relative_path"] = "../Rubber.blend"
        runner.atomic_json(self.output / runner.CONTRACT_NAME, changed)
        runner.atomic_json(self.output / runner.PLAN_NAME, dict(self.plan, contract_sha256=runner.canonical_sha256(changed)))
        with self.assertRaisesRegex(runner.ReducedDirectError, "path differs"):
            runner._load_plan_contract(self.output)

    def test_three_views_and_float32_roundtrip(self):
        for request in self.plan["requests"][:3]:
            metadata = self._metadata(request)
            runner._verify_runtime_metadata(metadata, request, self.contract, 128 * 128)
            delta = [a - b for a, b in zip(metadata["camera"]["location"], metadata["camera"]["look_at"])]
            azimuth = math.degrees(math.atan2(delta[1], delta[0]))
            self.assertAlmostEqual((azimuth - 45 - 18 * request["view_index"] + 180) % 360 - 180, 0, delta=2e-6)
        with self.assertRaises(runner.ReducedDirectError):
            runner.view_location(self._base()["camera"], 1)

    def test_metadata_rejects_runtime_semantic_drift(self):
        cases = [("request", "render_seed", 0), ("camera", "view_index", 8),
                 ("camera", "jitter", "random"), ("camera", "location", [0, 0, 0]),
                 ("camera", "forward", [0, 0, 1]), ("camera", "shift_x", .1),
                 ("material", "socket_rgba", [v / 255 for v in self.request["preview_srgb_uint8"]] + [1]),
                 ("material", "node_group", "Emission"), ("renderer", "samples", 64),
                 ("object", "scale", [1, 1, 1]), ("lights", "jitter", "random")]
        for section, key, value in cases:
            with self.subTest(section=section, key=key):
                metadata = self._metadata(self.request)
                metadata[section][key] = value
                with self.assertRaises(runner.ReducedDirectError):
                    runner._verify_runtime_metadata(metadata, self.request, self.contract, 128 * 128)
        for field, value in [("render_seed", 0), ("material_asset_sha256", "0" * 64), ("foreground_pixels", 10)]:
            metadata = self._metadata(self.request)
            metadata[field] = value
            with self.assertRaises(runner.ReducedDirectError):
                runner._verify_runtime_metadata(metadata, self.request, self.contract, 128 * 128)
        metadata = self._metadata(self.request)
        metadata["lights"]["records"][0]["energy"] += 1
        with self.assertRaisesRegex(runner.ReducedDirectError, "lights changed"):
            runner._verify_runtime_metadata(metadata, self.request, self.contract, 128 * 128)

    def test_render_rejects_reuse_before_rendering(self):
        (self.output / ".partial").mkdir()
        with patch.object(runner.shared, "bpy", object()), patch.object(runner, "_render_one") as render_one:
            with self.assertRaisesRegex(runner.ReducedDirectError, "existing output"):
                runner.render(self.output, self.assets)
            render_one.assert_not_called()

    def test_rubber_graph_assigns_frozen_linear_socket(self):
        class Socket:
            @property
            def default_value(self):
                return runner._float32(self.assigned)

            @default_value.setter
            def default_value(self, value):
                self.assigned = value

        class Groups(list):
            def keys(self):
                return [group.name for group in self]

            def get(self, name):
                return next((group for group in self if group.name == name), None)

        socket = Socket()
        output = SimpleNamespace(inputs={"Surface": object()})
        group_node = SimpleNamespace(inputs={"Color": socket}, outputs={"Shader": object()})
        created, links, appended = [], [], []

        class Nodes:
            def clear(self):
                created.clear()

            def new(self, kind):
                created.append(kind)
                return output if kind == "ShaderNodeOutputMaterial" else group_node

        groups = Groups()
        material = SimpleNamespace(use_nodes=False, node_tree=SimpleNamespace(
            nodes=Nodes(), links=SimpleNamespace(new=lambda source, target: links.append((source, target)))))

        def append(**kwargs):
            appended.append(kwargs)
            groups.append(SimpleNamespace(name="Rubber"))

        bpy = SimpleNamespace(data=SimpleNamespace(node_groups=groups, materials=SimpleNamespace(new=lambda name: material)),
                              ops=SimpleNamespace(wm=SimpleNamespace(append=append)))
        obj = SimpleNamespace(data=SimpleNamespace(materials=[]))
        asset = self.assets / runner.ASSETS["material_asset"]
        with patch.object(runner.shared, "bpy", bpy):
            result = runner._configure_rubber(obj, self.request, asset)
        self.assertEqual(socket.assigned, tuple(self.request["socket_rgba"]))
        self.assertEqual(created, ["ShaderNodeOutputMaterial", "ShaderNodeGroup"])
        self.assertEqual(links, [(group_node.outputs["Shader"], output.inputs["Surface"])])
        self.assertEqual(obj.data.materials, [material])
        self.assertEqual(appended[0]["filename"], "Rubber")
        self.assertTrue(appended[0]["directory"].startswith(str(asset)))
        self.assertEqual(result["socket_rgba"], runner._float32(self.request["socket_rgba"]))

    def test_binary_mask_tight_bbox_and_exact_edt(self):
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[192:320, 192:320] = 255
        interior, geometry = runner.shared._mask_interior((mask.reshape(-1) == 255).tolist(), 512, 512)
        self.assertEqual((geometry["bbox_w"], geometry["bbox_h"], geometry["radius"]), (128, 128, 3))
        self.assertEqual(interior, np.flatnonzero(exact_edt(mask != 0) >= 3).tolist())
        pixels = np.full((512, 512, 3), self.request["preview_srgb_uint8"], dtype=np.uint8)
        row = runner.measure_pixels(pixels, mask, self.request)
        self.assertTrue(row["eligible_ok"])
        self.assertEqual(row["measurement_stats"]["object_area"] + row["measurement_stats"]["background_area"], 512 * 512)
        mask[200, 200] = 128
        with self.assertRaisesRegex(runner.ReducedDirectError, "binary"):
            runner.measure_pixels(pixels, mask, self.request)
        for values in (np.zeros((512, 512)), np.full((512, 512), 255)):
            with self.assertRaises(runner.ReducedDirectError):
                runner.measure_pixels(pixels, values, self.request)

    def test_measure_pixels_accepts_two_dimensional_unique_inverse(self):
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[192:320, 192:320] = 255
        pixels = np.full((512, 512, 3), self.request["preview_srgb_uint8"], dtype=np.uint8)
        pixels[256:, :, 0] += 1
        expected = runner.measure_pixels(pixels, mask, self.request)
        unique = np.unique

        def unique_with_column_inverse(*args, **kwargs):
            result = unique(*args, **kwargs)
            if kwargs.get("return_inverse"):
                values, inverse = result
                return values, inverse.reshape(-1, 1)
            return result

        with patch.object(np, "unique", side_effect=unique_with_column_inverse), \
                patch.object(runner, "_measure_lab", wraps=runner._measure_lab) as measure_lab:
            actual = runner.measure_pixels(pixels, mask, self.request)
        self.assertEqual(actual, expected)
        self.assertEqual(measure_lab.call_args.args[0].shape,
                         (expected["measurement_stats"]["interior_area"], 3))

    def test_eligible_strict_bounds_and_statistics(self):
        geometry = calibration.derive_interior_contract(2000, 50, 40, background_area=512 * 512 - 2000)
        lab = np.tile([50., self.request["target_a"] + 2, self.request["target_b"] + 1], (1000, 1))
        lab[:100, 0] = 5
        lab[100:200, 0] = 95
        lab[200:600, 1] += 2
        row = runner._measure_lab(lab, geometry, self.request)
        self.assertEqual(row["measurement_stats"]["eligible_count"], 800)
        self.assertEqual(row["median_L"], 50)
        self.assertAlmostEqual(row["median_a"], self.request["target_a"] + 3)
        self.assertAlmostEqual(row["iqr_a"], 2)
        self.assertAlmostEqual(row["e_ch"], math.sqrt(10))
        lab[200, 0] = 0
        with self.assertRaisesRegex(runner.ReducedDirectError, "Eligible"):
            runner._measure_lab(lab, geometry, self.request)

    def test_complete_artifacts_frozen_gate_and_no_analysis_overwrite(self):
        self._artifacts()
        result = runner.analyze(self.output)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["measurements"]), 18)
        self.assertEqual(len(result["gate"]["targets"]), 6)
        self.assertEqual(result["gate"], calibration.summarize_chroma_measurements(result["measurements"], "reduced_direct"))
        with self.assertRaisesRegex(runner.ReducedDirectError, "already exists"):
            runner.analyze(self.output)

    def test_full_gate_failure_is_preserved(self):
        self._artifacts(failing=True)
        result = runner.analyze(self.output)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["gate"]["overall_pass"])
        self.assertEqual(len(result["measurements"]), 18)
        self.assertTrue((self.output / runner.SUMMARY_NAME).is_file())

    def test_manifest_hash_payload_and_coverage_rejections(self):
        manifest = self._artifacts()
        for change in ("hash", "duplicate", "path", "contract"):
            changed = deepcopy(manifest)
            if change == "hash":
                changed["records"][0]["image_sha256"] = "0" * 64
            elif change == "duplicate":
                changed["records"][1] = deepcopy(changed["records"][0])
            elif change == "path":
                changed["records"][0]["image_relative_path"] = "../image.png"
            else:
                changed["contract_sha256"] = "0" * 64
            runner.atomic_json(self.output / runner.MANIFEST_NAME, changed)
            with self.subTest(change=change), self.assertRaises(runner.ReducedDirectError):
                runner.analyze(self.output)
        runner.atomic_json(self.output / runner.MANIFEST_NAME, manifest)
        record = manifest["records"][0]
        path = self.output / record["metadata_relative_path"]
        metadata = runner.load_json(path)
        metadata["request"]["socket_rgba"][0] += .01
        runner.atomic_json(path, metadata)
        record["metadata_sha256"] = runner.file_sha256(path)
        runner.atomic_json(self.output / runner.MANIFEST_NAME, manifest)
        with self.assertRaisesRegex(runner.ReducedDirectError, "request payload"):
            runner.analyze(self.output)

    def test_cli_scope_and_failure_evidence(self):
        parser = runner.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["full90"])
        self.assertEqual(runner.main(["render", "--output-root", str(self.output), "--asset-root", str(self.assets)]), 2)
        failure = runner.load_json(self.output / "reduced_direct_render_failure.json")
        self.assertEqual(failure["status"], "failed")
        self.assertNotIn(str(self.root), str(failure))


if __name__ == "__main__":
    unittest.main()
