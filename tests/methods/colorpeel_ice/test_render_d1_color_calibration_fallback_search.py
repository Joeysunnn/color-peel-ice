from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from scripts.methods.colorpeel_ice import render_d1_color_calibration_fallback_search as runner
from src.methods.colorpeel_ice import renderer_color_calibration as calibration
from tests.methods.colorpeel_ice import test_render_d1_color_calibration_reduced_direct as fixtures

direct = runner.direct
ERRORS = (ValueError, direct.ReducedDirectError)


class FallbackSearchTests(unittest.TestCase):
    """Full orchestration uses explicit synthetic measured rows, not Blender.

    All PNG/metadata/manifest checks and frozen candidate helpers run normally.
    A separate test exercises the actual pixel/EDT/Lab decoder. This avoids
    thousands of identical pure-Python EDT evaluations in local unit tests.
    """

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.base = Path(cls.temp.name)
        cls.assets = cls.base / "assets"
        for name, relative in direct.ASSETS.items():
            path = cls.assets / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
        cls.previous = cls.base / "direct"
        plan, contract = direct.make_plan(cls.previous, cls.assets)
        fixture = fixtures.ReducedDirectTests()
        fixture.output, fixture.plan, fixture.contract = cls.previous, plan, contract
        fixture._artifacts(failing=True)
        # RC-2B predecessor is measured from real PNGs once, with no seam.
        direct.analyze(cls.previous)
        cls.previous_sha = direct.file_sha256(cls.previous / direct.SUMMARY_NAME)
        cls.template = cls.base / "template"
        cls.plan, cls.contract = runner.make_plan(cls.template, cls.assets,
            cls.previous / direct.SUMMARY_NAME, cls.previous_sha)

    def setUp(self):
        self.work = tempfile.TemporaryDirectory(dir=self.base)
        self.addCleanup(self.work.cleanup)
        self.output = Path(self.work.name) / "search"
        shutil.copytree(self.template, self.output)

    @staticmethod
    def _synthetic_measurement(pixels, mask, request, *, failing=False):
        geometry = calibration.derive_interior_contract(128 * 128, 128, 128,
            background_area=512 * 512 - 128 * 128, interior_area=124 * 124, eligible_count=124 * 124)
        return calibration.summarize_render_measurement({**request,
            "median_L": 50., "median_a": request["target_a"] + (20 if failing else 0),
            "median_b": request["target_b"], "iqr_L": 0., "iqr_a": 0., "iqr_b": 0.,
            "object_ratio_ok": True, "interior_ok": True, "eligible_ok": True, "measurement_stats": geometry})

    def _render_fixture(self, root, request, contract, *assets):
        fixture = fixtures.ReducedDirectTests()
        fixture.contract = contract
        metadata = fixture._metadata(request)
        relative = f"renders/{request['request_id']}"
        directory = root / relative
        directory.mkdir(parents=True)
        Image.new("RGB", (512, 512), tuple(request["preview_srgb_uint8"])).save(directory / "image.png")
        mask = Image.new("L", (512, 512), 0)
        mask.paste(255, (192, 192, 320, 320))
        mask.save(directory / "mask.png")
        record = {"request_id": request["request_id"]}
        for kind in ("image", "mask"):
            record[f"{kind}_relative_path"] = f"{relative}/{kind}.png"
            record[f"{kind}_sha256"] = direct.file_sha256(directory / f"{kind}.png")
            metadata.update({key: value for key, value in record.items() if key.startswith(kind)})
        direct.atomic_json(directory / "metadata.json", metadata)
        record.update(metadata_relative_path=f"{relative}/metadata.json",
                      metadata_sha256=direct.file_sha256(directory / "metadata.json"))
        return record

    def _render(self, stage):
        with patch.object(runner.shared, "bpy", object()), \
                patch.object(direct, "_render_one", side_effect=self._render_fixture) as render_one, \
                patch.object(runner, "_calibration", side_effect=AssertionError("Blender imported RC-1")):
            result = runner.render(self.output, self.assets, stage)
        self.assertEqual(render_one.call_count, result["request_count"])
        return result

    def test_coarse_payload_hue_gamut_and_blender_import(self):
        self.assertEqual(len(self.plan["targets"]), 6)
        expected = []
        for target in self.plan["targets"]:
            grid = calibration.coarse_fallback_candidates(target["target_a"], target["target_b"])
            self.assertEqual(len(grid), 63)
            self.assertEqual([entry["candidate"] for entry in target["candidates"]], grid)
            for entry in target["candidates"]:
                candidate = entry["candidate"]
                self.assertEqual(candidate["candidate_a"], candidate["chroma_scale"] * target["target_a"])
                self.assertEqual(candidate["candidate_b"], candidate["chroma_scale"] * target["target_b"])
                if candidate["gamut"] == "out":
                    self.assertEqual(entry["decision"], "invalid_gamut")
                    self.assertNotIn("socket_rgba", candidate)
                else:
                    expected.extend(calibration.build_fallback_candidate_requests(target["target_a"], target["target_b"], candidate))
                    self.assertEqual(candidate["socket_rgba"], candidate["linear_rgb"] + [1.])
        self.assertEqual(self.plan["requests"], expected)
        self.assertEqual(self.contract["coarse_requests_sha256"], runner.canonical_sha256(expected))
        code = ("import sys; sys.modules['PIL']=None; sys.modules['numpy']=None; "
                "sys.modules['src.methods.colorpeel_ice.renderer_color_calibration']=None; "
                "import scripts.methods.colorpeel_ice.render_d1_color_calibration_fallback_search")
        subprocess.run([sys.executable, "-c", code], cwd=runner.REPO_ROOT, check=True)

    def test_full_plan_analyze_flow_and_insufficient_status(self):
        # Two complete candidate matrices, including every unique refine member.
        # Only the measured Lab row is synthetic; artifact and request validation
        # plus plan/analyze/selection code are the real runner in both cases.
        for failing in (False, True):
            with self.subTest(failing=failing):
                if failing:
                    self.output = Path(self.work.name) / "insufficient"
                    shutil.copytree(self.template, self.output)
                coarse_manifest = self._render("coarse")
                self.assertEqual(coarse_manifest["request_count"], len(self.plan["requests"]))
                with patch.object(direct, "measure_pixels", side_effect=lambda p, m, r:
                                  self._synthetic_measurement(p, m, r, failing=failing)):
                    coarse = runner.analyze(self.output, "coarse")
                    refine_plan = runner.plan_refine(self.output)
                    for target in refine_plan["targets"]:
                        summaries = coarse["candidate_summaries"][target["stable_id"]]
                        search = calibration.fallback_candidate_search_plan(target["target_a"], target["target_b"], summaries)
                        self.assertEqual([entry["candidate"] for entry in target["candidates"] if entry["decision"] == "render"],
                            [candidate for candidate in search["candidates"] if candidate["source"] == "refine"])
                    self._render("refine")
                    code = runner.main(["analyze-final", "--output-root", str(self.output)])
                    self.assertEqual(code, 1 if failing else 0)
                    result = runner.load_json(self.output / runner.SELECTION)
                self.assertEqual(result["status"], "l_chroma_inverse_insufficient" if failing else "accepted_candidates")
                self.assertEqual(result["selection_count"], 0 if failing else 6)
                self.assertFalse(result["confirmation_performed"])
                self.assertEqual(len(result["targets"]), 6)
                for row in result["targets"]:
                    self.assertEqual(row["accepted_fallback"], not failing)
                    best = row["diagnostic_best"]
                    self.assertEqual((best["L_input"], best["chroma_scale"]), (50., 1.))
                    if failing:
                        self.assertIsNone(row["accepted"])
                    else:
                        self.assertEqual(row["accepted"], best)
                with self.assertRaises(runner.shared.PreflightError):
                    runner.analyze(self.output, "refine")

    def test_predecessor_failure_provenance_and_plan_reuse(self):
        bundle = runner.load_json(self.output / runner.PREDECESSOR)
        for change in (lambda b: b["analysis"].update(status="passed"),
                       lambda b: b["analysis"]["measurements"].pop(),
                       lambda b: b["manifest"]["records"].pop(),
                       lambda b: b["plan"]["requests"][0].update(render_seed=0),
                       lambda b: b["analysis"].update(contract_sha256="0" * 64)):
            changed = deepcopy(bundle)
            change(changed)
            with self.assertRaises(ERRORS + (KeyError,)):
                runner._check_predecessor(changed)
        with self.assertRaisesRegex(direct.ReducedDirectError, "new or empty"):
            runner.make_plan(self.output, self.assets, self.previous / direct.SUMMARY_NAME, self.previous_sha)
        with self.assertRaisesRegex(direct.ReducedDirectError, "Trusted predecessor"):
            runner.make_plan(Path(self.work.name) / "wrong", self.assets, self.previous / direct.SUMMARY_NAME, "0" * 64)

    def test_drift_missing_duplicates_and_partial_outputs(self):
        with self.assertRaises(ERRORS):
            runner.plan_refine(self.output)
        for field, value in (("render_seed", 0), ("shape", "cube"), ("view_index", 4),
                             ("stage", "fallback_confirmation"), ("socket_rgba", [0, 0, 0, 1])):
            changed = deepcopy(self.plan)
            changed["requests"][0][field] = value
            changed["requests_sha256"] = runner.canonical_sha256(changed["requests"])
            direct.atomic_json(self.output / "coarse" / runner.PLAN, changed)
            with self.assertRaises(ERRORS):
                runner._render_plan(self.output, "coarse", self.contract)
            with self.assertRaises(ERRORS):
                runner._load_stage(self.output, "coarse", self.contract)
        direct.atomic_json(self.output / "coarse" / runner.PLAN, self.plan)
        for records in ([], [{"request_id": self.plan["requests"][0]["request_id"]}] * len(self.plan["requests"])):
            manifest = {"schema": f"{runner.PREFIX}_manifest/v1", "contract_sha256": runner.canonical_sha256(self.contract),
                        "request_count": len(self.plan["requests"]), "records": records}
            direct.atomic_json(self.output / "coarse" / runner.MANIFEST, manifest)
            with self.assertRaises(ERRORS):
                runner.analyze(self.output, "coarse")
        with patch.object(runner.shared, "bpy", object()), patch.object(direct, "_render_one") as render_one:
            with self.assertRaisesRegex(direct.ReducedDirectError, "existing/partial"):
                runner.render(self.output, self.assets, "coarse")
            render_one.assert_not_called()
        with self.assertRaises(ERRORS):
            runner._render_plan(self.output, "confirmation", self.contract)
        for command in ("confirmation", "render-full", "full90"):
            with self.assertRaises(SystemExit):
                runner.build_parser().parse_args([command, "--output-root", str(self.output)])

    def test_actual_png_decoder_and_artifact_metadata_rejections(self):
        request = self.plan["requests"][0]
        directory = self.output / "coarse"
        record = self._render_fixture(directory, request, self.contract)
        row, state = direct._decode_record(directory, request, self.contract, record)
        self.assertEqual(row["request_id"], request["request_id"])
        self.assertTrue(row["eligible_ok"])
        self.assertEqual(state, self.contract["base_scene_state"])
        for kind in ("image", "mask", "metadata"):
            changed = dict(record, **{f"{kind}_sha256": "0" * 64})
            with self.assertRaisesRegex(direct.ReducedDirectError, "hash differs"):
                direct._decode_record(directory, request, self.contract, changed)
        metadata_path = directory / record["metadata_relative_path"]
        metadata = runner.load_json(metadata_path)
        for section, field, value in (("material", "socket_rgba", [0, 0, 0, 1]),
                                      ("camera", "view_index", 4), ("renderer", "samples", 1)):
            changed = deepcopy(metadata)
            changed[section][field] = value
            direct.atomic_json(metadata_path, changed)
            updated = dict(record, metadata_sha256=runner.file_sha256(metadata_path))
            with self.assertRaises(ERRORS):
                direct._decode_record(directory, request, self.contract, updated)


if __name__ == "__main__":
    unittest.main()
