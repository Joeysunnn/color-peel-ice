from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.methods.colorpeel_ice import natural_image_target_selection as selection
from src.methods.colorpeel_ice import renderer_color_calibration as calibration
from src.methods.colorpeel_ice.natural_image_masks import canonical_sha256


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = REPO_ROOT / calibration.PROTOCOL_RELPATH
EXPECTED_PROTOCOL_HASH = "476d01c5a383ead7d1ceec5c276bb26d809cf2fabf51cd81e0c49695cc4aa48e"
EXPECTED_REQUEST_HASHES = {
    "preflight": "c2b5a4078f9407c2e92d555924a89ad5ba5640eeee1ef5c38e41402253ea08ff",
    "reduced_direct": "0cbbd38a7411fe02faaa4573130a40c3e6da53adc86fdeda6174af6f70254585",
    "full_invariance": "36cd0f957f8049e84d08e73cfa278155539c7aefa4951cf07dba1e01444a2aff",
    "full_expansion": "4ad0ad54e01634f0916c746372d089922a397b0e0b61a2c657d63cbf776c2b3a",
}


def measurement_rows(errors=(0, 3, 5), record=None) -> list[dict]:
    rows = [
        {"stable_id": "a", "shape": "sphere", "view_index": view,
         "target_a": 0, "target_b": 0, "median_a": error, "median_b": 0, "median_L": 50,
         "iqr_L": 1, "iqr_a": 1, "iqr_b": 1,
         "object_ratio_ok": True, "interior_ok": True, "eligible_ok": True}
        for view, error in zip([0, 8, 16], errors)
    ]
    if record is not None:
        for row, error in zip(rows, errors):
            row.update(stable_id=record["source"]["stable_id"], target_a=record["target"]["a"],
                       target_b=record["target"]["b"], median_a=record["target"]["a"] + error,
                       median_b=record["target"]["b"])
    return rows


def direct_measurements(requests) -> list[dict]:
    return [dict(request, median_L=50, median_a=request["target_a"], median_b=request["target_b"],
                 iqr_L=1, iqr_a=1, iqr_b=1, object_ratio_ok=True, interior_ok=True, eligible_ok=True)
            for request in requests]


def coarse_measurement_summaries(record) -> list[dict]:
    target = record["target"]
    return [calibration.summarize_fallback_candidate(candidate, fallback_measurements(candidate, record))
            for candidate in calibration.coarse_fallback_candidates(target["a"], target["b"])
            if candidate["gamut"] == "in"]


def fallback_measurements(candidate, record, errors=(1, 1, 1), coarse_summaries=None) -> list[dict]:
    target = record["target"]
    requests = calibration.build_fallback_candidate_requests(target["a"], target["b"], candidate,
                                                            coarse_summaries=coarse_summaries)
    rows = direct_measurements(requests)
    for row, error in zip(rows, errors):
        row["median_a"] += error
    return rows


def complete_fallback_summaries(record, errors=(1, 1, 1)):
    target = record["target"]
    coarse = [calibration.summarize_fallback_candidate(candidate, fallback_measurements(candidate, record, errors))
              for candidate in calibration.coarse_fallback_candidates(target["a"], target["b"])
              if candidate["gamut"] == "in"]
    plan = calibration.fallback_candidate_search_plan(target["a"], target["b"], coarse)
    refine = [calibration.summarize_fallback_candidate(
        candidate, fallback_measurements(candidate, record, errors, coarse), coarse_summaries=coarse)
        for candidate in plan["candidates"] if candidate["source"] == "refine"]
    return coarse, refine


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


class RendererColorCalibrationProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = calibration.load_protocol(PROTOCOL_PATH)

    def test_frozen_protocol_validates_and_links_selection(self) -> None:
        self.assertEqual(self.protocol["schema"], calibration.SCHEMA)
        self.assertEqual(canonical_sha256(self.protocol), calibration.FROZEN_PROTOCOL_CANONICAL_SHA256)
        self.assertEqual(canonical_sha256(self.protocol), EXPECTED_PROTOCOL_HASH)
        self.assertEqual(self.protocol["request_hashes"], EXPECTED_REQUEST_HASHES)
        linked = self.protocol["frozen_inputs"]["selection_config"]
        self.assertEqual(linked["repo_relative_path"], selection.CONFIG_RELPATH)
        self.assertEqual(linked["canonical_sha256"], selection.FROZEN_CONFIG_CANONICAL_SHA256)
        self.assertEqual(linked["source_audit_manifest_sha256"], selection.SOURCE_AUDIT_MANIFEST_SHA256)
        self.assertEqual(linked["selected_stable_id_order_sha256"], "0fcacd0759a34bb5979663a746d28e8f3e50d12f949e1d1f5242a06a63a4c67f")
        self.assertEqual(linked["pipeline_assignment_sha256"], "2c4efac08e969f80509e25fa81570058e0069c3f6684763d22e9cd5af0078a90")
        self.assertEqual([len(selection.records_for_pipeline_role(role)) for role in selection.PIPELINE_ROLES], [6, 3, 2])
        self.assertFalse(self.protocol["approval_state"]["recoloring_approved"])
        self.assertFalse(self.protocol["approval_state"]["training_approved"])

    def test_frozen_path_canonical_hash_and_strict_schema_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "d1_renderer_color_calibration_protocol_v1.json"
            write_json(copied, self.protocol)
            with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "frozen repository protocol"):
                calibration.load_protocol(copied)

            repo = Path(temporary) / "repo"
            protocol_path = repo / calibration.PROTOCOL_RELPATH
            mutated = deepcopy(self.protocol)
            mutated["renderer_contract"]["samples"] = 256
            write_json(protocol_path, mutated)
            with patch.object(calibration, "REPO_ROOT", repo):
                with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "canonical hash drift"):
                    calibration.load_protocol(protocol_path)

        cases = []
        extra = deepcopy(self.protocol)
        extra["unexpected"] = True
        cases.append((extra, "fields differ"))
        nested_extra = deepcopy(self.protocol)
        nested_extra["planning"]["preflight"]["unexpected"] = True
        cases.append((nested_extra, "planning.preflight fields differ"))
        absolute = deepcopy(self.protocol)
        absolute["renderer_contract"]["legacy_renderer_note"] = "C:\\server\\artifact"
        cases.append((absolute, "absolute path"))
        bad_hash = deepcopy(self.protocol)
        bad_hash["frozen_inputs"]["selection_config"]["source_audit_manifest_sha256"] = "0" * 64
        cases.append((bad_hash, "source audit manifest hash"))
        for value, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(calibration.RendererColorCalibrationError, message):
                    calibration._validate_protocol_structure(value)

    def test_every_protocol_leaf_is_frozen_even_for_direct_structure_validation(self) -> None:
        def leaves(value, path=()):
            if isinstance(value, dict):
                for key, nested in value.items():
                    yield from leaves(nested, path + (key,))
            elif isinstance(value, list):
                for index, nested in enumerate(value):
                    yield from leaves(nested, path + (index,))
            else:
                yield path, value

        for path, original in leaves(self.protocol):
            with self.subTest(path=path):
                mutated = deepcopy(self.protocol)
                parent = mutated
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = not original if isinstance(original, bool) else (
                    original + 1 if isinstance(original, (int, float)) else "DRIFT")
                with self.assertRaises(calibration.RendererColorCalibrationError):
                    calibration._validate_protocol_structure(mutated)

    def test_metadata_order_and_nested_extra_fields_are_rejected(self) -> None:
        mutated = deepcopy(self.protocol)
        mutated["metadata_contract"]["required_fields"].reverse()
        with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "metadata required fields"):
            calibration._validate_protocol_structure(mutated)
        for key in ["object", "camera", "lights", "background", "runtime_assets"]:
            mutated = deepcopy(self.protocol)
            mutated["scene_contract"][key]["extra"] = True
            with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "fields differ"):
                calibration._validate_protocol_structure(mutated)


class RendererColorConversionTests(unittest.TestCase):
    def assert_close_list(self, observed, expected, places=5) -> None:
        self.assertEqual(len(observed), len(expected))
        for got, want in zip(observed, expected):
            self.assertAlmostEqual(got, want, places=places)

    def test_known_lab_to_linear_srgb_references_and_rounding(self) -> None:
        self.assert_close_list(calibration.lab_d65_to_linear_srgb(0.0, 0.0, 0.0), [0.0, 0.0, 0.0])
        self.assert_close_list(calibration.lab_d65_to_linear_srgb(100.0, 0.0, 0.0), [1.0, 1.0, 1.0])
        self.assert_close_list(
            calibration.lab_d65_to_linear_srgb(53.2408, 80.0925, 67.2032),
            [1.0, 0.0, 0.0],
            places=4,
        )
        half_byte_linear = calibration.srgb_to_linear_channel(10.5 / 255.0)
        self.assertEqual(calibration.linear_rgb_to_srgb_uint8([half_byte_linear, 0.0, 1.0]), [11, 0, 255])

    def test_preview_linear_socket_are_separate_and_gamut_is_never_clipped(self) -> None:
        record = selection.pipeline_development_records()[0]
        payload = calibration.color_payload_for_target(record)
        self.assertEqual(payload["L_input"], 50.0)
        self.assertEqual(len(payload["linear_rgb"]), 3)
        self.assertEqual(len(payload["socket_rgba"]), 4)
        self.assertEqual(payload["socket_rgba"][-1], 1.0)
        self.assertEqual(len(payload["preview_srgb_uint8"]), 3)
        self.assertNotEqual(payload["preview_srgb_uint8"], payload["linear_rgb"])
        with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "out of gamut"):
            calibration.assert_linear_rgb_in_gamut([-0.01, 0.0, 1.0])
        with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "out of gamut"):
            calibration.assert_linear_rgb_in_gamut([0.0, 1.01, 0.0])


class RendererColorRequestPlanTests(unittest.TestCase):
    def test_exact_request_counts_order_hashes_and_role_exclusion(self) -> None:
        preflight = calibration.build_preflight_requests()
        reduced = calibration.build_reduced_direct_requests()
        full = calibration.build_full_invariance_requests()
        expansion = calibration.build_full_expansion_requests()
        self.assertEqual([len(preflight), len(reduced), len(full), len(expansion)], [4, 18, 90, 72])
        self.assertEqual([row["expected_decoded_srgb_uint8"] for row in preflight],
                         [[118, 118, 118], [188, 188, 188], [225, 89, 63], [63, 196, 243]])

        development_ids = [row["source"]["stable_id"] for row in selection.pipeline_development_records()]
        self.assertEqual([row["stable_id"] for row in reduced[0::3]], development_ids)
        self.assertEqual(reduced[0]["stable_id"], "D1GT:86/139.png")
        self.assertEqual([row["view_index"] for row in reduced[:3]], [0, 8, 16])
        self.assertEqual([row["shape"] for row in full[:15]], ["cube"] * 5 + ["sphere"] * 5 + ["cylinder"] * 5)

        reduced_keys = {calibration.render_key(row) for row in reduced}
        full_keys = {calibration.render_key(row) for row in full}
        expansion_keys = {calibration.render_key(row) for row in expansion}
        self.assertTrue(reduced_keys.issubset(full_keys))
        self.assertTrue(expansion_keys.isdisjoint(reduced_keys))
        self.assertEqual(expansion_keys, full_keys - reduced_keys)
        self.assertFalse(any(row["stable_id"] in selection.PIPELINE_ROLES["protocol_confirmation"] for row in full))
        self.assertFalse(any(row["stable_id"] in selection.PIPELINE_ROLES["stress_test"] for row in full))

        summary = calibration.request_plan_summary()
        self.assertEqual(summary["counts"], {"preflight": 4, "reduced_direct": 18, "full_invariance": 90, "full_expansion": 72})
        self.assertEqual(summary["hashes"]["preflight"], canonical_sha256(preflight))
        self.assertEqual(summary["hashes"]["full_expansion"], canonical_sha256(expansion))
        self.assertEqual(summary["hashes"], EXPECTED_REQUEST_HASHES)
        for stage, rows in zip(EXPECTED_REQUEST_HASHES, [preflight, reduced, full, expansion]):
            self.assertEqual(canonical_sha256(rows), EXPECTED_REQUEST_HASHES[stage])

    def test_all_overlap_payload_fields_are_reusable_except_stage(self) -> None:
        reduced = calibration.build_reduced_direct_requests()
        full = {calibration.render_key(row): row for row in calibration.build_full_invariance_requests()}
        expansion = calibration.build_full_expansion_requests()
        for row in reduced + expansion:
            expected = dict(full[calibration.render_key(row)])
            expected["stage"] = row["stage"]
            self.assertEqual(row, expected)
        self.assertEqual([row["render_seed"] for row in reduced[:3]], [820020, 820028, 820036])

    def test_builders_reject_request_payload_drift(self) -> None:
        with patch.object(calibration, "REQUEST_BASE_SEED", 830000):
            for builder in [calibration.build_reduced_direct_requests,
                            calibration.build_full_invariance_requests, calibration.build_full_expansion_requests]:
                with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "request hash drifted"):
                    builder()
        with patch.object(calibration, "PREFLIGHT_BASE_SEED", 830000):
            with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "request hash drifted"):
                calibration.build_preflight_requests()


class RendererColorPreflightGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [dict(request, decoded_png_rgb_uint8=list(request["expected_decoded_srgb_uint8"]),
                          decoded_lab_delta=0.0) for request in calibration.build_preflight_requests()]

    def test_complete_preflight_and_exact_error_boundaries_pass(self) -> None:
        self.assertTrue(calibration.summarize_preflight_measurements(self.rows)["overall_pass"])
        for row in self.rows:
            row["decoded_png_rgb_uint8"] = [byte + delta for byte, delta in zip(row["expected_decoded_srgb_uint8"], [-1, 0, 1])]
            row["decoded_lab_delta"] = 0.5
        summary = calibration.summarize_preflight_measurements(list(reversed(self.rows)))
        self.assertEqual(summary["probe_count"], 4)
        self.assertTrue(summary["overall_pass"])
        self.assertTrue(all(probe["passed"] for probe in summary["probes"]))

    def test_preflight_threshold_failures_are_decisions(self) -> None:
        for key, value in [("decoded_png_rgb_uint8", [120, 118, 118]), ("decoded_lab_delta", 0.50001),
                           ("decoded_lab_delta", -0.01)]:
            rows = deepcopy(self.rows)
            rows[0][key] = value
            summary = calibration.summarize_preflight_measurements(rows)
            self.assertFalse(summary["overall_pass"])
            self.assertFalse(summary["probes"][0]["passed"])

    def test_preflight_incomplete_duplicate_wrong_stage_and_payload_reject(self) -> None:
        cases = [self.rows[:1], self.rows[:-1], self.rows + [self.rows[0]], self.rows[:-1] + [self.rows[0]]]
        for key, value in [("request_id", "unknown"), ("stage", "reduced_direct"), ("render_seed", 0),
                           ("linear_rgb", [0, 0, 0]), ("expected_byte_tolerance", 99)]:
            rows = deepcopy(self.rows)
            rows[0][key] = value
            cases.append(rows)
        for rows in cases:
            with self.assertRaises(calibration.RendererColorCalibrationError):
                calibration.summarize_preflight_measurements(rows)

    def test_preflight_decoded_values_must_be_valid_bytes_and_finite_delta(self) -> None:
        for key, values in [("decoded_png_rgb_uint8", [[118, 118], [256, 0, 0], [-1, 0, 0], [1.0, 0, 0], [True, 0, 0], None]),
                            ("decoded_lab_delta", [float("nan"), float("inf"), None, "0.0"])]:
            for value in values:
                rows = deepcopy(self.rows)
                rows[0][key] = value
                with self.assertRaises(calibration.RendererColorCalibrationError):
                    calibration.summarize_preflight_measurements(rows)


class RendererColorMeasurementTests(unittest.TestCase):
    def test_adaptive_radius_and_gate_boundaries(self) -> None:
        small = calibration.derive_interior_contract(100, 10, 12, image_area=10000, background_area=9900)
        large = calibration.derive_interior_contract(10000, 100, 120, image_area=100000, background_area=90000)
        self.assertEqual(small["radius"], 1)
        self.assertEqual(large["radius"], 3)
        self.assertGreater(large["radius"], small["radius"])
        self.assertTrue(calibration.derive_interior_contract(50, 10, 10, image_area=10000)["object_ratio_ok"])
        self.assertFalse(calibration.derive_interior_contract(49, 10, 10, image_area=10000)["object_ratio_ok"])
        self.assertFalse(calibration.derive_interior_contract(9000, 100, 100, image_area=10000)["object_ratio_ok"])
        self.assertTrue(calibration.derive_interior_contract(1000, 50, 50, image_area=10000, interior_area=150, eligible_count=120)["interior_ok"])
        self.assertTrue(calibration.derive_interior_contract(1000, 50, 50, image_area=10000, interior_area=150, eligible_count=120)["eligible_ok"])
        self.assertFalse(calibration.derive_interior_contract(1000, 50, 50, image_area=10000, interior_area=149)["interior_ok"])
        self.assertFalse(calibration.derive_interior_contract(1000, 50, 50, image_area=10000, interior_area=150, eligible_count=119)["eligible_ok"])
        with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "complement"):
            calibration.derive_interior_contract(100, 10, 10, image_area=1000, background_area=901)
        with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "requires interior_area"):
            calibration.derive_interior_contract(100, 10, 10, image_area=1000, eligible_count=80)

    def test_nearest_rank_p90_and_threshold_boundaries(self) -> None:
        self.assertEqual(calibration.nearest_rank_p90([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 9)
        passing_rows = measurement_rows()
        summary = calibration._summarize_chroma_measurements(passing_rows)
        self.assertEqual(summary["targets"]["a"]["median_e_ch"], 3)
        self.assertEqual(summary["targets"]["a"]["p90_e_ch"], 5)
        self.assertTrue(summary["overall_pass"])
        failing = deepcopy(passing_rows)
        failing[-1]["median_a"] = 5.1
        self.assertFalse(calibration._summarize_chroma_measurements(failing)["overall_pass"])

    def test_nonfinite_scalar_summaries_and_missing_or_failed_gates_are_rejected(self) -> None:
        for invalid in [float("nan"), float("inf"), -float("inf")]:
            for helper in [calibration.median, calibration.nearest_rank_p90]:
                with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "finite"):
                    helper([0, invalid, 1])
            for key in ["median_L", "median_a", "median_b", "target_a", "target_b", "iqr_L", "iqr_a", "iqr_b", "e_ch"]:
                rows = measurement_rows()
                rows[0][key] = invalid
                with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "finite"):
                    calibration._summarize_chroma_measurements(rows)
        for key in ["object_ratio_ok", "interior_ok", "eligible_ok"]:
            for gate in [None, False, 1]:
                rows = measurement_rows()
                if gate is None:
                    del rows[0][key]
                else:
                    rows[0][key] = gate
                with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "explicitly true"):
                    calibration._summarize_chroma_measurements(rows)

    def test_duplicate_identity_and_inconsistent_target_are_rejected(self) -> None:
        rows = measurement_rows()
        with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "duplicate render identity"):
            calibration._summarize_chroma_measurements(rows + [rows[0]])
        rows[1]["target_a"] = 1
        with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "inconsistent target"):
            calibration._summarize_chroma_measurements(rows)

    def test_complete_direct_matrices_and_merged_expansion_pass(self) -> None:
        for stage, builder in [("reduced_direct", calibration.build_reduced_direct_requests),
                               ("full_invariance", calibration.build_full_invariance_requests)]:
            rows = direct_measurements(builder())
            summary = calibration.summarize_chroma_measurements(rows, stage)
            self.assertTrue(summary["overall_pass"])
            self.assertEqual(len(summary["targets"]), 6)
            self.assertEqual(summary["render_count"], len(rows))
        merged = direct_measurements(calibration.build_reduced_direct_requests() + calibration.build_full_expansion_requests())
        self.assertTrue(calibration.summarize_chroma_measurements(merged, "full_invariance")["overall_pass"])

    def test_zero_iqr_passes_and_negative_iqr_rejects_complete_direct_data(self) -> None:
        rows = direct_measurements(calibration.build_reduced_direct_requests())
        for row in rows:
            row.update(iqr_L=0, iqr_a=0, iqr_b=0)
        self.assertTrue(calibration.summarize_chroma_measurements(rows, "reduced_direct")["overall_pass"])
        for key in ["iqr_L", "iqr_a", "iqr_b"]:
            mutated = deepcopy(rows)
            mutated[0][key] = -0.001
            with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "nonnegative"):
                calibration.summarize_chroma_measurements(mutated, "reduced_direct")

    def test_incomplete_extra_duplicate_wrong_stage_and_payload_direct_rows_reject(self) -> None:
        reduced = direct_measurements(calibration.build_reduced_direct_requests())
        expansion = direct_measurements(calibration.build_full_expansion_requests())
        cases = [(reduced[:1], "reduced_direct"), (reduced[1:], "reduced_direct"),
                 (reduced[3:], "reduced_direct"), (expansion, "full_invariance"), (expansion, "full_expansion"),
                 (reduced + [reduced[0]], "reduced_direct"),
                 (reduced[:-1] + [reduced[0]], "reduced_direct")]
        for key, value in [("stage", "full_invariance"), ("view_index", 4), ("request_id", "forged"),
                           ("stable_id", "unknown"), ("target_a", 999)]:
            mutated = deepcopy(reduced)
            mutated[0][key] = value
            cases.append((mutated, "reduced_direct"))
        missing_stage = deepcopy(reduced)
        del missing_stage[0]["stage"]
        cases.append((missing_stage, "reduced_direct"))
        merged = direct_measurements(calibration.build_reduced_direct_requests() + calibration.build_full_expansion_requests())
        merged[0]["stage"], merged[-1]["stage"] = merged[-1]["stage"], merged[0]["stage"]
        cases.append((merged, "full_invariance"))
        for rows, stage in cases:
            with self.subTest(stage=stage, count=len(rows)):
                with self.assertRaises(calibration.RendererColorCalibrationError):
                    calibration.summarize_chroma_measurements(rows, stage)


class RendererColorFallbackTests(unittest.TestCase):
    def test_fallback_candidate_counts_bounds_gamut_and_dedup(self) -> None:
        record = selection.pipeline_development_records()[4]
        target = record["target"]
        coarse = calibration.coarse_fallback_candidates(target["a"], target["b"])
        self.assertEqual(len(coarse), 63)
        self.assertTrue(any(row["gamut"] == "out" for row in coarse))
        summaries = coarse_measurement_summaries(record)
        refine = calibration.refine_fallback_candidates(target["a"], target["b"], summaries)
        self.assertLessEqual(len(refine), 25)
        self.assertTrue(all(35.0 <= row["L_input"] <= 75.0 for row in refine))
        self.assertTrue(all(0.60 <= row["chroma_scale"] <= 1.20 for row in refine))
        self.assertEqual(len({(row["L_input"], row["chroma_scale"]) for row in refine}), len(refine))
        plan = calibration.fallback_candidate_search_plan(target["a"], target["b"], summaries)
        self.assertEqual(plan["coarse_count"], 63)
        self.assertEqual(plan["evaluated_coarse_count"], len(summaries))
        self.assertEqual(plan["selected_anchor"]["L_input"], 50)
        self.assertEqual(plan["selected_anchor"]["chroma_scale"], 1)
        self.assertLessEqual(plan["evaluation_count"], 88)
        self.assertTrue(all(row["gamut"] == "in" for row in plan["candidates"]))

    def test_fallback_selection_tie_break_and_no_pass_diagnostic(self) -> None:
        no_pass = [
            {"L_input": 50.0, "chroma_scale": 1.0, "objective": 2.0, "p90_e_ch": 6.0, "passes": False},
            {"L_input": 48.0, "chroma_scale": 1.0, "objective": 3.0, "p90_e_ch": 5.0, "passes": False},
        ]
        rejected = calibration._rank_fallback_candidates(no_pass)
        self.assertFalse(rejected["accepted_fallback"])
        self.assertIsNone(rejected["accepted"])
        self.assertEqual(rejected["diagnostic_best"]["objective"], 2.0)

        tied = [
            {"L_input": 52.0, "chroma_scale": 1.0, "objective": 1.0, "p90_e_ch": 4.0, "passes": True},
            {"L_input": 51.0, "chroma_scale": 1.0, "objective": 1.0, "p90_e_ch": 4.0, "passes": True},
            {"L_input": 51.0, "chroma_scale": 0.98, "objective": 1.0, "p90_e_ch": 4.1, "passes": True},
        ]
        accepted = calibration._rank_fallback_candidates(tied)
        self.assertTrue(accepted["accepted_fallback"])
        self.assertEqual(accepted["accepted"]["L_input"], 51.0)
        self.assertEqual(accepted["accepted"]["chroma_scale"], 1.0)

    def test_fallback_measurement_objective(self) -> None:
        record = selection.pipeline_development_records()[0]
        candidate = next(row for row in calibration.coarse_fallback_candidates(record["target"]["a"], record["target"]["b"])
                         if row["L_input"] == 50 and row["chroma_scale"] == 1)
        rows = fallback_measurements(candidate, record, (1, 3, 5))
        rows[0]["median_L"] = 49
        rows[2]["median_L"] = 51
        summary = calibration.summarize_fallback_candidate(candidate, rows)
        self.assertEqual(summary["median_e_ch"], 3)
        self.assertEqual(summary["p90_e_ch"], 5)
        self.assertEqual(summary["median_render_L"], 50)
        self.assertEqual(summary["objective"], 4.25)
        self.assertTrue(summary["passes"])

    def test_refine_rejects_incomplete_duplicate_extra_and_tampered_coarse_summaries(self) -> None:
        record = selection.pipeline_development_records()[4]
        target = record["target"]
        a, b = target["a"], target["b"]
        coarse = calibration.coarse_fallback_candidates(a, b)
        summaries = coarse_measurement_summaries(record)
        out_of_gamut = next(row for row in coarse if row["gamut"] == "out")
        cases = [summaries[0], summaries[:-1], summaries + [summaries[0]],
                 summaries[:-1] + [summaries[0]], summaries[:-1] + [out_of_gamut]]
        for key, value in [("candidate_a", 999), ("linear_rgb", [0, 0, 0]), ("source", "refine"),
                           ("L_input", 51), ("stable_id", "other"), ("target_a", 0),
                           ("objective", -1), ("passes", False), ("best", True)]:
            mutated = deepcopy(summaries)
            mutated[0][key] = value
            cases.append(mutated)
        for invalid in cases:
            for helper in [calibration.refine_fallback_candidates, calibration.fallback_candidate_search_plan]:
                with self.assertRaises(calibration.RendererColorCalibrationError):
                    helper(a, b, invalid)

    def test_planner_selects_best_from_complete_summaries_without_caller_anchor(self) -> None:
        record = selection.pipeline_development_records()[0]
        a, b = record["target"]["a"], record["target"]["b"]
        summaries = coarse_measurement_summaries(record)
        first_core = next(row for row in calibration.coarse_fallback_candidates(a, b) if row["gamut"] == "in")
        summaries[0] = calibration.summarize_fallback_candidate(first_core, fallback_measurements(first_core, record, (0, 0, 0)))
        plan = calibration.fallback_candidate_search_plan(a, b, list(reversed(summaries)))
        self.assertEqual(plan["selected_anchor"], summaries[0])
        self.assertEqual(plan["evaluated_coarse_count"], len(summaries))
        with self.assertRaises(calibration.RendererColorCalibrationError):
            calibration.fallback_candidate_search_plan(a, b, summaries[-1])

    def test_fallback_summary_validates_frozen_target_core_and_selected_refine_grid(self) -> None:
        record = selection.pipeline_development_records()[0]
        a, b = record["target"]["a"], record["target"]["b"]
        summaries = coarse_measurement_summaries(record)
        refine = next(row for row in calibration.refine_fallback_candidates(a, b, summaries) if row["gamut"] == "in")
        rows = fallback_measurements(refine, record, coarse_summaries=summaries)
        self.assertTrue(calibration.summarize_fallback_candidate(refine, rows, coarse_summaries=summaries)["passes"])
        with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "complete coarse summaries"):
            calibration.summarize_fallback_candidate(refine, rows)
        coarse = next(row for row in calibration.coarse_fallback_candidates(a, b) if row["gamut"] == "in")
        for candidate in [dict(coarse, candidate_b=999), dict(coarse, L_input=51),
                          calibration.coarse_fallback_candidates(0, 0)[0], dict(refine, L_input=74)]:
            with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "candidate core"):
                calibration.summarize_fallback_candidate(candidate, rows, coarse_summaries=summaries)
        for key, value in [("stable_id", "unknown"), ("target_a", 0)]:
            mutated = deepcopy(rows)
            for row in mutated:
                row[key] = value
            with self.assertRaises(calibration.RendererColorCalibrationError):
                calibration.summarize_fallback_candidate(coarse, mutated)
        for role in ["protocol_confirmation", "stress_test"]:
            excluded = selection.records_for_pipeline_role(role)[0]
            with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "pipeline_development target"):
                calibration.summarize_fallback_candidate(coarse, measurement_rows((1, 1, 1), excluded))

    def test_fallback_planning_excludes_confirmation_and_stress_targets(self) -> None:
        for role in ["protocol_confirmation", "stress_test"]:
            for record in selection.records_for_pipeline_role(role):
                a, b = record["target"]["a"], record["target"]["b"]
                anchor = next(row for row in calibration.coarse_fallback_candidates(a, b) if row["gamut"] == "in")
                with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "pipeline_development target"):
                    calibration.fallback_candidate_search_plan(a, b, anchor)

    def test_fallback_summary_requires_one_target_and_exact_sphere_views(self) -> None:
        candidate = {"L_input": 50.0, "chroma_scale": 1.0, "gamut": "in"}
        rows = measurement_rows()
        cases = [rows[:2]]
        for key, value in [("stable_id", "other"), ("shape", "cube"), ("view_index", 4)]:
            mutated = deepcopy(rows)
            mutated[0][key] = value
            cases.append(mutated)
        for invalid in cases:
            with self.assertRaises(calibration.RendererColorCalibrationError):
                calibration.summarize_fallback_candidate(candidate, invalid)

    def test_fallback_selection_rejects_nonfinite_sort_inputs(self) -> None:
        row = {"L_input": 50.0, "chroma_scale": 1.0, "objective": 2.0, "p90_e_ch": 3.0, "passes": True}
        for key in ["L_input", "chroma_scale", "objective", "p90_e_ch"]:
            with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "finite"):
                calibration._rank_fallback_candidates([dict(row, **{key: float("nan")})])


class RendererColorFallbackWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.record = selection.pipeline_development_records()[0]
        cls.a, cls.b = cls.record["target"]["a"], cls.record["target"]["b"]
        cls.coarse, cls.refine = complete_fallback_summaries(cls.record)
        cls.plan = calibration.fallback_candidate_search_plan(cls.a, cls.b, cls.coarse)

    def test_candidate_measurements_cannot_be_reused_for_another_candidate(self) -> None:
        for source in ["coarse", "refine"]:
            candidates = [row for row in self.plan["candidates"] if row["source"] == source]
            rows = fallback_measurements(candidates[0], self.record, coarse_summaries=self.coarse)
            self.assertTrue(calibration.summarize_fallback_candidate(candidates[0], rows, coarse_summaries=self.coarse)["passes"])
            with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "request identity"):
                calibration.summarize_fallback_candidate(candidates[1], rows, coarse_summaries=self.coarse)
            with self.assertRaises(calibration.RendererColorCalibrationError):
                calibration.summarize_fallback_candidate(candidates[0], measurement_rows(record=self.record), coarse_summaries=self.coarse)

    def test_candidate_request_core_stage_seed_and_unique_identity_are_strict(self) -> None:
        for source in ["coarse", "refine"]:
            candidate = next(row for row in self.plan["candidates"] if row["source"] == source)
            original = fallback_measurements(candidate, self.record, coarse_summaries=self.coarse)
            cases = [original[:-1], original[:-1] + [original[0]]]
            for key, value in [("L_input", 51), ("chroma_scale", 0.99), ("candidate_a", 999), ("candidate_b", 999),
                               ("linear_rgb", [0, 0, 0]), ("socket_rgba", [0, 0, 0, 1]), ("preview_srgb_uint8", [0, 0, 0]),
                               ("source", "forged"), ("stage", "full_invariance"), ("shape", "cube"), ("view_index", 4),
                               ("render_seed", 0), ("stable_id", "unknown"), ("request_id", "forged")]:
                rows = deepcopy(original)
                rows[0][key] = value
                cases.append(rows)
            for rows in cases:
                with self.assertRaises(calibration.RendererColorCalibrationError):
                    calibration.summarize_fallback_candidate(candidate, rows, coarse_summaries=self.coarse)
        requests = [request for candidate in self.plan["candidates"]
                    for request in calibration.build_fallback_candidate_requests(self.a, self.b, candidate, coarse_summaries=self.coarse)]
        self.assertEqual(len(requests), len({row["request_id"] for row in requests}))
        self.assertEqual(len(requests), len({row["render_seed"] for row in requests}))

    def test_complete_search_accepts_only_after_all_unique_candidates(self) -> None:
        selected = calibration.choose_fallback_candidate(self.a, self.b, self.coarse, self.refine)
        self.assertTrue(selected["accepted_fallback"])
        self.assertEqual(selected["evaluated_coarse_count"], len(self.coarse))
        self.assertEqual(selected["evaluated_refine_count"], len(self.refine))
        self.assertEqual(selected["evaluated_candidate_count"], len(self.coarse) + len(self.refine))
        self.assertLessEqual(selected["evaluated_candidate_count"], 88)
        self.assertEqual(selected["accepted"]["L_input"], 50)
        self.assertEqual(selected["accepted"]["chroma_scale"], 1)

    def test_final_selector_rejects_subsets_extra_duplicate_tampered_and_wrong_target(self) -> None:
        cases = [(self.coarse[:1], []), (self.coarse[:-1], self.refine), (self.coarse, self.refine[:-1]),
                 (self.coarse, self.refine + [self.refine[0]]),
                 (self.coarse, self.refine[:-1] + [self.refine[0]]),
                 (self.coarse, self.refine[:-1] + [self.coarse[0]])]
        for key, value in [("candidate_a", 999), ("target_a", 0), ("stable_id", "unknown"),
                           ("request_payloads_sha256", "0" * 64), ("objective", -1)]:
            mutated = deepcopy(self.refine)
            mutated[0][key] = value
            cases.append((self.coarse, mutated))
        for coarse, refine in cases:
            with self.assertRaises(calibration.RendererColorCalibrationError):
                calibration.choose_fallback_candidate(self.a, self.b, coarse, refine)
        other = selection.pipeline_development_records()[1]["target"]
        with self.assertRaises(calibration.RendererColorCalibrationError):
            calibration.choose_fallback_candidate(other["a"], other["b"], self.coarse, self.refine)

    def test_complete_search_with_no_pass_is_diagnostic_and_cannot_confirm(self) -> None:
        coarse, refine = complete_fallback_summaries(self.record, (6, 6, 6))
        selected = calibration.choose_fallback_candidate(self.a, self.b, coarse, refine)
        self.assertFalse(selected["accepted_fallback"])
        self.assertIsNone(selected["accepted"])
        self.assertIsNotNone(selected["diagnostic_best"])
        with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "accepted complete-search candidate"):
            calibration.build_fallback_confirmation_requests(self.a, self.b, coarse, refine)

    def test_selected_candidate_exact_fifteen_render_confirmation(self) -> None:
        requests = calibration.build_fallback_confirmation_requests(self.a, self.b, self.coarse, self.refine)
        self.assertEqual(len(requests), 15)
        self.assertEqual({(row["shape"], row["view_index"]) for row in requests},
                         {(shape, view) for shape in ["cube", "sphere", "cylinder"] for view in [0, 4, 8, 12, 16]})
        self.assertEqual(len({row["request_id"] for row in requests}), 15)
        rows = direct_measurements(requests)
        summary = calibration.summarize_fallback_confirmation(rows, self.a, self.b, self.coarse, self.refine)
        self.assertTrue(summary["confirmed"])
        self.assertTrue(summary["overall_pass"])
        for row in rows:
            row["median_a"] += 6
        failed = calibration.summarize_fallback_confirmation(rows, self.a, self.b, self.coarse, self.refine)
        self.assertFalse(failed["confirmed"])
        self.assertFalse(failed["overall_pass"])

    def test_refine_winner_flows_from_measurements_to_complete_confirmation(self) -> None:
        candidate = next(row for row in self.plan["candidates"] if row["source"] == "refine")
        measurements = fallback_measurements(candidate, self.record, (0, 0, 0), self.coarse)
        winner_summary = calibration.summarize_fallback_candidate(candidate, measurements, coarse_summaries=self.coarse)
        refine = deepcopy(self.refine)
        index = next(index for index, row in enumerate(refine)
                     if (row["L_input"], row["chroma_scale"]) == (candidate["L_input"], candidate["chroma_scale"]))
        refine[index] = winner_summary
        selected = calibration.choose_fallback_candidate(self.a, self.b, self.coarse, refine)
        self.assertEqual(selected["accepted"], winner_summary)
        requests = calibration.build_fallback_confirmation_requests(self.a, self.b, self.coarse, refine)
        self.assertTrue(all(row["source"] == "refine" and row["L_input"] == candidate["L_input"]
                            and row["chroma_scale"] == candidate["chroma_scale"] for row in requests))
        result = calibration.summarize_fallback_confirmation(direct_measurements(requests), self.a, self.b, self.coarse, refine)
        self.assertTrue(result["confirmed"])

    def test_confirmation_rejects_partial_three_view_direct_other_candidate_and_incomplete_search(self) -> None:
        requests = calibration.build_fallback_confirmation_requests(self.a, self.b, self.coarse, self.refine)
        rows = direct_measurements(requests)
        other_candidate = self.plan["candidates"][0]
        other_requests = calibration._fallback_requests(self.record, other_candidate, "fallback_confirmation",
                                                        ["cube", "sphere", "cylinder"], [0, 4, 8, 12, 16])
        cases = [rows[:1], rows[:-1], rows[:3], rows[:-1] + [rows[0]],
                 direct_measurements(calibration.build_full_invariance_requests()[:15]), direct_measurements(other_requests)]
        for key, value in [("stage", "fallback_coarse"), ("render_seed", 0), ("candidate_a", 999),
                           ("L_input", 51), ("chroma_scale", 0.99), ("target_b", 999)]:
            mutated = deepcopy(rows)
            mutated[0][key] = value
            cases.append(mutated)
        for invalid in cases:
            with self.assertRaises(calibration.RendererColorCalibrationError):
                calibration.summarize_fallback_confirmation(invalid, self.a, self.b, self.coarse, self.refine)
        with self.assertRaises(calibration.RendererColorCalibrationError):
            calibration.summarize_fallback_confirmation(rows, self.a, self.b, self.coarse, self.refine[:-1])
        for role in ["protocol_confirmation", "stress_test"]:
            target = selection.records_for_pipeline_role(role)[0]["target"]
            with self.assertRaisesRegex(calibration.RendererColorCalibrationError, "pipeline_development target"):
                calibration.build_fallback_confirmation_requests(target["a"], target["b"], self.coarse, self.refine)


if __name__ == "__main__":
    unittest.main()
