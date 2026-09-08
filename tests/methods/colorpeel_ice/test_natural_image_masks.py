import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.methods.colorpeel_ice import natural_image_masks as masks
from src.methods.colorpeel_ice import prepare_natural_image_pilot as prepare


def brute_force_edt(binary: np.ndarray) -> np.ndarray:
    false_points = np.argwhere(~binary)
    out = np.zeros(binary.shape, dtype=np.float64)
    if len(false_points) == 0:
        out.fill(np.inf)
        return out
    for y, x in np.argwhere(binary):
        distances = (false_points[:, 0] - y) ** 2 + (false_points[:, 1] - x) ** 2
        out[y, x] = math.sqrt(float(distances.min()))
    return out


class ExactEdtTests(unittest.TestCase):
    def test_exact_edt_matches_brute_force_for_constructed_inside_and_outside(self):
        cases = []
        base = np.zeros((9, 11), dtype=bool)
        base[2:7, 3:9] = True
        cases.append(base)
        diagonal = np.zeros((10, 10), dtype=bool)
        np.fill_diagonal(diagonal, True)
        diagonal[4:7, 2:8] = True
        cases.append(diagonal)
        touching = np.zeros((8, 12), dtype=bool)
        touching[:5, :4] = True
        touching[6, 10] = True
        cases.append(touching)
        for case in cases:
            np.testing.assert_allclose(masks.exact_edt(case), brute_force_edt(case), atol=1e-12)
            np.testing.assert_allclose(masks.exact_edt(~case), brute_force_edt(~case), atol=1e-12)

    def test_exact_edt_matches_brute_force_for_random_inside_and_outside(self):
        rng = np.random.default_rng(1234)
        for shape in ((5, 7), (8, 8), (9, 6)):
            for _ in range(8):
                case = rng.random(shape) > 0.42
                case[0, 0] = False
                case[-1, -1] = True
                np.testing.assert_allclose(masks.exact_edt(case), brute_force_edt(case), atol=1e-12)
                np.testing.assert_allclose(masks.exact_edt(~case), brute_force_edt(~case), atol=1e-12)


class NaturalImageMaskDerivationTests(unittest.TestCase):
    def test_prepare_cli_requires_explicit_roots_and_rejects_repo_output(self):
        parser = prepare.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        with self.assertRaisesRegex(masks.NaturalImageMaskError, "must not be inside repo"):
            prepare.reject_repo_output_dir(prepare.REPO_ROOT / "experiments" / "natural_image_subject_color_pilot")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = root / "empty"
            empty.mkdir()
            self.assertEqual(prepare.reject_repo_output_dir(empty), empty.resolve())
            nonempty = root / "nonempty"
            nonempty.mkdir()
            (nonempty / "marker.txt").write_text("existing output", encoding="utf-8")
            with self.assertRaisesRegex(
                masks.NaturalImageMaskError,
                "non-empty output directory will not be overwritten",
            ):
                prepare.reject_repo_output_dir(nonempty)

    def test_rectangle_color_mask_uses_largest_safe_edt_radius_deterministically(self):
        raw = np.zeros((512, 512), dtype=bool)
        raw[140:372, 180:340] = True
        bbox = masks.half_open_bbox(raw)
        d_in = masks.exact_edt(raw)

        first = masks.derive_color_mask(raw, bbox, d_in)
        second = masks.derive_color_mask(raw, bbox, d_in)

        self.assertEqual(first["selected_radius_px"], 2)
        self.assertEqual(first["status"], "PASS")
        self.assertEqual(first["candidates_desc"], [2, 1, 0])
        self.assertEqual(int(first["mask"].sum()), int(second["mask"].sum()))
        np.testing.assert_array_equal(first["mask"], second["mask"])
        self.assertTrue(first["candidate_evaluations"][0]["passed"])

    def test_thin_object_falls_back_to_review_without_disappearing_silently(self):
        raw = np.zeros((512, 512), dtype=bool)
        raw[200:202, 56:456] = True
        bbox = masks.half_open_bbox(raw)
        result = masks.derive_color_mask(raw, bbox, masks.exact_edt(raw))

        self.assertEqual(result["selected_radius_px"], 0)
        self.assertEqual(result["status"], "REVIEW")
        self.assertIn("REVIEW: erosion_fallback", result["flags"])
        self.assertFalse(result["candidate_evaluations"][0]["passed"])
        self.assertTrue(result["candidate_evaluations"][-1]["passed"])

    def test_multicomponent_gate_rejects_component_loss_before_fallback(self):
        raw = np.zeros((512, 512), dtype=bool)
        raw[80:120, 80:120] = True
        raw[300:305, 300:460] = True
        bbox = masks.half_open_bbox(raw)
        result = masks.derive_color_mask(raw, bbox, masks.exact_edt(raw))

        radius_one = next(row for row in result["candidate_evaluations"] if row["radius"] == 1)
        self.assertFalse(radius_one["passed"])
        self.assertFalse(radius_one["gates"]["significant_component_retention_ok"])
        self.assertEqual(result["selected_radius_px"], 0)
        self.assertEqual(result["status"], "REVIEW")

    def test_significant_component_split_is_rejected_even_when_total_component_count_does_not_increase(self):
        raw = np.zeros((512, 512), dtype=bool)
        raw[100:300, 100:300] = True
        raw[20:25, 20:25] = True
        candidate = raw.copy()
        candidate[100:300, 199:201] = False
        candidate[20:25, 20:25] = False
        raw_labels, raw_components = masks.connected_components(raw)

        evaluation = masks.evaluate_color_candidate(raw, candidate, raw_labels, raw_components)

        self.assertTrue(evaluation["gates"]["area_ok"])
        self.assertTrue(evaluation["gates"]["total_retention_ok"])
        self.assertTrue(evaluation["gates"]["component_count_ok"])
        self.assertFalse(evaluation["gates"]["significant_component_single_subcomponent_ok"])
        self.assertFalse(evaluation["passed"])
        significant = evaluation["significant_component_retentions"][0]
        self.assertEqual(significant["retained_subcomponent_count"], 2)
        self.assertGreater(significant["retention"], 0.98)

    def test_touching_edge_bbox_margin_records_clipping(self):
        raw = np.zeros((512, 512), dtype=bool)
        raw[255:512, 0:120] = True
        bbox = masks.half_open_bbox(raw)
        crop = masks.compute_tight_crop(bbox, (512, 512), w_out=0)

        self.assertEqual(crop["bbox_half_open"], [0, 255, 120, 512])
        self.assertEqual(crop["requested_margin_px"], 13)
        self.assertEqual(crop["crop_box_half_open"], [0, 242, 133, 512])
        self.assertTrue(crop["boundary_clipped"])
        self.assertEqual(crop["applied_margin_px"]["left"], 0)
        self.assertEqual(crop["applied_margin_px"]["bottom"], 0)

    def test_alpha_range_and_inside_outside_invariants(self):
        raw = np.zeros((512, 512), dtype=bool)
        raw[128:384, 128:384] = True
        bbox = masks.half_open_bbox(raw)
        alpha = masks.derive_alpha(raw, bbox, masks.exact_edt(raw))
        alpha_u16 = alpha["alpha_u16"]

        self.assertEqual(alpha["dtype"], "uint16")
        self.assertEqual(alpha["range"], [0, 65535])
        self.assertEqual(int(alpha_u16[~raw].max()), 0)
        self.assertEqual(int(alpha_u16[200, 200]), 65535)
        self.assertGreater(int(alpha_u16[128, 128]), 0)
        self.assertLess(int(alpha_u16[128, 128]), 65535)
        self.assertGreaterEqual(alpha["w_in_px"], 1)
        self.assertEqual(alpha["w_out_px"], 0)

    def test_alpha_min_width_override_marks_review_in_merged_status(self):
        raw = np.zeros((32, 32), dtype=bool)
        raw[10:11, 4:28] = True
        bbox = masks.half_open_bbox(raw)
        alpha = masks.derive_alpha(raw, bbox, masks.exact_edt(raw))

        self.assertEqual(alpha["requested_w_in_px"], 1)
        self.assertEqual(alpha["q50_cap_floor"], 0)
        self.assertEqual(alpha["capped_w_in_px"], 0)
        self.assertTrue(alpha["min_width_override"])
        self.assertEqual(alpha["w_in_px"], 1)
        self.assertEqual(alpha["status"], "REVIEW")
        self.assertIn("REVIEW: alpha_min_width_override", alpha["flags"])

        status, flags = masks.merge_sample_status("PASS", [], alpha["status"], alpha["flags"])
        self.assertEqual(status, "REVIEW")
        self.assertEqual(flags, ["REVIEW: alpha_min_width_override"])

    def test_canonical_alpha_png_validator_accepts_portable_pillow_integer_decode(self):
        alpha = np.array(
            [
                [0, 1, 65535],
                [1024, 32768, 50000],
            ],
            dtype=np.uint16,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "alpha_u16.png"
            masks.save_alpha_u16(path, alpha)

            ihdr = prepare.read_png_ihdr(path)
            self.assertEqual(ihdr["bit_depth"], 16)
            self.assertEqual(ihdr["color_type"], 0)
            metadata, decoded = prepare.validate_canonical_alpha_png(path)

        self.assertEqual(metadata["storage_dtype"], "uint16")
        self.assertEqual(metadata["storage_bit_depth"], 16)
        self.assertEqual(metadata["storage_color_type"], 0)
        self.assertTrue(decoded.dtype == np.uint16 or np.issubdtype(decoded.dtype, np.signedinteger))
        self.assertGreaterEqual(int(decoded.min()), 0)
        self.assertLessEqual(int(decoded.max()), 65535)
        np.testing.assert_array_equal(decoded.astype(np.uint16), alpha)


class SelectionJsonTests(unittest.TestCase):
    selection_path = (
        prepare.REPO_ROOT
        / "experiments"
        / "natural_image_subject_color_pilot"
        / "configs"
        / "d1_replacement_selection_3_plus_3.json"
    )

    def load_payload(self) -> dict:
        return json.loads(self.selection_path.read_text(encoding="utf-8"))

    def test_default_locked_selection_preserves_the_12_plus_6_order(self):
        prepare.validate_selection(prepare.LOCKED_SELECTION)

        self.assertEqual(len(prepare.LOCKED_SELECTION), 18)
        self.assertEqual([row["rank"] for row in prepare.LOCKED_SELECTION], list(range(1, 19)))
        self.assertEqual([row["group"] for row in prepare.LOCKED_SELECTION[:12]], ["pilot"] * 12)
        self.assertEqual([row["group"] for row in prepare.LOCKED_SELECTION[12:]], ["reserve"] * 6)

    def test_default_report_and_readme_keep_the_historical_12_plus_6_wording(self):
        manifest = {
            "selection": {
                "locked_count": 18,
                "pilot_count": 12,
                "reserve_count": 6,
                "stable_id_order_sha256": "test",
            },
            "summary": {"PASS": 18, "REVIEW": 0, "FAIL": 0},
            "contact_sheet": "qc/production_edt_contact_sheet.png",
            "samples": [],
            "verification": {"passed": True},
        }
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            prepare.write_readme(output_dir, manifest)
            prepare.write_report(output_dir, manifest)
            readme = (output_dir / "README.md").read_text()
            report = (output_dir / "reports" / "01_mask_derivation_batch.md").read_text()

        self.assertIn("D1 pilot/reserve mask derivation batch. The locked\ninput list", readme)
        self.assertIn("- 12 pilot samples and 6 reserve samples.", readme)
        self.assertIn("fixed 12 pilot plus 6 reserve samples", report)

    def test_replacement_selection_json_is_valid_and_records_required_provenance(self):
        selection, metadata = prepare.load_selection_json(self.selection_path)

        self.assertEqual([row["stable_id"] for row in selection], [
            "D1GT:13/30.png",
            "D1GT:11/180.png",
            "D1GT:9/205.png",
            "D1GT:2/62.png",
            "D1GT:5/101.png",
            "D1GT:24/198.png",
        ])
        self.assertEqual(metadata["selection_source_repo_relative_path"],
                         "experiments/natural_image_subject_color_pilot/configs/d1_replacement_selection_3_plus_3.json")
        self.assertEqual(metadata["selection_file_sha256"], masks.file_sha256(self.selection_path))
        self.assertEqual(metadata["group_counts"], {"replacement": 3, "reserve": 3})
        self.assertEqual(len(metadata["human_selection_reasons"]), 6)

    def test_selection_json_rejects_duplicate_overlap_mismatch_and_bad_groups(self):
        cases = {
            "duplicate_sample": lambda payload: payload["samples"].__setitem__(1, {
                **payload["samples"][1], "sample_id": "13", "mask_name": "180.png", "stable_id": "D1GT:13/180.png",
            }),
            "locked_overlap": lambda payload: payload["samples"].__setitem__(0, {
                **payload["samples"][0], "sample_id": "3", "mask_name": "30.png", "stable_id": "D1GT:3/30.png",
            }),
            "stable_id_mismatch": lambda payload: payload["samples"][0].__setitem__("sample_id", "999"),
            "bad_groups": lambda payload: payload["samples"][2].__setitem__("group", "reserve"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selection.json"
            for name, mutate in cases.items():
                with self.subTest(name=name):
                    payload = self.load_payload()
                    mutate(payload)
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(masks.NaturalImageMaskError):
                        prepare.load_selection_json(path)

    def test_external_selection_builds_a_3_plus_3_manifest_and_report_in_tempdirs(self):
        selection, metadata = prepare.load_selection_json(self.selection_path)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            mask_root = root / "masks"
            output_dir = root / "output"
            image_root.mkdir()
            mask_root.mkdir()
            for index, row in enumerate(selection):
                image = np.full((512, 512, 3), 32 + index * 20, dtype=np.uint8)
                Image.fromarray(image, mode="RGB").save(image_root / f"{row['sample_id']}.jpg", "JPEG")
                mask = np.zeros((512, 512), dtype=np.uint8)
                mask[128:384, 128:384] = 255
                sample_mask_root = mask_root / row["sample_id"]
                sample_mask_root.mkdir()
                Image.fromarray(mask, mode="L").save(sample_mask_root / row["mask_name"])

            manifest = prepare.build_manifest(
                image_root,
                mask_root,
                "flat",
                output_dir,
                selection,
                metadata,
            )
            prepare.write_readme(output_dir, manifest)
            prepare.write_report(output_dir, manifest)

            self.assertEqual(manifest["selection"], metadata)
            self.assertEqual([row["group"] for row in manifest["samples"]],
                             ["replacement"] * 3 + ["reserve"] * 3)
            self.assertTrue(all(row["reconstruction_mask"] is None for row in manifest["samples"]))
            self.assertTrue((output_dir / manifest["contact_sheet"]).is_file())
            self.assertIn("3 replacement samples and 3 reserve samples", (output_dir / "README.md").read_text())
            report = (output_dir / "reports" / "01_mask_derivation_batch.md").read_text()
            self.assertIn("fixed 3 replacement plus 3 reserve samples", report)
            self.assertNotIn("12 pilot plus 6 reserve", report)


if __name__ == "__main__":
    unittest.main()
