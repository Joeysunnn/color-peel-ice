from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.methods.colorpeel_ice import target_color_audit as audit
from src.methods.colorpeel_ice.natural_image_masks import canonical_sha256, file_sha256
from src.methods.colorpeel_ice.natural_image_cohort import NaturalImageCohortError


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class TargetColorMathTests(unittest.TestCase):
    def test_known_d65_primaries(self):
        rgb = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        expected = [[0, 0, 0], [100, 0, 0], [53.2408, 80.0925, 67.2032],
                    [87.7347, -86.1827, 83.1793], [32.2970, 79.1875, -107.8602]]
        result = audit.rgb_uint8_to_lab(rgb)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_breakpoint_and_lch_quadrants(self):
        values = np.array([0.04045 - 1e-9, 0.04045, 0.04045 + 1e-9, 10 / 255, 11 / 255])
        expected = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in values]
        np.testing.assert_array_equal(audit.srgb_to_linear(values), expected)
        for ab, hue in [((1, 1), 45), ((-1, 1), 135), ((-1, -1), 225), ((1, -1), 315)]:
            self.assertEqual(audit.ab_to_lch(*ab)["h_degrees"], hue)
        self.assertIsNone(audit.ab_to_lch(0, 0)["h_degrees"])
        self.assertIsNone(audit.ab_to_lch(1e-9, 0)["h_degrees"])
        self.assertEqual(audit.ab_to_lch(1e-7, 0)["h_degrees"], 0)

    def test_l_filter_exact_boundaries_and_full_mask_denominator(self):
        lab = np.zeros((1, 7, 3), dtype=np.float64)
        lab[0, :, 0] = [0, 5, 5.0001, 94.9999, 95, 100, 50]
        mask = np.array([[True] * 6 + [False]])
        eligible, metrics = audit.filter_lightness(lab, mask)
        np.testing.assert_array_equal(eligible, [[False, False, True, True, False, False, False]])
        self.assertEqual(metrics["color_area_px"], 6)
        self.assertEqual(metrics["eligible_px"], 2)
        for key in ("valid_ratio", "dark_ratio", "light_ratio"):
            self.assertEqual(metrics[key], 1 / 3)
        self.assertAlmostEqual(metrics["l_percentiles"]["p50"], 50)

    def test_three_clusters_determinism_permutation_medians_compactness(self):
        lab = np.array([[20, -21, 0], [80, -19, 0], [40, 0, 20], [60, 0, 20],
                        [30, 20, 0], [50, 20, 0], [70, 20, 0]], dtype=float)
        yx = np.array([[0, i] for i in range(len(lab))])
        metrics, labels = audit.cluster_lab(lab, yx)
        self.assertEqual(metrics["status"], "MEASURED")
        self.assertEqual([row["population"] for row in metrics["clusters"]], [2, 2, 3])
        np.testing.assert_array_equal([row["mean_ab"] for row in metrics["clusters"]], [[-20, 0], [0, 20], [20, 0]])
        self.assertEqual(metrics["clusters"][0]["compactness"], {"rms": 1, "median": 1, "p90": 1})
        self.assertEqual(metrics["provisional_target"]["observed_median_L"], 50)
        self.assertEqual(metrics["provisional_target"]["a"], 20)
        self.assertEqual(metrics["dominant_ratio_over_eligible"], 3 / 7)
        for permutation in [np.arange(7)[::-1], np.array([2, 6, 1, 0, 5, 3, 4])]:
            reordered, permuted_labels = audit.cluster_lab(lab[permutation], yx[permutation])
            self.assertEqual(metrics, reordered)
            np.testing.assert_array_equal(labels[permutation], permuted_labels)
        tied, _ = audit.cluster_lab(lab[:6], yx[:6])
        self.assertEqual(tied["dominant_cluster"], 0)
        np.testing.assert_array_equal(audit._assign(np.array([[0, 0]]), np.array([[-1, 0], [1, 0], [0, 2]])), [0])

    def test_initializer_lower_median_and_farthest_ties(self):
        lab = np.array([[50, -10, 0], [50, 0, 0], [50, 10, 0], [50, 20, 0]], dtype=float)
        yx = np.array([[0, i] for i in range(4)])
        with patch.object(audit, "_assign", wraps=audit._assign) as assignment:
            audit.cluster_lab(lab, yx)
        np.testing.assert_array_equal(assignment.call_args_list[0].args[1], [[0, 0], [20, 0], [-10, 0]])

    def test_degenerate_empty_and_nonconvergence_are_review(self):
        for lab in [np.empty((0, 3)), np.ones((2, 3)), np.ones((4, 3))]:
            result, labels = audit.cluster_lab(lab, np.zeros((len(lab), 2)))
            self.assertEqual(result["reason"], "cluster_degenerate")
            self.assertIsNone(labels)
            self.assertIsNone(result["provisional_target"])
        lab = np.array([[50, -20, 0], [50, 0, 20], [50, 20, 0]], dtype=float)
        yx = np.array([[0, i] for i in range(3)])
        result, labels = audit.cluster_lab(lab, yx, max_iterations=1)
        self.assertEqual((result["status"], result["reason"], result["iterations"]), ("REVIEW", "nonconvergence", 1))
        self.assertIsNone(labels)
        # Exercise the explicit empty-cluster guard using a controlled assignment.
        with patch.object(audit, "_assign", return_value=np.zeros(3, dtype=int)):
            result, _ = audit.cluster_lab(lab, yx)
        self.assertEqual((result["status"], result["reason"], result["iterations"]), ("REVIEW", "empty_cluster", 1))


class TargetColorInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = audit.read_json(audit.REPO_ROOT / audit.COHORT_RELPATH)
        self.protocol = audit.read_json(audit.REPO_ROOT / audit.PROTOCOL_RELPATH)
        self.runs = {}
        self.source_manifests = {}
        for alias, spec in self.config["source_runs"].items():
            run_root = self.root / alias
            run_root.mkdir()
            self.runs[alias] = run_root
            cohort_rows = sorted([row for rows in (self.config["active_records"], self.config["excluded_records"])
                                  for row in rows if row["source_alias"] == alias], key=lambda row: row["source_rank"])
            sources = []
            for row in cohort_rows:
                rgb = np.tile(np.array([[170, 30, 30], [30, 170, 30], [30, 30, 170]], dtype=np.uint8), (3, 1, 1))
                raw_path = run_root / f"{row['sample_id']}.png"
                mask_path = run_root / f"{row['sample_id']}_mask.png"
                Image.fromarray(rgb).save(raw_path)
                Image.fromarray(np.full((3, 3), 255, dtype=np.uint8)).save(mask_path)
                sources.append({"rank": row["source_rank"], "stable_id": row["stable_id"], "sample_id": row["sample_id"],
                                "mask_name": row["mask_name"], "group": row["source_group"], "status": row["source_status"],
                                "human_review": deepcopy(row["source_manifest_human_review"]),
                                "source": {"image": {"sha256": file_sha256(raw_path)}},
                                "raw_copy": {"image_bytes_identical": True}, "geometry": {"color_area_px": 9},
                                "outputs": {"raw_image": raw_path.name, "color_mask": mask_path.name},
                                "output_metadata": {"raw_image": {"mode": "RGB", "size": [3, 3]},
                                                    "color_mask": {"mode": "L", "size": [3, 3]}}})
            self.source_manifests[alias] = {"samples": sources, "verification": deepcopy(spec["verification"]),
                                            "selection": {"stable_id_order_sha256": canonical_sha256([r["stable_id"] for r in sources])}}
        self.repo_root = self.root / "repo"
        self.cohort_path = self.repo_root / audit.COHORT_RELPATH
        self.protocol_path = self.repo_root / audit.PROTOCOL_RELPATH
        self.output = self.root / "output"
        self.commit = "a" * 40
        self.git_status = ""
        self.sync()
        repo_patch = patch.object(audit, "REPO_ROOT", self.repo_root)
        repo_patch.start()
        self.addCleanup(repo_patch.stop)
        git_patch = patch.object(audit.subprocess, "check_output", side_effect=self.synthetic_git)
        self.git_mock = git_patch.start()
        self.addCleanup(git_patch.stop)

    def synthetic_git(self, command, **kwargs):
        if command == ["git", "-C", str(self.repo_root), "rev-parse", "HEAD"]:
            return self.commit + "\n"
        self.assertEqual(command, ["git", "-C", str(self.repo_root), "status", "--porcelain=v1", "--untracked-files=all"])
        return self.git_status

    def sync(self):
        for alias, manifest in self.source_manifests.items():
            path = self.runs[alias] / self.config["source_runs"][alias]["manifest_relpath"]
            write_json(path, manifest)
            self.config["source_runs"][alias]["manifest_sha256"] = file_sha256(path)
        write_json(self.cohort_path, self.config)
        self.protocol["cohort"]["sha256"] = file_sha256(self.cohort_path)
        write_json(self.protocol_path, self.protocol)
        protocol_patch = patch.object(audit, "FROZEN_PROTOCOL_CANONICAL_SHA256", canonical_sha256(self.protocol))
        protocol_patch.start()
        self.addCleanup(protocol_patch.stop)

    def run_audit(self, **overrides):
        arguments = dict(cohort_path=self.cohort_path, protocol_path=self.protocol_path,
                         pairs=[[alias, str(root)] for alias, root in self.runs.items()],
                         output=self.output, expected_commit=self.commit)
        arguments.update(overrides)
        return audit.run_audit(**arguments)

    def test_end_to_end_main_21_all_pending_18_candidates_outputs_and_hashes(self):
        args = ["--cohort-config", str(self.cohort_path), "--protocol-config", str(self.protocol_path),
                "--output-dir", str(self.output), "--expected-git-commit", self.commit]
        for alias, root in self.runs.items():
            args += ["--mask-run", alias, str(root)]
        self.assertEqual(audit.main(args), 0)
        manifest = audit.read_json(self.output / "manifests/target_color_audit_manifest.json")
        self.assertEqual(manifest["summary"], {"measurement_count": 21, "downstream_primary_count": 18,
                                              "measured_count": 21, "review_count": 0, "pending_human_count": 21, "approved_count": 0})
        self.assertEqual(len(manifest["downstream_candidate_ids"]), 18)
        self.assertFalse(manifest["approved_for_downstream"])
        for row in manifest["records"]:
            self.assertFalse(row["approved_for_downstream"])
            self.assertEqual(row["qc_decision"], "PENDING_HUMAN")
            self.assertFalse(row["inputs"]["color_mask"]["expected_hash_available"])
        masks = list(self.output.glob("samples/*/*_mask.png"))
        self.assertEqual(len(masks), 42)
        for path in masks:
            with Image.open(path) as image:
                self.assertEqual(image.mode, "L")
                self.assertTrue(set(np.asarray(image).ravel()).issubset({0, 255}))
        self.assertEqual(len(list(self.output.glob("samples/*/qc/sample_qc.png"))), 21)
        self.assertTrue((self.output / "qc/target_color_audit_contact_sheet.png").is_file())
        self.assertTrue((self.output / "README.md").is_file())
        self.assertTrue((self.output / "reports/01_target_color_audit.md").is_file())
        for relative, digest in manifest["output_sha256"].items():
            self.assertEqual(file_sha256(self.output / relative), digest)
        with (self.output / "metrics/target_color_audit.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 21)
        self.assertEqual(rows[0]["qc_decision"], "PENDING_HUMAN")

    def test_degenerate_record_has_empty_dominant_mask_and_explanatory_qc(self):
        alias = next(iter(self.runs))
        source = self.source_manifests[alias]["samples"][0]
        raw = self.runs[alias] / source["outputs"]["raw_image"]
        Image.new("RGB", (3, 3), "black").save(raw)
        source["source"]["image"]["sha256"] = file_sha256(raw)
        self.sync()
        result = self.run_audit()
        row = next(row for row in result["records"] if row["stable_id"] == source["stable_id"])
        self.assertEqual(row["reason"], "cluster_degenerate")
        self.assertIsNone(row["provisional_target"])
        with Image.open(self.output / row["outputs"]["dominant_mask"]) as image:
            self.assertEqual(np.asarray(image).sum(), 0)
        self.assertTrue((self.output / row["outputs"]["sample_qc"]).is_file())

    def test_partial_outputs_survive_failure_and_cannot_be_overwritten(self):
        with patch.object(audit, "sample_qc", side_effect=OSError("synthetic output failure")):
            with self.assertRaisesRegex(OSError, "synthetic output failure"):
                self.run_audit()
        self.assertTrue(self.output.exists())
        self.assertEqual(len(list(self.output.glob("samples/*/*_mask.png"))), 2)
        with self.assertRaisesRegex(audit.TargetColorAuditError, "already exists"):
            self.run_audit()

    def test_invalid_image_and_mask_artifacts(self):
        alias = next(iter(self.runs))
        original = self.source_manifests[alias]["samples"][0]
        root = self.runs[alias]
        raw, mask = root / original["outputs"]["raw_image"], root / original["outputs"]["color_mask"]
        raw_bytes, mask_bytes = raw.read_bytes(), mask.read_bytes()
        cases = ["raw_mode", "mask_mode", "shape", "binary", "empty", "raw_hash", "copy_false", "area", "metadata", "escape", "missing"]
        for case in cases:
            with self.subTest(case=case):
                raw.write_bytes(raw_bytes)
                mask.write_bytes(mask_bytes)
                source = deepcopy(original)
                if case == "raw_mode": Image.new("L", (3, 3)).save(raw)
                if case == "mask_mode": Image.new("RGB", (3, 3)).save(mask)
                if case == "shape":
                    Image.new("L", (4, 3), 255).save(mask)
                    source["output_metadata"]["color_mask"]["size"] = [4, 3]
                if case == "binary": Image.new("L", (3, 3), 127).save(mask)
                if case == "empty": Image.new("L", (3, 3), 0).save(mask)
                if case == "raw_hash": source["source"]["image"]["sha256"] = "0" * 64
                if case == "copy_false": source["raw_copy"]["image_bytes_identical"] = False
                if case == "area": source["geometry"]["color_area_px"] = 8
                if case == "metadata": source["output_metadata"]["raw_image"]["size"] = [2, 3]
                if case == "escape": source["outputs"]["raw_image"] = "../outside.png"
                if case == "missing": source["outputs"]["color_mask"] = "missing.png"
                with self.assertRaises(audit.TargetColorAuditError): audit.validate_artifacts(root, source)

    def test_portable_safe_paths(self):
        for relative in ["../escape", "/absolute", "C:/outside", "C:relative", "folder\\..\\outside", "file:stream", ""]:
            with self.subTest(relative=relative), self.assertRaises(audit.TargetColorAuditError):
                audit.safe_input_path(self.root, relative)

    def test_output_constraints_and_validate_all_before_create(self):
        for output in [audit.REPO_ROOT / "unused_tc2_output", self.root, next(iter(self.runs.values())) / "nested"]:
            with self.subTest(output=output), self.assertRaises(audit.TargetColorAuditError):
                self.run_audit(output=output)
        with self.assertRaisesRegex(audit.TargetColorAuditError, "contain"):
            audit.validate_output_dir(self.root / "new_parent", {"a": self.root / "new_parent" / "run"})
        last = self.config["active_records"][-1]
        source = next(row for row in self.source_manifests[last["source_alias"]]["samples"] if row["stable_id"] == last["stable_id"])
        source["geometry"]["color_area_px"] = 999
        self.sync()
        with self.assertRaisesRegex(audit.TargetColorAuditError, "area"):
            self.run_audit()
        self.assertFalse(self.output.exists())

    def test_protocol_schema_semantics_cohort_hash_and_git_drift(self):
        original = deepcopy(self.protocol)
        for change in ["extra", "missing", "iterations", "gate", "bool_type", "hash"]:
            self.protocol = deepcopy(original)
            if change == "extra": self.protocol["unexpected"] = 1
            if change == "missing": del self.protocol["study"]
            if change == "iterations": self.protocol["clustering"]["max_iterations"] = 101
            if change == "gate": self.protocol["target"]["dominant_ratio_gate"] = 0.7
            if change == "bool_type": self.protocol["target"]["approved_for_downstream"] = 0
            if change == "hash": self.protocol["cohort"]["sha256"] = "0" * 64
            write_json(self.protocol_path, self.protocol)
            with self.subTest(change=change), self.assertRaises(audit.TargetColorAuditError): self.run_audit()
            self.assertFalse(self.output.exists())
        self.protocol = original
        self.sync()
        with patch.object(audit, "FROZEN_PROTOCOL_CANONICAL_SHA256", "0" * 64):
            with self.assertRaisesRegex(audit.TargetColorAuditError, "protocol hash drift"): self.run_audit()
        with self.assertRaisesRegex(audit.TargetColorAuditError, "git commit mismatch"):
            self.run_audit(expected_commit="0" * 40)
        self.assertFalse(self.output.exists())

    def test_substitute_cohort_and_protocol_rejected_before_output_creation(self):
        substitute_cohort = self.root / "alternative_cohort.json"
        substitute_protocol = self.root / "alternative_protocol.json"
        config = deepcopy(self.config)
        config["active_records"][0]["decision_reason"] = "Substitute cohort with self-consistent protocol hash"
        write_json(substitute_cohort, config)
        protocol = deepcopy(self.protocol)
        protocol["cohort"]["sha256"] = file_sha256(substitute_cohort)
        write_json(substitute_protocol, protocol)
        for overrides in [{"cohort_path": substitute_cohort}, {"protocol_path": substitute_protocol},
                          {"cohort_path": substitute_cohort, "protocol_path": substitute_protocol}]:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(audit.TargetColorAuditError, "path must be the frozen"):
                self.run_audit(**overrides)
            self.assertFalse(self.output.exists())
        # Even a byte-identical copy is not an accepted production input path.
        substitute_protocol.write_bytes(self.protocol_path.read_bytes())
        with self.assertRaisesRegex(audit.TargetColorAuditError, "protocol path"):
            self.run_audit(protocol_path=substitute_protocol)
        # Editing the frozen cohort cannot be legitimized by editing its protocol SHA.
        write_json(self.cohort_path, config)
        with self.assertRaisesRegex(audit.TargetColorAuditError, "cohort hash drift"):
            self.run_audit()
        write_json(self.protocol_path, protocol)
        with self.assertRaisesRegex(audit.TargetColorAuditError, "protocol hash drift"):
            self.run_audit()
        self.assertFalse(self.output.exists())

    def test_dirty_git_status_including_untracked_rejected_before_output_creation(self):
        for status in [" M tracked.py\n", "M  staged.py\n", "?? untracked.txt\n"]:
            self.git_status = status
            with self.subTest(status=status), self.assertRaisesRegex(audit.TargetColorAuditError, "must be clean"):
                self.run_audit()
            self.assertFalse(self.output.exists())

    def test_source_hash_duplicate_and_missing_alias_fail_before_creation(self):
        pairs = [[alias, str(root)] for alias, root in self.runs.items()]
        with self.assertRaisesRegex(NaturalImageCohortError, "Duplicate"):
            self.run_audit(pairs=pairs + [pairs[0]])
        with self.assertRaisesRegex(audit.TargetColorAuditError, "aliases differ"):
            self.run_audit(pairs=pairs[:1])
        alias = next(iter(self.runs))
        path = self.runs[alias] / self.config["source_runs"][alias]["manifest_relpath"]
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(NaturalImageCohortError, "hash drift"): self.run_audit()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
