from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.methods.colorpeel_ice import natural_image_target_selection as selection
from src.methods.colorpeel_ice.natural_image_masks import canonical_sha256, file_sha256


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / selection.CONFIG_RELPATH
INTEGRATION_MANIFEST_ENV = "D1_TARGET_COLOR_AUDIT_MANIFEST"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def contains_absolute_string(value) -> bool:
    if isinstance(value, dict):
        return any(contains_absolute_string(nested) for nested in value.values())
    if isinstance(value, list):
        return any(contains_absolute_string(nested) for nested in value)
    if isinstance(value, str):
        return selection._looks_absolute_path(value)
    return False


def synthetic_audit_manifest(config: dict) -> dict:
    records = []
    for row in config["records"]:
        source = deepcopy(row["source"])
        audit_row = {
            **source,
            "status": "MEASURED",
            "reason": None,
            "qc_decision": "PENDING_HUMAN",
            "approved_for_downstream": False,
        }
        if row["human_verdict"] == "PASS":
            dominant = row["metrics"]["dominant_cluster"]
            clusters = [{"ratio_over_eligible": 0.0, "compactness": {"rms": 0.0, "median": 0.0, "p90": 0.0}}
                        for _ in range(max(3, dominant + 1))]
            clusters[dominant] = {
                "ratio_over_eligible": row["metrics"]["dominant_cluster_ratio_over_eligible"],
                "compactness": deepcopy(row["metrics"]["dominant_cluster_compactness"]),
            }
            audit_row.update({
                "provisional_target": deepcopy(row["target"]),
                "clusters": clusters,
            })
            for key in selection.METRIC_KEYS - {"dominant_cluster_compactness", "dominant_cluster_ratio_over_eligible"}:
                audit_row[key] = row["metrics"][key]
        records.append(audit_row)
    return {
        "schema": selection.SOURCE_AUDIT_SCHEMA,
        "git_commit": selection.GIT_COMMIT,
        "protocol": {
            "path": selection._expected_audit_provenance()["protocol"]["path"],
            "sha256": selection.PROTOCOL_SHA256,
        },
        "cohort": deepcopy(selection._expected_audit_provenance()["cohort"]),
        "source_runs": deepcopy(selection.SOURCE_RUNS),
        "versions": {},
        "records": records,
        "summary": deepcopy(selection.SOURCE_SUMMARY),
        "downstream_candidate_ids": [],
        "qc_decision": "PENDING_HUMAN",
        "approved_for_downstream": False,
        "output_sha256": {},
    }


class NaturalImageTargetSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = selection.load_selection_config(CONFIG_PATH)

    def test_frozen_config_validates_and_contains_no_absolute_paths(self) -> None:
        self.assertEqual(self.config["schema"], "natural_image_target_color_selection/v1")
        self.assertEqual(self.config["source_audit"]["manifest_sha256"], selection.SOURCE_AUDIT_MANIFEST_SHA256)
        self.assertEqual(canonical_sha256(self.config), selection.FROZEN_CONFIG_CANONICAL_SHA256)
        self.assertFalse(contains_absolute_string(self.config))
        self.assertEqual(self.config["source_audit"]["summary"], selection.SOURCE_SUMMARY)
        self.assertEqual(self.config["source_audit"]["qc_decision"], "PENDING_HUMAN")
        self.assertFalse(self.config["source_audit"]["approved_for_downstream"])
        self.assertEqual(len(self.config["records"]), 21)

    def test_frozen_path_and_canonical_hash_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "d1_target_color_selection_v1.json"
            write_json(copied, self.config)
            with self.assertRaisesRegex(selection.NaturalImageTargetSelectionError, "frozen repository config"):
                selection.load_selection_config(copied)

            repo = Path(temporary) / "repo"
            config_path = repo / selection.CONFIG_RELPATH
            mutated = deepcopy(self.config)
            mutated["expected_counts"]["selected"] = 12
            write_json(config_path, mutated)
            with patch.object(selection, "REPO_ROOT", repo):
                with self.assertRaisesRegex(selection.NaturalImageTargetSelectionError, "canonical hash drift"):
                    selection.load_selection_config(config_path)

    def test_strict_config_schema_counts_sets_and_hashes(self) -> None:
        cases = []
        missing = deepcopy(self.config)
        del missing["study"]
        cases.append((missing, "fields differ"))
        extra = deepcopy(self.config)
        extra["unexpected"] = True
        cases.append((extra, "fields differ"))
        bad_count = deepcopy(self.config)
        bad_count["expected_counts"]["human_pass"] = 12
        cases.append((bad_count, "expected_counts"))
        bad_selected = deepcopy(self.config)
        bad_selected["selected_stable_ids"] = list(reversed(bad_selected["selected_stable_ids"]))
        cases.append((bad_selected, "selected stable-id order"))
        bad_failed = deepcopy(self.config)
        bad_failed["failed_stable_ids"].pop()
        cases.append((bad_failed, "failed stable-id order"))
        bad_hash = deepcopy(self.config)
        bad_hash["canonical_hashes"]["selected_stable_id_order_sha256"] = "0" * 64
        cases.append((bad_hash, "selected-order hash"))
        absolute = deepcopy(self.config)
        absolute["selection_policy"]["held_out_condition"] = "/absolute/path"
        cases.append((absolute, "absolute path"))
        for value, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(selection.NaturalImageTargetSelectionError, message):
                    selection._validate_config_structure(value)

    def test_pass_fail_membership_roles_and_no_replacements(self) -> None:
        source_order = [row["source"]["stable_id"] for row in self.config["records"]]
        self.assertEqual([stable_id for stable_id in source_order if stable_id in selection.SELECTED_STABLE_IDS],
                         selection.SELECTED_STABLE_IDS)
        self.assertEqual([stable_id for stable_id in source_order if stable_id in selection.FAILED_STABLE_IDS],
                         selection.FAILED_STABLE_IDS)
        self.assertEqual(set(source_order), set(selection.SELECTED_STABLE_IDS).union(selection.FAILED_STABLE_IDS))
        self.assertEqual(set(selection.SELECTED_STABLE_IDS).intersection(selection.FAILED_STABLE_IDS), set())
        self.assertEqual([row["source"]["stable_id"] for row in selection.selected_records(self.config)],
                         selection.SELECTED_STABLE_IDS)
        for role, stable_ids in selection.PIPELINE_ROLES.items():
            self.assertEqual([row["source"]["stable_id"] for row in selection.records_for_pipeline_role(role, self.config)],
                             stable_ids)
        self.assertEqual(len(selection.pipeline_development_records(self.config)), 6)
        self.assertEqual(len(selection.protocol_confirmation_records(self.config)), 3)
        self.assertEqual(len(selection.stress_test_records(self.config)), 2)

    def test_all_selected_downstream_eligible_and_recoloring_training_pending(self) -> None:
        for row in self.config["records"]:
            selected = row["source"]["stable_id"] in selection.SELECTED_STABLE_IDS
            self.assertEqual(row["target_color_qc_approved"], selected)
            self.assertEqual(row["downstream_eligible"], selected)
            self.assertFalse(row["recoloring_approved"])
            self.assertEqual(row["recoloring_status"], "PENDING")
            self.assertFalse(row["training_approved"])
            self.assertEqual(row["training_status"], "PENDING")
            self.assertIsNone(row["failure_reason"])
            if selected:
                self.assertIn("target", row)
                self.assertIn("metrics", row)
            else:
                self.assertNotIn("target", row)
                self.assertNotIn("metrics", row)

    def test_config_rejects_membership_approval_and_metric_drift(self) -> None:
        cases = []
        bad_role = deepcopy(self.config)
        bad_role["records"][2]["pipeline_role"] = "stress_test"
        cases.append((bad_role, "pipeline_role"))
        bad_fail_approval = deepcopy(self.config)
        bad_fail_approval["records"][0]["target_color_qc_approved"] = True
        cases.append((bad_fail_approval, "target_color_qc_approved"))
        bad_downstream = deepcopy(self.config)
        bad_downstream["records"][2]["downstream_eligible"] = False
        cases.append((bad_downstream, "downstream_eligible"))
        bad_recoloring = deepcopy(self.config)
        bad_recoloring["records"][2]["recoloring_approved"] = True
        cases.append((bad_recoloring, "recoloring"))
        bad_training = deepcopy(self.config)
        bad_training["records"][2]["training_approved"] = True
        cases.append((bad_training, "training"))
        bad_metric_type = deepcopy(self.config)
        bad_metric_type["records"][2]["metrics"]["valid_ratio"] = 1
        cases.append((bad_metric_type, "valid_ratio"))
        bad_target_type = deepcopy(self.config)
        bad_target_type["records"][2]["target"]["a"] = 1
        cases.append((bad_target_type, "target.a"))
        for value, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(selection.NaturalImageTargetSelectionError, message):
                    selection._validate_config_structure(value)

    def mutated_manifest(self, mutator) -> Path:
        manifest = synthetic_audit_manifest(self.config)
        mutator(manifest)
        path = Path(self.temporary.name) / "target_color_audit_manifest.json"
        write_json(path, manifest)
        return path

    def assert_manifest_rejected(self, mutator, message: str) -> None:
        path = self.mutated_manifest(mutator)
        patched_config = deepcopy(self.config)
        patched_config["source_audit"]["manifest_sha256"] = file_sha256(path)
        with patch.object(selection, "SOURCE_AUDIT_MANIFEST_SHA256", file_sha256(path)):
            with self.assertRaisesRegex(selection.NaturalImageTargetSelectionError, message):
                selection.validate_source_audit_manifest(path, patched_config)

    def test_synthetic_source_audit_manifest_validates_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "target_color_audit_manifest.json"
            manifest = synthetic_audit_manifest(self.config)
            write_json(path, manifest)
            patched_config = deepcopy(self.config)
            patched_config["source_audit"]["manifest_sha256"] = file_sha256(path)
            with patch.object(selection, "SOURCE_AUDIT_MANIFEST_SHA256", file_sha256(path)):
                validated = selection.validate_source_audit_manifest(path, patched_config)
        self.assertEqual([row["source"]["stable_id"] for row in validated["records"] if row["human_verdict"] == "PASS"],
                         selection.SELECTED_STABLE_IDS)

    def test_real_source_audit_manifest_validates_read_only_when_env_is_set(self) -> None:
        manifest_path = os.environ.get(INTEGRATION_MANIFEST_ENV)
        if not manifest_path:
            self.skipTest(f"{INTEGRATION_MANIFEST_ENV} is not set")
        path = Path(manifest_path)
        self.assertEqual(file_sha256(path), selection.SOURCE_AUDIT_MANIFEST_SHA256)
        validated = selection.validate_source_audit_manifest(path, self.config)
        self.assertEqual([row["source"]["stable_id"] for row in selection.selected_records(validated)],
                         selection.SELECTED_STABLE_IDS)
        self.assertEqual(selection.main(["--audit-manifest", str(path), "--selection-config", str(CONFIG_PATH)]), 0)

    def test_source_audit_byte_hash_and_root_provenance_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.temporary = tempfile.TemporaryDirectory(dir=temporary)
            self.addCleanup(self.temporary.cleanup)
            path = self.mutated_manifest(lambda manifest: manifest["records"][0].__setitem__("source_rank", 99))
            with self.assertRaisesRegex(selection.NaturalImageTargetSelectionError, "byte hash drift"):
                selection.validate_source_audit_manifest(path, self.config)
            self.assert_manifest_rejected(
                lambda manifest: manifest["source_runs"]["initial_d3263a6"].__setitem__("root", "/different/root"),
                "root provenance",
            )

    def test_source_audit_schema_row_order_and_identity_drift_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.temporary = tempfile.TemporaryDirectory(dir=temporary)
            self.addCleanup(self.temporary.cleanup)
            self.assert_manifest_rejected(lambda manifest: manifest.__setitem__("schema", "wrong/v1"), "schema")
            self.assert_manifest_rejected(lambda manifest: manifest["records"].pop(), "exactly 21 records")
            self.assert_manifest_rejected(
                lambda manifest: manifest["records"][1].__setitem__("stable_id", manifest["records"][0]["stable_id"]),
                "duplicated",
            )
            self.assert_manifest_rejected(
                lambda manifest: manifest["records"].reverse(),
                "source row order",
            )

    def test_source_audit_selected_target_metric_and_source_drift_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.temporary = tempfile.TemporaryDirectory(dir=temporary)
            self.addCleanup(self.temporary.cleanup)
            self.assert_manifest_rejected(
                lambda manifest: manifest["records"][2].__setitem__("source_rank", 99),
                "source_rank",
            )
            self.assert_manifest_rejected(
                lambda manifest: manifest["records"][2]["provisional_target"].__setitem__("a", 0.0),
                "target differs",
            )
            self.assert_manifest_rejected(
                lambda manifest: manifest["records"][2].__setitem__("valid_ratio", 0.0),
                "metrics differ",
            )
            self.assert_manifest_rejected(
                lambda manifest: manifest["records"][2]["clusters"][1]["compactness"].__setitem__("rms", 0.0),
                "metrics differ",
            )


if __name__ == "__main__":
    unittest.main()
