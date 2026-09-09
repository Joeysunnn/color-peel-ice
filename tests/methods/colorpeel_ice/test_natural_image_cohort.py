from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from src.methods.colorpeel_ice import natural_image_cohort as cohort
from src.methods.colorpeel_ice.natural_image_masks import canonical_sha256, file_sha256


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    REPO_ROOT
    / "experiments"
    / "natural_image_subject_color_pilot"
    / "configs"
    / "d1_target_color_audit_cohort_v1.json"
)


def source_samples(config: dict, alias: str) -> list[dict]:
    rows = [
        row
        for collection in (config["active_records"], config["excluded_records"])
        for row in collection
        if row["source_alias"] == alias
    ]
    return [
        {
            "rank": row["source_rank"],
            "stable_id": row["stable_id"],
            "sample_id": row["sample_id"],
            "mask_name": row["mask_name"],
            "group": row["source_group"],
            "status": row["source_status"],
            "human_review": deepcopy(row["source_manifest_human_review"]),
        }
        for row in sorted(rows, key=lambda item: item["source_rank"])
    ]


class NaturalImageCohortTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = cohort.load_cohort_config(CONFIG_PATH)

    def make_synthetic_runs(self, root: Path, config: dict) -> dict[str, Path]:
        runs: dict[str, Path] = {}
        for alias in sorted(config["source_runs"]):
            run_root = root / alias
            manifest_path = run_root / "manifests" / "pilot_mask_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            samples = source_samples(config, alias)
            manifest = {
                "samples": samples,
                "verification": deepcopy(config["source_runs"][alias]["verification"]),
                "selection": {"stable_id_order_sha256": canonical_sha256([row["stable_id"] for row in samples])},
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            config["source_runs"][alias]["manifest_sha256"] = file_sha256(manifest_path)
            runs[alias] = run_root
        return runs

    def rewrite_manifest(self, config: dict, alias: str, manifest_path: Path, manifest: dict) -> None:
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        config["source_runs"][alias]["manifest_sha256"] = file_sha256(manifest_path)

    def test_config_and_synthetic_provenance_validate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = deepcopy(self.config)
            runs = self.make_synthetic_runs(Path(temporary), config)
            validated = cohort.validate_cohort(config, runs)
        self.assertEqual(validated["cohort_id"], "d1_target_color_audit_cohort_v1")

    def test_strict_schema_rejects_missing_and_unknown_fields(self) -> None:
        missing = deepcopy(self.config)
        del missing["study"]
        with self.assertRaisesRegex(cohort.NaturalImageCohortError, "fields differ"):
            cohort.select_measurement_records(missing)
        unknown = deepcopy(self.config)
        unknown["unexpected"] = True
        with self.assertRaisesRegex(cohort.NaturalImageCohortError, "fields differ"):
            cohort.select_measurement_records(unknown)
        unknown_record = deepcopy(self.config)
        unknown_record["active_records"][0]["unexpected"] = True
        with self.assertRaisesRegex(cohort.NaturalImageCohortError, "fields differ"):
            cohort.select_measurement_records(unknown_record)
        bad_verification = deepcopy(self.config)
        bad_verification["source_runs"]["initial_d3263a6"]["verification"]["unexpected"] = True
        with self.assertRaisesRegex(cohort.NaturalImageCohortError, "fields differ"):
            cohort.select_measurement_records(bad_verification)
        bad_stable_id = deepcopy(self.config)
        bad_stable_id["active_records"][0]["stable_id"] = "not-a-stable-id"
        with self.assertRaisesRegex(cohort.NaturalImageCohortError, "stable_id is invalid"):
            cohort.select_measurement_records(bad_stable_id)

    def test_alias_sets_and_duplicate_pair_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = deepcopy(self.config)
            runs = self.make_synthetic_runs(Path(temporary), config)
            with self.assertRaisesRegex(cohort.NaturalImageCohortError, "aliases differ"):
                cohort.validate_cohort(config, {"initial_d3263a6": runs["initial_d3263a6"]})
            extra_runs = dict(runs, unexpected=Path(temporary))
            with self.assertRaisesRegex(cohort.NaturalImageCohortError, "aliases differ"):
                cohort.validate_cohort(config, extra_runs)
        with self.assertRaisesRegex(cohort.NaturalImageCohortError, "Duplicate"):
            cohort.parse_mask_run_pairs((("initial_d3263a6", "a"), ("initial_d3263a6", "b")))
        parsed = cohort.parse_mask_run_pairs([
            ["initial_d3263a6", "a"],
            ["replacements_71e40c1", Path("b")],
        ])
        self.assertEqual(parsed, {"initial_d3263a6": Path("a"), "replacements_71e40c1": Path("b")})

    def test_manifest_hash_count_verification_and_order_drift_rejected(self) -> None:
        alias = "initial_d3263a6"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = deepcopy(self.config)
            runs = self.make_synthetic_runs(root, config)
            manifest_path = runs[alias] / "manifests" / "pilot_mask_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["samples"][0]["status"] = "REVIEW"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(cohort.NaturalImageCohortError, "hash drift"):
                cohort.validate_cohort(config, runs)

            config = deepcopy(self.config)
            runs = self.make_synthetic_runs(root, config)
            manifest_path = runs[alias] / "manifests" / "pilot_mask_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["samples"].pop()
            manifest["selection"]["stable_id_order_sha256"] = canonical_sha256(
                [row["stable_id"] for row in manifest["samples"]]
            )
            self.rewrite_manifest(config, alias, manifest_path, manifest)
            with self.assertRaisesRegex(cohort.NaturalImageCohortError, "record count drift"):
                cohort.validate_cohort(config, runs)

            config = deepcopy(self.config)
            runs = self.make_synthetic_runs(root, config)
            manifest_path = runs[alias] / "manifests" / "pilot_mask_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["verification"]["passed"] = False
            self.rewrite_manifest(config, alias, manifest_path, manifest)
            with self.assertRaisesRegex(cohort.NaturalImageCohortError, "verification drift"):
                cohort.validate_cohort(config, runs)

            config = deepcopy(self.config)
            runs = self.make_synthetic_runs(root, config)
            manifest_path = runs[alias] / "manifests" / "pilot_mask_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["samples"] = list(reversed(manifest["samples"]))
            manifest["selection"]["stable_id_order_sha256"] = canonical_sha256(
                [row["stable_id"] for row in manifest["samples"]]
            )
            self.rewrite_manifest(config, alias, manifest_path, manifest)
            with self.assertRaisesRegex(cohort.NaturalImageCohortError, "stable order drift"):
                cohort.validate_cohort(config, runs)

    def test_source_identity_and_source_field_drift_rejected(self) -> None:
        alias = "initial_d3263a6"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = deepcopy(self.config)
            runs = self.make_synthetic_runs(root, config)
            manifest_path = runs[alias] / "manifests" / "pilot_mask_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["samples"][1]["stable_id"] = manifest["samples"][0]["stable_id"]
            manifest["samples"][1]["sample_id"] = manifest["samples"][0]["sample_id"]
            manifest["samples"][1]["mask_name"] = manifest["samples"][0]["mask_name"]
            self.rewrite_manifest(config, alias, manifest_path, manifest)
            with self.assertRaisesRegex(cohort.NaturalImageCohortError, "stable IDs must be unique"):
                cohort.validate_cohort(config, runs)

            config = deepcopy(self.config)
            runs = self.make_synthetic_runs(root, config)
            config["active_records"][0]["source_rank"] = 2
            with self.assertRaisesRegex(cohort.NaturalImageCohortError, "source_rank does not match"):
                cohort.validate_cohort(config, runs)

            config = deepcopy(self.config)
            runs = self.make_synthetic_runs(root, config)
            manifest_path = runs[alias] / "manifests" / "pilot_mask_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["samples"][0]["status"] = "REVIEW"
            self.rewrite_manifest(config, alias, manifest_path, manifest)
            with self.assertRaisesRegex(cohort.NaturalImageCohortError, "source_status does not match"):
                cohort.validate_cohort(config, runs)

            config = deepcopy(self.config)
            runs = self.make_synthetic_runs(root, config)
            initial_manifest_path = runs[alias] / "manifests" / "pilot_mask_manifest.json"
            initial = json.loads(initial_manifest_path.read_text(encoding="utf-8"))
            replacement_alias = "replacements_71e40c1"
            replacement_path = runs[replacement_alias] / "manifests" / "pilot_mask_manifest.json"
            replacement = json.loads(replacement_path.read_text(encoding="utf-8"))
            replacement["samples"][0]["stable_id"] = initial["samples"][0]["stable_id"]
            replacement["samples"][0]["sample_id"] = initial["samples"][0]["sample_id"]
            replacement["samples"][0]["mask_name"] = initial["samples"][0]["mask_name"]
            replacement["selection"]["stable_id_order_sha256"] = canonical_sha256(
                [row["stable_id"] for row in replacement["samples"]]
            )
            config["source_runs"][replacement_alias]["stable_id_order_sha256"] = (
                replacement["selection"]["stable_id_order_sha256"]
            )
            self.rewrite_manifest(config, replacement_alias, replacement_path, replacement)
            with self.assertRaisesRegex(cohort.NaturalImageCohortError, "unique across aliases"):
                cohort.validate_cohort(config, runs)

    def test_rank_role_crosstab_verdict_and_overlap_rejected(self) -> None:
        bad_rank = deepcopy(self.config)
        bad_rank["active_records"][0]["cohort_rank"] = 2
        with self.assertRaisesRegex(cohort.NaturalImageCohortError, "ranks must be continuous"):
            cohort.select_measurement_records(bad_rank)
        bad_role = deepcopy(self.config)
        bad_role["active_records"][0]["role"] = "reserve"
        with self.assertRaisesRegex(cohort.NaturalImageCohortError, "roles must be"):
            cohort.select_measurement_records(bad_role)
        bad_group = deepcopy(self.config)
        bad_group["active_records"][0]["source_group"] = "reserve"
        with self.assertRaisesRegex(cohort.NaturalImageCohortError, "crosstab"):
            cohort.select_measurement_records(bad_group)
        bad_verdict = deepcopy(self.config)
        bad_verdict["active_records"][0]["human_verdict"] = "FAIL"
        with self.assertRaisesRegex(cohort.NaturalImageCohortError, "active records"):
            cohort.select_measurement_records(bad_verdict)
        overlap = deepcopy(self.config)
        overlap["excluded_records"][0]["stable_id"] = overlap["active_records"][0]["stable_id"]
        overlap["excluded_records"][0]["sample_id"] = overlap["active_records"][0]["sample_id"]
        overlap["excluded_records"][0]["mask_name"] = overlap["active_records"][0]["mask_name"]
        with self.assertRaisesRegex(cohort.NaturalImageCohortError, "must not overlap"):
            cohort.select_measurement_records(overlap)
        duplicate_excluded = deepcopy(self.config)
        duplicate_excluded["excluded_records"][1]["stable_id"] = duplicate_excluded["excluded_records"][0]["stable_id"]
        duplicate_excluded["excluded_records"][1]["sample_id"] = duplicate_excluded["excluded_records"][0]["sample_id"]
        duplicate_excluded["excluded_records"][1]["mask_name"] = duplicate_excluded["excluded_records"][0]["mask_name"]
        with self.assertRaisesRegex(cohort.NaturalImageCohortError, "Excluded stable IDs must be unique"):
            cohort.select_measurement_records(duplicate_excluded)

    def test_measurement_and_downstream_selection_are_frozen(self) -> None:
        measurement = cohort.select_measurement_records(self.config)
        downstream = cohort.select_downstream_records(self.config)
        self.assertEqual(len(measurement), 21)
        self.assertEqual(len(downstream), 18)
        self.assertEqual([row["cohort_rank"] for row in measurement], list(range(1, 22)))
        self.assertEqual([row["cohort_rank"] for row in downstream], list(range(1, 19)))
        self.assertEqual(measurement[4]["stable_id"], "D1GT:13/30.png")
        self.assertEqual(measurement[-1]["stable_id"], "D1GT:17/74.png")


if __name__ == "__main__":
    unittest.main()
