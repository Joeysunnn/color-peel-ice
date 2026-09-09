"""Strict provenance validation for the frozen D1 target-color audit cohort."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.methods.colorpeel_ice.natural_image_masks import (
    NaturalImageMaskError,
    canonical_sha256,
    file_sha256,
    parse_stable_id,
)


class NaturalImageCohortError(ValueError):
    """Raised when cohort metadata or its source-manifest provenance drifts."""


SCHEMA = "natural_image_target_color_audit_cohort/v1"
COHORT_ID = "d1_target_color_audit_cohort_v1"
SOURCE_ALIASES = {"initial_d3263a6", "replacements_71e40c1"}
CONFIG_KEYS = {
    "schema", "cohort_id", "study", "source_runs", "measurement_roles", "downstream_roles",
    "expected_counts", "expected_source_group_role_counts", "active_stable_id_order_sha256",
    "human_review_provenance", "active_records", "excluded_records",
}
SOURCE_RUN_KEYS = {
    "manifest_relpath", "manifest_sha256", "record_count", "verification", "stable_id_order_sha256",
}
COUNT_KEYS = {"active", "primary", "reserve", "excluded"}
VERIFICATION_KEYS = {"passed", "checked_samples", "errors"}
RECORD_KEYS = {
    "cohort_rank", "role", "stable_id", "sample_id", "mask_name", "source_alias", "source_rank",
    "source_group", "source_status", "source_manifest_human_review", "human_verdict", "decision_reason",
}
REVIEW_KEYS = {"extraction", "color", "compositing", "overall"}
EXPECTED_CROSSTAB = {
    "initial_d3263a6:pilot:primary": 10,
    "initial_d3263a6:reserve:primary": 5,
    "replacements_71e40c1:replacement:primary": 3,
    "replacements_71e40c1:reserve:reserve": 3,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NaturalImageCohortError(message)


def _strict_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    actual = set(value)
    _require(actual == expected, f"{label} fields differ; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    return value


def _string(value: Any, label: str) -> str:
    _require(isinstance(value, str) and value, f"{label} must be a nonempty string")
    return value


def _integer(value: Any, label: str) -> int:
    _require(type(value) is int, f"{label} must be an integer")
    return value


def _validate_review(value: Any, label: str) -> dict[str, Any]:
    review = _strict_keys(value, REVIEW_KEYS, label)
    _require(all(review[name] == "PASS" for name in REVIEW_KEYS), f"{label} must retain PASS source review")
    return review


def _parse_cohort_stable_id(value: Any, label: str) -> tuple[str, str]:
    stable_id = _string(value, label)
    try:
        return parse_stable_id(stable_id)
    except NaturalImageMaskError as exc:
        raise NaturalImageCohortError(f"{label} is invalid: {exc}") from exc


def _validate_record_shape(record: Any, label: str, aliases: set[str]) -> dict[str, Any]:
    row = _strict_keys(record, RECORD_KEYS, label)
    _integer(row["cohort_rank"], f"{label}.cohort_rank")
    _require(row["role"] in {"primary", "reserve"}, f"{label}.role is invalid")
    stable_id = row["stable_id"]
    sample_id, mask_name = _parse_cohort_stable_id(stable_id, f"{label}.stable_id")
    _require(sample_id == row["sample_id"] and mask_name == row["mask_name"],
             f"{label}.stable_id must match sample_id and mask_name")
    _require(row["source_alias"] in aliases, f"{label}.source_alias is unknown")
    _integer(row["source_rank"], f"{label}.source_rank")
    _require(row["source_rank"] > 0, f"{label}.source_rank must be positive")
    _require(row["source_group"] in {"pilot", "reserve", "replacement"}, f"{label}.source_group is invalid")
    _require(row["source_status"] == "PASS", f"{label}.source_status must be PASS")
    _validate_review(row["source_manifest_human_review"], f"{label}.source_manifest_human_review")
    _require(row["human_verdict"] in {"PASS", "FAIL"}, f"{label}.human_verdict is invalid")
    _string(row["decision_reason"], f"{label}.decision_reason")
    return row


def _validate_config_structure(config: Any) -> dict[str, Any]:
    value = _strict_keys(config, CONFIG_KEYS, "cohort config")
    _require(value["schema"] == SCHEMA, "Unexpected cohort schema")
    _require(value["cohort_id"] == COHORT_ID, "Unexpected cohort_id")
    _require(value["study"] == "natural_image_subject_color_pilot", "Unexpected study")
    source_runs = _strict_keys(value["source_runs"], SOURCE_ALIASES, "source_runs")
    for alias, source in source_runs.items():
        spec = _strict_keys(source, SOURCE_RUN_KEYS, f"source_runs.{alias}")
        relpath = _string(spec["manifest_relpath"], f"source_runs.{alias}.manifest_relpath")
        path = Path(relpath)
        _require(not path.is_absolute() and ".." not in path.parts, f"source_runs.{alias}.manifest_relpath must be safe relative")
        _string(spec["manifest_sha256"], f"source_runs.{alias}.manifest_sha256")
        _require(len(spec["manifest_sha256"]) == 64, f"source_runs.{alias}.manifest_sha256 must be SHA-256")
        _require(_integer(spec["record_count"], f"source_runs.{alias}.record_count") > 0,
                 f"source_runs.{alias}.record_count must be positive")
        verification = _strict_keys(spec["verification"], VERIFICATION_KEYS, f"source_runs.{alias}.verification")
        _require(type(verification["passed"]) is bool and verification["passed"] is True,
                 f"source_runs.{alias}.verification.passed must be true")
        _require(_integer(verification["checked_samples"],
                          f"source_runs.{alias}.verification.checked_samples") == spec["record_count"],
                 f"source_runs.{alias}.verification.checked_samples must equal record_count")
        _require(isinstance(verification["errors"], list) and all(isinstance(error, str) for error in verification["errors"]),
                 f"source_runs.{alias}.verification.errors must be a string list")
        _string(spec["stable_id_order_sha256"], f"source_runs.{alias}.stable_id_order_sha256")
    _require(value["measurement_roles"] == ["primary", "reserve"], "measurement_roles must be [primary, reserve]")
    _require(value["downstream_roles"] == ["primary"], "downstream_roles must be [primary]")
    _require(value["expected_counts"] == {"active": 21, "primary": 18, "reserve": 3, "excluded": 3},
             "expected_counts must freeze 21 active, 18 primary, 3 reserve, and 3 excluded")
    _strict_keys(value["expected_counts"], COUNT_KEYS, "expected_counts")
    _require(value["expected_source_group_role_counts"] == EXPECTED_CROSSTAB,
             "expected_source_group_role_counts must match the frozen source crosstab")
    provenance = _strict_keys(value["human_review_provenance"], {
        "source_manifest_field", "source_manifest_requirement", "audit_verdict_precedence", "audit_scope",
    }, "human_review_provenance")
    for name, field in provenance.items():
        _string(field, f"human_review_provenance.{name}")

    active = value["active_records"]
    excluded = value["excluded_records"]
    _require(isinstance(active, list) and isinstance(excluded, list), "cohort record collections must be lists")
    active = [_validate_record_shape(row, f"active_records[{index}]", SOURCE_ALIASES) for index, row in enumerate(active)]
    excluded = [_validate_record_shape(row, f"excluded_records[{index}]", SOURCE_ALIASES) for index, row in enumerate(excluded)]
    _require([row["cohort_rank"] for row in active] == list(range(1, 22)), "Active ranks must be continuous from 1 through 21")
    _require([row["role"] for row in active] == ["primary"] * 18 + ["reserve"] * 3,
             "Active roles must be 18 primary rows followed by 3 reserve rows")
    _require(all(row["human_verdict"] == "PASS" for row in active), "All active records must have human PASS verdicts")
    _require(all(row["human_verdict"] == "FAIL" for row in excluded), "All excluded records must have human FAIL verdicts")
    _require(len(excluded) == 3, "Exactly three records must be excluded")
    active_stable_ids = [row["stable_id"] for row in active]
    excluded_stable_ids = [row["stable_id"] for row in excluded]
    active_sample_ids = [row["sample_id"] for row in active]
    excluded_sample_ids = [row["sample_id"] for row in excluded]
    _require(len(set(active_stable_ids)) == len(active_stable_ids), "Active stable IDs must be unique")
    _require(len(set(active_sample_ids)) == len(active_sample_ids), "Active sample IDs must be unique")
    _require(len(set(excluded_stable_ids)) == len(excluded_stable_ids), "Excluded stable IDs must be unique")
    _require(len(set(excluded_sample_ids)) == len(excluded_sample_ids), "Excluded sample IDs must be unique")
    _require(not set(active_stable_ids).intersection(excluded_stable_ids), "Active and excluded stable IDs must not overlap")
    _require(not set(active_sample_ids).intersection(excluded_sample_ids), "Active and excluded sample IDs must not overlap")
    _require(value["active_stable_id_order_sha256"] == canonical_sha256(active_stable_ids),
             "active_stable_id_order_sha256 does not match the active order")
    crosstab: dict[str, int] = {}
    for row in active:
        key = f"{row['source_alias']}:{row['source_group']}:{row['role']}"
        crosstab[key] = crosstab.get(key, 0) + 1
    _require(crosstab == EXPECTED_CROSSTAB, "Active source group/role crosstab drifted")
    return value


def load_cohort_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the config itself, without accessing run artifacts."""
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NaturalImageCohortError(f"Cannot load cohort config {config_path}: {exc}") from exc
    return _validate_config_structure(value)


def parse_mask_run_pairs(pairs: Iterable[Sequence[str | Path]]) -> dict[str, Path]:
    """Parse repeated ``--mask-run ALIAS RUN_ROOT`` values without defining a CLI."""
    result: dict[str, Path] = {}
    for pair in pairs:
        _require(isinstance(pair, Sequence) and not isinstance(pair, (str, bytes)) and len(pair) == 2,
                 "Each mask run must be a non-string (alias, run_root) pair")
        alias, run_root = pair
        _string(alias, "mask run alias")
        _require(alias not in result, f"Duplicate mask run alias: {alias}")
        _require(isinstance(run_root, (str, Path)), f"Mask run root for {alias} must be a path")
        result[alias] = Path(run_root)
    return result


def _load_source_manifest(alias: str, spec: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / spec["manifest_relpath"]
    _require(manifest_path.is_file(), f"Source manifest is missing for {alias}: {manifest_path}")
    actual_hash = file_sha256(manifest_path)
    _require(actual_hash == spec["manifest_sha256"], f"Source manifest hash drift for {alias}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NaturalImageCohortError(f"Cannot parse source manifest for {alias}: {exc}") from exc
    _require(isinstance(manifest, dict), f"Source manifest for {alias} must be an object")
    samples = manifest.get("samples")
    _require(isinstance(samples, list), f"Source manifest samples must be a list for {alias}")
    _require(len(samples) == spec["record_count"], f"Source manifest record count drift for {alias}")
    _require(manifest.get("verification") == spec["verification"], f"Source manifest verification drift for {alias}")
    stable_ids: list[str] = []
    sample_ids: list[str] = []
    for index, sample in enumerate(samples):
        _require(isinstance(sample, dict), f"Source sample {alias}[{index}] must be an object")
        for field in ("rank", "stable_id", "sample_id", "mask_name", "group", "status", "human_review"):
            _require(field in sample, f"Source sample {alias}[{index}] missing {field}")
        sample_id, mask_name = _parse_cohort_stable_id(sample["stable_id"], f"Source sample {alias}[{index}].stable_id")
        _require(sample_id == sample["sample_id"] and mask_name == sample["mask_name"],
                 f"Source stable ID mismatch for {alias}[{index}]")
        stable_ids.append(sample["stable_id"])
        sample_ids.append(sample["sample_id"])
    _require(len(set(stable_ids)) == len(stable_ids), f"Source stable IDs must be unique for {alias}")
    _require(len(set(sample_ids)) == len(sample_ids), f"Source sample IDs must be unique for {alias}")
    stable_order_hash = canonical_sha256(stable_ids)
    _require(manifest.get("selection", {}).get("stable_id_order_sha256") == stable_order_hash,
             f"Source manifest stable order field drift for {alias}")
    _require(stable_order_hash == spec["stable_id_order_sha256"], f"Source manifest stable order drift for {alias}")
    return manifest


def validate_cohort(config: Mapping[str, Any], mask_runs: Mapping[str, str | Path]) -> dict[str, Any]:
    """Fail closed unless this config and both caller-supplied source runs match exactly."""
    value = _validate_config_structure(dict(config))
    _require(isinstance(mask_runs, Mapping), "mask_runs must map source aliases to run roots")
    _require(set(mask_runs) == SOURCE_ALIASES,
             f"mask_runs aliases differ; expected={sorted(SOURCE_ALIASES)}, actual={sorted(mask_runs)}")
    manifests = {
        alias: _load_source_manifest(alias, value["source_runs"][alias], Path(mask_runs[alias]))
        for alias in sorted(SOURCE_ALIASES)
    }
    source_stable_ids = [
        sample["stable_id"]
        for alias in sorted(SOURCE_ALIASES)
        for sample in manifests[alias]["samples"]
    ]
    source_sample_ids = [
        sample["sample_id"]
        for alias in sorted(SOURCE_ALIASES)
        for sample in manifests[alias]["samples"]
    ]
    _require(len(set(source_stable_ids)) == len(source_stable_ids),
             "Source stable IDs must be unique across aliases")
    _require(len(set(source_sample_ids)) == len(source_sample_ids),
             "Source sample IDs must be unique across aliases")
    source_rows = {
        (alias, sample["stable_id"])
        for alias in sorted(SOURCE_ALIASES)
        for sample in manifests[alias]["samples"]
    }
    seen_source_rows: set[tuple[str, str]] = set()
    for collection_name in ("active_records", "excluded_records"):
        for row in value[collection_name]:
            source = manifests[row["source_alias"]]
            matches = [sample for sample in source["samples"] if sample["stable_id"] == row["stable_id"]]
            _require(len(matches) == 1,
                     f"{collection_name} {row['stable_id']} must match exactly one source row")
            sample = matches[0]
            for field, source_field in (("source_rank", "rank"), ("source_group", "group"), ("source_status", "status")):
                _require(row[field] == sample[source_field],
                         f"{collection_name} {row['stable_id']} {field} does not match source manifest")
            _require(row["source_manifest_human_review"] == sample["human_review"],
                     f"{collection_name} {row['stable_id']} source human review does not match")
            _require(sample["human_review"].get("overall") == "PASS",
                     f"{collection_name} {row['stable_id']} source human review must be PASS")
            source_key = (row["source_alias"], row["stable_id"])
            _require(source_key not in seen_source_rows,
                     f"Source row is repeated in active/excluded cohort records: {source_key}")
            seen_source_rows.add(source_key)
    _require(seen_source_rows == source_rows,
             f"Cohort records must exactly cover source rows; missing={sorted(source_rows - seen_source_rows)}, "
             f"unexpected={sorted(seen_source_rows - source_rows)}")
    return value


def load_and_validate_cohort(config_path: str | Path, mask_runs: Mapping[str, str | Path]) -> dict[str, Any]:
    """Load a cohort file and validate its declared provenance against source manifests."""
    return validate_cohort(load_cohort_config(config_path), mask_runs)


def select_measurement_records(cohort: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return all 21 active records in frozen cohort order, including measurement reserves."""
    value = _validate_config_structure(dict(cohort))
    return deepcopy(value["active_records"])


def select_downstream_records(cohort: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return only the 18 primary active records in frozen cohort order."""
    return [row for row in select_measurement_records(cohort) if row["role"] == "primary"]
