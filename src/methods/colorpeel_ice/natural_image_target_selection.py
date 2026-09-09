"""Frozen D1 natural-image target-color selection; validation only."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from src.methods.colorpeel_ice.natural_image_masks import canonical_sha256, file_sha256


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_RELPATH = "experiments/natural_image_subject_color_pilot/configs/d1_target_color_selection_v1.json"
FROZEN_CONFIG_CANONICAL_SHA256 = "33d9b2d5fb404786d63ac23b161c7578a08442ecd24691ca4c221721e88ee0a6"
SOURCE_AUDIT_MANIFEST_SHA256 = "24eff6caf66dc6c938e888e5acc6a738ec9ff2a8ddd5c1da6cc602120861fc5a"

SCHEMA = "natural_image_target_color_selection/v1"
SOURCE_AUDIT_SCHEMA = "natural_image_target_color_audit/v1"
GIT_COMMIT = "dbd4281c9926e2dd5fce523ebe3b9820fa1c6fdf"
PROTOCOL_SHA256 = "11c159706e64d41c224627fce1ce38c89ad31e1596da6225a5aecc1a000aa798"
COHORT_SHA256 = "3e54ab3451e6525b5b14e202a7efa0b4494f2160c3bfc9c9ebd0ccfe8eb38280"
SOURCE_STABLE_ID_ORDER_SHA256 = "cd7840f5003462b7a22c49180ffe3d70d3bbdc3f1046a878b964ff0cd5c0634a"

SELECTED_STABLE_IDS = [
    "D1GT:86/139.png",
    "D1GT:76/120.png",
    "D1GT:13/30.png",
    "D1GT:19/31.png",
    "D1GT:77/53.png",
    "D1GT:81/130.png",
    "D1GT:5/101.png",
    "D1GT:43/156.png",
    "D1GT:71/152.png",
    "D1GT:31/98.png",
    "D1GT:2/62.png",
]
FAILED_STABLE_IDS = [
    "D1GT:3/207.png",
    "D1GT:7/116.png",
    "D1GT:46/97.png",
    "D1GT:11/180.png",
    "D1GT:74/104.png",
    "D1GT:53/196.png",
    "D1GT:10/91.png",
    "D1GT:1/8.png",
    "D1GT:24/198.png",
    "D1GT:17/74.png",
]
PIPELINE_ROLES = {
    "pipeline_development": [
        "D1GT:86/139.png",
        "D1GT:13/30.png",
        "D1GT:19/31.png",
        "D1GT:77/53.png",
        "D1GT:81/130.png",
        "D1GT:2/62.png",
    ],
    "protocol_confirmation": ["D1GT:43/156.png", "D1GT:71/152.png", "D1GT:31/98.png"],
    "stress_test": ["D1GT:76/120.png", "D1GT:5/101.png"],
}
EXPECTED_COUNTS = {
    "source_records": 21,
    "human_pass": 11,
    "human_fail": 10,
    "selected": 11,
    "downstream_eligible": 11,
    "pipeline_development": 6,
    "protocol_confirmation": 3,
    "stress_test": 2,
}
SOURCE_SUMMARY = {
    "measurement_count": 21,
    "downstream_primary_count": 18,
    "measured_count": 21,
    "review_count": 0,
    "pending_human_count": 21,
    "approved_count": 0,
}
SOURCE_RUNS = {
    "initial_d3263a6": {
        "manifest_relpath": "manifests/pilot_mask_manifest.json",
        "manifest_sha256": "073bc48cfd9120c230f9fe03b2c654f27421ab09cf4b968b103331c6060c558c",
        "record_count": 18,
        "verification": {"passed": True, "checked_samples": 18, "errors": []},
        "stable_id_order_sha256": "a2e910c08e327343c5247fffb5d4b469efc202457bac4626db4d00e1123a9d8b",
        "root": "/home/r12user5/Documents/Jiawei/colorpeel-runs/natural_image_subject_color_pilot/mask_derivation_d3263a6_20260908",
        "current_manifest_sha256": "073bc48cfd9120c230f9fe03b2c654f27421ab09cf4b968b103331c6060c558c",
    },
    "replacements_71e40c1": {
        "manifest_relpath": "manifests/pilot_mask_manifest.json",
        "manifest_sha256": "f792d7372ffae85132d87c52735295c52b829a99f08e4f7dcc427bf761654e4c",
        "record_count": 6,
        "verification": {"passed": True, "checked_samples": 6, "errors": []},
        "stable_id_order_sha256": "6beb65d94720f18dcac8d1a6750fced2578c92a43f92fe5e07f53721c32ef95f",
        "root": "/home/r12user5/Documents/Jiawei/colorpeel-runs/natural_image_subject_color_pilot/mask_derivation_replacements_71e40c1_20260909",
        "current_manifest_sha256": "f792d7372ffae85132d87c52735295c52b829a99f08e4f7dcc427bf761654e4c",
    },
}

CONFIG_KEYS = {
    "schema",
    "selection_id",
    "study",
    "source_audit",
    "selection_policy",
    "expected_counts",
    "selected_stable_ids",
    "failed_stable_ids",
    "pipeline_roles",
    "canonical_hashes",
    "records",
}
SOURCE_AUDIT_KEYS = {
    "schema",
    "manifest_sha256",
    "git_commit",
    "protocol_sha256",
    "cohort_sha256",
    "record_count",
    "summary",
    "qc_decision",
    "approved_for_downstream",
    "source_stable_id_order_sha256",
    "source_run_manifest_sha256",
}
POLICY_KEYS = {
    "no_replacements",
    "automatic_threshold_gate",
    "target_color_qc_approved_scope",
    "all_selected_remain_downstream_eligible",
    "not_diffusion_train_test_split",
    "held_out_condition",
    "failure_reason_source",
    "recoloring_approved",
    "training_approved",
}
HASH_KEYS = {
    "selected_stable_id_order_sha256",
    "pipeline_assignment_sha256",
    "human_verdict_order_sha256",
    "source_stable_id_order_sha256",
}
SOURCE_KEYS = {
    "cohort_rank",
    "role",
    "stable_id",
    "sample_id",
    "mask_name",
    "source_alias",
    "source_rank",
    "source_group",
    "source_status",
    "source_manifest_human_review",
}
BASE_RECORD_KEYS = {
    "source",
    "human_verdict",
    "failure_reason",
    "target_color_qc_approved",
    "downstream_eligible",
    "pipeline_role",
    "recoloring_approved",
    "recoloring_status",
    "training_approved",
    "training_status",
}
TARGET_KEYS = {"a", "b", "C", "h_degrees", "observed_median_L", "lightness_is_fixed_target_token"}
METRIC_KEYS = {
    "color_area_px",
    "eligible_px",
    "dark_px",
    "light_px",
    "valid_ratio",
    "dark_ratio",
    "light_ratio",
    "dominant_cluster",
    "dominant_ratio_over_eligible",
    "dominant_ratio_over_color_mask",
    "dominant_cluster_ratio_over_eligible",
    "dominant_cluster_compactness",
}
COMPACTNESS_KEYS = {"rms", "median", "p90"}
AUDIT_KEYS = {
    "schema",
    "git_commit",
    "protocol",
    "cohort",
    "source_runs",
    "versions",
    "records",
    "summary",
    "downstream_candidate_ids",
    "qc_decision",
    "approved_for_downstream",
    "output_sha256",
}


class NaturalImageTargetSelectionError(ValueError):
    """Raised when target selection config or source audit provenance drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NaturalImageTargetSelectionError(message)


def _strict_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    actual = set(value)
    _require(actual == expected, f"{label} fields differ; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    return value


def _stable_id_role_map() -> dict[str, str]:
    return {stable_id: role for role, stable_ids in PIPELINE_ROLES.items() for stable_id in stable_ids}


def _assignment_rows() -> list[dict[str, str]]:
    return [
        {"stable_id": stable_id, "pipeline_role": role}
        for role, stable_ids in PIPELINE_ROLES.items()
        for stable_id in stable_ids
    ]


def _looks_absolute_path(value: str) -> bool:
    return value.startswith("/") or value.startswith("\\") or re.match(r"^[A-Za-z]:[\\/]", value) is not None


def _reject_absolute_strings(value: Any, label: str = "config") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _reject_absolute_strings(nested, f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_absolute_strings(nested, f"{label}[{index}]")
    elif isinstance(value, str):
        _require(not _looks_absolute_path(value), f"{label} must not contain an absolute path")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NaturalImageTargetSelectionError(f"Cannot read JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON must be an object: {path}")
    return value


def _validate_source_fields(source: Any, label: str) -> dict[str, Any]:
    value = _strict_keys(source, SOURCE_KEYS, label)
    _require(type(value["cohort_rank"]) is int, f"{label}.cohort_rank must be an integer")
    _require(value["role"] in {"primary", "reserve"}, f"{label}.role is invalid")
    _require(isinstance(value["stable_id"], str) and value["stable_id"].startswith("D1GT:"), f"{label}.stable_id is invalid")
    _require(isinstance(value["sample_id"], str) and value["sample_id"], f"{label}.sample_id must be nonempty")
    _require(isinstance(value["mask_name"], str) and value["mask_name"].endswith(".png"), f"{label}.mask_name is invalid")
    _require(value["stable_id"] == f"D1GT:{value['sample_id']}/{value['mask_name']}", f"{label}.stable_id does not match sample fields")
    _require(value["source_alias"] in SOURCE_RUNS, f"{label}.source_alias is unknown")
    _require(type(value["source_rank"]) is int and value["source_rank"] > 0, f"{label}.source_rank must be positive integer")
    _require(value["source_group"] in {"pilot", "replacement", "reserve"}, f"{label}.source_group is invalid")
    _require(value["source_status"] == "PASS", f"{label}.source_status must be PASS")
    review = _strict_keys(value["source_manifest_human_review"], {"extraction", "color", "compositing", "overall"}, f"{label}.source_manifest_human_review")
    _require(all(verdict == "PASS" for verdict in review.values()), f"{label}.source_manifest_human_review must retain PASS values")
    return value


def _validate_record(row: Any, label: str, role_of: Mapping[str, str]) -> dict[str, Any]:
    _require(isinstance(row, dict), f"{label} must be an object")
    source = _validate_source_fields(row.get("source"), f"{label}.source")
    stable_id = source["stable_id"]
    selected = stable_id in SELECTED_STABLE_IDS
    expected_keys = BASE_RECORD_KEYS | ({"target", "metrics"} if selected else set())
    _require(set(row) == expected_keys, f"{label} fields differ; missing={sorted(expected_keys - set(row))}, extra={sorted(set(row) - expected_keys)}")
    _require(row["human_verdict"] == ("PASS" if selected else "FAIL"), f"{label}.human_verdict drifted")
    _require(row["failure_reason"] is None, f"{label}.failure_reason must remain null")
    _require(row["target_color_qc_approved"] is selected, f"{label}.target_color_qc_approved drifted")
    _require(row["downstream_eligible"] is selected, f"{label}.downstream_eligible drifted")
    _require(row["pipeline_role"] == role_of.get(stable_id), f"{label}.pipeline_role drifted")
    _require(row["recoloring_approved"] is False and row["recoloring_status"] == "PENDING", f"{label}.recoloring must remain pending/unapproved")
    _require(row["training_approved"] is False and row["training_status"] == "PENDING", f"{label}.training must remain pending/unapproved")
    if selected:
        target = _strict_keys(row["target"], TARGET_KEYS, f"{label}.target")
        _require(target["lightness_is_fixed_target_token"] is False, f"{label}.target lightness flag must be false")
        for key in TARGET_KEYS - {"lightness_is_fixed_target_token"}:
            _require(type(target[key]) is float, f"{label}.target.{key} must be a float")
        metrics = _strict_keys(row["metrics"], METRIC_KEYS, f"{label}.metrics")
        for key in ("color_area_px", "eligible_px", "dark_px", "light_px", "dominant_cluster"):
            _require(type(metrics[key]) is int, f"{label}.metrics.{key} must be an integer")
        for key in (
            "valid_ratio",
            "dark_ratio",
            "light_ratio",
            "dominant_ratio_over_eligible",
            "dominant_ratio_over_color_mask",
            "dominant_cluster_ratio_over_eligible",
        ):
            _require(type(metrics[key]) is float, f"{label}.metrics.{key} must be a float")
        compactness = _strict_keys(metrics["dominant_cluster_compactness"], COMPACTNESS_KEYS, f"{label}.metrics.dominant_cluster_compactness")
        _require(all(type(value) is float for value in compactness.values()), f"{label}.metrics.dominant_cluster_compactness values must be floats")
    return row


def _validate_config_structure(config: Any) -> dict[str, Any]:
    value = _strict_keys(config, CONFIG_KEYS, "selection config")
    _reject_absolute_strings(value)
    _require(value["schema"] == SCHEMA, "Unexpected selection schema")
    _require(value["selection_id"] == "d1_target_color_selection_v1", "Unexpected selection_id")
    _require(value["study"] == "natural_image_subject_color_pilot", "Unexpected study")
    audit = _strict_keys(value["source_audit"], SOURCE_AUDIT_KEYS, "source_audit")
    _require(audit == {
        "schema": SOURCE_AUDIT_SCHEMA,
        "manifest_sha256": SOURCE_AUDIT_MANIFEST_SHA256,
        "git_commit": GIT_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "cohort_sha256": COHORT_SHA256,
        "record_count": 21,
        "summary": SOURCE_SUMMARY,
        "qc_decision": "PENDING_HUMAN",
        "approved_for_downstream": False,
        "source_stable_id_order_sha256": SOURCE_STABLE_ID_ORDER_SHA256,
        "source_run_manifest_sha256": {
            "initial_d3263a6": SOURCE_RUNS["initial_d3263a6"]["manifest_sha256"],
            "replacements_71e40c1": SOURCE_RUNS["replacements_71e40c1"]["manifest_sha256"],
        },
    }, "source_audit drifted")
    policy = _strict_keys(value["selection_policy"], POLICY_KEYS, "selection_policy")
    _require(policy["no_replacements"] is True, "selection_policy.no_replacements must be true")
    _require(policy["automatic_threshold_gate"] is None, "selection_policy must not contain an automatic threshold gate")
    _require(policy["target_color_qc_approved_scope"] == "selection_layer_only", "selection approval scope drifted")
    _require(policy["all_selected_remain_downstream_eligible"] is True, "selected rows must remain downstream eligible")
    _require(policy["not_diffusion_train_test_split"] is True, "pipeline roles are not a diffusion train/test split")
    _require("original <S*> + <C*> must not appear during training" in policy["held_out_condition"], "held-out condition drifted")
    _require(policy["recoloring_approved"] is False and policy["training_approved"] is False, "recoloring/training approval must stay false")
    _require(value["expected_counts"] == EXPECTED_COUNTS, "expected_counts drifted")
    _require(value["selected_stable_ids"] == SELECTED_STABLE_IDS, "selected stable-id order drifted")
    _require(value["failed_stable_ids"] == FAILED_STABLE_IDS, "failed stable-id order drifted")
    _require(value["pipeline_roles"] == PIPELINE_ROLES, "pipeline role membership drifted")
    records = value["records"]
    _require(isinstance(records, list) and len(records) == 21, "records must contain exactly 21 rows")
    role_of = _stable_id_role_map()
    records = [_validate_record(row, f"records[{index}]", role_of) for index, row in enumerate(records)]
    source_order = [row["source"]["stable_id"] for row in records]
    _require(len(set(source_order)) == 21, "records must contain 21 unique stable IDs")
    _require([row["source"]["cohort_rank"] for row in records] == list(range(1, 22)), "source cohort ranks must be continuous")
    _require(set(SELECTED_STABLE_IDS).union(FAILED_STABLE_IDS) == set(source_order), "PASS and FAIL sets must cover all source rows")
    _require(set(SELECTED_STABLE_IDS).isdisjoint(FAILED_STABLE_IDS), "PASS and FAIL sets must be disjoint")
    _require([stable_id for stable_id in source_order if stable_id in SELECTED_STABLE_IDS] == SELECTED_STABLE_IDS, "selected IDs must preserve source cohort order")
    _require([stable_id for stable_id in source_order if stable_id in FAILED_STABLE_IDS] == FAILED_STABLE_IDS, "failed IDs must cover the source-order remainder")
    _require(sum(row["human_verdict"] == "PASS" for row in records) == 11, "PASS count drifted")
    _require(sum(row["human_verdict"] == "FAIL" for row in records) == 10, "FAIL count drifted")
    _require(sum(row["downstream_eligible"] is True for row in records) == 11, "downstream eligibility count drifted")
    hashes = _strict_keys(value["canonical_hashes"], HASH_KEYS, "canonical_hashes")
    verdict_rows = [{"stable_id": row["source"]["stable_id"], "human_verdict": row["human_verdict"]} for row in records]
    _require(hashes["selected_stable_id_order_sha256"] == canonical_sha256(SELECTED_STABLE_IDS), "selected-order hash drifted")
    _require(hashes["pipeline_assignment_sha256"] == canonical_sha256(_assignment_rows()), "assignment hash drifted")
    _require(hashes["human_verdict_order_sha256"] == canonical_sha256(verdict_rows), "verdict-order hash drifted")
    _require(hashes["source_stable_id_order_sha256"] == canonical_sha256(source_order), "source-order hash drifted")
    _require(hashes["source_stable_id_order_sha256"] == SOURCE_STABLE_ID_ORDER_SHA256, "source-order hash does not match audit")
    for role, stable_ids in PIPELINE_ROLES.items():
        _require(len(stable_ids) == EXPECTED_COUNTS[role], f"{role} count drifted")
    return value


def load_selection_config(path: str | Path = REPO_ROOT / CONFIG_RELPATH) -> dict[str, Any]:
    """Load only the frozen repository config path and validate its canonical hash."""
    config_path = Path(path)
    _require(config_path.resolve() == (REPO_ROOT / CONFIG_RELPATH).resolve(), "selection config path must be the frozen repository config")
    value = read_json(config_path)
    _require(canonical_sha256(value) == FROZEN_CONFIG_CANONICAL_SHA256, "selection config canonical hash drift")
    return _validate_config_structure(value)


def _expected_audit_provenance() -> dict[str, Any]:
    return {
        "schema": SOURCE_AUDIT_SCHEMA,
        "git_commit": GIT_COMMIT,
        "protocol": {
            "path": "/home/r12user5/Documents/Jiawei/colorpeel/experiments/natural_image_subject_color_pilot/configs/d1_target_color_audit_protocol_v1.json",
            "sha256": PROTOCOL_SHA256,
        },
        "cohort": {
            "path": "/home/r12user5/Documents/Jiawei/colorpeel/experiments/natural_image_subject_color_pilot/configs/d1_target_color_audit_cohort_v1.json",
            "sha256": COHORT_SHA256,
            "cohort_id": "d1_target_color_audit_cohort_v1",
        },
        "source_runs": SOURCE_RUNS,
        "summary": SOURCE_SUMMARY,
        "qc_decision": "PENDING_HUMAN",
        "approved_for_downstream": False,
    }


def _audit_row_by_stable_id(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("records")
    _require(isinstance(rows, list) and len(rows) == 21, "audit manifest must contain exactly 21 records")
    result = {}
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), f"audit records[{index}] must be an object")
        stable_id = row.get("stable_id")
        _require(isinstance(stable_id, str), f"audit records[{index}].stable_id must be a string")
        _require(stable_id not in result, f"audit stable_id is duplicated: {stable_id}")
        result[stable_id] = row
    _require(canonical_sha256([row["stable_id"] for row in rows]) == SOURCE_STABLE_ID_ORDER_SHA256, "audit source row order drifted")
    return result


def _validate_audit_provenance(manifest: Mapping[str, Any]) -> None:
    _strict_keys(manifest, AUDIT_KEYS, "audit manifest")
    expected = _expected_audit_provenance()
    for key in ("schema", "git_commit", "summary", "qc_decision", "approved_for_downstream"):
        _require(manifest[key] == expected[key], f"audit manifest {key} drifted")
    for key in ("path", "sha256"):
        _require(manifest["protocol"].get(key) == expected["protocol"][key], f"audit protocol {key} drifted")
    for key in ("path", "sha256", "cohort_id"):
        _require(manifest["cohort"].get(key) == expected["cohort"][key], f"audit cohort {key} drifted")
    _require(manifest["source_runs"] == expected["source_runs"], "audit source run root provenance drifted")


def _compare_source(config_row: Mapping[str, Any], audit_row: Mapping[str, Any], label: str) -> None:
    for key in SOURCE_KEYS - {"source_manifest_human_review"}:
        _require(config_row["source"][key] == audit_row.get(key), f"{label}.{key} differs from audit row")
    _require(config_row["source"]["source_manifest_human_review"] == audit_row.get("source_manifest_human_review"),
             f"{label}.source_manifest_human_review differs from audit row")


def _compare_selected_fields(config_row: Mapping[str, Any], audit_row: Mapping[str, Any], label: str) -> None:
    _require(audit_row.get("status") == "MEASURED", f"{label} audit row must remain MEASURED")
    _require(audit_row.get("qc_decision") == "PENDING_HUMAN", f"{label} audit row qc_decision drifted")
    _require(audit_row.get("approved_for_downstream") is False, f"{label} audit row approval drifted")
    _require(config_row["target"] == {key: audit_row["provisional_target"][key] for key in TARGET_KEYS}, f"{label}.target differs from audit row")
    metrics = {key: audit_row[key] for key in METRIC_KEYS if key not in {"dominant_cluster_compactness", "dominant_cluster_ratio_over_eligible"}}
    dominant = audit_row["clusters"][audit_row["dominant_cluster"]]
    metrics["dominant_cluster_ratio_over_eligible"] = dominant["ratio_over_eligible"]
    metrics["dominant_cluster_compactness"] = dominant["compactness"]
    _require(config_row["metrics"] == metrics, f"{label}.metrics differ from audit row")


def validate_source_audit_manifest(audit_manifest_path: str | Path, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate a caller-supplied audit manifest against the frozen config and byte hash."""
    selected_config = _validate_config_structure(dict(config)) if config is not None else load_selection_config()
    path = Path(audit_manifest_path)
    _require(path.is_file(), f"audit manifest is missing: {path}")
    _require(file_sha256(path) == SOURCE_AUDIT_MANIFEST_SHA256, "audit manifest byte hash drift")
    manifest = read_json(path)
    _validate_audit_provenance(manifest)
    audit_rows = _audit_row_by_stable_id(manifest)
    _require(set(audit_rows) == set(SELECTED_STABLE_IDS).union(FAILED_STABLE_IDS), "audit rows must exactly cover PASS and FAIL sets")
    for index, row in enumerate(selected_config["records"]):
        stable_id = row["source"]["stable_id"]
        audit_row = audit_rows[stable_id]
        label = f"records[{index}] {stable_id}"
        _compare_source(row, audit_row, label)
        if stable_id in SELECTED_STABLE_IDS:
            _compare_selected_fields(row, audit_row, label)
    return selected_config


def selected_records(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    value = _validate_config_structure(dict(config)) if config is not None else load_selection_config()
    return [deepcopy(row) for row in value["records"] if row["human_verdict"] == "PASS"]


def records_for_pipeline_role(role: str, config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    _require(role in PIPELINE_ROLES, f"Unknown pipeline role: {role}")
    selected = selected_records(config)
    by_id = {row["source"]["stable_id"]: row for row in selected}
    return [deepcopy(by_id[stable_id]) for stable_id in PIPELINE_ROLES[role]]


def pipeline_development_records(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    return records_for_pipeline_role("pipeline_development", config)


def protocol_confirmation_records(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    return records_for_pipeline_role("protocol_confirmation", config)


def stress_test_records(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    return records_for_pipeline_role("stress_test", config)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-manifest", required=True, type=Path)
    parser.add_argument("--selection-config", default=REPO_ROOT / CONFIG_RELPATH, type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_selection_config(args.selection_config)
        validate_source_audit_manifest(args.audit_manifest, config)
    except NaturalImageTargetSelectionError as exc:
        parser.exit(2, f"target selection validation aborted: {exc}\n")
    summary = {
        "schema": config["schema"],
        "selected_count": len(SELECTED_STABLE_IDS),
        "failed_count": len(FAILED_STABLE_IDS),
        "pipeline_roles": {role: len(stable_ids) for role, stable_ids in PIPELINE_ROLES.items()},
        "downstream_eligible_count": sum(row["downstream_eligible"] is True for row in config["records"]),
        "recoloring_approved": False,
        "training_approved": False,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
