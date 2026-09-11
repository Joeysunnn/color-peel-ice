#!/usr/bin/env python3
"""RC-2C-1 deterministic Rubber L/chroma search; no confirmation rendering.

Ordinary Python plans/analyzes; Blender 4.2.11 renders. Every stage is
single-use. The predecessor analysis SHA must be obtained from trusted RC-2B
evidence; its complete neighboring plan, contract, manifest and renders are
verified when initializing the new search root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.methods.colorpeel_ice import render_d1_color_calibration_reduced_direct as direct

shared = direct.shared
require = direct.require
canonical_sha256 = direct.canonical_sha256
file_sha256 = direct.file_sha256
load_json = direct.load_json
PREFIX = "d1_rubber_fallback_search"
CONTRACT = "search_contract.json"
PREDECESSOR = "predecessor.json"
PLAN = "plan.json"
MANIFEST = "render_manifest.json"
ANALYSIS = "analysis.json"
SELECTION = "selection_manifest.json"
COMPATIBILITY = "legacy_gpu_inventory_compatibility.json"
# Only failure records from stages that write no rendered artifacts may be retried.
# Render failures remain deliberate stop evidence.
RETRYABLE_FAILURES = {"analyze-coarse_failure.json", "analyze-final_failure.json", "plan-refine_failure.json"}
# The only RC-2C contract eligible for this GPU-inventory compatibility path.
# Its completed coarse renders predate the GPU-inventory comparison fix.
LEGACY_GPU_INVENTORY_GIT_COMMIT = "408efc838dffbaa30fdf30c5fa02a9c63404678f"
LEGACY_GPU_INVENTORY_ADAPTER_SHA256 = "e3ea29454a97e043aa2a2c3330392dcebae1f95598c874628d3c64db833d1203"
INITIAL_COMPATIBILITY_ANALYZER_GIT_COMMIT = "2fd71e96c195fe51f0b4c1c3c0cc9cda8cfaf7b6"
INITIAL_COMPATIBILITY_ANALYZER_ADAPTER_SHA256 = "d0ea7a6714a4035c81c96241404c973dddb7db5faf6f4d65bdb06baf53ccafeb"


def _calibration():
    # RC-1 imports Pillow; Blender only consumes the sealed, persisted plan.
    from src.methods.colorpeel_ice import renderer_color_calibration
    return renderer_color_calibration


def _write(path: Path, value: Any) -> None:
    require(not path.exists(), f"Output already exists: {path.name}")
    direct.atomic_json(path, value)


def _json_sha(value: Any, newline: str | None = None) -> str:
    # Frozen RC-2B atomic_json serialization, including the trailing newline.
    serialized = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(serialized.replace("\n", os.linesep if newline is None else newline).encode()).hexdigest()


def _code_hashes() -> dict[str, str]:
    return {"adapter": file_sha256(Path(__file__)), "rc2b": file_sha256(Path(direct.__file__)),
            "rc2a": file_sha256(Path(shared.__file__)),
            "rc1": file_sha256(REPO_ROOT / "src/methods/colorpeel_ice/renderer_color_calibration.py")}


def _targets() -> list[dict[str, Any]]:
    return [row for row in direct._requests() if row["view_index"] == 0]


def _runtime_contract(identity: dict[str, Any]) -> dict[str, Any]:
    """Keep GPU inventory as provenance, not a renderer-semantics equality key."""
    return {**identity, "renderer": {key: value for key, value in identity["renderer"].items()
                                       if key != "cuda_devices"}}


def _legacy_gpu_inventory_contract(contract: dict[str, Any]) -> bool:
    current = _code_hashes()
    legacy = {**current, "adapter": LEGACY_GPU_INVENTORY_ADAPTER_SHA256}
    return (contract["git_commit"] == LEGACY_GPU_INVENTORY_GIT_COMMIT
            and contract["code_sha256"] == legacy)


def _compatibility_evidence(contract: dict[str, Any]) -> dict[str, Any]:
    return {"schema": f"{PREFIX}_legacy_gpu_inventory_compatibility/v1",
            "reason": "cuda_devices_inventory_is_provenance_only",
            "legacy_contract_git_commit": contract["git_commit"],
            "legacy_adapter_sha256": contract["code_sha256"]["adapter"],
            "analyzer_git_commit": shared._git_commit(),
            "analyzer_adapter_sha256": _code_hashes()["adapter"]}


def _valid_compatibility_evidence(contract: dict[str, Any], evidence: Any) -> bool:
    # The sidecar records the analyzer that performed coarse analysis. It must
    # remain valid after a later, non-rendering recovery patch changes HEAD.
    initial = {"schema": f"{PREFIX}_legacy_gpu_inventory_compatibility/v1",
               "reason": "cuda_devices_inventory_is_provenance_only",
               "legacy_contract_git_commit": contract["git_commit"],
               "legacy_adapter_sha256": contract["code_sha256"]["adapter"],
               "analyzer_git_commit": INITIAL_COMPATIBILITY_ANALYZER_GIT_COMMIT,
               "analyzer_adapter_sha256": INITIAL_COMPATIBILITY_ANALYZER_ADAPTER_SHA256}
    return evidence == _compatibility_evidence(contract) or evidence == initial


def _records(root: Path, requests: list[dict[str, Any]], contract: dict[str, Any],
             manifest: dict[str, Any], prefix: str, expected_base=None, expected_runtime=None):
    require(set(manifest) == {"schema", "contract_sha256", "request_count", "records"}, "Manifest fields differ")
    require(manifest["schema"] == f"{prefix}_manifest/v1"
            and manifest["contract_sha256"] == canonical_sha256(contract)
            and manifest["request_count"] == len(requests), "Manifest contract/count differs")
    records = manifest["records"]
    require(isinstance(records, list) and len(records) == len(requests), "Missing candidate rows")
    by_id = {row["request_id"]: row for row in requests}
    identities = [record.get("request_id") for record in records if isinstance(record, dict)]
    require(len(identities) == len(records) and all(isinstance(identity, str) for identity in identities)
            and len(set(identities)) == len(identities) and set(identities) == set(by_id),
            "Missing, duplicate or extra candidate rows")
    seen, measured = set(), {}
    base, runtime = expected_base, expected_runtime
    partial = root / ".partial"
    require(not partial.exists() or not any(partial.iterdir()), "Partial previous outputs remain")
    for record in records:
        require(isinstance(record, dict) and isinstance(record.get("request_id"), str), "Invalid manifest record")
        identity = record["request_id"]
        require(identity in by_id and identity not in seen, "Duplicate or extra candidate rows")
        seen.add(identity)
        row, state = direct._decode_record(root, by_id[identity], contract, record)
        metadata = load_json(shared._artifact_path(root, record["metadata_relative_path"], "metadata"))
        identity_runtime = {key: metadata[key] for key in ("blender_version", "blender_build_identifier", "renderer")}
        require(base is None or base == state, "Base scene metadata drift")
        require(runtime is None or _runtime_contract(runtime) == _runtime_contract(identity_runtime),
                "Runtime metadata drift")
        base, runtime = state, identity_runtime
        measured[identity] = row
    render_root = root / "renders"
    require((not render_root.exists() and not seen)
            or (render_root.is_dir() and {p.name for p in render_root.iterdir()} == seen), "Render coverage differs")
    return [measured[row["request_id"]] for row in requests], base, runtime


def _check_predecessor(bundle: dict[str, Any], *, frozen: bool = True) -> None:
    require(set(bundle) == {"plan", "contract", "manifest", "analysis", "analysis_file_sha256"}, "Predecessor fields differ")
    plan, contract, manifest, analysis = (bundle[key] for key in ("plan", "contract", "manifest", "analysis"))
    requests = direct._requests() if frozen else plan["requests"]
    require(len(requests) == 18 and canonical_sha256(requests) == direct.REQUEST_HASH, "Predecessor requests hash differs")
    require(set(contract) == {"schema", "git_commit", "adapter_script_sha256", "assets", "material",
                              "protocol_canonical_sha256", "requests_sha256", "request_count"}, "Predecessor contract fields differ")
    require(contract["schema"] == f"{direct.PREFIX}_contract/v1" and contract["material"] == "Rubber"
            and contract["protocol_canonical_sha256"] == shared.FROZEN_PROTOCOL_SHA256
            and contract["requests_sha256"] == direct.REQUEST_HASH and contract["request_count"] == 18
            and contract["adapter_script_sha256"] == direct._script_hash(), "Predecessor contract differs")
    require(isinstance(contract["git_commit"], str) and len(contract["git_commit"]) == 40
            and all(c in "0123456789abcdef" for c in contract["git_commit"]), "Predecessor git commit differs")
    _check_assets(contract["assets"])
    require(plan == {"schema": f"{direct.PREFIX}_plan/v1", "requests": requests,
                     "requests_sha256": direct.REQUEST_HASH, "protocol_canonical_sha256": shared.FROZEN_PROTOCOL_SHA256,
                     "contract_sha256": canonical_sha256(contract)}, "Predecessor frozen request payload differs")
    require(set(manifest) == {"schema", "contract_sha256", "request_count", "records"}
            and manifest["schema"] == f"{direct.PREFIX}_manifest/v1" and manifest["request_count"] == 18
            and manifest["contract_sha256"] == canonical_sha256(contract), "Predecessor manifest differs")
    records = manifest["records"]
    require(isinstance(records, list) and len(records) == 18
            and {r["request_id"] for r in records} == {r["request_id"] for r in requests}, "Predecessor incomplete/duplicate manifest")
    require(set(analysis) == {"schema", "status", "contract_sha256", "manifest_sha256", "measurements", "gate"}
            and analysis["schema"] == f"{direct.PREFIX}_analysis/v1" and analysis["status"] == "failed"
            and analysis["contract_sha256"] == canonical_sha256(contract)
            and analysis["manifest_sha256"] in {_json_sha(manifest, "\n"), _json_sha(manifest, "\r\n")}
            and bundle["analysis_file_sha256"] in {_json_sha(analysis, "\n"), _json_sha(analysis, "\r\n")},
            "Predecessor failure/hash provenance differs")
    gate = (_calibration().summarize_chroma_measurements(analysis["measurements"], "reduced_direct")
            if frozen else analysis["gate"])
    require(gate == analysis["gate"] and gate["overall_pass"] is False, "Predecessor must be a complete failed direct gate")
    by_id = {row["request_id"]: row for row in records}
    for row in analysis["measurements"]:
        require(row.get("decision") == "measurement_valid" and row.get("provenance") == {
            "render_contract_sha256": canonical_sha256(contract), **by_id[row["request_id"]]}, "Predecessor row provenance differs")
    if frozen:
        _calibration()._reject_absolute_strings(bundle, "predecessor")


def _check_assets(assets: dict[str, Any]) -> None:
    require(isinstance(assets, dict) and set(assets) == set(direct.ASSETS), "Assets differ")
    for name, relative in direct.ASSETS.items():
        row = assets[name]
        require(set(row) == {"relative_path", "sha256"} and row["relative_path"] == relative,
                "Asset relative path differs")
        require(isinstance(row["sha256"], str) and len(row["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in row["sha256"]), "Asset hash differs")


def _stage_plan(stage: str, contract: dict[str, Any], coarse=None) -> dict[str, Any]:
    calibration = _calibration()
    require(stage in {"coarse", "refine"}, "Only coarse/refine search is permitted; confirmation is excluded")
    targets, requests = [], []
    for target in _targets():
        stable_id, a, b = target["stable_id"], target["target_a"], target["target_b"]
        summaries = None if coarse is None else coarse["candidate_summaries"][stable_id]
        if stage == "coarse":
            grid = calibration.coarse_fallback_candidates(a, b)
            renderable = [c for c in grid if c["gamut"] == "in"]
        else:
            search = calibration.fallback_candidate_search_plan(a, b, summaries)
            grid = calibration.refine_fallback_candidates(a, b, summaries)
            renderable = [c for c in search["candidates"] if c["source"] == "refine"]
        entries = []
        for candidate in grid:
            decision = "invalid_gamut" if candidate["gamut"] == "out" else (
                "render" if candidate in renderable else "reused_coarse")
            entries.append({"candidate": candidate, "decision": decision})
            if decision == "render":
                requests.extend(calibration.build_fallback_candidate_requests(a, b, candidate, coarse_summaries=summaries))
        targets.append({"stable_id": stable_id, "target_a": a, "target_b": b, "candidates": entries})
    return {"schema": f"{PREFIX}_plan/v1", "stage": stage, "contract_sha256": canonical_sha256(contract),
            "coarse_analysis_sha256": None if coarse is None else _json_sha(coarse),
            "targets": targets, "requests": requests, "requests_sha256": canonical_sha256(requests)}


def make_plan(root: Path, asset_root: Path, direct_analysis: Path, direct_analysis_sha256: str):
    shared._new_or_empty(root)
    require(file_sha256(direct_analysis) == direct_analysis_sha256, "Trusted predecessor analysis SHA differs")
    previous = direct_analysis.parent
    bundle = {key: load_json(previous / name) for key, name in (
        ("plan", direct.PLAN_NAME), ("contract", direct.CONTRACT_NAME),
        ("manifest", direct.MANIFEST_NAME), ("analysis", direct.SUMMARY_NAME))}
    require(direct_analysis.resolve() == (previous / direct.SUMMARY_NAME).resolve(), "Predecessor analysis filename differs")
    bundle["analysis_file_sha256"] = direct_analysis_sha256
    _check_predecessor(bundle)
    direct._verify_runtime_assets(asset_root, bundle["contract"])
    rows, base, runtime = _records(previous, direct._requests(), bundle["contract"], bundle["manifest"], direct.PREFIX)
    require({r["request_id"]: r for r in rows} == {r["request_id"]: r for r in bundle["analysis"]["measurements"]},
            "Predecessor measurements differ from artifacts")
    contract = {"schema": f"{PREFIX}_contract/v1", "git_commit": shared._git_commit(), "code_sha256": _code_hashes(),
                "protocol_canonical_sha256": shared.FROZEN_PROTOCOL_SHA256, "assets": bundle["contract"]["assets"],
                "material": "Rubber", "predecessor_sha256": canonical_sha256(bundle),
                "predecessor_stage": "reduced_direct", "predecessor_status": "failed",
                "base_scene_state": base, "runtime_identity": runtime}
    contract["coarse_requests_sha256"] = _stage_plan("coarse", contract)["requests_sha256"]
    plan = _stage_plan("coarse", contract)
    _write(root / PREDECESSOR, bundle)
    _write(root / CONTRACT, contract)
    _write(root / "coarse" / PLAN, plan)
    return plan, contract


def _load_contract(root: Path, *, frozen: bool = True, allow_legacy_gpu_inventory: bool = False,
                   require_compatibility_evidence: bool = False):
    children = {p.name for p in root.iterdir()}
    require(children <= {CONTRACT, PREDECESSOR, "coarse", "refine", "refine_plan_binding.json", SELECTION,
                         COMPATIBILITY} | RETRYABLE_FAILURES,
            "Partial previous outputs or failed run evidence present")
    for name in RETRYABLE_FAILURES & children:
        failure = load_json(root / name)
        command = name.removesuffix("_failure.json")
        require(set(failure) == {"schema", "status", "command", "error_type"}
                and failure["schema"] == f"{PREFIX}_failure/v1" and failure["status"] == "failed"
                and failure["command"] == command and isinstance(failure["error_type"], str) and failure["error_type"],
                "Retryable failure provenance differs")
    contract, bundle = load_json(root / CONTRACT), load_json(root / PREDECESSOR)
    _check_predecessor(bundle, frozen=frozen)
    require(set(contract) == {"schema", "git_commit", "code_sha256", "protocol_canonical_sha256", "assets", "material",
                              "predecessor_sha256", "predecessor_stage", "predecessor_status", "base_scene_state", "runtime_identity",
                              "coarse_requests_sha256"},
            "Search contract fields differ")
    require(contract["schema"] == f"{PREFIX}_contract/v1" and contract["material"] == "Rubber"
            and contract["predecessor_stage"] == "reduced_direct" and contract["predecessor_status"] == "failed"
            and contract["predecessor_sha256"] == canonical_sha256(bundle), "Search predecessor contract differs")
    exact_code = (contract["git_commit"] == shared._git_commit()
                  and contract["code_sha256"] == _code_hashes())
    legacy = _legacy_gpu_inventory_contract(contract)
    require(exact_code or (allow_legacy_gpu_inventory and legacy), "Adapter or frozen code hash differs")
    evidence = root / COMPATIBILITY
    if legacy and require_compatibility_evidence:
        require(evidence.is_file() and _valid_compatibility_evidence(contract, load_json(evidence)),
                "Legacy GPU inventory compatibility provenance differs")
    if exact_code:
        require(not evidence.exists(), "Unexpected legacy compatibility evidence")
    protocol = load_json(REPO_ROOT / "experiments/natural_image_subject_color_pilot/configs/d1_renderer_color_calibration_protocol_v1.json")
    require(contract["protocol_canonical_sha256"] == canonical_sha256(protocol) == shared.FROZEN_PROTOCOL_SHA256,
            "Protocol hash differs")
    require(contract["assets"] == bundle["contract"]["assets"], "Assets changed from predecessor")
    if frozen:
        _calibration()._reject_absolute_strings(contract, "contract")
    return contract


def _load_stage(root: Path, stage: str, contract, coarse=None):
    plan = load_json(root / stage / PLAN)
    require(plan == _stage_plan(stage, contract, coarse), "Frozen candidate request/seed/payload or stage plan drift")
    _render_plan(root, stage, contract)
    return plan


def _measure_stage(root: Path, stage: str, contract, coarse=None):
    calibration = _calibration()
    plan = _load_stage(root, stage, contract, coarse)
    directory = root / stage
    manifest = load_json(directory / MANIFEST)
    rows, _, _ = _records(directory, plan["requests"], contract, manifest, PREFIX,
                          contract["base_scene_state"], contract["runtime_identity"])
    by_key = {}
    for row in rows:
        by_key.setdefault((row["stable_id"], row["L_input"], row["chroma_scale"]), []).append(row)
    summaries = {}
    for target in plan["targets"]:
        stable_id = target["stable_id"]
        prior = None if coarse is None else coarse["candidate_summaries"][stable_id]
        summaries[stable_id] = [calibration.summarize_fallback_candidate(
            entry["candidate"], by_key[(stable_id, entry["candidate"]["L_input"], entry["candidate"]["chroma_scale"])],
            coarse_summaries=prior) for entry in target["candidates"] if entry["decision"] == "render"]
    return {"schema": f"{PREFIX}_analysis/v1", "status": "complete", "stage": stage,
            "contract_sha256": canonical_sha256(contract), "plan_sha256": file_sha256(directory / PLAN),
            "manifest_sha256": file_sha256(directory / MANIFEST), "measurements": rows, "candidate_summaries": summaries}


def _verified_coarse(root: Path, contract):
    saved = load_json(root / "coarse" / ANALYSIS)
    actual = _measure_stage(root, "coarse", contract)
    require(saved == actual, "Complete coarse analysis differs from artifacts")
    return actual


def plan_refine(root: Path):
    require(not (root / "refine").exists(), "Refine output already exists")
    contract = _load_contract(root, allow_legacy_gpu_inventory=True, require_compatibility_evidence=True)
    coarse = _verified_coarse(root, contract)
    plan = _stage_plan("refine", contract, coarse)
    _write(root / "refine_plan_binding.json", {"plan_sha256": _json_sha(plan),
        "coarse_analysis_sha256": file_sha256(root / "coarse" / ANALYSIS),
        "contract_sha256": canonical_sha256(contract)})
    _write(root / "refine" / PLAN, plan)
    return plan


def _render_plan(root: Path, stage: str, contract):
    require(stage in {"coarse", "refine"}, "Only coarse/refine search is permitted; confirmation is excluded")
    plan = load_json(root / stage / PLAN)
    require(set(plan) == {"schema", "stage", "contract_sha256", "coarse_analysis_sha256", "targets", "requests", "requests_sha256"}
            and plan["schema"] == f"{PREFIX}_plan/v1" and plan["stage"] == stage
            and plan["contract_sha256"] == canonical_sha256(contract), "Stage plan contract differs")
    require(canonical_sha256(plan["requests"]) == plan["requests_sha256"], "Candidate request payload hash differs")
    if stage == "coarse":
        require(plan["requests_sha256"] == contract["coarse_requests_sha256"]
                and plan["coarse_analysis_sha256"] is None, "Frozen coarse payload hash differs")
    else:
        require(load_json(root / "refine_plan_binding.json") == {
            "plan_sha256": file_sha256(root / stage / PLAN),
            "coarse_analysis_sha256": file_sha256(root / "coarse" / ANALYSIS),
            "contract_sha256": canonical_sha256(contract)}, "Refine plan binding differs")
        require(plan["coarse_analysis_sha256"] == file_sha256(root / "coarse" / ANALYSIS), "Coarse analysis hash differs")
        coarse = load_json(root / "coarse" / ANALYSIS)
        require(coarse["status"] == "complete" and coarse["stage"] == "coarse"
                and coarse["contract_sha256"] == canonical_sha256(contract)
                and coarse["plan_sha256"] == file_sha256(root / "coarse" / PLAN)
                and coarse["manifest_sha256"] == file_sha256(root / "coarse" / MANIFEST), "Coarse evidence hash differs")
        _render_plan(root, "coarse", contract)
        partial = root / "coarse" / ".partial"
        require(not partial.exists() or not any(partial.iterdir()), "Partial previous coarse outputs remain")
        for row in coarse["measurements"]:
            record = row["provenance"]
            for kind in ("image", "mask", "metadata"):
                path = shared._artifact_path(root / "coarse", record[f"{kind}_relative_path"], kind)
                require(file_sha256(path) == record[f"{kind}_sha256"], "Coarse artifact hash differs")
    ids, groups = set(), {}
    for row in plan["requests"]:
        require(row["request_id"] not in ids and row["stage"] == f"fallback_{stage}"
                and row["source"] == stage and row["shape"] == "sphere" and row["view_index"] in (0, 8, 16)
                and row["pipeline_role"] == "pipeline_development", "Duplicate request or wrong stage/shape/view")
        ids.add(row["request_id"])
        direct.rubber_socket_rgba(row)
        groups.setdefault((row["stable_id"], row["L_input"], row["chroma_scale"]), []).append(row["view_index"])
    require(all(views == [0, 8, 16] for views in groups.values()), "Missing/duplicate candidate views")
    return plan


def render(root: Path, asset_root: Path, stage: str):
    require(shared.bpy is not None, "Render requires Blender")
    contract = _load_contract(root, frozen=False, allow_legacy_gpu_inventory=True,
                              require_compatibility_evidence=True)
    plan = _render_plan(root, stage, contract)
    paths = direct._verify_runtime_assets(asset_root, contract)
    directory = root / stage
    require({p.name for p in directory.iterdir()} == {PLAN}, "Stage contains existing/partial outputs; resume is forbidden")
    records = [direct._render_one(directory, request, contract, *paths) for request in plan["requests"]]
    manifest = {"schema": f"{PREFIX}_manifest/v1", "contract_sha256": canonical_sha256(contract),
                "request_count": len(records), "records": records}
    _write(directory / MANIFEST, manifest)
    return manifest


def analyze(root: Path, stage: str):
    calibration = _calibration()
    require(stage in {"coarse", "refine"}, "Only coarse/refine search is permitted")
    require(not (root / stage / ANALYSIS).exists() and not (root / SELECTION).exists(), "Analysis output already exists")
    contract = _load_contract(root, allow_legacy_gpu_inventory=True,
                              require_compatibility_evidence=stage == "refine")
    coarse = _verified_coarse(root, contract) if stage == "refine" else None
    result = _measure_stage(root, stage, contract, coarse)
    if stage == "coarse":
        _write(root / stage / ANALYSIS, result)
        if _legacy_gpu_inventory_contract(contract):
            _write(root / COMPATIBILITY, _compatibility_evidence(contract))
        return result
    selections = []
    for target in _targets():
        stable_id = target["stable_id"]
        choice = calibration.choose_fallback_candidate(target["target_a"], target["target_b"],
            coarse["candidate_summaries"][stable_id], result["candidate_summaries"][stable_id])
        selections.append({"stable_id": stable_id, "target_a": target["target_a"], "target_b": target["target_b"], **choice})
    passed = all(row["accepted_fallback"] for row in selections)
    selection = {"schema": f"{PREFIX}_selection/v1", "stage": "candidate_search_only",
                 "status": "accepted_candidates" if passed else "l_chroma_inverse_insufficient",
                 "contract_sha256": canonical_sha256(contract), "predecessor_sha256": contract["predecessor_sha256"],
                 "coarse_analysis_sha256": _json_sha(coarse), "refine_analysis_sha256": _json_sha(result),
                 "selection_count": 6 if passed else 0, "targets": selections,
                 "confirmation_performed": False}
    _write(root / stage / ANALYSIS, result)
    _write(root / SELECTION, selection)
    return selection


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan-coarse", "render-coarse", "analyze-coarse", "plan-refine", "render-refine", "analyze-final"):
        command = commands.add_parser(name)
        command.add_argument("--output-root", required=True, type=Path)
        if name in {"plan-coarse", "render-coarse", "render-refine"}:
            command.add_argument("--asset-root", required=True, type=Path)
        if name == "plan-coarse":
            command.add_argument("--direct-analysis", required=True, type=Path)
            command.add_argument("--direct-analysis-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(shared._extract_args(None) if argv is None else list(argv))
    try:
        if args.command == "plan-coarse":
            make_plan(args.output_root, args.asset_root, args.direct_analysis, args.direct_analysis_sha256)
        elif args.command == "plan-refine":
            plan_refine(args.output_root)
        elif args.command.startswith("render-"):
            render(args.output_root, args.asset_root, args.command.split("-")[1])
        else:
            result = analyze(args.output_root, "coarse" if args.command == "analyze-coarse" else "refine")
            print(result["status"])
            return 1 if result["status"] == "l_chroma_inverse_insufficient" else 0
        print(f"{args.command} complete")
        return 0
    except Exception as exc:
        failure = args.output_root / f"{args.command}_failure.json"
        if args.command != "plan-coarse" and args.output_root.is_dir() and not failure.exists():
            _write(failure, {"schema": f"{PREFIX}_failure/v1", "status": "failed", "command": args.command,
                             "error_type": type(exc).__name__})
        print(f"fallback search aborted: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
