"""Frozen request and measurement definitions for the Emission color-branch pilot."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.methods.colorpeel_ice import renderer_color_calibration as calibration
from src.methods.colorpeel_ice import natural_image_target_selection as selection


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_RELPATH = "experiments/natural_image_subject_color_pilot/configs/d1_emission_color_branch_pilot_protocol_v1.json"
SHAPE_ORDINAL = {"cube": 0, "sphere": 1, "cylinder": 2}


class EmissionColorBranchError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EmissionColorBranchError(message)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def load_protocol() -> dict[str, Any]:
    try:
        value = json.loads((REPO_ROOT / PROTOCOL_RELPATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EmissionColorBranchError(f"Cannot load emission protocol: {exc}") from exc
    require(value.get("schema") == "emission_color_branch_pilot_protocol/v1", "Protocol schema differs")
    require(value.get("protocol_id") == "d1_emission_color_branch_pilot_protocol_v1", "Protocol id differs")
    require(value.get("frozen_inputs", {}).get("selection_canonical_sha256") == "33d9b2d5fb404786d63ac23b161c7578a08442ecd24691ca4c221721e88ee0a6", "Selection identity differs")
    require(value.get("frozen_inputs", {}).get("renderer_protocol_canonical_sha256") == "476d01c5a383ead7d1ceec5c276bb26d809cf2fabf51cd81e0c49695cc4aa48e", "Renderer protocol identity differs")
    require(value.get("renderer_contract", {}).get("material") == "built_in_blender_emission", "Material differs")
    require(value.get("renderer_contract", {}).get("emission_strength") == 1.0, "Emission strength differs")
    pilot = value.get("pilot", {})
    require(pilot.get("shapes") == ["cube", "sphere", "cylinder"] and pilot.get("view_indices") == [0, 8, 16] and pilot.get("request_count") == 18, "Pilot matrix differs")
    require(pilot.get("median_e_ch_max") == 3.0 and pilot.get("p90_e_ch_max") == 5.0 and pilot.get("overall_render_p90_e_ch_max") == 5.0, "Pilot gate differs")
    require(value.get("approval_state") == {"pilot_approved": True, "full_dataset_approved": False, "training_approved": False}, "Approval state differs")
    development = {record["source"]["stable_id"]: record for record in selection.pipeline_development_records()}
    targets = value.get("targets")
    require(isinstance(targets, list) and len(targets) == 2, "Target count differs")
    for target in targets:
        source = development.get(target.get("stable_id"))
        require(source is not None and target.get("a") == source["target"]["a"] and target.get("b") == source["target"]["b"],
                "Target is not the frozen natural-image color")
    return value


def _slug(stable_id: str) -> str:
    return stable_id.replace("D1GT:", "D1GT_").replace("/", "_").replace(".png", "")


def target_lch(a: float, b: float) -> tuple[float, float]:
    chroma = math.hypot(a, b)
    require(math.isfinite(chroma) and chroma > 0.0, "Target chroma differs")
    return chroma, math.degrees(math.atan2(b, a))


def pilot_requests() -> list[dict[str, Any]]:
    protocol = load_protocol()
    rows = []
    for target_index, target in enumerate(protocol["targets"]):
        stable_id, a, b = target.get("stable_id"), target.get("a"), target.get("b")
        require(isinstance(stable_id, str) and type(a) in {int, float} and type(b) in {int, float}, "Target differs")
        chroma, hue = target_lch(float(a), float(b))
        linear = calibration.assert_linear_rgb_in_gamut(calibration.lab_d65_to_linear_srgb(50.0, float(a), float(b)))
        for shape in protocol["pilot"]["shapes"]:
            for view in protocol["pilot"]["view_indices"]:
                rows.append({"request_id": f"emission__{_slug(stable_id)}__{shape}__v{view:02d}", "stable_id": stable_id,
                             "target_index": target_index, "target_a": float(a), "target_b": float(b), "target_C": chroma,
                             "target_h_degrees": hue, "L_input": 50.0, "linear_rgb": linear, "socket_rgba": linear + [1.0],
                             "shape": shape, "view_index": view, "material": "Emission", "emission_strength": 1.0,
                             "render_seed": 910000 + target_index * 100 + SHAPE_ORDINAL[shape] * 20 + view})
    require(len(rows) == 18 and len({row["request_id"] for row in rows}) == 18, "Pilot request identity differs")
    return rows


def nearest_rank_p90(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    require(ordered and all(math.isfinite(value) for value in ordered), "P90 inputs differ")
    return ordered[math.ceil(0.9 * len(ordered)) - 1]


def summarize_measurements(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {row["request_id"]: row for row in pilot_requests()}
    require(len(rows) == 18 and {row.get("request_id") for row in rows} == set(expected), "Measurement coverage differs")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        request = expected[row["request_id"]]
        require(all(row.get(key) == value for key, value in request.items()), "Measurement request payload differs")
        require(all(row.get(key) is True for key in ("object_ratio_ok", "interior_ok", "eligible_ok")), "Mask QC differs")
        require(int(row.get("outside_mask_changed_pixel_count", -1)) == 0 and int(row.get("rgb_clipping_pixel_count", -1)) == 0, "Render safety QC differs")
        e_ch = float(row.get("e_ch", float("nan")))
        require(math.isfinite(e_ch), "Measurement e_ch differs")
        groups.setdefault(str(row["stable_id"]), []).append(row)
    targets = []
    for stable_id, group in sorted(groups.items()):
        require(len(group) == 9, "Target coverage differs")
        errors = [float(row["e_ch"]) for row in group]
        median, p90 = float(sorted(errors)[4]), nearest_rank_p90(errors)
        targets.append({"stable_id": stable_id, "render_count": 9, "median_e_ch": median, "p90_e_ch": p90,
                        "passes": median <= 3.0 and p90 <= 5.0})
    overall_p90 = nearest_rank_p90([float(row["e_ch"]) for row in rows])
    return {"schema": "d1_emission_color_branch_pilot_analysis/v1", "request_count": 18, "targets": targets,
            "overall_render_p90_e_ch": overall_p90, "overall_pass": all(row["passes"] for row in targets) and overall_p90 <= 5.0}
