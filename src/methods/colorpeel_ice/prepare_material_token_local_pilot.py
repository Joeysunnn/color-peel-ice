"""Plan isolated controlled assets for the material token-local pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from src.methods.colorpeel_ice.multiview_render_contract import EXPECTED_PROFILE_V5, canonical_sha256


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "material_token_local_pilot_v1"
DEFAULT_PROTOCOL = EXPERIMENT_ROOT / "protocols" / "material_token_local_pilot_v1.json"
SHAPES = ("cube", "sphere", "cylinder")
FULL_COLORS = ("red", "blue", "green", "yellow")
LIGHTING = ("soft_front", "side_directional", "warm_top")
VIEWPOINTS = ("frontish", "oblique_45")
RENDERER_FIELDS = ("camera", "light", "background", "scene_json", "image", "mask", "background_mask")


class ProtocolError(ValueError):
    """Raised when a pilot contract or realization is inconsistent."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read JSON {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ProtocolError(f"Expected object at {path}:{line_number}")
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read JSONL {path}: {exc}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    if protocol.get("schema") != "material_token_local_pilot/v1":
        raise ProtocolError("Unexpected protocol schema")
    if protocol.get("protocol_id") != "material_token_local_pilot_v1":
        raise ProtocolError("Unexpected protocol ID")
    target = protocol.get("target_material")
    if target != {
        "name": "metal", "token": "<M*>", "initializer_token": "metal", "renderer_asset": "MyMetal",
        "scope": "controlled metallic surface appearance; no texture token and no natural-image BRDF claim",
    }:
        raise ProtocolError("Target material contract changed")
    if protocol.get("shapes") != list(SHAPES) or protocol.get("colors") != {
        "red": [173, 35, 35], "blue": [42, 75, 215],
        "green": [29, 105, 20], "yellow": [255, 238, 51],
    }:
        raise ProtocolError("Shape or color contract changed")
    if protocol.get("preview_colors") != ["red", "blue"]:
        raise ProtocolError("Preview color contract changed")
    if protocol.get("lighting_conditions") != list(LIGHTING) or protocol.get("viewpoints") != list(VIEWPOINTS):
        raise ProtocolError("Lighting or viewpoint contract changed")
    if protocol.get("rendering") != {
        "preview_image_count": 36, "full_image_count": 72,
        "render_seed_formula": "730000 + cell_index * 10 + view_index",
        "training_authorization": "blocked_pending_preview_human_review",
    }:
        raise ProtocolError("Rendering and training gate contract changed")
    if protocol.get("training") != {
        "adaptation_mode": "token_local_kv", "modifier_token": "<M*>",
        "initializer_token": "metal", "caption": "a photo of an object made of <M*>",
        "forbidden": ["subject_token", "color_token", "joint_training", "AlignIT",
                      "texture_token", "additional_semantic_loss"],
    }:
        raise ProtocolError("Training contract changed")
    return protocol


def build_render_requests(protocol: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    validate_protocol(protocol)
    if mode not in {"preview", "full"}:
        raise ProtocolError("mode must be preview or full")
    colors = tuple(protocol["preview_colors"] if mode == "preview" else FULL_COLORS)
    rgb_by_color = protocol["colors"]
    rows: list[dict[str, Any]] = []
    for shape_index, shape in enumerate(SHAPES):
        for color in colors:
            color_index = FULL_COLORS.index(color)
            cell_index = shape_index * len(FULL_COLORS) + color_index
            for lighting_index, lighting_condition in enumerate(LIGHTING):
                for viewpoint_index, viewpoint in enumerate(VIEWPOINTS):
                    view_index = lighting_index * len(VIEWPOINTS) + viewpoint_index
                    rows.append({
                        "cell_id": f"{shape}_{color}_metal",
                        "cell_index": cell_index,
                        "shape": shape,
                        "color": color,
                        "material": "metal",
                        "material_token": "<M*>",
                        "nominal_rgb": rgb_by_color[color],
                        "view_index": view_index,
                        "lighting_condition": lighting_condition,
                        "viewpoint": viewpoint,
                        "dataset_split": mode,
                        "render_seed": 730000 + cell_index * 10 + view_index,
                        "renderer_profile_id": EXPECTED_PROFILE_V5["profile_id"],
                        "renderer_profile_sha256": canonical_sha256(EXPECTED_PROFILE_V5),
                        **{field: None for field in RENDERER_FIELDS},
                    })
    expected = 36 if mode == "preview" else 72
    if len(rows) != expected or len({(row["cell_id"], row["view_index"]) for row in rows}) != expected:
        raise AssertionError("Pilot render grid is incomplete")
    return rows


def preview_checklist() -> str:
    return """# Material pilot preview review\n\nReview all 36 images before training. Pass only if each condition is visibly present and no renderer issue is observed.\n\n- Native metal appearance is coherent across cube, sphere, and cylinder.\n- Red and blue base colors remain visibly distinct under every lighting condition.\n- `soft_front`, `side_directional`, and `warm_top` produce distinguishable illumination; the last is visibly warm.\n- `frontish` and `oblique_45` differ in viewpoint while keeping one centered object.\n- There are no empty frames, mask failures, duplicate geometry, fused objects, or background contamination.\n\nA pass approves only full controlled rendering and staging. It does not authorize training until the reviewer explicitly changes the authorization state in a new run record.\n"""


def validate_preview_approval(protocol: dict[str, Any], preview_root: Path, approval_path: Path) -> str:
    requests = build_render_requests(protocol, "preview")
    contract = read_json(preview_root / "render_contract.json")
    if contract.get("profile_id") != EXPECTED_PROFILE_V5["profile_id"] or contract.get("requests_sha256") != canonical_sha256(requests):
        raise ProtocolError("Preview render contract does not match this pilot")
    manifest_path = preview_root / "renderer_realization.jsonl"
    realized = read_jsonl(manifest_path)
    expected = {(row["cell_id"], row["view_index"]): row for row in requests}
    if len(realized) != 36 or {(row.get("cell_id"), row.get("view_index")) for row in realized} != set(expected):
        raise ProtocolError("Preview must contain all 36 realized images")
    for row in realized:
        request = expected[(row["cell_id"], row["view_index"])]
        if any(row.get(key) != value for key, value in request.items() if key not in RENDERER_FIELDS):
            raise ProtocolError("Preview realization differs from the locked request")
        if row.get("render_contract_sha256") != canonical_sha256(contract):
            raise ProtocolError("Preview realization has a different render contract")
        for field in ("image", "mask", "background_mask", "scene_json"):
            path = (preview_root / row.get(field, "")).resolve()
            if not path.is_relative_to(preview_root.resolve()) or not path.is_file():
                raise ProtocolError(f"Missing preview artifact: {field}")
            if row.get("artifact_sha256", {}).get(field) != sha256(path):
                raise ProtocolError(f"Preview artifact hash differs: {field}")
    approval = read_json(approval_path)
    if (approval.get("verdict") != "pass" or not approval.get("reviewer") or not approval.get("reviewed_at")
            or approval.get("renderer_realization_sha256") != sha256(manifest_path)):
        raise ProtocolError("Human preview approval is absent or does not match rendered images")
    return sha256(approval_path)


def plan(protocol: dict[str, Any], mode: str, output_dir: Path,
         preview_root: Path | None = None, approval_path: Path | None = None) -> dict[str, Any]:
    approval_sha256 = None
    if mode == "full":
        if preview_root is None or approval_path is None:
            raise ProtocolError("Full grid requires the rendered preview and its human approval record")
        approval_sha256 = validate_preview_approval(protocol, preview_root, approval_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ProtocolError(f"Output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_render_requests(protocol, mode)
    write_jsonl(output_dir / "render_requests.jsonl", rows)
    if mode == "preview":
        (output_dir / "preview_review_checklist.md").write_text(preview_checklist(), encoding="utf-8")
        write_json(output_dir / "review_record_template.json", {
            "verdict": "pending", "reviewer": "", "reviewed_at": "",
            "renderer_realization_sha256": None,
        })
    status = {
        "status": "planned_pending_human_preview" if mode == "preview" else "planned_pending_preview_approval",
        "mode": mode,
        "request_count": len(rows),
        "renderer_profile_id": EXPECTED_PROFILE_V5["profile_id"],
        "renderer_profile_sha256": canonical_sha256(EXPECTED_PROFILE_V5),
        "training_authorization": protocol["rendering"]["training_authorization"],
        "preview_approval_sha256": approval_sha256,
    }
    write_json(output_dir / "protocol_status.json", status)
    return status


def _copy_asset(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise ProtocolError(f"Training asset already exists: {destination}")
    shutil.copy2(source, destination)


def stage_training_assets(protocol: dict[str, Any], render_root: Path, output_dir: Path,
                          preview_root: Path, approval_path: Path) -> dict[str, Any]:
    from src.train.instance_mask_utils import load_latent_instance_mask

    validate_protocol(protocol)
    approval_sha256 = validate_preview_approval(protocol, preview_root, approval_path)
    requests = build_render_requests(protocol, "full")
    contract = read_json(render_root / "render_contract.json")
    if contract.get("profile_id") != EXPECTED_PROFILE_V5["profile_id"] or contract.get("requests_sha256") != canonical_sha256(requests):
        raise ProtocolError("Full render contract does not match this pilot")
    realized = read_jsonl(render_root / "renderer_realization.jsonl")
    by_key = {(row.get("cell_id"), row.get("view_index")): row for row in realized}
    expected_keys = {(row["cell_id"], row["view_index"]) for row in requests}
    if len(realized) != 72 or set(by_key) != expected_keys:
        raise ProtocolError("Rendered full grid does not exactly match the locked requests")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ProtocolError(f"Output directory must be new or empty: {output_dir}")
    image_dir, mask_dir = output_dir / "images", output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir()
    staged: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        row = by_key[(request["cell_id"], request["view_index"])]
        for field, value in request.items():
            if field in RENDERER_FIELDS:
                continue
            if row.get(field) != value:
                raise ProtocolError(f"Realized request changed {field}: {request['cell_id']} view {request['view_index']}")
        if row.get("render_contract_sha256") != canonical_sha256(contract):
            raise ProtocolError("Full realization has a different render contract")
        for field in ("image", "mask", "background_mask", "scene_json"):
            source = (render_root / row.get(field, "")).resolve()
            if not source.is_relative_to(render_root.resolve()) or not source.is_file():
                raise ProtocolError(f"Missing rendered {field}: {source}")
            if row.get("artifact_sha256", {}).get(field) != sha256(source):
                raise ProtocolError(f"Rendered {field} hash differs from the ledger")
        stem = f"{index:03d}_{request['cell_id']}_view_{request['view_index']:02d}"
        record = {"request": request}
        for field, destination in (("image", image_dir / f"{stem}.jpg"), ("mask", mask_dir / f"{stem}.png")):
            source = (render_root / row.get(field, "")).resolve()
            source_hash = sha256(source)
            if field == "mask":
                load_latent_instance_mask(source, 512, 64)
            _copy_asset(source, destination)
            if sha256(destination) != source_hash:
                raise ProtocolError(f"Staged {field} hash differs from the rendered source")
            record[f"source_{field}"] = str(source)
            record[f"{field}_sha256"] = source_hash
            record[f"staged_{field}"] = str(destination)
        staged.append(record)
    concepts = [{"instance_prompt": [protocol["training"]["caption"]],
                 "instance_data_dir": str(image_dir.resolve()), "instance_mask_dir": str(mask_dir.resolve())}]
    write_json(output_dir / "concepts.json", concepts)
    write_jsonl(output_dir / "training_assets_manifest.jsonl", staged)
    write_json(output_dir / "staging_provenance.json", {
        "protocol_sha256": canonical_sha256(protocol),
        "preview_approval_sha256": approval_sha256,
        "render_contract_sha256": canonical_sha256(contract),
        "renderer_realization_sha256": sha256(render_root / "renderer_realization.jsonl"),
        "concepts_sha256": sha256(output_dir / "concepts.json"),
        "training_assets_manifest_sha256": sha256(output_dir / "training_assets_manifest.jsonl"),
    })
    status = {"status": "staged_pending_separate_training_authorization", "image_count": len(staged), "concepts": str((output_dir / "concepts.json").resolve())}
    write_json(output_dir / "staging_status.json", status)
    return status


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--mode", choices=("preview", "full"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-training-from", type=Path)
    parser.add_argument("--preview-root", type=Path)
    parser.add_argument("--review-record", type=Path)
    args = parser.parse_args(argv)
    if (args.mode is None) == (args.stage_training_from is None):
        parser.error("provide exactly one of --mode or --stage-training-from")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = validate_protocol(read_json(args.protocol))
    if args.mode:
        result = plan(protocol, args.mode, args.output_dir, args.preview_root, args.review_record)
    else:
        if args.preview_root is None or args.review_record is None:
            raise ProtocolError("Staging requires the rendered preview and its human approval record")
        result = stage_training_assets(protocol, args.stage_training_from, args.output_dir,
                                       args.preview_root, args.review_record)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
