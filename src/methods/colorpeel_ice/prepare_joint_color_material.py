"""Plan and stage matched color/material counterfactuals without changing old assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from src.methods.colorpeel_ice.multiview_render_contract import (
    EXPECTED_PROFILE_CM_JOINT, EXPECTED_PROFILE_V5_GROUND_REFLECTION, canonical_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = REPO_ROOT / "experiments/color_material_composition_v1/protocols/joint_color_material_data_v1.json"
RENDERER_FIELDS = ("camera", "light", "background", "scene_json", "image", "mask", "background_mask")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_protocol(path: Path):
    protocol = read_json(path)
    source = REPO_ROOT / protocol["selected_scene_profile"]
    if (protocol.get("schema") != "joint_color_material_data/v1"
            or canonical_sha256(read_json(source)) != protocol["selected_scene_profile_sha256"]
            or read_json(source) != EXPECTED_PROFILE_V5_GROUND_REFLECTION
            or protocol.get("shapes") != ["cube", "sphere", "cylinder"]
            or protocol.get("materials") != ["metal", "rubber"]
            or protocol.get("lighting_conditions") != ["soft_front", "side_directional", "warm_top"]
            or protocol.get("viewpoints") != ["frontish", "oblique_45"]
            or protocol.get("source_emission_target") != "D1GT:81/130.png"
            or protocol.get("source_emission_lab") != {"L": 50.0, "a": 63.21805661727065, "b": 54.85769274644318}
            or protocol.get("render_seed_formula") != "740000 + shape_index * 10 + view_index"
            or protocol.get("full_image_count") != 72 or protocol.get("preview_prefix_count") != 12
            or protocol.get("training") != {"architecture": "shared_custom_diffusion_kv",
                                            "modifier_token": "<C*>+<M*>", "initializer_token": "orange+metal",
                                            "use_object_masks": True, "caa_weights": [0.0, 0.2]}
            or protocol.get("evaluation_holdout_noun") != "mug"):
        raise ValueError("joint C/M data protocol differs")
    expected_colors = {
        "orange": {"nominal_rgb": [223, 54, 25], "material_socket_rgba": [0.738731741987261, 0.2, 0.009661907129827573, 1.0]},
        "blue": {"nominal_rgb": [42, 75, 215], "material_socket_rgba": [42 / 255, 75 / 255, 215 / 255, 1.0]},
    }
    if protocol.get("colors") != expected_colors:
        raise ValueError("joint C/M color socket differs")
    return protocol


def build_requests(protocol: dict):
    rows = []
    for view_index in range(6):
        for shape_index, shape in enumerate(protocol["shapes"]):
            for color_index, color in enumerate(("orange", "blue")):
                for material_index, material in enumerate(protocol["materials"]):
                    color_source = protocol["colors"][color]
                    rows.append({
                        "cell_id": f"{shape}_{color}_{material}",
                        "cell_index": shape_index * 4 + color_index * 2 + material_index,
                        "shape": shape, "color": color, "material": material,
                        "color_token": "<C*>" if color == "orange" else None,
                        "material_token": "<M*>" if material == "metal" else None,
                        "nominal_rgb": color_source["nominal_rgb"],
                        "material_socket_rgba": color_source["material_socket_rgba"],
                        "view_index": view_index,
                        "lighting_condition": protocol["lighting_conditions"][view_index // 2],
                        "viewpoint": protocol["viewpoints"][view_index % 2],
                        "dataset_split": "full",
                        "render_seed": 740000 + shape_index * 10 + view_index,
                        "renderer_profile_id": EXPECTED_PROFILE_CM_JOINT["profile_id"],
                        "renderer_profile_sha256": canonical_sha256(EXPECTED_PROFILE_CM_JOINT),
                        **{field: None for field in RENDERER_FIELDS},
                    })
    if len(rows) != 72:
        raise AssertionError("joint C/M request grid differs")
    return rows


def plan(protocol_path: Path, output_dir: Path):
    protocol = validate_protocol(protocol_path)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    rows = build_requests(protocol)
    write_json(output_dir / "render_profile.json", EXPECTED_PROFILE_CM_JOINT)
    write_jsonl(output_dir / "render_requests.jsonl", rows)
    write_json(output_dir / "plan_provenance.json", {
        "protocol_sha256": sha256(protocol_path),
        "profile_sha256": canonical_sha256(EXPECTED_PROFILE_CM_JOINT),
        "requests_sha256": canonical_sha256(rows),
        "preview_prefix_count": 12, "full_image_count": 72,
    })
    write_json(output_dir / "preview_review_template.json", {
        "verdict": "pending", "reviewer": "", "reviewed_at": "",
        "renderer_realization_sha256": None,
    })
    (output_dir / "preview_review_checklist.md").write_text(
        "# Joint color/material preview\n\n"
        "Review all 12 soft-front images: cube, sphere, cylinder crossed with "
        "orange/blue and metal/rubber. The orange metal should remain visibly "
        "orange rather than gold; the same shape and view should match across "
        "the four cells. Check the expected ground reflection, object masks, "
        "and absence of extra geometry. Record a pass only for this exact "
        "renderer realization before the full render and training.\n",
        encoding="utf-8",
    )
    return rows


def caption(shape: str, color: str, material: str):
    c = "<C*>" if color == "orange" else "blue"
    m = "<M*>" if material == "metal" else "rubber"
    return f"a photo of a {shape} with {c} color and {m} material"


def validate_preview_review(plan_dir: Path, preview_root: Path, review_record: Path):
    requests = read_jsonl(plan_dir / "render_requests.jsonl")
    preview = read_json(preview_root / "renderer_status.json")
    contract = read_json(preview_root / "render_contract.json")
    if (preview.get("status") != "partial_smoke" or preview.get("completed_count") != 12
            or preview.get("selected_count") != 12 or preview.get("request_count") != 72
            or contract.get("profile_sha256") != canonical_sha256(EXPECTED_PROFILE_CM_JOINT)
            or contract.get("requests_sha256") != canonical_sha256(requests)):
        raise ValueError("joint C/M preview does not match the 12-image plan")
    realized_path = preview_root / "renderer_realization.jsonl"
    realized = read_jsonl(realized_path)
    if len(realized) != 12:
        raise ValueError("joint C/M preview must contain exactly 12 images")
    for expected, row in zip(requests[:12], realized):
        if any(row.get(key) != value for key, value in expected.items() if key not in RENDERER_FIELDS):
            raise ValueError("joint C/M preview request differs")
        if row.get("render_contract_sha256") != canonical_sha256(contract):
            raise ValueError("joint C/M preview contract differs")
        for field in ("image", "mask", "background_mask", "scene_json"):
            source = (preview_root / row[field]).resolve()
            if (not source.is_relative_to(preview_root.resolve())
                    or sha256(source) != row["artifact_sha256"][field]):
                raise ValueError(f"joint C/M preview {field} differs")
    review = read_json(review_record)
    if (review.get("verdict") != "pass" or not review.get("reviewer")
            or not review.get("reviewed_at")
            or review.get("renderer_realization_sha256") != sha256(realized_path)):
        raise ValueError("joint C/M preview has no matching pass review")
    return sha256(review_record)


def stage(protocol_path: Path, plan_dir: Path, render_root: Path, output_dir: Path,
          preview_root: Path, review_record: Path):
    from src.train.instance_mask_utils import load_latent_instance_mask

    validate_protocol(protocol_path)
    requests = build_requests(read_json(protocol_path))
    if (read_json(plan_dir / "plan_provenance.json")["protocol_sha256"] != sha256(protocol_path)
            or read_jsonl(plan_dir / "render_requests.jsonl") != requests):
        raise ValueError("render plan differs")
    review_sha256 = validate_preview_review(plan_dir, preview_root, review_record)
    contract = read_json(render_root / "render_contract.json")
    if (contract["profile_sha256"] != canonical_sha256(EXPECTED_PROFILE_CM_JOINT)
            or contract["requests_sha256"] != canonical_sha256(requests)
            or read_json(render_root / "renderer_status.json")["status"] != "succeeded"):
        raise ValueError("full render is incomplete or differs")
    realized = read_jsonl(render_root / "renderer_realization.jsonl")
    by_key = {(row["cell_id"], row["view_index"]): row for row in realized}
    if len(realized) != 72 or len(by_key) != 72:
        raise ValueError("expected 72 unique realized images")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    concepts = []
    staged = []
    for shape in ("cube", "sphere", "cylinder"):
        for color in ("orange", "blue"):
            for material in ("metal", "rubber"):
                cell_id = f"{shape}_{color}_{material}"
                image_dir = output_dir / cell_id / "images"
                mask_dir = output_dir / cell_id / "masks"
                image_dir.mkdir(parents=True)
                mask_dir.mkdir()
                for view in range(6):
                    request = next(row for row in requests if row["cell_id"] == cell_id and row["view_index"] == view)
                    row = by_key[cell_id, view]
                    if any(row.get(key) != value for key, value in request.items() if key not in RENDERER_FIELDS):
                        raise ValueError(f"realized request differs: {cell_id}/{view}")
                    if row.get("render_contract_sha256") != canonical_sha256(contract):
                        raise ValueError(f"realized contract differs: {cell_id}/{view}")
                    for field, destination in (("image", image_dir / f"view_{view:02d}.jpg"),
                                               ("mask", mask_dir / f"view_{view:02d}.png")):
                        source = (render_root / row[field]).resolve()
                        if not source.is_relative_to(render_root.resolve()) or sha256(source) != row["artifact_sha256"][field]:
                            raise ValueError(f"realized artifact differs: {cell_id}/{view}/{field}")
                        if field == "mask":
                            load_latent_instance_mask(source, 512, 64)
                        shutil.copy2(source, destination)
                        if sha256(destination) != row["artifact_sha256"][field]:
                            raise ValueError("staged artifact hash differs")
                        staged.append({"cell_id": cell_id, "view_index": view, "field": field,
                                       "path": str(destination), "sha256": sha256(destination)})
                concepts.append({"instance_prompt": [caption(shape, color, material)],
                                 "instance_data_dir": str(image_dir.resolve()),
                                 "instance_mask_dir": str(mask_dir.resolve())})
    write_json(output_dir / "concepts.json", concepts)
    write_jsonl(output_dir / "training_assets_manifest.jsonl", staged)
    write_json(output_dir / "staging_provenance.json", {
        "protocol_sha256": sha256(protocol_path), "render_contract_sha256": canonical_sha256(contract),
        "renderer_realization_sha256": sha256(render_root / "renderer_realization.jsonl"),
        "preview_review_sha256": review_sha256,
        "concepts_sha256": sha256(output_dir / "concepts.json"),
        "assets_sha256": sha256(output_dir / "training_assets_manifest.jsonl"),
        "image_count": 72,
    })
    return concepts


def validate_staging(protocol_path: Path, plan_dir: Path, preview_root: Path,
                     review_record: Path, staging_root: Path):
    protocol = validate_protocol(protocol_path)
    requests = build_requests(protocol)
    if (read_json(plan_dir / "plan_provenance.json")["protocol_sha256"] != sha256(protocol_path)
            or read_jsonl(plan_dir / "render_requests.jsonl") != requests):
        raise ValueError("joint C/M training plan differs")
    review_sha256 = validate_preview_review(plan_dir, preview_root, review_record)
    provenance = read_json(staging_root / "staging_provenance.json")
    concepts_path = staging_root / "concepts.json"
    assets_path = staging_root / "training_assets_manifest.jsonl"
    if (provenance.get("protocol_sha256") != sha256(protocol_path)
            or provenance.get("preview_review_sha256") != review_sha256
            or provenance.get("concepts_sha256") != sha256(concepts_path)
            or provenance.get("assets_sha256") != sha256(assets_path)
            or provenance.get("image_count") != 72):
        raise ValueError("joint C/M staging provenance differs")
    expected_concepts = []
    expected_assets = set()
    for shape in protocol["shapes"]:
        for color in ("orange", "blue"):
            for material in protocol["materials"]:
                cell_id = f"{shape}_{color}_{material}"
                root = staging_root.resolve() / cell_id
                expected_concepts.append({
                    "instance_prompt": [caption(shape, color, material)],
                    "instance_data_dir": str(root / "images"),
                    "instance_mask_dir": str(root / "masks"),
                })
                for view in range(6):
                    for field in ("image", "mask"):
                        expected_assets.add((cell_id, view, field))
    if read_json(concepts_path) != expected_concepts:
        raise ValueError("joint C/M training concepts differ")
    assets = read_jsonl(assets_path)
    if len(assets) != 144 or {(row["cell_id"], row["view_index"], row["field"])
                                  for row in assets} != expected_assets:
        raise ValueError("joint C/M training assets differ")
    for row in assets:
        subdir, extension = (("images", ".jpg") if row["field"] == "image" else ("masks", ".png"))
        expected = staging_root.resolve() / row["cell_id"] / subdir / f"view_{row['view_index']:02d}{extension}"
        if Path(row["path"]).resolve() != expected or sha256(expected) != row["sha256"]:
            raise ValueError("joint C/M staged artifact differs")
    return concepts_path, assets_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path)
    parser.add_argument("--render-root", type=Path)
    parser.add_argument("--preview-root", type=Path)
    parser.add_argument("--review-record", type=Path)
    args = parser.parse_args()
    if (args.plan_dir is None) != (args.render_root is None):
        parser.error("--plan-dir and --render-root must be provided together for staging")
    if args.plan_dir and (args.preview_root is None or args.review_record is None):
        parser.error("staging requires --preview-root and --review-record")
    result = (stage(args.protocol, args.plan_dir, args.render_root, args.output_dir,
                    args.preview_root, args.review_record) if args.plan_dir
              else plan(args.protocol, args.output_dir))
    print(f"{len(result)} {'concepts' if args.plan_dir else 'render requests'}: {args.output_dir}")


if __name__ == "__main__":
    main()
