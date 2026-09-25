"""Prepare fresh metal renders and unpaired emission-C / native-metal-M training."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.methods.colorpeel_ice.multiview_render_contract import (
    EXPECTED_PROFILE_M_UNPAIRED, EXPECTED_PROFILE_V5_GROUND_REFLECTION, canonical_sha256,
)
from src.methods.colorpeel_ice.prepare_joint_color_material import read_json, read_jsonl, sha256, write_json, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = REPO_ROOT / "experiments/color_material_composition_v1/protocols/unpaired_emission_material_v1.json"
RENDER_FIELDS = ("camera", "light", "background", "scene_json", "image", "mask", "background_mask")
SHAPES = ("cube", "sphere", "cylinder")
COLORS = ("red", "blue", "green", "yellow")
LIGHTS = ("soft_front", "side_directional", "warm_top")
VIEWS = ("frontish", "oblique_45")


def protocol(path: Path):
    value = read_json(path)
    color_source = REPO_ROOT / value["color_source_protocol"]
    material_scene = REPO_ROOT / value["material_scene_profile"]
    if (value.get("schema") != "unpaired_emission_material/v1"
            or value.get("color_source_protocol") != "experiments/natural_image_subject_color_pilot/configs/d1_emission_color_transfer_orange_protocol_v1.json"
            or sha256(color_source) != value["color_source_protocol_sha256"]
            or value.get("color_source_root_relative_to_COLORPEEL_RUN_ROOT") != "natural_image_subject_color_pilot/emission_color_branch_4c9e55d_pilot"
            or read_json(material_scene) != EXPECTED_PROFILE_V5_GROUND_REFLECTION
            or canonical_sha256(read_json(material_scene)) != value["material_scene_profile_canonical_sha256"]
            or value.get("material_shapes") != list(SHAPES)
            or value.get("material_colors") != {"red": [173, 35, 35], "blue": [42, 75, 215],
                                                "green": [29, 105, 20], "yellow": [255, 238, 51]}
            or value.get("material_lights") != list(LIGHTS)
            or value.get("material_views") != list(VIEWS)
            or value.get("material_seed_formula") != "760000 + cell_index * 10 + view_index"
            or value.get("material_image_count") != 72 or value.get("preview_prefix_count") != 12
            or value.get("training") != {
                "color_token": "<C*>", "material_token": "<M*>", "shared_kv": True,
                "caa_weight": 0.0, "color_steps": 100, "material_steps": 5000,
                "schedule": "50 material rows then 1 emission color row, repeated 100 times",
                "strict_unpaired_modifier_updates": True,
            }):
        raise ValueError("unpaired emission/material protocol differs")
    return value


def requests(value: dict):
    rows = []
    for view_index in range(6):
        for shape_index, shape in enumerate(SHAPES):
            for color_index, color in enumerate(COLORS):
                cell_index = shape_index * 4 + color_index
                rows.append({
                    "cell_id": f"{shape}_{color}_metal", "cell_index": cell_index,
                    "shape": shape, "color": color, "material": "metal", "material_token": "<M*>",
                    "nominal_rgb": value["material_colors"][color],
                    "view_index": view_index, "lighting_condition": LIGHTS[view_index // 2],
                    "viewpoint": VIEWS[view_index % 2], "dataset_split": "full",
                    "render_seed": 760000 + cell_index * 10 + view_index,
                    "renderer_profile_id": EXPECTED_PROFILE_M_UNPAIRED["profile_id"],
                    "renderer_profile_sha256": canonical_sha256(EXPECTED_PROFILE_M_UNPAIRED),
                    **{field: None for field in RENDER_FIELDS},
                })
    assert len(rows) == 72 and len({(row["cell_id"], row["view_index"]) for row in rows}) == 72
    return rows


def plan(protocol_path: Path, output_dir: Path):
    value = protocol(protocol_path)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    rows = requests(value)
    write_json(output_dir / "render_profile.json", EXPECTED_PROFILE_M_UNPAIRED)
    write_jsonl(output_dir / "render_requests.jsonl", rows)
    write_json(output_dir / "plan_provenance.json", {
        "protocol_sha256": sha256(protocol_path), "requests_sha256": canonical_sha256(rows),
        "profile_sha256": canonical_sha256(EXPECTED_PROFILE_M_UNPAIRED),
    })
    write_json(output_dir / "preview_review_template.json", {
        "verdict": "pending", "reviewer": "", "reviewed_at": "",
        "renderer_realization_sha256": None,
    })
    (output_dir / "preview_review_checklist.md").write_text(
        "# Material-only preview\n\nReview the first 12 soft-front images: three shapes and four literal colors. "
        "Check the selected ground reflection, recognizable metal, color variation, one object per image, "
        "and correct nonempty masks. A pass record must match the preview realization SHA-256.\n",
        encoding="utf-8",
    )
    return rows


def verified_render(root: Path, expected: list[dict], expected_count: int):
    contract = read_json(root / "render_contract.json")
    status = read_json(root / "renderer_status.json")
    rows = read_jsonl(root / "renderer_realization.jsonl")
    if (contract.get("profile_sha256") != canonical_sha256(EXPECTED_PROFILE_M_UNPAIRED)
            or contract.get("requests_sha256") != canonical_sha256(expected)
            or status.get("completed_count") != expected_count
            or status.get("status") != ("partial_smoke" if expected_count == 12 else "succeeded")
            or len(rows) != expected_count):
        raise ValueError("material render status or contract differs")
    for request, row in zip(expected[:expected_count], rows):
        if (any(row.get(key) != value for key, value in request.items() if key not in RENDER_FIELDS)
                or row.get("render_contract_sha256") != canonical_sha256(contract)):
            raise ValueError("material realization differs from its request")
        for field in ("image", "mask", "background_mask", "scene_json"):
            artifact = (root / row[field]).resolve()
            if (not artifact.is_relative_to(root.resolve())
                    or sha256(artifact) != row["artifact_sha256"][field]):
                raise ValueError(f"material {field} hash differs")
    return rows


def verified_review(preview_root: Path, review_record: Path, expected: list[dict]):
    verified_render(preview_root, expected, 12)
    review = read_json(review_record)
    if (review.get("verdict") != "pass" or not review.get("reviewer")
            or not review.get("reviewed_at")
            or review.get("renderer_realization_sha256") != sha256(preview_root / "renderer_realization.jsonl")):
        raise ValueError("material preview has no matching pass review")
    return sha256(review_record)


def schedule(color: list[dict], material: list[dict]):
    if (len(color) != 9 or len(material) != 72
            or any("<C*>" not in row["instance_prompt"][0] or "<M*>" in row["instance_prompt"][0] for row in color)
            or any("<M*>" not in row["instance_prompt"][0] or "<C*>" in row["instance_prompt"][0] for row in material)):
        raise ValueError("unpaired color/material physical rows differ")
    rows = []
    for cycle in range(100):
        rows.extend(material[(cycle * 50 + index) % 72] for index in range(50))
        rows.append(color[cycle % 9])
    assert len(rows) == 5100
    return rows


def stage(protocol_path: Path, plan_dir: Path, preview_root: Path, review_record: Path,
          full_root: Path, run_root: Path, output_dir: Path):
    from scripts.methods.colorpeel_ice import stage_d1_emission_color_transfer as emission
    from src.train.instance_mask_utils import load_latent_instance_mask

    value = protocol(protocol_path)
    expected = requests(value)
    if (read_json(plan_dir / "plan_provenance.json")["protocol_sha256"] != sha256(protocol_path)
            or read_jsonl(plan_dir / "render_requests.jsonl") != expected):
        raise ValueError("material render plan differs")
    review_sha = verified_review(preview_root, review_record, expected)
    rendered = verified_render(full_root, expected, 72)
    color_protocol = emission.protocol(REPO_ROOT / value["color_source_protocol"])
    color_source = run_root / value["color_source_root_relative_to_COLORPEEL_RUN_ROOT"]
    color_images = emission.verified_source(color_source, color_protocol)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    color_concepts, material_concepts, assets = [], [], []
    for request_id, source in sorted(color_images.items()):
        target = output_dir / "color" / request_id / "images"
        target.mkdir(parents=True)
        path = target / "image.png"
        shutil.copy2(source["image"], path)
        if sha256(path) != source["image_sha256"]:
            raise ValueError("emission color artifact differs")
        color_concepts.append({"instance_prompt": [source["prompt"]], "instance_data_dir": str(target.resolve())})
        assets.append({"branch": "color", "id": request_id, "field": "image", "path": str(path.resolve()),
                       "sha256": sha256(path)})
    for row in rendered:
        item = output_dir / "material" / row["cell_id"] / f"view_{row['view_index']:02d}"
        image_dir, mask_dir = item / "images", item / "masks"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir()
        for field, target in (("image", image_dir / "image.jpg"), ("mask", mask_dir / "image.png")):
            source = full_root / row[field]
            if field == "mask":
                load_latent_instance_mask(source, 512, 64)
            shutil.copy2(source, target)
            if sha256(target) != row["artifact_sha256"][field]:
                raise ValueError("staged material artifact differs")
            assets.append({"branch": "material", "id": f"{row['cell_id']}/view_{row['view_index']:02d}",
                           "field": field, "path": str(target.resolve()), "sha256": sha256(target)})
        material_concepts.append({
            "instance_prompt": [f"a photo of a {row['shape']} in {row['color']} color with <M*> material"],
            "instance_data_dir": str(image_dir.resolve()), "instance_mask_dir": str(mask_dir.resolve()),
        })
    concepts = schedule(color_concepts, material_concepts)
    write_json(output_dir / "concepts.json", concepts)
    write_jsonl(output_dir / "training_assets_manifest.jsonl", assets)
    write_json(output_dir / "staging_provenance.json", {
        "protocol_sha256": sha256(protocol_path), "review_sha256": review_sha,
        "full_realization_sha256": sha256(full_root / "renderer_realization.jsonl"),
        "concepts_sha256": sha256(output_dir / "concepts.json"),
        "assets_sha256": sha256(output_dir / "training_assets_manifest.jsonl"),
        "color_physical_images": 9, "material_physical_images": 72,
        "color_training_steps": 100, "material_training_steps": 5000,
    })
    return concepts


def validate_staging(protocol_path: Path, plan_dir: Path, preview_root: Path,
                     review_record: Path, full_root: Path, run_root: Path, staging_root: Path):
    from scripts.methods.colorpeel_ice import stage_d1_emission_color_transfer as emission

    value = protocol(protocol_path)
    expected = requests(value)
    if (read_json(plan_dir / "plan_provenance.json")["protocol_sha256"] != sha256(protocol_path)
            or read_jsonl(plan_dir / "render_requests.jsonl") != expected):
        raise ValueError("unpaired training plan differs")
    review_sha = verified_review(preview_root, review_record, expected)
    rendered = verified_render(full_root, expected, 72)
    color_protocol = emission.protocol(REPO_ROOT / value["color_source_protocol"])
    color_images = emission.verified_source(
        run_root / value["color_source_root_relative_to_COLORPEEL_RUN_ROOT"], color_protocol,
    )
    concepts_path = staging_root / "concepts.json"
    assets_path = staging_root / "training_assets_manifest.jsonl"
    provenance = read_json(staging_root / "staging_provenance.json")
    if (provenance.get("protocol_sha256") != sha256(protocol_path)
            or provenance.get("review_sha256") != review_sha
            or provenance.get("full_realization_sha256") != sha256(full_root / "renderer_realization.jsonl")
            or provenance.get("concepts_sha256") != sha256(concepts_path)
            or provenance.get("assets_sha256") != sha256(assets_path)
            or provenance.get("color_physical_images") != 9
            or provenance.get("material_physical_images") != 72
            or provenance.get("color_training_steps") != 100
            or provenance.get("material_training_steps") != 5000):
        raise ValueError("unpaired training staging provenance differs")
    color_concepts = [{
        "instance_prompt": [source["prompt"]],
        "instance_data_dir": str((staging_root.resolve() / "color" / request_id / "images")),
    } for request_id, source in sorted(color_images.items())]
    material_concepts = []
    for row in rendered:
        item = staging_root.resolve() / "material" / row["cell_id"] / f"view_{row['view_index']:02d}"
        material_concepts.append({
            "instance_prompt": [f"a photo of a {row['shape']} in {row['color']} color with <M*> material"],
            "instance_data_dir": str(item / "images"), "instance_mask_dir": str(item / "masks"),
        })
    if read_json(concepts_path) != schedule(color_concepts, material_concepts):
        raise ValueError("unpaired training captions or step schedule differ")
    expected_assets = [{
        "branch": "color", "id": request_id, "field": "image",
        "path": str(staging_root.resolve() / "color" / request_id / "images" / "image.png"),
        "sha256": source["image_sha256"],
    } for request_id, source in sorted(color_images.items())]
    for row in rendered:
        item = staging_root.resolve() / "material" / row["cell_id"] / f"view_{row['view_index']:02d}"
        for field, path in (("image", item / "images" / "image.jpg"),
                            ("mask", item / "masks" / "image.png")):
            expected_assets.append({
                "branch": "material", "id": f"{row['cell_id']}/view_{row['view_index']:02d}",
                "field": field, "path": str(path), "sha256": row["artifact_sha256"][field],
            })
    assets = read_jsonl(assets_path)
    if assets != expected_assets:
        raise ValueError("unpaired training asset sources differ")
    for asset in assets:
        path = Path(asset["path"]).resolve()
        if not path.is_relative_to(staging_root.resolve()) or sha256(path) != asset["sha256"]:
            raise ValueError("unpaired training asset hash differs")
    return concepts_path, assets_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path)
    parser.add_argument("--preview-root", type=Path)
    parser.add_argument("--review-record", type=Path)
    parser.add_argument("--full-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    if args.plan_dir:
        if any(value is None for value in (args.preview_root, args.review_record, args.full_root, args.run_root)):
            parser.error("staging requires preview, review, full render, and run roots")
        result = stage(args.protocol, args.plan_dir, args.preview_root, args.review_record,
                       args.full_root, args.run_root, args.output_dir)
        print(f"{len(result)} scheduled training rows: {args.output_dir}")
    else:
        print(f"{len(plan(args.protocol, args.output_dir))} material render requests: {args.output_dir}")


if __name__ == "__main__":
    main()
