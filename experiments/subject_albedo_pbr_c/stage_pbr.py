"""Stage all thirty visually accepted PBR candidates without starting training."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd_native_comparison_v1.generate import write_json
from experiments.subject_albedo_pbr_c import pbr_data
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]


def _finite(value, minimum, maximum):
    return type(value) in (int, float) and np.isfinite(value) and minimum <= value <= maximum


def run(candidates, review, output):
    candidates, review = Path(candidates).resolve(), Path(review).resolve()
    output = resolve_run_dir(Path(output))
    if (output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(candidates)
            or candidates.is_relative_to(output)):
        raise ValueError("staged dataset must be fresh and outside the repository and old candidates")
    revision = git_info()  # Also refuses a dirty worktree.
    status, plan = read_json(candidates / "status.json"), read_json(candidates / "plan.json")
    preview = Path(plan["preview"]).resolve()
    if output.is_relative_to(preview) or preview.is_relative_to(output):
        raise ValueError("staged dataset must be outside the immutable source preview")
    if (status.get("status") != "pbr_candidates_numeric_verified_pending_visual_review"
            or status.get("mode") != "full30" or status.get("expected_rgb") != 30
            or status.get("generated_rgb") != 30 or status.get("training_started") is not False
            or status.get("training_ready") is not False or status.get("git") != revision
            or plan.get("git") != revision):
        raise ValueError("expected thirty completed PBR candidates from the current clean checkout")
    manifest = candidates / "dataset_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf8").splitlines()]
    expected = set(itertools.product(pbr_data.COLORS, pbr_data.METALLIC, pbr_data.ROUGHNESS))
    keys = [(row["color"], row["metallic_state"], row["roughness_state"]) for row in rows]
    if (len(rows) != 30 or len({row["id"] for row in rows}) != 30 or len(set(keys)) != 30
            or set(keys) != expected):
        raise ValueError("PBR candidates must contain thirty unique complete A/M/R combinations")
    reviewed = read_json(review)
    decisions = reviewed.get("decisions", [])
    if (reviewed.get("candidate_manifest_sha256") != sha256(manifest)
            or not isinstance(reviewed.get("reviewer"), str) or not reviewed["reviewer"].strip()
            or not isinstance(reviewed.get("limitations"), list)
            or any(not isinstance(value, str) for value in reviewed["limitations"])
            or not isinstance(decisions, list) or len(decisions) != 30
            or any(not isinstance(item, dict) or item.get("accepted") is not True
                   or not isinstance(item.get("reasons"), list)
                   or any(not isinstance(reason, str) for reason in item["reasons"]) for item in decisions)
            or len({item.get("id") for item in decisions}) != 30
            or {item.get("id") for item in decisions} != {row["id"] for row in rows}):
        raise ValueError("all thirty actual candidates need an explicit matching per-image accepted visual review")
    locks = dict(plan["source_locks"])
    pbr_data._check_locks(locks)
    for path in [candidates / name for name in ("status.json", "plan.json", "dataset_manifest.jsonl",
                "geometry.npz", "geometry_audit.json", "projection_audit.json", "renders/renders.json",
                "renders/scene_audit.json", "renders/fixed_scene.blend", "clay.png", "normal.png", "uv.png")] + [review]:
        locks[str(path)] = sha256(path)
    protocol_hash = hashlib.sha256(json.dumps(plan["protocol"], sort_keys=True).encode()).hexdigest()
    mesh_path = Path(plan["mesh"])
    if mesh_path.resolve() != (candidates / "geometry.npz").resolve() or sha256(mesh_path) != plan["mesh_sha256"]:
        raise ValueError("candidate scene geometry differs from its fixed plan")
    with np.load(mesh_path, allow_pickle=False) as mesh:
        array_order = ["vertices", "faces", "uv"] + (["face_uv"] if "face_uv" in mesh.files else [])
        geometry_hash = hashlib.sha256(b"".join(mesh[name].tobytes() for name in array_order)).hexdigest()
    geometry_audit = read_json(candidates / "geometry_audit.json")
    scene = read_json(candidates / "renders/scene_audit.json")
    if (plan.get("protocol_sha256") != protocol_hash or scene.get("protocol_sha256") != protocol_hash
            or scene.get("mesh_sha256") != plan["mesh_sha256"]
            or geometry_audit.get("array_content_sha256") != geometry_hash
            or scene.get("render_engine") != "CYCLES" or scene.get("normal_input_linked") is not False):
        raise ValueError("fixed scene/geometry provenance differs")
    projection = read_json(candidates / "projection_audit.json")
    if (not _finite(projection.get("UV_linear_error_p95"), 0, .01)
            or not _finite(projection.get("mask_IoU"), .995, 1)):
        raise ValueError("source projection or silhouette audit failed")
    planned, rendered = plan["rows"], read_json(candidates / "renders/renders.json")
    if (len(planned) != 30 or len(rendered) != 30 or len({row["id"] for row in planned}) != 30
            or len({row["id"] for row in rendered}) != 30
            or {row["id"] for row in planned} != {row["id"] for row in rows}
            or {row["id"] for row in rendered} != {row["id"] for row in rows}):
        raise ValueError("candidate plan or original EXR receipt grid differs")
    planned, rendered = ({row["id"]: row for row in records} for records in (planned, rendered))
    original_path, mask_path = (candidates / "source" / name for name in ("original.jpg", "object_mask.png"))
    render_mask = plan.get("render_mask")
    for path in (original_path,) if render_mask is not None else (original_path, mask_path):
        source_path = str(Path(plan["preview"]) / "source" / path.name)
        if sha256(path) != plan["source_locks"].get(source_path):
            raise ValueError("candidate original RGB/mask differs from the immutable source preview")
        locks[str(path)] = sha256(path)
    with Image.open(original_path) as image:
        if image.mode != "RGB" or image.size != (512, 512):
            raise ValueError("original source must be 512px RGB")
        original = np.asarray(image).copy()
    with Image.open(mask_path) as image:
        if image.mode != "L" or image.size != (512, 512) or set(np.unique(image)) != {0, 255}:
            raise ValueError("original source mask must be 512px nonempty binary")
        mask = np.asarray(image).copy()
    if render_mask is not None:
        repair_source = Path(render_mask["source"])
        base_path = candidates / "source/original_mask.png"
        base_hash = plan["source_locks"].get(str(preview / "source/object_mask.png"))
        if (not repair_source.is_absolute() or render_mask.get("sha256") != sha256(mask_path)
                or plan["source_locks"].get(str(repair_source)) != render_mask.get("sha256")
                or sha256(repair_source) != render_mask.get("sha256")
                or render_mask.get("base_mask_sha256") != base_hash or sha256(base_path) != base_hash
                or render_mask.get("added_pixels") != 1450 or not render_mask.get("uv_policy")):
            raise ValueError("fixed repaired render mask, original mask or UV repair provenance differs")
        with Image.open(base_path) as image:
            if image.mode != "L" or image.size != (512, 512) or set(np.unique(image)) != {0, 255}:
                raise ValueError("pre-repair source mask must be 512px nonempty binary")
            base_mask = np.asarray(image).copy()
        if (np.any((base_mask == 255) & (mask != 255))
                or int(np.count_nonzero((mask == 255) & (base_mask == 0))) != render_mask["added_pixels"]):
            raise ValueError("repaired render mask must add exactly 1450 pixels without deleting original foreground")
        locks[str(base_path)], locks[str(mask_path)] = sha256(base_path), sha256(mask_path)
    albedos = {}
    frozen_keys = ("id", "color", "hue_degrees", "metallic_state", "metallic", "roughness_state",
                   "roughness", "proposed_caption", "maps")
    for row in rows:
        frozen, exr = planned[row["id"]], rendered[row["id"]]
        if (any(row.get(key) != frozen.get(key) for key in frozen_keys)
                or row.get("hue_degrees") != pbr_data.COLORS[row["color"]]
                or row.get("metallic") != pbr_data.METALLIC[row["metallic_state"]]
                or row.get("roughness") != pbr_data.ROUGHNESS[row["roughness_state"]]
                or row.get("scene_sha256") != protocol_hash or row.get("geometry_content_sha256") != geometry_hash
                or row.get("render_status") != "true_cycles_pbr" or row.get("training_eligible") is not False
                or row.get("caption") != frozen["proposed_caption"]
                or not isinstance(row["caption"], str) or row["caption"].count("<S*>") != 1
                or any(token in row["caption"] for token in ("<C*>", "<M*>", "<T*>"))):
            raise ValueError("candidate parameters, Subject-only caption or fixed scene differ")
        qc = row["qc"]
        if (not _finite(qc.get("core_coverage_fraction"), .995, 1)
                or not _finite(qc.get("clipped_pixel_fraction"), 0, .01)
                or qc.get("outside_mask_changed_pixels") != 0):
            raise ValueError("candidate coverage, clipping or background QC failed")
        image_path, row_mask = Path(row["image"]), Path(row["mask"])
        if (image_path.resolve() != (candidates / "images" / f"{row['id']}.png").resolve()
                or row_mask.resolve() != mask_path.resolve() or sha256(image_path) != row["image_sha256"]
                or row.get("mask_sha256", sha256(mask_path)) != sha256(mask_path)):
            raise ValueError("candidate image/mask path or hash differs")
        with Image.open(image_path) as image:
            if (image.mode != "RGB" or image.size != (512, 512)
                    or not np.array_equal(np.asarray(image)[mask == 0], original[mask == 0])):
                raise ValueError("candidate RGB dimensions/mode or original outside-mask background differ")
        albedo = row["maps"]["albedo"]
        if sha256(albedo["path"]) != albedo["sha256"] or plan["source_locks"].get(albedo["path"]) != albedo["sha256"]:
            raise ValueError("candidate Albedo is not bound to the original source locks")
        if row["color"] in albedos and albedos[row["color"]] != albedo["sha256"]:
            raise ValueError("one color changes Albedo with material parameters")
        albedos[row["color"]] = albedo["sha256"]
        exr_path = Path(exr["path"])
        if (not exr_path.resolve().is_relative_to((candidates / "renders").resolve())
                or sha256(exr_path) != exr["sha256"] or exr.get("albedo_sha256") != albedo["sha256"]
                or exr.get("metallic") != row["metallic"] or exr.get("roughness") != row["roughness"]
                or exr.get("mesh_sha256") != plan["mesh_sha256"] or exr.get("protocol_sha256") != protocol_hash):
            raise ValueError("original EXR hash or fixed render parameters differ")
        locks[str(image_path)], locks[str(exr_path)] = sha256(image_path), sha256(exr_path)
    if len(set(albedos.values())) != 5:
        raise ValueError("five colors must use five distinct fixed Albedo maps")
    pbr_data._check_locks(locks)
    output.mkdir(parents=True)
    staged = []
    for row in rows:
        root = output / "assets" / row["id"]
        target_image, target_mask = root / "images/image.png", root / "masks/image.png"
        for source, target in ((Path(row["image"]), target_image), (mask_path, target_mask)):
            target.parent.mkdir(parents=True)
            shutil.copyfile(source, target)
            if sha256(source) != sha256(target):
                raise ValueError("staging changed actual candidate RGB/mask bytes")
        staged.append({**row, "source_image": row["image"], "source_image_sha256": row["image_sha256"],
            "source_mask": row["mask"], "source_mask_sha256": sha256(mask_path),
            "image": str(target_image), "image_sha256": sha256(target_image),
            "mask": str(target_mask), "mask_sha256": sha256(target_mask), "training_eligible": True,
            "render_mask": render_mask})
    concepts = [{"instance_prompt": [row["caption"]], "instance_data_dir": str(Path(row["image"]).parent),
                 "instance_mask_dir": str(Path(row["mask"]).parent)} for row in staged]
    write_json(output / "concepts.json", concepts)
    (output / "training_assets_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in staged), encoding="utf8")
    shutil.copyfile(review, output / "visual_review.json")
    write_json(output / "source_locks.json", locks)
    from experiments.subject_albedo_pbr_c.pbr import review as write_review
    write_review(output, staged, accepted=True, reference_root=candidates)
    sheets = ["contact_sheet.png", "index.html"]
    if not all((output / name).is_file() for name in sheets):
        raise ValueError("accepted data review did not produce its actual contact sheet and index")
    if git_info() != revision:
        raise ValueError("checkout changed while staging visually accepted data")
    pbr_data._check_locks(locks)
    audit = {"schema": "subject_c_visually_accepted_pbr_dataset/v1", "status": "verified", "git": revision,
             "rows": 30, "training_eligible_images": 30, "training_started": False,
             "candidate_manifest_sha256": sha256(manifest), "visual_review_sha256": sha256(review),
             "scene_sha256": protocol_hash, "geometry_content_sha256": geometry_hash,
             "geometry_array_order": array_order, "render_mask": render_mask,
             "limitations": reviewed["limitations"], "contact_sheets": sheets,
             "files": {name: sha256(output / name) for name in
                       ("concepts.json", "training_assets_manifest.jsonl", "source_locks.json", "visual_review.json")}}
    write_json(output / "dataset_audit.json", audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.candidates, args.review, args.output)))
