"""Only a complete reviewed PBR factorial may become an isolated training dataset."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from experiments.subject_albedo_pbr_c import pbr, stage_pbr as module


@pytest.fixture
def candidates(tmp_path, monkeypatch):
    root, preview = tmp_path / "candidates", tmp_path / "preview"
    for directory in (root / "source", root / "renders", root / "images", preview / "source"):
        directory.mkdir(parents=True)
    original = np.full((512, 512, 3), (20, 40, 60), np.uint8)
    mask = np.zeros((512, 512), np.uint8)
    mask[80:470, 140:450] = 255
    Image.fromarray(original).save(preview / "source/original.jpg")
    Image.fromarray(mask).save(preview / "source/object_mask.png")
    for name in ("original.jpg", "object_mask.png"):
        (root / "source" / name).write_bytes((preview / "source" / name).read_bytes())
    original = np.asarray(Image.open(root / "source/original.jpg")).copy()
    revision = {"commit": "a" * 40, "branch": "codex/subject-c-pbr-scene"}
    protocol = {"renderer": {"engine": "CYCLES"}, "camera": {}, "lights": [], "shader": {}}
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    arrays = {"vertices": np.ones((3, 3), np.float32), "faces": np.array([[0, 1, 2]], np.int32),
              "uv": np.ones((3, 2), np.float32)}
    mesh = root / "geometry.npz"
    np.savez_compressed(mesh, **arrays)
    geometry_hash = hashlib.sha256(b"".join(arrays[name].tobytes() for name in ("vertices", "faces", "uv"))).hexdigest()
    module.write_json(root / "geometry_audit.json", {"array_content_sha256": geometry_hash})
    module.write_json(root / "projection_audit.json", {"UV_linear_error_p95": .001, "mask_IoU": 1})
    module.write_json(root / "renders/scene_audit.json", {"protocol_sha256": protocol_hash,
        "mesh_sha256": module.sha256(mesh), "render_engine": "CYCLES", "normal_input_linked": False})
    (root / "renders/fixed_scene.blend").write_bytes(b"fixed blender scene")
    for name in ("clay.png", "normal.png", "uv.png"):
        Image.new("RGB", (512, 512), "gray").save(root / name)
    albedos = {}
    for number, color in enumerate(module.pbr_data.COLORS):
        path = preview / f"{color}.png"
        Image.new("RGB", (512, 512), (30 + number * 30, 60, 90)).save(path)
        albedos[color] = {"path": str(path), "sha256": module.sha256(path)}
    planned, rows, rendered = [], [], []
    for number, (color, metallic, roughness) in enumerate(itertools.product(
            module.pbr_data.COLORS, module.pbr_data.METALLIC, module.pbr_data.ROUGHNESS)):
        identity = f"{color}__{metallic}__{roughness}"
        caption = f"a photo of <S*> mailbox in {color} color with a {roughness} {metallic} surface"
        frozen = {"id": identity, "color": color, "hue_degrees": module.pbr_data.COLORS[color],
            "metallic_state": metallic, "metallic": module.pbr_data.METALLIC[metallic], "roughness_state": roughness,
            "roughness": module.pbr_data.ROUGHNESS[roughness], "proposed_caption": caption,
            "maps": {"albedo": albedos[color]}}
        planned.append(frozen)
        image = original.copy()
        image[mask == 255] = (40 + number * 5, 100, 130)
        image_path = root / "images" / f"{identity}.png"
        Image.fromarray(image).save(image_path)
        rows.append({**frozen, "image": str(image_path), "image_sha256": module.sha256(image_path),
            "mask": str(root / "source/object_mask.png"), "caption": caption, "render_status": "true_cycles_pbr",
            "training_eligible": False, "qc": {"core_coverage_fraction": 1, "clipped_pixel_fraction": 0,
                "outside_mask_changed_pixels": 0}, "scene_sha256": protocol_hash, "geometry_content_sha256": geometry_hash})
        exr = root / "renders" / f"{identity}.exr"
        exr.write_bytes(identity.encode())
        rendered.append({"id": identity, "path": str(exr), "sha256": module.sha256(exr),
            "albedo_sha256": albedos[color]["sha256"], "metallic": frozen["metallic"], "roughness": frozen["roughness"],
            "mesh_sha256": module.sha256(mesh), "protocol_sha256": protocol_hash})
    source_locks = {str(path): module.sha256(path) for path in preview.rglob("*") if path.is_file()}
    plan = {"git": revision, "preview": str(preview), "rows": planned, "protocol": protocol,
            "protocol_sha256": protocol_hash, "mesh": str(mesh), "mesh_sha256": module.sha256(mesh), "source_locks": source_locks}
    module.write_json(root / "plan.json", plan)
    module.write_json(root / "renders/renders.json", rendered)
    module.write_json(root / "status.json", {"status": "pbr_candidates_numeric_verified_pending_visual_review",
        "mode": "full30", "expected_rgb": 30, "generated_rgb": 30, "training_started": False,
        "training_ready": False, "git": revision})
    review = tmp_path / "visual_review.json"
    def reseal():
        (root / "dataset_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        module.write_json(review, {"candidate_manifest_sha256": module.sha256(root / "dataset_manifest.jsonl"),
            "reviewer": "project owner", "decisions": [{"id": row["id"], "accepted": True, "reasons": []} for row in rows],
            "limitations": ["Single-view manually fitted scene; not recovered ground truth geometry or illumination."]})
    reseal()
    monkeypatch.setattr(module, "git_info", lambda: revision)
    monkeypatch.setattr(module, "resolve_run_dir", lambda path: path.resolve())
    reviews = []
    def show(output, staged, accepted, reference_root):
        reviews.append((staged, accepted, reference_root))
        Image.new("RGB", (64, 64), "blue").save(output / "contact_sheet.png")
        (output / "index.html").write_text("accepted actual training images")
    monkeypatch.setattr(pbr, "review", show)
    return root, review, tmp_path / "staged", rows, reseal, reviews, revision


def test_stages_all_actual_rgb_pairs_and_frozen_captions_without_touching_candidates(candidates):
    root, review, output, rows, _, reviews, _ = candidates
    before = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    result = module.run(root, review, output)
    assert result["status"] == "verified" and result["rows"] == result["training_eligible_images"] == 30
    assert result["training_started"] is False
    concepts = module.read_json(output / "concepts.json")
    staged = [json.loads(line) for line in (output / "training_assets_manifest.jsonl").read_text().splitlines()]
    assert len(concepts) == len(staged) == 30
    for original, actual, concept in zip(rows, staged, concepts):
        assert actual["training_eligible"] is True and original["training_eligible"] is False
        assert actual["caption"] == original["caption"] == concept["instance_prompt"][0]
        assert actual["source_image"] == original["image"]
        assert module.sha256(actual["image"]) == original["image_sha256"] == actual["image_sha256"]
        assert module.sha256(actual["mask"]) == actual["mask_sha256"]
        assert Path(concept["instance_data_dir"]) / "image.png" == Path(actual["image"])
    assert (output / "visual_review.json").read_bytes() == review.read_bytes()
    assert reviews[0][1:] == (True, root)
    assert {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("failure", ("review_reject", "review_missing", "review_wrong_hash", "not_complete", "git",
    "background", "rgb_hash", "clip", "coverage", "scene", "geometry", "albedo", "exr", "old_source"))
def test_any_failed_review_numeric_check_or_changed_input_refuses_entire_dataset(candidates, failure):
    root, review, output, rows, reseal, _, revision = candidates
    if failure.startswith("review_"):
        value = module.read_json(review)
        if failure == "review_reject":
            value["decisions"][0]["accepted"] = False
        elif failure == "review_missing":
            value["decisions"].pop()
        else:
            value["candidate_manifest_sha256"] = "bad"
        module.write_json(review, value)
    elif failure == "not_complete":
        value = module.read_json(root / "status.json")
        value["generated_rgb"] = 6
        module.write_json(root / "status.json", value)
    elif failure == "git":
        revision["commit"] = "b" * 40
    elif failure in ("background", "rgb_hash"):
        with Image.open(rows[0]["image"]) as image:
            image = image.copy()
        image.putpixel((0, 0), (255, 0, 0))
        image.save(rows[0]["image"])
        if failure == "background":
            rows[0]["image_sha256"] = module.sha256(rows[0]["image"])
            reseal()
    elif failure in ("clip", "coverage", "scene", "geometry", "albedo"):
        if failure == "clip":
            rows[0]["qc"]["clipped_pixel_fraction"] = .02
        elif failure == "coverage":
            rows[0]["qc"]["core_coverage_fraction"] = .99
        elif failure == "scene":
            rows[0]["scene_sha256"] = "changed"
        elif failure == "geometry":
            rows[0]["geometry_content_sha256"] = "changed"
        else:
            rows[0]["maps"]["albedo"] = rows[-1]["maps"]["albedo"]
        reseal()
    elif failure == "exr":
        receipt = module.read_json(root / "renders/renders.json")[0]
        Path(receipt["path"]).write_bytes(b"changed EXR")
    else:
        plan = module.read_json(root / "plan.json")
        next(iter(map(Path, plan["source_locks"]))).write_bytes(b"changed old source")
    with pytest.raises((ValueError, RuntimeError)):
        module.run(root, review, output)
    assert not output.exists()


def test_existing_output_is_preserved_and_never_reused(candidates):
    root, review, output, _, _, _, _ = candidates
    output.mkdir()
    marker = output / "old.txt"
    marker.write_text("old results")
    with pytest.raises(ValueError, match="fresh"):
        module.run(root, review, output)
    assert marker.read_text() == "old results" and len(list(output.iterdir())) == 1


def test_staging_cannot_add_files_inside_the_immutable_parameter_preview(candidates):
    root, review, _, _, _, _, _ = candidates
    preview = Path(module.read_json(root / "plan.json")["preview"])
    output = preview / "new_dataset"
    with pytest.raises(ValueError, match="immutable source preview"):
        module.run(root, review, output)
    assert not output.exists()


@pytest.fixture
def repaired(candidates):
    root, review, output, rows, reseal, reviews, revision = candidates
    plan = module.read_json(root / "plan.json")
    base_path = root / "source/original_mask.png"
    base_path.write_bytes((root / "source/object_mask.png").read_bytes())
    mask = np.asarray(Image.open(base_path)).copy()
    mask[470:475, 140:430] = 255  # Exactly 1450 newly covered bottom-edge pixels.
    repair_source = root.parent / "approved_b_repaired_mask.png"
    Image.fromarray(mask).save(repair_source)
    (root / "source/object_mask.png").write_bytes(repair_source.read_bytes())
    plan["source_locks"][str(repair_source)] = module.sha256(repair_source)
    plan["render_mask"] = {"source": str(repair_source), "sha256": module.sha256(repair_source),
        "base_mask_sha256": module.sha256(base_path), "added_pixels": 1450,
        "uv_policy": "fixed nearest-valid-image UV extension; original five Albedo files remain unchanged"}
    mesh_path = root / "geometry.npz"
    with np.load(mesh_path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in ("vertices", "faces", "uv")}
    arrays["face_uv"] = np.full((1, 3, 2), .25, np.float32)
    np.savez_compressed(mesh_path, **arrays)
    geometry_hash = hashlib.sha256(b"".join(arrays[name].tobytes() for name in ("vertices", "faces", "uv", "face_uv"))).hexdigest()
    plan["mesh_sha256"] = module.sha256(mesh_path)
    module.write_json(root / "plan.json", plan)
    module.write_json(root / "geometry_audit.json", {"array_content_sha256": geometry_hash})
    scene = module.read_json(root / "renders/scene_audit.json")
    scene["mesh_sha256"] = plan["mesh_sha256"]
    module.write_json(root / "renders/scene_audit.json", scene)
    rendered = module.read_json(root / "renders/renders.json")
    for row, receipt in zip(rows, rendered):
        row["geometry_content_sha256"] = geometry_hash
        receipt["mesh_sha256"] = plan["mesh_sha256"]
    module.write_json(root / "renders/renders.json", rendered)
    reseal()
    return candidates, plan, repair_source, arrays


def test_repaired_render_mask_and_face_uv_are_bound_and_retained_in_staged_audit(repaired):
    (root, review, output, rows, _, _, _), plan, repair_source, _ = repaired
    original_albedos = {row["color"]: row["maps"]["albedo"]["sha256"] for row in rows}
    result = module.run(root, review, output)
    assert result["render_mask"] == plan["render_mask"]
    assert result["geometry_array_order"] == ["vertices", "faces", "uv", "face_uv"]
    locks = module.read_json(output / "source_locks.json")
    assert locks[str(repair_source)] == plan["render_mask"]["sha256"]
    assert str(root / "source/original_mask.png") in locks
    staged = [json.loads(line) for line in (output / "training_assets_manifest.jsonl").read_text().splitlines()]
    assert all(row["render_mask"] == plan["render_mask"] for row in staged)
    assert all(row["mask_sha256"] == module.sha256(repair_source) for row in staged)
    assert {row["color"]: row["maps"]["albedo"]["sha256"] for row in staged} == original_albedos


@pytest.mark.parametrize("failure", ("source_lock", "base_hash", "deletes_foreground", "added_count", "face_uv_hash"))
def test_repaired_mask_or_face_uv_provenance_errors_are_rejected(repaired, failure):
    (root, review, output, _, _, _, _), plan, repair_source, arrays = repaired
    if failure == "source_lock":
        del plan["source_locks"][str(repair_source)]
    elif failure == "base_hash":
        plan["render_mask"]["base_mask_sha256"] = plan["render_mask"]["sha256"]
    elif failure == "deletes_foreground":
        mask = np.asarray(Image.open(repair_source)).copy()
        mask[100, 200] = 0
        Image.fromarray(mask).save(repair_source)
        (root / "source/object_mask.png").write_bytes(repair_source.read_bytes())
        plan["render_mask"]["sha256"] = module.sha256(repair_source)
        plan["source_locks"][str(repair_source)] = module.sha256(repair_source)
    elif failure == "added_count":
        plan["render_mask"]["added_pixels"] = 1449
    else:
        old_three_array_hash = hashlib.sha256(b"".join(arrays[name].tobytes() for name in ("vertices", "faces", "uv"))).hexdigest()
        module.write_json(root / "geometry_audit.json", {"array_content_sha256": old_three_array_hash})
    module.write_json(root / "plan.json", plan)
    with pytest.raises(ValueError, match="mask|geometry|UV"):
        module.run(root, review, output)
    assert not output.exists()
