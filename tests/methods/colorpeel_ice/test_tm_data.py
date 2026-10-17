"""Control integrity tests for the independent Texture/Material oracle pilot."""

import collections
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.texture_material_disentangled_v1 import data
from src.train.instance_mask_utils import load_latent_instance_mask, pair_instance_images_and_masks


def test_plan_has_balanced_nuisances_and_truly_unseen_learned_pairs(tmp_path):
    result = data.make_plan(tmp_path / "plan")
    protocol = data.read_protocol()
    train = [row for row in result["records"] if row["split"] == "train"]
    evaluation = [row for row in result["records"] if row["split"] == "eval"]
    assert (len(train), len(evaluation)) == (64, 16)
    assert len({row["id"] for row in result["records"]}) == 80
    assert collections.Counter(row["concept_id"] for row in train) == {
        "vertical_stripes": 16, "checkerboard": 16,
        "polished_metal": 16, "rough_dielectric": 16,
    }
    for row in train:
        roles = (protocol["textures"][row["texture_id"]]["role"],
                 protocol["materials"][row["material_id"]]["role"])
        assert sorted(roles) == ["nuisance", "target"]
    learned_pairs = {(row["texture_id"], row["material_id"]) for row in evaluation}
    assert len(learned_pairs) == 4
    assert not learned_pairs.intersection((row["texture_id"], row["material_id"]) for row in train)
    for branch in ("texture", "material"):
        groups = collections.defaultdict(list)
        for row in train:
            if row["branch"] == branch:
                nuisance = row["material_id"] if branch == "texture" else row["texture_id"]
                groups[row["concept_id"]].append((row["shape"], row["color"], row["light"], nuisance, row["seed"]))
        a, b = groups.values()
        assert a == b
    with pytest.raises(FileExistsError):
        data.make_plan(tmp_path / "plan")


def test_extraction_preserves_order_removes_hue_and_ignores_mask_outliers():
    rgb = np.zeros((10, 10, 3), dtype="uint8")
    rgb[:5, :5] = [40, 20, 10]
    rgb[5:, :5] = [180, 90, 45]
    rgb[:, 5:] = 255
    mask = np.zeros((10, 10), dtype="uint8")
    mask[:, :5] = 255
    texture, stats = data.extract_texture(Image.fromarray(rgb), Image.fromarray(mask))
    pixels = np.asarray(texture)
    assert texture.mode == "L"
    assert set(np.unique(pixels[:, :5])) == {64, 191}
    assert np.all(pixels[:, 5:] == 128)
    assert stats["valid_pixels"] == 50
    assert stats["upper_luminance"] < 0.3  # White pixels outside mask do not enter normalization.
    assert pixels[0, 0] < pixels[-1, 0]
    with pytest.raises(ValueError, match="Near-constant"):
        data.extract_texture(Image.new("RGB", (8, 8), "gray"))
    with pytest.raises(ValueError, match="dimensions"):
        data.extract_texture(Image.new("RGB", (8, 8)), Image.new("L", (4, 4), 255))


def render_fixture(tmp_path):
    plan = data.make_plan(tmp_path / "plan")
    root = tmp_path / "renders"
    root.mkdir()
    image, mask = root / "image.png", root / "mask.png"
    Image.new("RGB", (512, 512), (100, 70, 40)).save(image)
    mask_pixels = np.zeros((512, 512), dtype="uint8")
    mask_pixels[100:412, 100:412] = 127
    mask_pixels[101:411, 101:411] = 128
    mask_pixels[102:410, 102:410] = 255
    Image.fromarray(mask_pixels).save(mask)
    rows = []
    for row in plan["records"]:
        material = data.read_protocol()["materials"][row["material_id"]]
        record = {**row, "image": str(image), "mask": str(mask),
                     "image_sha256": data.sha256(image), "mask_sha256": data.sha256(mask),
                     "roughness": material["roughness"], "metallic": material["metallic"],
                     "texture_projection": "UV", "uv_repeats": 1,
                     "texture_colorspace": "Non-Color", "normal_bump_displacement": False,
                     "base_color_linear_rgb": data.read_protocol()["dataset"]["colors_linear_rgb"][row["color"]],
                     "uv_mapping": {"policy": "fixture", "layer": "UVMap", "loop_count": 4},
                     "uv_sha256": "a" * 64}
        metadata = root / f'{row["id"]}.json'
        data.write_json(metadata, record)
        rows.append({**record, "metadata": str(metadata), "metadata_sha256": data.sha256(metadata)})
    manifest = {"status": "complete", "protocol_sha256": data.protocol_sha256(), "records": rows}
    data.write_json(root / "manifest.json", manifest)
    return root, manifest


def test_stage_verifies_frozen_sampling_groups_and_drift(tmp_path):
    root, _ = render_fixture(tmp_path)
    staged = tmp_path / "staged"
    audit = data.stage(root, staged)
    assert audit["learned_texture_material_pairs_in_training"] == 0
    assert data.verify_staging(staged) == audit
    for concept, receipt in audit["datasets"].items():
        assert receipt["image_count"] == 16
        concepts = json.loads((staged / concept / "concepts.json").read_text())
        assert len(concepts) == 16
        token = "<T*>" if receipt["branch"] == "texture" else "<M*>"
        assert all(token in row["instance_prompt"][0] for row in concepts)
        assets = [json.loads(line) for line in
                  (staged / concept / "training_assets_manifest.jsonl").read_text().splitlines()]
        for group, asset in zip(concepts, assets):
            pairs = pair_instance_images_and_masks(Path(group["instance_data_dir"]),
                                                  Path(group["instance_mask_dir"]))
            assert len(pairs) == 1
            assert pairs[0][0].stem == pairs[0][1].stem == "image"
            assert load_latent_instance_mask(pairs[0][1], 512, 64).sum() > 0
            assert asset["source_image_sha256"] == asset["image_sha256"]
            assert asset["source_mask_sha256"] != asset["mask_sha256"]
            with Image.open(pairs[0][1]) as binary:
                assert set(np.unique(np.asarray(binary))) == {0, 255}
    extra = staged / "vertical_stripes" / "images" / "unexpected.png"
    Image.new("RGB", (8, 8)).save(extra)
    with pytest.raises(ValueError, match="Unexpected staged files"):
        data.verify_staging(staged)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "wrong_seed", "wrong_material", "wrong_uv", "changed_image"])
def test_stage_rejects_bad_counterfactual_or_files_before_writing(tmp_path, failure):
    root, manifest = render_fixture(tmp_path)
    broken = copy.deepcopy(manifest)
    if failure == "missing":
        broken["records"].pop()
    elif failure == "duplicate":
        broken["records"][-1] = broken["records"][0]
    elif failure == "wrong_seed":
        broken["records"][0]["seed"] += 99
    elif failure == "wrong_material":
        broken["records"][0]["metallic"] = 0.123
    elif failure == "wrong_uv":
        broken["records"][0]["texture_projection"] = "BOX"
    else:
        Image.new("RGB", (512, 512), "blue").save(root / "image.png")
    data.write_json(root / "manifest.json", broken)
    output = tmp_path / "staged"
    with pytest.raises(ValueError):
        data.stage(root, output)
    assert not output.exists()
