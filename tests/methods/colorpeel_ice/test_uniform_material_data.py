"""Uniform-albedo controls cannot accidentally retain old texture supervision."""

import collections
import copy
import json

import numpy as np
import pytest
from PIL import Image

from experiments.material_uniform_albedo_v1 import data
from experiments.material_uniform_albedo_v1.protocol import PARENT_SHA256


def test_material_plan_is_paired_and_has_no_texture_dimension(tmp_path):
    plan = data.make_plan(tmp_path / "plan")
    assert (plan["train_images"], plan["eval_images"]) == (16, 0)
    assert collections.Counter(row["concept_id"] for row in plan["records"]) == {
        "polished_metal": 8, "rough_dielectric": 8,
    }
    contexts = collections.defaultdict(list)
    for row in plan["records"]:
        assert row["branch"] == "material"
        assert row["albedo_mode"] == "uniform" and row["albedo_level"] == 0.5
        assert not any("texture" in key for key in row)
        assert "texture" not in data.caption(row)
        assert "<M*>" in data.caption(row) and "<T*>" not in data.caption(row)
        contexts[row["concept_id"]].append(tuple(row[key] for key in ("shape", "color", "light", "seed")))
    assert contexts["polished_metal"] == contexts["rough_dielectric"]
    protocol = data.read_protocol()
    assert protocol["parent_protocol_sha256"] == PARENT_SHA256
    assert "textures" not in protocol and "texture_extraction" not in protocol
    assert protocol["training"]["checkpoint_steps"] == [500, 750, 1000]
    with pytest.raises(FileExistsError):
        data.make_plan(tmp_path / "plan")


def render_fixture(tmp_path):
    root = tmp_path / "renders"
    root.mkdir()
    image, mask, rgba = (root / name for name in ("image.png", "mask.png", "rgba.png"))
    Image.new("RGB", (512, 512), (100, 70, 40)).save(image)
    pixels = np.zeros((512, 512), dtype="uint8")
    pixels[100:412, 100:412] = 127
    pixels[101:411, 101:411] = 128
    pixels[102:410, 102:410] = 255
    Image.fromarray(pixels).save(mask)
    Image.new("RGBA", (512, 512), (100, 70, 40, 255)).save(rgba)
    protocol = data.read_protocol()
    rows = []
    for row in data.records():
        material = protocol["materials"][row["material_id"]]
        color = protocol["dataset"]["colors_linear_rgb"][row["color"]]
        record = {**row, "roughness": material["roughness"], "metallic": material["metallic"],
                  "base_color_linear_rgb": color,
                  "uniform_base_color_linear_rgba": [0.5 * value for value in color] + [1.0],
                  "texture_image_nodes": 0, "normal_bump_displacement": False,
                  "shader": "Principled BSDF", "ior": 1.5, "specular_ior_level": 0.5,
                  "coat_weight": 0.0, "transmission_weight": 0.0,
                  "cycles_samples": 64, "resolution": [512, 512], "blender_version": [4, 2, 11],
                  "render_seed": row["seed"], "object_scale": 1.3}
        for key, path in (("image", image), ("mask", mask), ("input_rgba", rgba)):
            record.update({key: str(path), key + "_sha256": data.sha256(path)})
        metadata = root / (row["id"] + ".json")
        data.write_json(metadata, record)
        rows.append({**record, "metadata": str(metadata), "metadata_sha256": data.sha256(metadata)})
    manifest = {"status": "complete", "protocol_sha256": data.protocol_sha256(), "records": rows}
    data.write_json(root / "manifest.json", manifest)
    return root, manifest


def test_stage_binary_mask_and_exact_sampling_groups(tmp_path):
    root, _ = render_fixture(tmp_path)
    staged = tmp_path / "staged"
    audit = data.stage(root, staged)
    assert data.verify_staging(staged) == audit
    assert audit["train_images"] == 16
    for concept, receipt in audit["datasets"].items():
        assert receipt["sample_count"] == 8
        groups = json.loads((staged / concept / "concepts.json").read_text())
        assert len(groups) == 8
        assert all("texture" not in row["instance_prompt"][0] for row in groups)
        mask = next((staged / concept / "masks").rglob("image.png"))
        with Image.open(mask) as opened:
            assert set(np.unique(np.asarray(opened))) == {0, 255}
    extra = staged / "polished_metal" / "images" / "extra.png"
    Image.new("RGB", (8, 8)).save(extra)
    with pytest.raises(ValueError, match="Unexpected staged files"):
        data.verify_staging(staged)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "seed", "material", "texture",
                                     "uniform_color", "samples", "changed_image", "changed_mask", "caption"])
def test_reject_changed_data_without_touching_source(tmp_path, failure):
    root, manifest = render_fixture(tmp_path)
    broken = copy.deepcopy(manifest)
    if failure == "missing":
        broken["records"].pop()
    elif failure == "duplicate":
        broken["records"][-1] = broken["records"][0]
    elif failure == "seed":
        broken["records"][0]["seed"] += 10
    elif failure == "material":
        broken["records"][0]["roughness"] = 0.123
    elif failure == "texture":
        broken["records"][0]["texture_image_nodes"] = 1
    elif failure == "uniform_color":
        broken["records"][0]["uniform_base_color_linear_rgba"][0] += 0.1
    elif failure == "samples":
        broken["records"][0]["cycles_samples"] = 32
    elif failure == "changed_image":
        Image.new("RGB", (512, 512), "blue").save(root / "image.png")
    elif failure == "changed_mask":
        Image.new("L", (512, 512)).save(root / "mask.png")
    elif failure == "caption":
        staged = tmp_path / "staged"
        data.stage(root, staged)
        path = staged / "polished_metal" / "concepts.json"
        groups = json.loads(path.read_text())
        groups[0]["instance_prompt"] = ["a striped sphere"]
        data.write_json(path, groups)
        with pytest.raises(ValueError, match="staging metadata"):
            data.verify_staging(staged)
        return
    data.write_json(root / "manifest.json", broken)
    output = tmp_path / "staged"
    with pytest.raises(ValueError):
        data.stage(root, output)
    assert not output.exists()
