"""Full natural factorial staging, actual RGB references, and corruption gates."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.natural_texture_material_v1 import dataset_protocol as schema
from experiments.natural_texture_material_v1 import dataset_stage as module
from experiments.texture_material_disentangled_v1.data import read_protocol, write_json


@pytest.fixture
def rendered(tmp_path):
    protocol = read_protocol()
    protocol["phase"] = 2
    protocol["sources"], protocol["textures"], protocol["source_locks"] = {}, {}, {}
    protocol["materials"] = {name: value for name, value in protocol["materials"].items() if value["role"] == "nuisance"}
    source = tmp_path / "accepted_extraction"
    source.mkdir()
    for index, name in enumerate(schema.SOURCES):
        folder = source / name
        folder.mkdir()
        Image.new("RGBA", (64, 64), (100, 150, 180, 255)).save(folder / "input.png")
        Image.fromarray(np.arange(64, dtype="uint8").reshape(8, 8)).save(folder / "texture.png")
        material = {"roughness": .2 + index * .1, "metallic": index * .3}
        write_json(folder / "material.json", material)
        protocol["sources"][name] = {**material, "input": str(folder / "input.png"), "texture": str(folder / "texture.png"), "material": str(folder / "material.json")}
        protocol["textures"][name + "_texture"] = {"role": "target", "source_id": name, "path": str(folder / "texture.png"), "sha256": schema.sha256(folder / "texture.png")}
        protocol["materials"][name + "_material"] = {**material, "role": "target", "source_id": name}
        protocol["source_locks"].update({str(path): schema.sha256(path) for path in folder.iterdir()})
    plan = tmp_path / "plan"
    plan.mkdir()
    write_json(plan / "protocol.json", protocol)
    write_json(plan / "plan.json", {"protocol_sha256": schema.protocol_sha256(protocol), "records": schema.records(protocol)})
    root = tmp_path / "renders"
    root.mkdir()
    Image.new("RGB", (512, 512), (90, 80, 70)).save(root / "image.png")
    Image.new("RGBA", (512, 512), (90, 80, 70, 255)).save(root / "input_rgba.png")
    mask = np.zeros((512, 512), dtype="uint8")
    mask[100:412, 100:412] = 127
    mask[101:411, 101:411] = 128
    mask[102:410, 102:410] = 255
    Image.fromarray(mask).save(root / "mask.png")
    rows = []
    for row in schema.records(protocol):
        material = protocol["materials"][row["material_id"]]
        color = protocol["dataset"]["colors_linear_rgb"][row["color"]]
        record = {**row, "roughness": material["roughness"], "metallic": material["metallic"],
            "base_color_linear_rgb": color, "normal_bump_displacement": False, "shader": "Principled BSDF",
            "ior": 1.5, "specular_ior_level": .5, "coat_weight": 0., "transmission_weight": 0.,
            "roughness_input_linked": False, "metallic_input_linked": False, "normal_input_linked": False,
            "base_color_input_linked": row["branch"] == "texture",
            "cycles_samples": 64, "resolution": [512, 512], "blender_version": [4, 2, 11],
            "render_seed": row["seed"], "object_scale": 1.3}
        if row["branch"] == "texture":
            record.update(texture_projection="UV", uv_repeats=1, texture_colorspace="Non-Color",
                          texture_image_nodes=1, uv_mapping={"policy": "fixture"}, uv_sha256="a" * 64)
        else:
            record.update(texture_image_nodes=0, uniform_base_color_linear_rgba=[.5 * value for value in color] + [1.])
        for key in ("image", "mask", "input_rgba"):
            record[key] = str(root / (key + ".png"))
            record[key + "_sha256"] = schema.sha256(record[key])
        metadata = root / (row["id"] + ".json")
        write_json(metadata, record)
        rows.append({**record, "metadata": str(metadata), "metadata_sha256": schema.sha256(metadata)})
    manifest = {"status": "complete", "protocol_path": str(plan / "protocol.json"),
        "plan_path": str(plan / "plan.json"), "plan_sha256": schema.sha256(plan / "plan.json"),
        "protocol_sha256": schema.protocol_sha256(protocol), "source_locks": protocol["source_locks"], "records": rows}
    write_json(root / "manifest.json", manifest)
    return root, manifest


def test_staging_all_actual_images_loader_groups_and_references(rendered, tmp_path):
    root, _ = rendered
    output = tmp_path / "staged"
    audit = module.stage(root, output)
    assert module.verify_staging(output) == audit
    assert (audit["train_images"], audit["texture_images"], audit["material_images"]) == (72, 48, 24)
    assert audit["training_started"] is False
    refs = module.read(output / "training_references.json")
    assert len(refs) == 6 and len(list((output / "contact_sheets").glob("*.jpg"))) == 6
    for concept, receipt in audit["datasets"].items():
        count = 16 if receipt["branch"] == "texture" else 8
        assert receipt["phase"] == 2 and receipt["image_count"] == count
        groups = module.read(output / concept / "concepts.json")
        rows = [json.loads(line) for line in (output / concept / "training_assets_manifest.jsonl").read_text().splitlines()]
        assert len(groups) == len(rows) == len(refs[concept]["images"]) == count
        assert refs[concept]["planned_methods"] == ["fullrank", "lora_r64"]
        assert refs[concept]["status"] == "dataset_prepared_training_not_started"
        assert [ref["image"] for ref in refs[concept]["images"]] == [row["image"] for row in rows]
        for group, row in zip(groups, rows):
            assert group["instance_prompt"] == [row["prompt"]]
            assert Path(row["image"]).name == Path(row["mask"]).name == "image.png"
            assert row["source_image_sha256"] == row["image_sha256"]
            assert row["source_mask_sha256"] != row["mask_sha256"]
            assert set(np.unique(np.asarray(Image.open(row["mask"])))) == {0, 255}
            assert str(output) in row["image"] and "texture.png" not in row["image"]
    assert "training NOT started" in (output / "index.html").read_text()
    with pytest.raises(FileExistsError):
        module.stage(root, output)
    extra = output / "mailbox_texture/images/extra.png"
    Image.new("RGB", (8, 8)).save(extra)
    with pytest.raises(ValueError, match="Unexpected staged files"):
        module.verify_staging(output)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "texture_on_material", "rm_map_link", "wrong_uv", "mask_pair", "changed_source", "changed_rgb"])
def test_reject_bad_renders_before_creating_staging(rendered, tmp_path, failure):
    root, original = rendered
    manifest = copy.deepcopy(original)
    if failure == "missing":
        manifest["records"].pop()
    elif failure == "duplicate":
        manifest["records"][-1] = manifest["records"][0]
    elif failure == "texture_on_material":
        next(row for row in manifest["records"] if row["branch"] == "material")["texture_image_nodes"] = 1
    elif failure == "rm_map_link":
        manifest["records"][0]["roughness_input_linked"] = True
    elif failure == "wrong_uv":
        manifest["records"][0]["texture_projection"] = "GENERATED"
    elif failure == "mask_pair":
        changed = root / "different_mask.png"
        Image.new("L", (512, 512), 255).save(changed)
        row = manifest["records"][0]
        row.update(mask=str(changed), mask_sha256=schema.sha256(changed))
        write_json(row["metadata"], {key: value for key, value in row.items() if key not in ("metadata", "metadata_sha256")})
        row["metadata_sha256"] = schema.sha256(row["metadata"])
    elif failure == "changed_source":
        Path(next(iter(manifest["source_locks"]))).write_bytes(b"changed")
    else:
        Image.new("RGB", (512, 512), "blue").save(root / "image.png")
    write_json(root / "manifest.json", manifest)
    output = tmp_path / "staged"
    with pytest.raises(ValueError):
        module.stage(root, output)
    assert not output.exists()


def test_staging_rejects_changed_training_references(rendered, tmp_path):
    output = tmp_path / "staged"
    module.stage(rendered[0], output)
    (output / "training_references.json").write_text("{}")
    with pytest.raises(ValueError, match="training references"):
        module.verify_staging(output)
