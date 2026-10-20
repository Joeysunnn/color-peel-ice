"""Accepted C assets are consumed read-only and reject altered training inputs."""

import json
from pathlib import Path

from PIL import Image
import pytest

from experiments.subject_albedo_pbr_c import data, stage_pbr
from tests.methods.colorpeel_ice.test_subject_experiment_c_stage_pbr import candidates


@pytest.fixture
def accepted(candidates):
    root, review, output, _, _, _, _ = candidates
    stage_pbr.run(root, review, output)
    return output


def snapshot(root):
    return {str(path): data.sha256(path) for path in root.rglob("*") if path.is_file()}


def reseal(dataset, name):
    audit = data.read_json(dataset / "dataset_audit.json")
    audit["files"][name] = data.sha256(dataset / name)
    stage_pbr.write_json(dataset / "dataset_audit.json", audit)


def test_accepted_dataset_reverifies_without_writing_any_files(accepted):
    before = snapshot(accepted.parent)
    audit = data.verify_dataset(accepted)
    assert audit["rows"] == audit["training_eligible_images"] == 30
    assert audit["training_started"] is False
    assert snapshot(accepted.parent) == before


@pytest.mark.parametrize("change", ("rgb", "extra_image", "extra_mask", "concept", "caption", "factor",
    "ineligible", "missing_row", "review", "audit", "original_source", "albedo_source", "metadata_hash"))
def test_changed_accepted_data_is_rejected_without_new_results(accepted, change):
    rows = data._jsonl(accepted / "training_assets_manifest.jsonl")
    first = rows[0]
    if change == "rgb":
        image = Image.open(first["image"]).copy()
        image.putpixel((0, 0), (250, 0, 0))
        image.save(first["image"])
    elif change in ("extra_image", "extra_mask"):
        field = "image" if change == "extra_image" else "mask"
        (Path(first[field]).parent / "accidental.png").write_bytes(Path(first[field]).read_bytes())
    elif change == "concept":
        concepts = data.read_json(accepted / "concepts.json")
        concepts[0]["instance_prompt"] = ["a photo of <S*> mailbox"]
        stage_pbr.write_json(accepted / "concepts.json", concepts)
        reseal(accepted, "concepts.json")
    elif change in ("caption", "factor", "ineligible", "missing_row"):
        if change == "caption":
            first["caption"] = "a photo of <S*> <M*> mailbox"
        elif change == "factor":
            first["metallic"] = .5
        elif change == "ineligible":
            first["training_eligible"] = False
        else:
            rows.pop()
        (accepted / "training_assets_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        reseal(accepted, "training_assets_manifest.jsonl")
    elif change == "review":
        review = data.read_json(accepted / "visual_review.json")
        review["decisions"][0]["accepted"] = False
        stage_pbr.write_json(accepted / "visual_review.json", review)
        reseal(accepted, "visual_review.json")
        audit = data.read_json(accepted / "dataset_audit.json")
        audit["visual_review_sha256"] = data.sha256(accepted / "visual_review.json")
        stage_pbr.write_json(accepted / "dataset_audit.json", audit)
    elif change == "audit":
        audit = data.read_json(accepted / "dataset_audit.json")
        audit["training_started"] = True
        stage_pbr.write_json(accepted / "dataset_audit.json", audit)
    elif change in ("original_source", "albedo_source"):
        source = (Path(first["source_image"]) if change == "original_source"
                  else Path(first["maps"]["albedo"]["path"]))
        source.write_bytes(b"altered immutable source")
    else:
        with (accepted / "concepts.json").open("a") as stream:
            stream.write("\n")
    before = snapshot(accepted.parent)
    with pytest.raises(ValueError):
        data.verify_dataset(accepted)
    assert snapshot(accepted.parent) == before
