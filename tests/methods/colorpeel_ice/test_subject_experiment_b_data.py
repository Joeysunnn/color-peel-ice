"""Exercise byte-preserving paired staging with real PNGs and frozen legacy receipts."""

import copy
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.subject_multicolor_multimaterial_b import data


@pytest.fixture
def source(tmp_path, monkeypatch):
    original, reviewed = tmp_path / "original", tmp_path / "reviewed"
    category, aligned = original / "category", reviewed / "caption_aligned_staged"
    (reviewed / "preview").mkdir(parents=True)
    raw_concepts, raw_records, ledger, concepts, preview = [], [], [], [], []
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:400, 100:400] = 255
    expected = {}
    for i, (old_color, color) in enumerate(zip(data.SOURCE_COLORS, data.COLORS)):
        raw_image, raw_mask = category / old_color / "images/image.png", category / old_color / "masks/image.png"
        raw_image.parent.mkdir(parents=True)
        raw_mask.parent.mkdir()
        rgb = np.full((512, 512, 3), 55, dtype=np.uint8)
        rgb[mask == 255] = (i * 20 + 80, 130, 160)
        Image.fromarray(rgb).save(raw_image)
        Image.fromarray(mask).save(raw_mask)
        expected[old_color] = data.sha256(raw_image)
        raw_concepts.append({"instance_prompt": [f"a photo of <S*> mailbox in {old_color} color"],
                             "instance_data_dir": str(raw_image.parent), "instance_mask_dir": str(raw_mask.parent)})
        raw_records.append({"name": old_color, "image_sha256": data.sha256(raw_image),
                            "mask_sha256": data.sha256(raw_mask)})
        for old_material in ("metal", "matte"):
            image = aligned / old_color / old_material / "images/image.png"
            binary = aligned / old_color / old_material / "masks/image.png"
            image.parent.mkdir(parents=True)
            binary.parent.mkdir()
            current = rgb.copy()
            if old_material == "matte":
                current[mask == 255] -= 10
            Image.fromarray(current).save(image)
            Image.fromarray(mask).save(binary)
            caption = f"a photo of <S*> mailbox in {color} color made of " + (
                "glossy painted metal" if old_material == "metal" else "matte plastic")
            concepts.append({"instance_prompt": [caption], "instance_data_dir": str(image.parent),
                             "instance_mask_dir": str(binary.parent)})
            ledger.append({"color": old_color, "material": old_material, "image": str(image),
                           "mask": str(binary), "image_sha256": data.sha256(image),
                           "mask_sha256": data.sha256(binary)})
            if old_material == "matte":
                candidate = reviewed / "preview" / f"{old_color}_matte.png"
                candidate.write_bytes(image.read_bytes())
                preview.append({"color": old_color, "matte_image": candidate.name})
    data.write_json(category / "concepts.json", raw_concepts)
    data.write_json(category / "staging_manifest.json", {"records": raw_records})
    data.write_json(aligned / "balanced_concepts.json", concepts)
    (aligned / "training_assets_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ledger), encoding="utf-8")
    data.write_json(reviewed / "preview/preview_manifest.json", {"records": preview})
    data.write_json(reviewed / "preview_review.json", {
        "verdict": "pass", "reviewer": "fixture", "reviewed_at": "2026-10-20",
        "preview_manifest_sha256": data.sha256(reviewed / "preview/preview_manifest.json")})
    old_protocol, caption_protocol, training_protocol, reference = [tmp_path / name for name in (
        "old_protocol.json", "caption_protocol.json", "training_protocol.json", "reference.png")]
    old_protocol.write_text("{}", encoding="utf-8")
    caption_protocol.write_text("{}", encoding="utf-8")
    reference.write_bytes(b"reference fixture")
    data.write_json(training_protocol, {"training_data": {"expected_image_sha256": expected},
                                       "source_pilot": {"repaired_mask_sha256": data.sha256(raw_mask)}})
    data.write_json(aligned / "staging_provenance.json", {
        "balanced_concepts_sha256": data.sha256(aligned / "balanced_concepts.json"),
        "training_assets_manifest_sha256": data.sha256(aligned / "training_assets_manifest.jsonl"),
        "caption_protocol_sha256": data.sha256(caption_protocol)})
    specification = copy.deepcopy(data.read_json(data.PROTOCOL))
    specification.update(legacy_protocol=str(old_protocol), caption_protocol=str(caption_protocol))
    for key, path in {
            "legacy_protocol_sha256": old_protocol, "caption_protocol_sha256": caption_protocol,
            "original_concepts_sha256": category / "concepts.json",
            "original_manifest_sha256": category / "staging_manifest.json",
            "review_sha256": reviewed / "preview_review.json",
            "preview_manifest_sha256": reviewed / "preview/preview_manifest.json",
            "reviewed_provenance_sha256": aligned / "staging_provenance.json",
            "reviewed_concepts_sha256": aligned / "balanced_concepts.json",
            "reviewed_manifest_sha256": aligned / "training_assets_manifest.jsonl"}.items():
        specification[key] = data.sha256(path)
    protocol_path = tmp_path / "protocol.json"
    data.write_json(protocol_path, specification)
    monkeypatch.setattr(data, "PROTOCOL", protocol_path)
    monkeypatch.setattr(data.legacy, "protocol", lambda _: {
        "source_training_protocol": str(training_protocol), "transform": {"reference_path": str(reference)}})
    calls = []
    monkeypatch.setattr(data.legacy, "validated_preview", lambda *args: calls.append(args))
    return original, reviewed, tmp_path / "new_dataset", calls


def test_staging_preserves_all_source_bytes_and_corrects_only_labels(source):
    original, reviewed, output, calls = source
    before = {str(p): data.sha256(p) for root in (original, reviewed) for p in root.rglob("*") if p.is_file()}
    rows = data.stage(original, reviewed, output)
    assert len(rows) == 10
    assert [(r["color"], r["material"]) for r in rows] == [
        (color, material) for color in data.COLORS for material in data.MATERIALS]
    assert rows[0]["source_color"] == "red" and rows[8]["source_color"] == "magenta"
    assert rows[0]["caption"] == "a photo of <S*> mailbox in pink color with a metallic surface"
    assert rows[9]["caption"] == "a photo of <S*> mailbox in purple color with a matte surface"
    for row in rows:
        assert Path(row["image"]).read_bytes() == Path(row["source_image"]).read_bytes()
        assert not Path(row["image"]).is_symlink()
    assert before == {str(p): data.sha256(p) for root in (original, reviewed) for p in root.rglob("*") if p.is_file()}
    audit = data.verify_dataset(output)
    assert audit["status"] == "verified" and audit["rows"] == 10
    assert audit["outside_mask_changed_pixels"] == 0
    assert len(calls) == 3


@pytest.mark.parametrize("mutation", ["caption", "extra_image", "rgb", "source_image", "lock", "audit"])
def test_verification_rejects_changed_supervision_or_source(source, mutation):
    original, reviewed, output, _ = source
    rows = data.stage(original, reviewed, output)
    if mutation == "caption":
        concepts = data.read_json(output / "concepts.json")
        concepts[0]["instance_prompt"] = ["a photo of <S*> mailbox with a matte surface"]
        data.write_json(output / "concepts.json", concepts)
    elif mutation == "extra_image":
        Path(rows[0]["image"]).with_name("unmanifested.png").write_bytes(Path(rows[0]["image"]).read_bytes())
    elif mutation == "rgb":
        Path(rows[0]["image"]).write_bytes(b"changed")
    elif mutation == "source_image":
        Path(rows[0]["source_image"]).write_bytes(b"changed")
    elif mutation == "lock":
        locks = data.read_json(output / "source_locks.json")
        locks["files"][str(data.PROTOCOL)] = "bad"
        data.write_json(output / "source_locks.json", locks)
    else:
        audit = data.read_json(output / "dataset_audit.json")
        audit["outside_mask_changed_pixels"] = 1
        data.write_json(output / "dataset_audit.json", audit)
    with pytest.raises(ValueError):
        data.verify_dataset(output)


def test_rejects_existing_output_or_nested_old_source(source):
    original, reviewed, output, _ = source
    with pytest.raises(ValueError, match="outside"):
        data.stage(original, reviewed, reviewed / "new_dataset")
    output.mkdir()
    with pytest.raises(FileExistsError):
        data.stage(original, reviewed, output)


def test_rejects_invalid_review_before_creating_output(source):
    original, reviewed, output, _ = source
    review = reviewed / "preview_review.json"
    value = data.read_json(review)
    value["verdict"] = "pending"
    data.write_json(review, value)
    with pytest.raises(ValueError, match="accepted matte preview"):
        data.stage(original, reviewed, output)
    assert not output.exists()


def test_rejects_broken_legacy_pairing(source):
    original, reviewed, output, _ = source
    ledger = reviewed / "caption_aligned_staged/training_assets_manifest.jsonl"
    rows = data._jsonl(ledger)
    rows[0]["material"] = "matte"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="staging differs"):
        data.stage(original, reviewed, output)
    assert not output.exists()


def test_protocol_digest_is_independent_of_json_line_endings(source):
    before = data.protocol_sha256()
    data.PROTOCOL.write_bytes(data.PROTOCOL.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert data.protocol_sha256() == before

def test_frozen_source_hashes_are_complete_sha256_values():
    specification = data.read_json(data.PROTOCOL)
    for key, value in specification.items():
        if key.endswith("_sha256"):
            assert re.fullmatch("[0-9a-f]{64}", value), key
