"""Verify reviewed natural-map provenance and exact mask-aware extraction."""

import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None
    # The existing source resolver imports OpenCV for its unrelated CLI. Source
    # provenance tests need no inpainting; retain real-OpenCV cases for the server.
    sys.modules["cv2"] = types.ModuleType("cv2")
    from experiments.natural_texture_material_v1 import extract
    del sys.modules["cv2"]
else:
    from experiments.natural_texture_material_v1 import extract


def saved(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


@pytest.fixture
def source(tmp_path):
    old, full = tmp_path / "old", tmp_path / "full"
    records, inference = {}, []
    for name in extract.MATERIALS:
        input_name = "spoon_positive_control" if name == "metal_spoon" else name
        maps = old / "supermat_outputs" / input_name
        maps.mkdir(parents=True)
        yy, xx = np.indices((512, 512))
        gray = ((xx + yy) % 180 + 35).astype("uint8")
        Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2)).save(maps / "albedo.png")
        Image.new("L", (512, 512), 75).save(maps / "roughness.png")
        Image.new("L", (512, 512), 2).save(maps / "metallic.png")
        valid = np.zeros((512, 512), dtype="uint8")
        valid[50:450, 80:400] = 255
        valid[50:100, 80:140] = 0
        mask_path = old / "masks" / f"{name}.png"
        mask_path.parent.mkdir(exist_ok=True)
        Image.fromarray(valid).save(mask_path)
        candidate = {"name": "reviewed", "mask": str(mask_path), "mask_path": str(mask_path),
            "measurements": {key: {"map_sha256": extract.data.sha256(maps / f"{key}.png")}
                             for key in ("roughness", "metallic")}}
        canonical = {"selected_region": "reviewed", "roughness": 75 / 255, "metallic": 2 / 255,
                     "semantic_status": "semantic_gloss_conflict" if name == "wood_spoon" else "accepted",
                     "candidates": [candidate]}
        canonical_path = old / "canonical" / f"{name}.json"
        if name != "wood_spoon":
            upstream = old / "canonical" / f"original_{name}.json"
            saved(upstream, canonical)
            canonical.update(source_canonical=str(upstream), source_canonical_sha256=extract.data.sha256(upstream))
        saved(canonical_path, canonical)
        # source_paths uses a dedicated wood output directory, matching the fixture's name.
        input_path = old / "supermat_inputs" / f"{input_name}.png"
        input_path.parent.mkdir(exist_ok=True)
        rgb = Image.open(maps / "albedo.png").convert("RGBA")
        rgb.putalpha(Image.fromarray(valid))
        rgb.save(input_path)
        receipt = {key: {"original_path": str((maps / f"{key}.png").resolve()),
                         "original_sha256": extract.data.sha256(maps / f"{key}.png")}
                   for key in ("albedo", "roughness", "metallic")}
        records[name] = {"source_canonical_path": str(canonical_path.resolve()),
            "source_canonical_sha256": extract.data.sha256(canonical_path),
            "mask_path": str(mask_path.resolve()), "mask_sha256": extract.data.sha256(mask_path),
            "selected_region": "reviewed", "maps": receipt}
        inference.append({"input": str(input_path), "input_sha256": extract.data.sha256(input_path),
            "maps": {key: {"sha256": value["original_sha256"]} for key, value in receipt.items()}})
    saved(old / "supermat_outputs/inference_manifest.json", {"status": "complete", "exit_code": 0,
          "image_size": 512, "seed": 42, "outputs": inference})
    saved(full / "maps/manifest.json", {"status": "complete", "previous_run": str(old), "materials": records})
    return full


def test_sources_are_receipt_bound_and_statistics_use_only_reviewed_mask(source):
    records, locks = extract.inspect_sources(source)
    assert set(records) == set(extract.MATERIALS)
    assert records["mailbox"]["distributions"]["roughness"]["median"] == 75 / 255
    assert records["wood_spoon"]["canonical"]["semantic_status"] == "semantic_gloss_conflict"
    assert any(path.endswith("inference_manifest.json") for path in locks)
    path = records["metal_spoon"]["paths"]["albedo"]
    Image.new("RGB", (512, 512), "white").save(path)
    with pytest.raises(ValueError, match="map changed"):
        extract.inspect_sources(source)


def test_changed_input_and_region_receipts_are_rejected(source):
    records, _ = extract.inspect_sources(source)
    Image.new("RGBA", (512, 512), (10, 20, 30, 255)).save(records["mailbox"]["paths"]["input"])
    with pytest.raises(ValueError, match="input receipt"):
        extract.inspect_sources(source)


@pytest.mark.skipif(cv2 is None, reason="real OpenCV inpainting verified in server environment")
def test_texture_completion_fills_holes_without_changing_valid_gray():
    yy, xx = np.indices((80, 100))
    pixels = ((xx + yy) % 128 + 64).astype("uint8")
    mask = np.zeros((80, 100), dtype="uint8")
    mask[5:75, 10:90] = 255
    mask[5:25, 10:30] = 0
    patch, receipt = extract.texture_patch(Image.fromarray(pixels), Image.fromarray(mask))
    valid = mask[5:75, 10:90] >= 128
    assert patch.size == (80, 70)
    np.testing.assert_array_equal(np.asarray(patch)[valid], pixels[5:75, 10:90][valid])
    assert receipt["inpainted_pixels"] == 400
    assert receipt["crop_xyxy"] == [10, 5, 90, 75]


def test_interior_patch_is_all_valid_and_preserves_original_gray_without_resampling():
    yy, xx = np.indices((18, 26))
    pixels = (yy * 10 + xx).astype("uint8")
    mask = np.zeros_like(pixels)
    mask[2:16, 3:23] = 255
    mask[2:7, 3:8] = 0
    patch, receipt = extract.interior_texture_patch(Image.fromarray(pixels), Image.fromarray(mask))
    left, top, right, bottom = receipt["crop_xyxy"]
    assert (mask[top:bottom, left:right] == 255).all()
    np.testing.assert_array_equal(np.asarray(patch), pixels[top:bottom, left:right])
    assert patch.size == (right - left, bottom - top)
    assert receipt["pixel_count"] == (right - left) * (bottom - top)
    assert receipt["retained_fraction_of_reviewed_region"] == receipt["pixel_count"] / (mask > 0).sum()
    assert receipt["no_inpainting"] is True


def test_interior_rectangle_matches_exhaustive_maximum_with_deterministic_ties():
    random = np.random.default_rng(42)
    masks = [np.ones((4, 5), dtype=bool),
             np.array([[1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [0, 0, 0, 0, 0]], dtype=bool)]
    masks += [random.random((4, 5)) > .35 for _ in range(50)]
    for valid in masks:
        candidates = []
        for top in range(valid.shape[0]):
            for left in range(valid.shape[1]):
                for bottom in range(top + 1, valid.shape[0] + 1):
                    for right in range(left + 1, valid.shape[1] + 1):
                        if valid[top:bottom, left:right].all():
                            candidates.append((-(bottom - top) * (right - left), top, left, bottom, right))
        _, top, left, bottom, right = min(candidates)
        image, mask = Image.new("L", valid.shape[::-1], 99), Image.fromarray(valid.astype("uint8") * 255)
        _, receipt = extract.interior_texture_patch(image, mask)
        assert receipt["crop_xyxy"] == [left, top, right, bottom]
        assert receipt == extract.interior_texture_patch(image, mask)[1]
    with pytest.raises(ValueError, match="nonempty"):
        extract.interior_texture_patch(Image.new("L", (3, 4)), Image.new("L", (3, 4)))


@pytest.mark.skipif(cv2 is None, reason="full extraction requires real OpenCV on server")
def test_run_writes_separate_scalar_materials_and_immutable_receipts(source, tmp_path, monkeypatch):
    monkeypatch.setenv("COLORPEEL_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(extract, "git_info", lambda: {"commit": "fake", "branch": "codex/test"})
    output = tmp_path / "extracted"
    result = extract.run(source, output)
    assert result["status"] == "complete"
    assert result["schema"] == "natural_texture_material/extraction_v2"
    assert (output / "overview.png").is_file() and (output / "index.html").is_file()
    assert Image.open(output / "mailbox/gray_full.png").size == (512, 512)
    assert (output / "mailbox/texture_inpaint_audit.png").is_file()
    record = extract.read(output / "mailbox/extraction.json")
    assert record["texture_completion"]["use"] == "audit_only_not_for_training"
    assert record["texture_statistics"]["patch_sampling"] == record["texture_sampling"]
    left, top, right, bottom = record["texture_sampling"]["crop_xyxy"]
    original = np.asarray(Image.open(output / "mailbox/gray_full.png"))
    np.testing.assert_array_equal(np.asarray(Image.open(output / "mailbox/texture.png")), original[top:bottom, left:right])
    assert record["texture_sampling"]["no_inpainting"] is True
    material = extract.read(output / "mailbox/material.json")
    assert material["roughness"] == 75 / 255
    assert "texture" not in material and "albedo" not in material
    assert material["distributions"]["roughness"]["iqr"] == 0
    for path, digest in extract.read(output / "source_locks.json").items():
        assert extract.data.sha256(Path(path)) == digest
    with pytest.raises(FileExistsError):
        extract.run(source, output)


def test_changed_region_and_canonical_values_are_rejected(source):
    manifest_path = source / "maps/manifest.json"
    manifest = extract.read(manifest_path)
    canonical_path = Path(manifest["materials"]["wood_spoon"]["source_canonical_path"])
    canonical = extract.read(canonical_path)
    canonical["roughness"] = .99
    saved(canonical_path, canonical)
    manifest["materials"]["wood_spoon"]["source_canonical_sha256"] = extract.data.sha256(canonical_path)
    saved(manifest_path, manifest)
    with pytest.raises(ValueError, match="R/M disagrees"):
        extract.inspect_sources(source)


def test_frozen_supermat_receipt_hash_is_enforced(source):
    manifest_path = source / "maps/manifest.json"
    manifest = extract.read(manifest_path)
    canonical_path = Path(manifest["materials"]["wood_spoon"]["source_canonical_path"])
    canonical = extract.read(canonical_path)
    receipt = canonical_path.parent.parent / "supermat_outputs/inference_manifest.json"
    canonical["supermat_manifest_sha256"] = extract.data.sha256(receipt)
    saved(canonical_path, canonical)
    manifest["materials"]["wood_spoon"]["source_canonical_sha256"] = extract.data.sha256(canonical_path)
    saved(manifest_path, manifest)
    extract.inspect_sources(source)
    changed = extract.read(receipt)
    changed["unapproved_change"] = True
    saved(receipt, changed)
    with pytest.raises(ValueError, match="Frozen SuperMat"):
        extract.inspect_sources(source)


def test_existing_output_rejected_before_git_or_source_reads(tmp_path, monkeypatch):
    monkeypatch.setenv("COLORPEEL_OUTPUT_ROOT", str(tmp_path))
    output = tmp_path / "already_there"
    output.mkdir()
    with pytest.raises(FileExistsError):
        extract.run(tmp_path / "does_not_exist", output)
