"""Checks cover edits hidden by compositing and limits of descriptive metrics."""

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from experiments.subject_marble_c_v1.quality import evaluate, write_review


@pytest.fixture
def source():
    image = np.full((32, 32, 3), 50, dtype=np.uint8)
    image[7:25, 7:25] = [160, 50, 30]
    image[10:22, 16:19] = [40, 10, 10]
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[5:27, 5:27] = 255
    return image, mask


def test_background_compositing_does_not_hide_raw_drift(source):
    image, mask = source
    raw = image.copy()
    raw[mask == 0] = 150
    qc = evaluate(image, raw, image, mask)
    assert qc["raw_background"]["changed_pixels"] == int((mask == 0).sum())
    assert qc["raw_background"]["mean_absolute_error_255"] == 100
    assert qc["final_background"]["changed_pixels"] == 0
    assert "raw_background_changed_before_compositing" in qc["flags"]
    assert qc["training_eligible"] is False


def test_identity_metrics_are_descriptive_and_do_not_approve(source):
    image, mask = source
    qc = evaluate(image, image, image, mask)
    assert qc["structure_proxy"]["source_edge_recall"] == 1
    assert qc["structure_proxy"]["raw_edge_precision"] == 1
    assert qc["color"]["hue_median_degrees"] == 0
    assert qc["manual_review_required"] is True
    assert qc["flags"] == ["manual_review_required"]


def test_hue_wrap_and_achromatic_exclusion(source):
    _, mask = source
    hsv = np.empty((32, 32, 3), dtype=np.uint8)
    hsv[:] = [254, 255, 200]
    original = np.asarray(Image.fromarray(hsv, "HSV").convert("RGB"))
    hsv[:] = [1, 255, 200]
    final = np.asarray(Image.fromarray(hsv, "HSV").convert("RGB"))
    qc = evaluate(original, final, final, mask)
    assert qc["color"]["hue_p95_degrees"] < 10
    gray = np.full_like(final, 150)
    qc = evaluate(original, gray, gray, mask)
    assert qc["color"]["chromatic_pixels"] == 0
    assert qc["color"]["hue_median_degrees"] is None


def test_edge_change_is_flagged_without_claiming_geometry_failure(source):
    image, mask = source
    final = np.full_like(image, 50)
    qc = evaluate(image, final, final, mask)
    assert qc["structure_proxy"]["source_edge_recall"] == 0
    assert "inspect_structure_or_reflection_change" in qc["flags"]


def test_restored_final_cannot_hide_raw_object_displacement():
    image = np.full((64, 64, 3), 50, dtype=np.uint8)
    image[18:32, 18:32] = [160, 50, 30]
    image[22:28, 24:26] = [40, 10, 10]
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[17:33, 17:33] = 255
    raw = np.full_like(image, 50)
    raw[48:62, 48:62] = image[18:32, 18:32]
    qc = evaluate(image, raw, image.copy(), mask)
    assert qc["final_background"]["changed_pixels"] == 0
    assert qc["color"]["hue_median_degrees"] == 0
    assert qc["structure_proxy"]["source_edge_recall"] == 0
    assert qc["structure_proxy"]["mask_boundary_band_edge_recall"] == 0
    assert qc["structure_proxy"]["evaluated_image"] == "raw_resized_before_background_composite"
    assert "inspect_structure_or_reflection_change" in qc["flags"]


@pytest.mark.parametrize("failure", ["soft_mask", "float_rgb", "wrong_size", "no_background"])
def test_bad_inputs_fail(source, failure):
    image, mask = source
    raw = image.copy()
    if failure == "soft_mask":
        mask[0, 0] = 128
    if failure == "float_rgb":
        raw = raw.astype(float)
    if failure == "wrong_size":
        raw = raw[:-1]
    if failure == "no_background":
        mask[:] = 255
    with pytest.raises(ValueError):
        evaluate(image, raw, image, mask)


def test_review_exact_grid_and_refuses_overwrite(tmp_path, source):
    image, mask = source
    Image.fromarray(image).save(tmp_path / "original.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    rows = []
    for metallic in range(2):
        for roughness in range(3):
            name = f"m{metallic}r{roughness}.png"
            Image.fromarray(image).save(tmp_path / name)
            rows.append({"id": name, "image": name, "raw_image": name,
                         "metallic_state": str(metallic), "metallic_control": metallic,
                         "roughness_state": str(roughness), "roughness_control": roughness,
                         "caption": "a photo of <S*> mailbox", "qc": evaluate(image, image, image, mask)})
    with pytest.raises(ValueError):
        write_review(tmp_path, "original.png", "mask.png", rows[:-1])
    result = write_review(tmp_path, "original.png", "mask.png", rows[::-1])
    assert Image.open(result["contact_sheet"]).size == (1536, 1810)
    assert "&lt;S*&gt;" in Path(result["index"]).read_text(encoding="utf-8")
    assert len(list(tmp_path.glob("m*r*.png"))) == 6
    with pytest.raises(FileExistsError):
        write_review(tmp_path, "original.png", "mask.png", rows)
