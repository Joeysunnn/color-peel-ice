"""Verify immutable preview inputs and physically ordered linear compositing."""

import itertools
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from experiments.subject_albedo_pbr_c import pbr_data as module


@pytest.fixture
def preview(tmp_path):
    root = tmp_path / "preview"
    source = root / "source"
    source.mkdir(parents=True)
    external = tmp_path / "old_source.txt"
    external.write_text("immutable old SuperMat provenance")
    inside = np.zeros((512, 512), np.uint8)
    inside[40:470, 60:450] = 255
    Image.fromarray(inside).save(source / "object_mask.png")
    Image.new("RGB", (512, 512), (60, 80, 100)).save(source / "original.jpg")
    Image.new("RGB", (512, 512), (100, 70, 40)).save(source / "albedo.png")
    Image.new("RGBA", (512, 512), (60, 80, 100, 255)).save(source / "input.png")
    for channel, value in (("metallic", 10), ("roughness", 90)):
        Image.new("L", (512, 512), value).save(source / f"{channel}.png")
    maps = {channel: {} for channel in ("albedo", "metallic", "roughness")}
    for number, color in enumerate(module.COLORS):
        path = root / f"{color}.png"
        Image.new("RGB", (512, 512), (30 + number * 30, 60, 90)).save(path)
        maps["albedo"][color] = {"path": str(path), "sha256": module.sha256(path)}
    for channel, factors, outside in (("metallic", module.METALLIC, 10), ("roughness", module.ROUGHNESS, 90)):
        for factor, value in factors.items():
            pixels = np.full((512, 512), outside, np.float32) / 255
            pixels[inside == 255] = value
            path = root / f"{channel}_{factor}.npy"
            np.save(path, pixels, allow_pickle=False)
            display = path.with_suffix(".png")
            Image.fromarray(np.rint(pixels * 255).astype(np.uint8)).save(display)
            maps[channel][factor] = {"path": str(path), "sha256": module.sha256(path), "value": value,
                "preview": str(display), "preview_sha256": module.sha256(display), "qc": "accepted_parameter_edit"}
    rows = [{"id": f"{color}__{metallic}__{roughness}", "color": color, "hue_degrees": module.COLORS[color],
             "metallic_state": metallic, "metallic": module.METALLIC[metallic], "roughness_state": roughness,
             "roughness": module.ROUGHNESS[roughness], "maps": {"albedo": maps["albedo"][color],
             "metallic": maps["metallic"][metallic], "roughness": maps["roughness"][roughness]},
             "qc": "accepted_parameter_edit", "render_status": "not_rendered", "rgb_image": None,
             "training_eligible": False}
            for color, metallic, roughness in itertools.product(module.COLORS, module.METALLIC, module.ROUGHNESS)]
    (root / "status.json").write_text(json.dumps({"status": "parameter_preview_verified_renderer_missing",
        "parameter_configurations": 30, "accepted_parameter_edits": 30, "rejected_parameter_edits": 0,
        "rendered_rgb": 0, "training_eligible_images": 0, "training_started": False}))
    (root / "protocol.json").write_text(json.dumps({"schema": "subject_albedo_pbr_c/v1", "colors": module.COLORS,
        "metallic": module.METALLIC, "roughness": module.ROUGHNESS, "training_enabled": False}))
    (root / "source_locks.json").write_text(json.dumps({str(external): module.sha256(external)}))
    for name in ("source_receipt.json", "albedo_qc.json", "renderer_readiness.json"):
        (root / name).write_text("{}")
    def reseal():
        (root / "parameter_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (root / "output_hashes.json").write_text(json.dumps({str(path): module.sha256(path)
            for path in root.rglob("*") if path.is_file() and path.name not in ("status.json", "output_hashes.json")}))
    reseal()
    return SimpleNamespace(root=root, rows=rows, maps=maps, mask=inside, external=external, reseal=reseal)


def test_preview_verifies_complete_factorial_and_freezes_all_sources_without_writes(preview):
    before = {str(path): path.read_bytes() for path in preview.root.rglob("*") if path.is_file()}
    rows, locks = module.verify_preview(preview.root)
    assert rows == preview.rows and len(rows) == 30
    assert set(before) <= set(locks)
    assert str(preview.external) in locks
    assert all(module.sha256(path) == value for path, value in locks.items())
    assert {str(path): path.read_bytes() for path in preview.root.rglob("*") if path.is_file()} == before
    assert preview.external.read_text() == "immutable old SuperMat provenance"


@pytest.mark.parametrize("kind", ("source", "map", "row", "factor", "npy"))
def test_preview_rejects_changed_sources_maps_or_false_factorial(preview, kind):
    if kind == "source":
        preview.external.write_text("changed")
    elif kind == "map":
        with open(preview.maps["albedo"]["pink"]["path"], "ab") as handle:
            handle.write(b"changed")
    elif kind == "row":
        preview.rows[0]["roughness"] = 0.3
        preview.reseal()
    elif kind == "factor":
        preview.rows[0]["maps"]["albedo"] = preview.maps["albedo"]["blue"]
        preview.reseal()
    else:
        record = preview.maps["metallic"]["nonmetallic"]
        values = np.load(record["path"])
        values[preview.mask == 255] = 0.3
        np.save(record["path"], values, allow_pickle=False)
        record["sha256"] = module.sha256(record["path"])
        preview.reseal()
    with pytest.raises(ValueError):
        module.verify_preview(preview.root)


def render_arrays():
    original = np.full((30, 30, 3), 255, np.uint8)
    mask = np.zeros((30, 30), np.uint8)
    mask[4:26, 4:26] = 255
    rgba = np.zeros((30, 30, 4), np.float32)
    rgba[mask == 255, :3] = 0.25
    rgba[mask == 255, 3] = 1
    return rgba, original, mask


def test_composite_blends_premultiplied_linear_edge_then_srgb_and_preserves_outside():
    rgba, original, mask = render_arrays()
    rgba[4, 10] = (0.125, 0.125, 0.125, 0.5)
    before = rgba.copy(), original.copy(), mask.copy()
    image, qc = module.composite(rgba, original, mask)
    assert np.all(image[10, 10] == 137)  # Linear .25 -> sRGB .537, not uint8 .25*255.
    assert np.all(image[4, 10] == 207)  # Premult .125 + white*(1-.5) = linear .625.
    assert np.array_equal(image[mask == 0], original[mask == 0])
    assert qc["core_coverage_fraction"] == 1 and qc["core_pixels"] == 400
    assert qc["outside_mask_changed_pixels"] == 0 and qc["clipped_pixel_fraction"] == 0
    for after, saved in zip((rgba, original, mask), before):
        assert np.array_equal(after, saved)


@pytest.mark.parametrize("holes,passes", ((2, True), (3, False)))
def test_core_coverage_enforces_995_per_thousand_without_source_foreground_fill(holes, passes):
    rgba, original, mask = render_arrays()
    rgba[10, 10:10 + holes] = 0
    if not passes:
        with pytest.raises(ValueError, match="below 99.5%"):
            module.composite(rgba, original, mask)
    else:
        _, qc = module.composite(rgba, original, mask)
        assert qc["core_coverage_fraction"] == 0.995


def test_hdr_clipping_is_reported_without_changing_other_pixels_by_normalization():
    rgba, original, mask = render_arrays()
    baseline, _ = module.composite(rgba, original, mask)
    rgba[10, 10, :3] = (2.0, -0.1, 0.25)
    image, qc = module.composite(rgba, original, mask)
    assert np.array_equal(image[10, 10], [255, 0, 137])
    assert np.array_equal(image[11, 11], baseline[11, 11])
    assert qc["clipped_pixel_fraction"] == pytest.approx(1 / 484)
    assert qc["clipped_channel_fraction"] == pytest.approx(2 / (484 * 3))


def test_fixed_background_plate_removes_source_foreground_color_from_transparent_edge_only():
    rgba, original, mask = render_arrays()
    original[mask == 255] = (255, 50, 0)
    rgba[4, 10] = (0.125, 0.125, 0.125, 0.5)
    plate = original.copy()
    plate[mask == 255] = (80, 80, 80)
    before = plate.copy()
    without_plate, _ = module.composite(rgba, original, mask)
    image, qc = module.composite(rgba, original, mask, background=plate)
    assert without_plate[4, 10, 0] > without_plate[4, 10, 2]
    assert len(set(image[4, 10])) == 1  # Neutral rendered edge no longer receives old orange foreground.
    assert np.array_equal(image[10, 10], without_plate[10, 10])  # Opaque core never exposes the plate.
    assert np.array_equal(image[mask == 0], original[mask == 0])
    assert np.array_equal(plate, before)
    assert qc["background_source"] == "fixed_background_plate" and qc["outside_mask_changed_pixels"] == 0


@pytest.mark.parametrize("failure", ("outside", "dtype", "shape"))
def test_background_plate_cannot_change_outside_pixels_or_input_contract(failure):
    rgba, original, mask = render_arrays()
    plate = original.copy()
    if failure == "outside":
        plate[0, 0] = (10, 20, 30)
    elif failure == "dtype":
        plate = plate.astype(np.float32)
    else:
        plate = plate[:-1]
    with pytest.raises(ValueError, match="background plate"):
        module.composite(rgba, original, mask, background=plate)


@pytest.mark.parametrize("failure", ("rgba_dtype", "rgb_dtype", "mask_shape", "mask_binary", "empty", "nan", "alpha", "straight_rgb", "no_core", "no_coverage"))
def test_invalid_numeric_inputs_and_missing_geometry_are_rejected(failure):
    rgba, original, mask = render_arrays()
    if failure == "rgba_dtype":
        rgba = rgba.astype(np.float64)
    elif failure == "rgb_dtype":
        original = original.astype(np.float32)
    elif failure == "mask_shape":
        mask = mask[:-1]
    elif failure == "mask_binary":
        mask[10, 10] = 128
    elif failure == "empty":
        mask[:] = 0
    elif failure == "nan":
        rgba[10, 10, 0] = np.nan
    elif failure == "alpha":
        rgba[10, 10, 3] = 1.1
    elif failure == "straight_rgb":
        rgba[0, 0] = (1, 1, 1, 0)
    elif failure == "no_coverage":
        rgba[:] = 0
    else:
        mask[:] = 0
        mask[5, 5:20] = 255
    with pytest.raises(ValueError):
        module.composite(rgba, original, mask)


def test_read_exr_enables_codec_before_lazy_import_and_converts_bgra_without_gamma(tmp_path, monkeypatch):
    path = tmp_path / "combined.exr"
    path.write_bytes(b"mock EXR")
    pixels = np.array([[[0.1, 0.2, 2.0, 0.5]]], np.float32)
    def read(actual_path, flags):
        assert actual_path == str(path) and flags == -1
        assert os.environ["OPENCV_IO_ENABLE_OPENEXR"] == "1"
        return pixels
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(IMREAD_UNCHANGED=-1, imread=read))
    rgba = module.read_exr(path)
    assert rgba.dtype == np.float32
    assert np.array_equal(rgba, pixels[:, :, [2, 1, 0, 3]])
    assert rgba[0, 0, 0] == 2.0  # Preserve HDR linear values; clipping is a later explicit display step.


@pytest.mark.parametrize("pixels", (None, np.zeros((2, 2, 3), np.float32), np.zeros((2, 2, 4), np.float64),
                                   np.full((2, 2, 4), np.nan, np.float32)))
def test_read_exr_rejects_missing_channels_nonfinite_or_non_float32(tmp_path, monkeypatch, pixels):
    path = tmp_path / "invalid.exr"
    path.write_bytes(b"mock EXR")
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(IMREAD_UNCHANGED=-1, imread=lambda *args: pixels))
    with pytest.raises(ValueError, match="four finite scene-linear float32"):
        module.read_exr(path)
