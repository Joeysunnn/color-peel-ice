"""Check C source adaptation without OpenCV, model inference or source mutation."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from experiments.subject_albedo_pbr_c import source


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    assets = root / source.ASSETS
    assets.mkdir(parents=True)
    original = assets / "source_mailbox.jpg"
    Image.new("RGB", (512, 512), (70, 90, 110)).save(original, format="PNG")
    pixels = np.zeros((512, 512), dtype=np.uint8)
    pixels[90:440, 70:470] = 255
    mask = assets / "object_mask.png"
    Image.fromarray(pixels).save(mask)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    paths = {name: upstream / f"{name}.png" for name in ("input", "albedo", "roughness", "metallic", "mask")}
    rgba = Image.open(original).convert("RGBA")
    rgba.putalpha(Image.fromarray(pixels))
    rgba.save(paths["input"])
    Image.new("RGB", (512, 512), (120, 100, 80)).save(paths["albedo"])
    Image.new("L", (512, 512), 90).save(paths["roughness"])
    Image.new("L", (512, 512), 10).save(paths["metallic"])
    Image.new("L", (512, 512), 255).save(paths["mask"])
    for name in ("canonical", "upstream_canonical", "supermat_manifest"):
        paths[name] = upstream / f"{name}.json"
        paths[name].write_text(json.dumps({"seed": 42, "image_size": 512, "checkpoint_sha256": "frozen-checkpoint"}))
    config = root / source.CONFIG
    config.parent.mkdir(parents=True)
    def refresh():
        config.write_text(json.dumps({"source_image_sha256": source.sha256(original),
            "original_object_mask_sha256": source.sha256(mask), "supermat_input_sha256": source.sha256(paths["input"])}))
    refresh()
    locks = {str(path): source.sha256(path) for path in paths.values()}
    records = {"mailbox": {"paths": paths, "canonical": {"selected_region": "small_painted_lid"}},
               "metal_spoon": {}, "wood_spoon": {}}
    calls = []
    def inspect(fullmaps):
        calls.append(fullmaps)
        return records, {str(path): source.sha256(path) for path in paths.values()}
    monkeypatch.setattr(source, "ROOT", root)
    monkeypatch.setitem(sys.modules, "experiments.natural_texture_material_v1.extract", SimpleNamespace(inspect_sources=inspect))
    return SimpleNamespace(root=root, original=original, mask=mask, paths=paths, records=records,
                           locks=locks, calls=calls, config=config, refresh=refresh, fullmaps=tmp_path / "fullmaps")


def test_reuses_original_full_maps_and_original_alpha_instead_of_measurement_region(frozen):
    before = {str(path): path.read_bytes() for path in frozen.root.rglob("*") if path.is_file()}
    paths, locks, record = source.load_source(frozen.fullmaps)
    assert frozen.calls == [frozen.fullmaps]
    assert set(paths) == {"original", "mask", "input", "albedo", "roughness", "metallic"}
    assert paths["mask"] == frozen.mask and paths["mask"] != frozen.paths["mask"]
    assert all(paths[name] == frozen.paths[name] for name in ("input", "albedo", "roughness", "metallic"))
    assert frozen.locks.items() <= locks.items()
    assert set(locks) - set(frozen.locks) == {str(frozen.original), str(frozen.mask), str(frozen.config)}
    assert record["extraction"] == "reuse_existing_supermat_no_rerun"
    assert record["reviewed_rm_region"] == "small_painted_lid"
    assert record["foreground_pixels"] == 350 * 400
    assert record["experiment_b_mask_difference"]["used_here"] is False
    assert record["experiment_b_mask_difference"]["added_bottom_pixels"] == 1450
    assert record["supermat"]["seed"] == 42
    assert {str(path): path.read_bytes() for path in frozen.root.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("name", ("original", "mask", "input"))
def test_rejects_changes_against_frozen_protocol(frozen, name):
    path = frozen.original if name == "original" else frozen.mask if name == "mask" else frozen.paths[name]
    with path.open("ab") as handle:
        handle.write(b"changed bytes")
    with pytest.raises(ValueError, match="frozen original mailbox"):
        source.load_source(frozen.fullmaps)


@pytest.mark.parametrize("kind", ("rgb", "alpha"))
def test_even_matching_file_receipts_cannot_hide_rgba_pixel_or_mask_mismatch(frozen, kind):
    rgba = np.asarray(Image.open(frozen.paths["input"])).copy()
    rgba[100, 100, 0 if kind == "rgb" else 3] = 0
    Image.fromarray(rgba).save(frozen.paths["input"])
    frozen.refresh()
    with pytest.raises(ValueError, match="RGB/alpha differ"):
        source.load_source(frozen.fullmaps)


@pytest.mark.parametrize("name,mode,size,value", [
    ("albedo", "L", (512, 512), 90), ("albedo", "RGB", (256, 512), (90, 90, 90)),
    ("roughness", "RGB", (512, 512), (10, 50, 90)),
    ("metallic", "L", (128, 512), 90),
])
def test_rejects_wrong_map_channels_and_dimensions(frozen, name, mode, size, value):
    Image.new(mode, size, value).save(frozen.paths[name])
    with pytest.raises(ValueError, match="512px"):
        source.load_source(frozen.fullmaps)


def test_scalar_rgb_maps_with_equal_channels_are_valid(frozen):
    Image.new("RGB", (512, 512), (90, 90, 90)).save(frozen.paths["roughness"])
    paths, _, _ = source.load_source(frozen.fullmaps)
    assert paths["roughness"] == frozen.paths["roughness"]


def test_upstream_changed_lock_is_rejected_after_resolving(frozen, monkeypatch):
    def inspect(fullmaps):
        return frozen.records, {**frozen.locks, str(frozen.original): "incorrect"}
    monkeypatch.setitem(sys.modules, "experiments.natural_texture_material_v1.extract", SimpleNamespace(inspect_sources=inspect))
    # The adapter's new locks cover original assets; alter an independently upstream-bound map instead.
    frozen.paths["albedo"].touch()
    Image.new("RGB", (512, 512), (30, 60, 90)).save(frozen.paths["albedo"])
    with pytest.raises(ValueError, match="locks changed"):
        source.load_source(frozen.fullmaps)
