"""Test independent parameter interventions without pretending to render RGB."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from experiments.subject_albedo_pbr_c import prepare


@pytest.fixture
def source(tmp_path, monkeypatch):
    original = tmp_path / "old_extraction"
    original.mkdir()
    yy, xx = np.indices((24, 24))
    albedo = np.stack((150 + xx * 2, 65 + yy * 2, 35 + xx), axis=-1).astype(np.uint8)
    mask = np.zeros((24, 24), np.uint8)
    mask[3:21, 3:21] = 255
    paths = {}
    for name, pixels in {"original": albedo, "mask": mask,
                          "input": np.dstack((albedo, mask)), "albedo": albedo,
                          "metallic": (xx * 7).astype(np.uint8),
                          "roughness": (yy * 7).astype(np.uint8)}.items():
        path = original / ("original.jpg" if name == "original" else f"{name}.png")
        Image.fromarray(pixels).save(path)
        paths[name] = path
    locks = {str(path): prepare.sha256(path) for path in paths.values()}
    monkeypatch.setattr(prepare, "load_source", lambda _: (paths, locks.copy(), {"extraction": "reused"}))
    monkeypatch.setattr(prepare, "git_info", lambda: {"commit": "a" * 40, "branch": "codex/subject-experiment-c"})
    monkeypatch.setenv("COLORPEEL_OUTPUT_ROOT", str(tmp_path))
    return original, tmp_path / "new_c", paths, locks


def test_prepare_creates_independent_factorial_without_training_or_rendered_rgb(source):
    original, output, paths, locks = source
    result = prepare.run(original, output)
    assert result["status"] == "parameter_preview_verified_renderer_missing"
    assert result["parameter_configurations"] == result["accepted_parameter_edits"] == 30
    assert result["rejected_parameter_edits"] == 0
    assert not result["training_started"]
    assert result["rendered_rgb"] == result["training_eligible_images"] == 0
    assert not (output / "concepts.json").exists() and not (output / "images").exists()
    rows = [json.loads(line) for line in (output / "parameter_manifest.jsonl").read_text().splitlines()]
    assert len({(r["color"], r["metallic"], r["roughness"]) for r in rows}) == 30
    assert all(r["rgb_image"] is None and not r["training_eligible"] and r["render_status"] == "not_rendered" for r in rows)
    # Changing M/R reuses the exact same A; each map varies only its own factor.
    for channel, factor in (("albedo", "color"), ("metallic", "metallic_state"), ("roughness", "roughness_state")):
        groups = {}
        for row in rows:
            groups.setdefault(row[factor], set()).add(row["maps"][channel]["sha256"])
        assert all(len(hashes) == 1 for hashes in groups.values())
        assert len(set.union(*groups.values())) == len(groups)
    for row in rows:
        for record in row["maps"].values():
            assert prepare.sha256(Path(record["path"])) == record["sha256"]
    for path, digest in locks.items():
        assert prepare.sha256(Path(path)) == digest
    assert prepare.read_json(output / "renderer_readiness.json")["training_ready"] is False
    assert {"extraction_grid.png", "albedo_grid.png", "parameter_grid.png"} <= set(result["grids"])
    assert all((output / name).is_file() for name in result["grids"])
    for path, digest in prepare.read_json(output / "output_hashes.json").items():
        assert prepare.sha256(Path(path)) == digest


def test_qc_failure_keeps_audit_maps_but_rejects_all_six_affected_combinations(source, monkeypatch):
    original, output, _, _ = source
    edit = prepare.edit_albedo

    def fail_one(albedo, mask, hue, protocol):
        pixels, qc = edit(albedo, mask, hue, protocol)
        if hue == 0:
            qc.update(status="rejected", reasons=["test failed lightness check"])
        return pixels, qc

    monkeypatch.setattr(prepare, "edit_albedo", fail_one)
    result = prepare.run(original, output)
    assert result["status"] == "parameter_preview_has_rejected_edits_renderer_missing"
    assert result["accepted_parameter_edits"] == 24 and result["rejected_parameter_edits"] == 6
    rows = [json.loads(line) for line in (output / "parameter_manifest.jsonl").read_text().splitlines()]
    assert all(r["qc"] == "rejected_parameter_edit" for r in rows if r["color"] == "pink")
    assert all(not r["training_eligible"] for r in rows)


def test_existing_output_is_never_changed(source):
    original, output, _, _ = source
    output.mkdir()
    marker = output / "old_result.png"
    marker.write_bytes(b"immutable result")
    with pytest.raises(ValueError, match="fresh external"):
        prepare.run(original, output)
    assert marker.read_bytes() == b"immutable result" and len(list(output.iterdir())) == 1


@pytest.mark.parametrize("value", (0.0, 0.15, 0.5, 0.85, 1.0))
def test_authoritative_parameter_maps_preserve_exact_float_intervention_and_outside(tmp_path, value):
    source = np.arange(36, dtype=np.uint8).reshape(6, 6)
    mask = np.zeros((6, 6), np.uint8)
    mask[1:5, 1:5] = 255
    record = prepare.parameter_map(source, mask, value, tmp_path / "edited")
    saved = np.load(record["path"], allow_pickle=False)
    assert saved.dtype == np.float32
    assert np.all(saved[mask == 255] == np.float32(value))
    assert np.array_equal(saved[mask == 0], (source.astype(np.float32) / 255)[mask == 0])
    assert record["qc"] == "accepted_parameter_edit"
    if value == 0.5:
        assert record["preview_foreground_value"] == 128 / 255
        assert saved[2, 2] == 0.5  # The preview must not replace the authoritative value.


def test_hue_edit_cannot_claim_color_success_for_achromatic_source():
    pixels = np.full((8, 8, 3), 128, np.uint8)
    mask = np.zeros((8, 8), np.uint8)
    mask[1:7, 1:7] = 255
    image, qc = prepare.edit_albedo(pixels, mask, 120, prepare.read_json(prepare.PROTOCOL)["qc"])
    assert np.array_equal(image[mask == 0], pixels[mask == 0])
    assert qc["status"] == "rejected" and qc["chromatic_support_fraction"] == 0
    assert qc["chromatic_hue_error_p95"] is None
