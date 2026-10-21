"""Protect six-case semantics and explicit background-restoration provenance."""

from pathlib import Path

import numpy as np
import pytest

from experiments.subject_marble_c_v1 import run


def test_phase1_grid_has_correct_official_sign_and_no_extra_cases():
    protocol = run.read_json(run.PROTOCOL)
    rows = run.grid(protocol)
    assert len(rows) == len({r["id"] for r in rows}) == 6
    assert {(r["metallic_control"], r["roughness_control"]) for r in rows} == {
        (m, r) for m in (20, -20) for r in (-1, 0, 1)}
    assert all(r["metallic_control"] == -20 for r in rows if r["metallic_state"] == "high")
    assert all(r["training_eligible"] is False and "orange color" in r["caption"] for r in rows)


def test_unverified_alternative_control_grid_is_rejected():
    protocol = run.read_json(run.PROTOCOL)
    protocol["metallic"] = {"low": 0, "high": 1}
    with pytest.raises(ValueError, match="signed"):
        run.grid(protocol)


def test_background_restore_keeps_exact_edit_foreground_and_unmodified_source():
    original = np.full((10, 10, 3), 33, dtype=np.uint8)
    edited = np.full_like(original, 99)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:8, 2:6] = 255
    result = run.restore_background(original, edited, mask)
    assert np.all(result[mask == 0] == 33) and np.all(result[mask > 0] == 99)
    assert np.all(original == 33) and np.all(edited == 99)


def test_existing_result_is_refused_before_model_or_source_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "resolve_run_dir", lambda p: p)
    output = tmp_path / "old_result"
    output.mkdir()
    marker = output / "old.png"
    marker.write_bytes(b"old experiment")
    with pytest.raises(ValueError, match="fresh"):
        run.run(tmp_path / "no_assets", output, 2)
    assert marker.read_bytes() == b"old experiment" and list(output.iterdir()) == [marker]


def test_modified_source_receipt_is_rejected(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"original")
    frozen = {str(source): run.sha256(source)}
    run.check_locks(frozen)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen"):
        run.check_locks(frozen)
