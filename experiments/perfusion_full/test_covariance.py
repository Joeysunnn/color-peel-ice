"""Focused checks for the independent full-Perfusion covariance artifact."""

import hashlib
import json

import pytest
import torch

from experiments.perfusion_full.covariance import (
    content_token_positions,
    inverse_with_minimal_ridge,
    sample_distinct_captions,
)


def test_hash_sample_is_distinct_order_independent_and_source_hashed(tmp_path):
    captions = ["red mailbox", "blue sphere", "red mailbox", "green cube", "yellow car"]
    first = tmp_path / "captions.txt"
    first.write_text("\n".join(captions) + "\n", encoding="utf-8")
    second = tmp_path / "reversed.txt"
    second.write_text("\n".join(reversed(captions)) + "\n", encoding="utf-8")
    chosen, info = sample_distinct_captions(first, 3, 42)
    other, other_info = sample_distinct_captions(second, 3, 42)
    assert chosen == other
    assert len(chosen) == len(set(chosen)) == 3
    assert info["source_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert info["source_sha256"] != other_info["source_sha256"]
    assert info["selected_captions_sha256"] == other_info["selected_captions_sha256"]


def test_jsonl_caption_fields_and_insufficient_unique_rejected(tmp_path):
    path = tmp_path / "captions.jsonl"
    rows = [{"caption": "first"}, {"text": "second"}, {"TEXT": " second "}]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    selected, info = sample_distinct_captions(path, 2, 1)
    assert set(selected) == {"first", "second"}
    assert info["selected_distinct_count"] == 2
    with pytest.raises(ValueError, match="fewer than 3 distinct"):
        sample_distinct_captions(path, 3, 1)


def test_content_position_never_selects_special_or_padding():
    ids = torch.tensor([[1, 11, 12, 2, 2], [1, 21, 2, 2, 2]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    positions = content_token_positions(ids, mask, {1, 2}, ["first", "second"], 42)
    assert int(positions[0]) in {1, 2}
    assert int(positions[1]) == 1
    assert torch.equal(positions, content_token_positions(ids, mask, {1, 2}, ["first", "second"], 42))
    with pytest.raises(ValueError, match="no content token"):
        content_token_positions(ids[:1, [0, 3, 4]], mask[:1, [0, 3, 4]], {1, 2}, ["empty"], 42)


def test_inverse_uses_no_ridge_when_well_conditioned():
    covariance = torch.tensor([[4.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    inverse, diagnostics = inverse_with_minimal_ridge(covariance)
    assert diagnostics["ridge"] == 0
    assert torch.allclose(covariance @ inverse, torch.eye(2, dtype=torch.float64), atol=1e-12)


def test_singular_covariance_uses_recorded_minimal_ridge():
    covariance = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    inverse, diagnostics = inverse_with_minimal_ridge(covariance)
    assert diagnostics["ridge"] == pytest.approx(2e-8, rel=1e-6)
    adjusted = covariance + diagnostics["ridge"] * torch.eye(2, dtype=torch.float64)
    assert torch.allclose(adjusted @ inverse, torch.eye(2, dtype=torch.float64), atol=1e-7)
    assert diagnostics["regularized_condition_number"] <= 100000001
