"""Lightweight CLIPSeg mask staging tests with a deterministic fake predictor."""

import json

import numpy as np
from PIL import Image
import pytest
import torch

from experiments.perfusion_full.prepare_soft_masks import (
    normalized_soft_mask,
    prepare,
    sha256,
)


def test_normalization_uses_soft_probabilities_without_threshold():
    logits = torch.tensor([[-2.0, -1.0], [0.0, 2.0]])
    mask, stats = normalized_soft_mask(logits, (2, 2))
    assert mask.shape == (2, 2)
    assert mask.max() == pytest.approx(1.0)
    assert 0 < mask.min() < mask[0, 1] < mask[1, 0] < mask[1, 1]
    assert stats["raw_probability_max"] < 1.0


def test_prepare_keeps_sources_read_only_and_one_directory_per_row(tmp_path):
    concepts = []
    source_hashes = {}
    for index in range(2):
        image_dir = tmp_path / f"images_{index}"
        binary_dir = tmp_path / f"binary_{index}"
        image_dir.mkdir()
        binary_dir.mkdir()
        image = image_dir / "sample.png"
        binary = binary_dir / "sample.png"
        Image.new("RGB", (4, 4), color=(index * 30, 40, 50)).save(image)
        Image.fromarray(np.array([[0, 0, 255, 255]] * 4, dtype=np.uint8)).save(binary)
        source_hashes[image] = sha256(image)
        source_hashes[binary] = sha256(binary)
        concepts.append({"instance_data_dir": str(image_dir), "instance_mask_dir": str(binary_dir),
                         "instance_prompt": ["a photo of <S*> mailbox"]})
    concepts_path = tmp_path / "concepts.json"
    concepts_path.write_text(json.dumps(concepts), encoding="utf-8")

    def fake_predict(image, query):
        assert query == "mailbox"
        assert image.size == (4, 4)
        return torch.tensor([[-2.0, -1.0], [0.0, 2.0]])

    output = tmp_path / "derived"
    provenance = prepare(concepts_path, "mailbox", output, fake_predict, {"model": "fake"})
    derived = json.loads((output / "derived_concepts.json").read_text(encoding="utf-8"))
    ledger = [json.loads(line) for line in (output / "soft_mask_manifest.jsonl").read_text().splitlines()]
    assert provenance["concept_rows"] == provenance["images"] == 2
    assert len({row["instance_mask_dir"] for row in derived}) == 2
    for index, row in enumerate(derived):
        assert row["instance_data_dir"] == concepts[index]["instance_data_dir"]
        assert row["instance_prompt"] == concepts[index]["instance_prompt"]
        mask_path = output / "masks" / f"row_{index:04d}" / "sample.png"
        with Image.open(mask_path) as image:
            values = np.unique(np.asarray(image))
        assert len(values) > 2
        assert int(values[-1]) == 255
        assert ledger[index]["binary_audit"]["soft_iou_with_binary"] is not None
    assert all(sha256(path) == digest for path, digest in source_hashes.items())
    with pytest.raises(FileExistsError):
        prepare(concepts_path, "mailbox", output, fake_predict, {"model": "fake"})
