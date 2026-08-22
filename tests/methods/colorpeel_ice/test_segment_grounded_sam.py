import importlib.util
import inspect
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).parents[3]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generate = load_module(
    "generate_for_segmentation",
    ROOT / "scripts" / "methods" / "colorpeel_ice" / "generate.py",
)
segmentation = load_module(
    "segment_grounded_sam",
    ROOT / "scripts" / "methods" / "colorpeel_ice" / "segment_grounded_sam.py",
)


def test_locked_models_thresholds_and_600_transfer_items():
    items = segmentation.transfer_items(generate.build_manifest())

    assert len(items) == 600
    assert segmentation.GROUNDING_DINO_MODEL == "IDEA-Research/grounding-dino-tiny"
    assert segmentation.SAM_MODEL == "facebook/sam-vit-base"
    assert segmentation.BOX_THRESHOLD == 0.25
    assert segmentation.TEXT_THRESHOLD == 0.25
    assert segmentation.MIN_MASK_RATIO == 0.005
    assert segmentation.MAX_MASK_RATIO == 0.90
    assert segmentation.LOCAL_FILES_ONLY is True
    assert {segmentation.query_for_item(item) for item in items} == set(
        segmentation.SEGMENTATION_QUERIES
    )


def test_grounding_dino_uses_transformers_448_box_threshold_keyword():
    source = inspect.getsource(segmentation.GroundedSamSegmenter.__call__)

    assert "box_threshold=BOX_THRESHOLD" in source
    assert "\n            threshold=BOX_THRESHOLD" not in source


def test_segmentation_saves_valid_mask_and_records_failures(tmp_path):
    items = segmentation.transfer_items(generate.build_manifest())[:3]
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    for item in (items[0], items[2]):
        path = image_dir / item["image_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100), (255, 255, 255)).save(path)

    calls = 0

    def fake_segmenter(image, query):
        nonlocal calls
        calls += 1
        mask = np.zeros((image.height, image.width), dtype=bool)
        if calls == 1:
            mask[:10, :10] = True
        else:
            mask[:] = True
        return mask

    statuses = segmentation.run_segmentation(items, image_dir, mask_dir, fake_segmenter)

    assert [status["status"] for status in statuses] == ["ok", "failure", "failure"]
    assert statuses[0]["mask_ratio"] == 0.01
    assert (mask_dir / items[0]["image_path"]).is_file()
    assert statuses[1]["failure_reason"] == "image_missing"
    assert statuses[2]["failure_reason"] == "mask_ratio_out_of_range"


def test_model_cache_failure_writes_600_failures_and_returns_nonzero(tmp_path):
    manifest = tmp_path / "generation.jsonl"
    output = tmp_path / "segmentation_status.jsonl"
    generate.write_manifest(generate.build_manifest(), manifest)

    class MissingCache:
        def __init__(self, device, dtype):
            raise OSError("cache miss")

    original = segmentation.GroundedSamSegmenter
    segmentation.GroundedSamSegmenter = MissingCache
    try:
        result = segmentation.main(
            [
                "--manifest",
                str(manifest),
                "--image-dir",
                str(tmp_path / "images"),
                "--mask-dir",
                str(tmp_path / "masks"),
                "--output",
                str(output),
            ]
        )
    finally:
        segmentation.GroundedSamSegmenter = original

    rows = segmentation.read_jsonl(output)
    assert result == 1
    assert len(rows) == 600
    assert all(row["status"] == "failure" for row in rows)
    assert all(row["failure_reason"].startswith("model_load_error:") for row in rows)
