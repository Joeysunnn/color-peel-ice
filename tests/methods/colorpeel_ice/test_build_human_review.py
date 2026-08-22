import csv
import importlib.util
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).parents[3]
SPEC = importlib.util.spec_from_file_location(
    "build_human_review",
    ROOT / "scripts" / "methods" / "colorpeel_ice" / "build_human_review.py",
)
review = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(review)

CYAN_SPEC = importlib.util.spec_from_file_location(
    "generate_cyan_diagnostic_for_review",
    ROOT
    / "scripts"
    / "methods"
    / "colorpeel_ice"
    / "generate_cyan_diagnostic.py",
)
cyan = importlib.util.module_from_spec(CYAN_SPEC)
assert CYAN_SPEC.loader is not None
CYAN_SPEC.loader.exec_module(cyan)


def _make_images(rows, root: Path) -> None:
    prototype = root / "prototype.png"
    prototype.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((512, 512, 3), 127, dtype=np.uint8), mode="RGB").save(
        prototype
    )
    for row in rows:
        path = root / row["image_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prototype, path)


def test_packet_is_deterministically_randomized_and_blinded(tmp_path):
    rows = cyan.build_manifest()
    image_dir = tmp_path / "generated"
    _make_images(rows, image_dir)

    review_rows, key_rows = review.build_packet(
        rows, image_dir, tmp_path / "blinded1", 20260822
    )
    review_rows_2, key_rows_2 = review.build_packet(
        rows, image_dir, tmp_path / "blinded2", 20260822
    )

    assert len(review_rows) == len(key_rows) == 540
    assert [row["pair_id"] for row in key_rows] == [row["pair_id"] for row in key_rows_2]
    assert [row["id"] for row in key_rows] == [row["id"] for row in key_rows_2]
    for row in review_rows:
        assert not any("condition" in field for field in row)
        path = Path(row["image_path"])
        assert path.name == f"{row['review_id']}.png"
        assert not any(
            value in path.name
            for value in ("trained", "vanilla", "cyan", "aqua", "teal", "turquoise")
        )
        assert path.is_file()
    assert {row["model_variant"] for row in key_rows} == {"vanilla", "trained"}
    assert {row["template_family"] for row in key_rows} == {
        "adjective_transfer",
        "training_suffix",
    }


def test_review_csv_exposes_only_empty_rating_fields(tmp_path):
    rows = cyan.build_manifest()
    image_dir = tmp_path / "generated"
    _make_images(rows, image_dir)
    review_rows, _ = review.build_packet(rows, image_dir, tmp_path / "blinded", 7)
    output = tmp_path / "review.csv"
    review.write_csv(output, review.REVIEW_FIELDS, review_rows)
    with output.open(encoding="utf-8", newline="") as stream:
        loaded = list(csv.DictReader(stream))
    assert tuple(loaded[0]) == review.REVIEW_FIELDS
    for field in (
        "color_fidelity_rating",
        "prompt_alignment_rating",
        "visual_quality_rating",
        "invalid_or_artifact",
        "reviewer_id",
        "notes",
    ):
        assert all(row[field] == "" for row in loaded)
