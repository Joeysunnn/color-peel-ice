import importlib.util
import json
from collections import Counter
from pathlib import Path

from PIL import Image
import pytest


ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts" / "methods" / "colorpeel_ice" / "generate.py"
SPEC = importlib.util.spec_from_file_location("clevr_inference", SCRIPT)
clevr_inference = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(clevr_inference)


def test_manifest_has_fixed_900_item_protocol():
    manifest = clevr_inference.build_manifest()

    assert len(manifest) == 900
    assert len({item["id"] for item in manifest}) == 900
    assert Counter(item["category"] for item in manifest) == {
        "grid": 180,
        "subject_only": 60,
        "color_only": 60,
        "transfer": 600,
    }
    assert set(item["seed"] for item in manifest) == set(range(42, 62))
    assert all(item["num_inference_steps"] == 100 for item in manifest)
    assert all(item["guidance_scale"] == 6.0 for item in manifest)


def test_manifest_has_all_grid_and_transfer_prompts():
    manifest = clevr_inference.build_manifest()
    grid = [item for item in manifest if item["category"] == "grid"]
    transfers = [item for item in manifest if item["category"] == "transfer"]

    assert {
        (item["subject_label"], item["color_label"]) for item in grid
    } == {
        (shape, color)
        for shape in ("cube", "sphere", "cylinder")
        for color in ("red", "cyan", "gray")
    }
    assert len({item["prompt"] for item in grid}) == 9
    assert len({item["prompt"] for item in transfers}) == 30
    assert {item["transfer_template_index"] for item in transfers} == set(range(10))


def test_dry_run_writes_manifest_without_model(tmp_path):
    clevr_inference.main(["--output-dir", str(tmp_path), "--dry-run"])

    manifest_path = tmp_path / "generation_manifest.jsonl"
    assert manifest_path.is_file()
    assert len(manifest_path.read_text(encoding="utf-8").splitlines()) == 900


def test_safety_disable_requires_acknowledgement_and_is_recorded(tmp_path):
    with pytest.raises(SystemExit):
        clevr_inference.parse_args(
            ["--output-dir", str(tmp_path), "--dry-run", "--disable-safety-checker"]
        )
    with pytest.raises(SystemExit):
        clevr_inference.parse_args(
            ["--output-dir", str(tmp_path), "--dry-run", "--acknowledge-safety-risk"]
        )

    clevr_inference.main(
        [
            "--output-dir",
            str(tmp_path),
            "--dry-run",
            "--disable-safety-checker",
            "--acknowledge-safety-risk",
        ]
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "generation_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 900
    assert all(row["safety_checker_disabled"] for row in rows)
    assert all(row["safety_risk_acknowledged"] for row in rows)


def test_skip_existing_only_skips_decodable_images(tmp_path):
    items = clevr_inference.build_manifest()[:4]
    valid_path = tmp_path / items[0]["image_path"]
    corrupt_path = tmp_path / items[1]["image_path"]
    wrong_size_path = tmp_path / items[2]["image_path"]
    valid_path.parent.mkdir(parents=True)
    Image.new("RGB", (512, 512), (255, 0, 0)).save(valid_path)
    corrupt_path.write_bytes(b"not a png")
    Image.new("RGB", (2, 2), (255, 0, 0)).save(wrong_size_path)

    pending = clevr_inference.pending_items(items, tmp_path, skip_existing=True)

    assert [item["id"] for item in pending] == [
        items[1]["id"],
        items[2]["id"],
        items[3]["id"],
    ]
    assert clevr_inference.pending_items(items, tmp_path, skip_existing=False) == items
