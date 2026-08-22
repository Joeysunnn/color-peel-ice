import importlib.util
from collections import Counter
from pathlib import Path


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
