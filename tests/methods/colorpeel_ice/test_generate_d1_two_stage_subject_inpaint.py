import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts" / "methods" / "colorpeel_ice" / "generate_d1_two_stage_subject_inpaint.py"
PROTOCOL = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_two_stage_subject_inpaint_context_transfer_protocol_v1.json"
SPEC = importlib.util.spec_from_file_location("two_stage_subject_inpaint", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_context_transfer_grid_is_complete_and_keeps_subject_out_of_stage_one_prompts():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rows = module.build_manifest(protocol)
    assert len(rows) == 45
    assert {row["group"] for row in rows} == {"context", "hard_compositional"}
    assert all("<S*>" not in row["background_prompt"] and "<S*>" in row["subject_prompt"] for row in rows)
    assert len({row["image_path"] for row in rows}) == 45


def test_mask_has_one_binary_subject_region_at_the_declared_foreground_location():
    mask = module.make_mask()
    assert mask.mode == "L" and mask.size == (512, 512)
    assert set(mask.getdata()) == {0, 255}
    assert mask.getpixel((256, 350)) == 255
    assert mask.getpixel((10, 10)) == 0


def test_dry_run_writes_the_bound_45_row_manifest(tmp_path):
    output = tmp_path / "inference"
    assert module.main(["--protocol", str(PROTOCOL), "--output-dir", str(output), "--dry-run"]) == 0
    assert len((output / "generation_manifest.jsonl").read_text(encoding="utf-8").splitlines()) == 45
