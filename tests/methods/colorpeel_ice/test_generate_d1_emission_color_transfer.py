from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[3]
PATH = ROOT / "scripts" / "methods" / "colorpeel_ice" / "generate_d1_emission_color_transfer.py"
SPEC = importlib.util.spec_from_file_location("emission_transfer_generate", PATH)
generator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(generator)


def test_generation_matrix_is_fixed_18_images_with_paired_controls():
    value = generator.protocol(ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_emission_color_transfer_orange_generation_protocol_v1.json")
    rows = generator.build_manifest(value)
    assert len(rows) == 18
    assert {row["noun"] for row in rows} == {"bowl", "parrot", "rose"}
    assert {row["seed"] for row in rows} == {42, 43}
    assert {row["condition"] for row in rows} == {"learned_color", "literal_orange_trained", "literal_orange_vanilla"}
    assert all(row["num_inference_steps"] == 100 and row["guidance_scale"] == 6.0 for row in rows)


def test_dry_run_writes_exact_manifest_without_parent_model(tmp_path):
    protocol = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_emission_color_transfer_orange_generation_protocol_v1.json"
    assert generator.main(["--output-dir", str(tmp_path), "--protocol", str(protocol), "--dry-run"]) == 0
    rows = [json.loads(line) for line in (tmp_path / "generation_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 18
    assert "<C*>" in next(row["prompt"] for row in rows if row["condition"] == "learned_color")
