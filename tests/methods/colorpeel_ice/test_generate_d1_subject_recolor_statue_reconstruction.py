import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts" / "methods" / "colorpeel_ice" / "generate_d1_subject_recolor_statue_reconstruction.py"
PROTOCOL = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_gorilla_statue_reconstruction_generation_protocol_v1.json"
ABLATION_PROTOCOL = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_statue_init_kv_ablation_reconstruction_protocol_v1.json"
STEP_DOSE_PROTOCOL = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_statue_init_kvlow_step_dose_reconstruction_protocol_v1.json"
INITIALIZER_STEP_SCREEN_PROTOCOL = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_no_gorilla_initializer_step_screen_reconstruction_protocol_v1.json"
SPEC = importlib.util.spec_from_file_location("statue_reconstruction", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_reconstruction_grid_is_45_training_prompt_images():
    rows = module.build_manifest(json.loads(PROTOCOL.read_text(encoding="utf-8")))
    assert len(rows) == 45
    assert {row["checkpoint_steps"] for row in rows} == {250, 500, 1000}
    assert {row["seed"] for row in rows} == {42, 43, 44, 45, 46}
    assert {row["prompt"] for row in rows} == {
        "a photo of <S*> statue in red color", "a photo of <S*> statue in green color", "a photo of <S*> statue in blue color",
    }


def test_dry_run_accepts_launcher_created_empty_output_directory(tmp_path):
    output = tmp_path / "inference"
    output.mkdir()
    assert module.main(["--protocol", str(PROTOCOL), "--output-dir", str(output), "--dry-run"]) == 0
    assert len((output / "generation_manifest.jsonl").read_text(encoding="utf-8").splitlines()) == 45


def test_checkpoint_ids_keep_same_step_ablation_outputs_distinct():
    protocol = {
        "source_checkpoints": [
            {"id": "kvfull-500", "steps": 500, "model_dir": "/tmp/full"},
            {"id": "kvlow-500", "steps": 500, "model_dir": "/tmp/low"},
        ],
        "sampling": {"seeds": [42], "num_inference_steps": 10, "guidance_scale": 1.0, "expected_image_count": 2},
        "prompts": [{"color": "red", "prompt": "a photo of <S*> gorilla statue in red color"}],
    }
    rows = module.build_manifest(protocol)
    assert {row["id"] for row in rows} == {"kvfull-500-red-seed-42", "kvlow-500-red-seed-42"}
    assert {row["image_path"] for row in rows} == {
        "images/kvfull-500/red-seed-42.png", "images/kvlow-500/red-seed-42.png",
    }


def test_statue_initializer_ablation_reconstruction_grid_is_bound_and_disjoint():
    rows = module.build_manifest(json.loads(ABLATION_PROTOCOL.read_text(encoding="utf-8")))
    assert len(rows) == 30
    assert {row["checkpoint_id"] for row in rows} == {"kvfull-500", "kvlow-500"}
    assert len({row["image_path"] for row in rows}) == 30
    assert {row["prompt"] for row in rows} == {
        "a photo of <S*> gorilla statue in red color",
        "a photo of <S*> gorilla statue in green color",
        "a photo of <S*> gorilla statue in blue color",
    }


def test_low_kv_step_dose_reconstruction_grid_is_bound_and_disjoint():
    rows = module.build_manifest(json.loads(STEP_DOSE_PROTOCOL.read_text(encoding="utf-8")))
    assert len(rows) == 30
    assert {row["checkpoint_id"] for row in rows} == {"kvlow-750", "kvlow-1000"}
    assert len({row["image_path"] for row in rows}) == 30
    assert all("transfer" not in row["id"] for row in rows)


def test_no_gorilla_initializer_step_screen_reconstruction_grid_is_complete_and_disjoint():
    rows = module.build_manifest(json.loads(INITIALIZER_STEP_SCREEN_PROTOCOL.read_text(encoding="utf-8")))
    assert len(rows) == 90
    assert {row["checkpoint_id"] for row in rows} == {
        "statue-500", "statue-750", "gorilla-500", "gorilla-750", "sculpture-500", "sculpture-750",
    }
    assert len({row["image_path"] for row in rows}) == 90
    assert {row["prompt"] for row in rows} == {
        "a photo of <S*> statue in red color",
        "a photo of <S*> statue in green color",
        "a photo of <S*> statue in blue color",
    }
