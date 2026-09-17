import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts" / "methods" / "colorpeel_ice" / "generate_d1_subject_recolor_statue_reconstruction.py"
PROTOCOL = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_gorilla_statue_reconstruction_generation_protocol_v1.json"
ABLATION_PROTOCOL = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_statue_init_kv_ablation_reconstruction_protocol_v1.json"
STEP_DOSE_PROTOCOL = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_statue_init_kvlow_step_dose_reconstruction_protocol_v1.json"
INITIALIZER_STEP_SCREEN_PROTOCOL = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_no_gorilla_initializer_step_screen_reconstruction_protocol_v1.json"
INFERENCE_STACK_ABLATION_PROTOCOL = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_inference_stack_ablation_protocol_v1.json"
UNSEEN_TRANSFER_PROTOCOL = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_v2_v6_unseen_transfer_legacy_protocol_v1.json"
MAILBOX_RECONSTRUCTION_PROTOCOLS = [
    ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_mailbox_category_750_reconstruction_protocol_v1.json",
    ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_mailbox_token_first_750_reconstruction_protocol_v1.json",
]
MAILBOX_1000_RECONSTRUCTION_PROTOCOLS = [
    ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_mailbox_category_1000_reconstruction_protocol_v1.json",
    ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_mailbox_token_first_1000_reconstruction_protocol_v1.json",
]
MAILBOX_TRANSFER_PROTOCOL = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_mailbox_750_transfer_protocol_v1.json"
MAILBOX_FREE_TRANSFER_PROTOCOL = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_mailbox_free_750_transfer_protocol_v1.json"
MAILBOX_1000_TRANSFER_PROTOCOLS = [
    ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / name
    for name in (
        "d1_subject_recolor_mailbox_category_1000_transfer_protocol_v1.json",
        "d1_subject_recolor_mailbox_category_free_1000_transfer_protocol_v1.json",
        "d1_subject_recolor_mailbox_token_first_1000_transfer_protocol_v1.json",
        "d1_subject_recolor_mailbox_token_first_free_1000_transfer_protocol_v1.json",
    )
]
LEGACY_REGENERATION_CONFIGS = [
    ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_statue_init_kv_ablation_reconstruction_legacy_generate.yaml",
    ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_statue_init_kvlow_step_dose_reconstruction_legacy_generate.yaml",
    ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_no_gorilla_initializer_step_screen_reconstruction_legacy_generate.yaml",
]
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


def test_checkpoint_hash_fields_reject_replaced_weight_or_manifest(tmp_path):
    run_dir, model_dir = tmp_path / "run", tmp_path / "run" / "checkpoints"
    model_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("manifest", encoding="utf-8")
    for name in ("<S*>.bin", module.WEIGHTS, "embedding_update_audit.json", "training_metrics.jsonl"):
        (model_dir / name).write_bytes(name.encode("utf-8"))
    protocol = {"source_checkpoints": [{
        "model_dir": str(model_dir), "run_dir": str(run_dir),
        "model_sha256": module.sha256(model_dir / module.WEIGHTS),
        "run_manifest_sha256": module.sha256(run_dir / "manifest.json"),
    }], "forbidden_token_artifacts": ["<C*>.bin"]}
    module.validate_model_dir(model_dir, protocol)
    (model_dir / module.WEIGHTS).write_bytes(b"replaced")
    with pytest.raises(ValueError, match="weights do not match"):
        module.validate_model_dir(model_dir, protocol)


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


def test_inference_stack_ablation_is_a_2_by_2_checkpoint_runtime_comparison():
    protocol = json.loads(INFERENCE_STACK_ABLATION_PROTOCOL.read_text(encoding="utf-8"))
    rows = module.build_manifest(protocol)
    assert len(rows) == 30
    assert {row["checkpoint_id"] for row in rows} == {"historical-750", "v6-gorilla-750"}
    assert {row["checkpoint_steps"] for row in rows} == {750}
    assert len({row["image_path"] for row in rows}) == 30
    assert protocol["prohibitions"] == [
        "no training", "no checkpoint modification", "no transfer prompts", "no shared output directories",
    ]


def test_legacy_regeneration_configs_bind_old_runtime_and_expected_grids():
    expected_counts = [30, 30, 90]
    for config, expected_count in zip(LEGACY_REGENERATION_CONFIGS, expected_counts):
        text = config.read_text(encoding="utf-8")
        assert "COLORPEEL_INFERENCE_RUNTIME: \"colorpeel017\"" in text
        assert "/envs/colorpeel017/bin/python" in text
        assert f"expected_image_count: {expected_count}" in text
        assert "training: forbidden" in text
        assert "transfer: forbidden" in text


def test_unseen_v2_v6_transfer_grid_excludes_tested_and_unbound_checkpoints():
    protocol = json.loads(UNSEEN_TRANSFER_PROTOCOL.read_text(encoding="utf-8"))
    rows = module.build_manifest(protocol)
    assert len(rows) == 1020
    assert {row["checkpoint_id"] for row in rows} == {
        "v2-250", "v2-500", "v3-kvfull-500", "v3-kvlow-500", "v4-kvlow-750", "v4-kvlow-1000",
        "v6-statue-500", "v6-statue-750", "v6-gorilla-500", "v6-gorilla-750", "v6-sculpture-500", "v6-sculpture-750",
    }
    assert "v2_750_1000" in protocol["excluded"]
    assert "v5" in protocol["excluded"]
    assert len({row["image_path"] for row in rows}) == 1020


def test_mailbox_reconstruction_protocols_are_exact_training_prompt_grids():
    for protocol_path in MAILBOX_RECONSTRUCTION_PROTOCOLS:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        rows = module.build_manifest(protocol)
        assert len(rows) == 25
        assert {row["checkpoint_steps"] for row in rows} == {750}
        assert {row["seed"] for row in rows} == {42, 43, 44, 45, 46}
        assert {row["color"] for row in rows} == {"seen_red", "seen_green", "seen_cyan", "seen_blue", "seen_magenta"}
        assert all("transfer" not in row["color"] for row in rows)


def test_mailbox_1000_reconstruction_protocols_bind_the_completed_full_kv_runs():
    for protocol_path in MAILBOX_1000_RECONSTRUCTION_PROTOCOLS:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        rows = module.build_manifest(protocol)
        assert len(rows) == 25
        assert {row["checkpoint_steps"] for row in rows} == {1000}
        assert len({row["image_path"] for row in rows}) == 25
        assert all(len(item["model_sha256"]) == 64 for item in protocol["source_checkpoints"])
        assert protocol["next_step"] == "human reconstruction review before transfer"


def test_mailbox_transfer_grid_includes_all_user_requested_hard_compositions():
    protocol = json.loads(MAILBOX_TRANSFER_PROTOCOL.read_text(encoding="utf-8"))
    rows = module.build_manifest(protocol)
    assert len(rows) == 230
    assert {row["checkpoint_id"] for row in rows} == {"category-750", "token-first-750"}
    assert {item["group"] for item in protocol["prompts"]} == {"unseen_color", "context", "viewpoint", "composition", "hard_compositional"}
    hard = {item["prompt"] for item in protocol["prompts"] if item["group"] == "hard_compositional"}
    assert hard == {
        "a photo of <S*> mailbox in purple color on a plain white background",
        "a side view photo of <S*> mailbox in orange color on a city street",
        "a photo of <S*> mailbox in pink color in a snowy environment",
        "a close-up photo of <S*> mailbox in white color in front of a house",
    }


def test_mailbox_free_transfer_grid_removes_the_ordinary_category_word_only():
    protocol = json.loads(MAILBOX_FREE_TRANSFER_PROTOCOL.read_text(encoding="utf-8"))
    rows = module.build_manifest(protocol)
    assert len(rows) == 230
    assert all("mailbox" not in item["prompt"].lower() for item in protocol["prompts"])
    assert {item["group"] for item in protocol["prompts"]} == {"unseen_color", "context", "viewpoint", "composition", "hard_compositional"}
    assert {row["checkpoint_id"] for row in rows} == {"category-750", "token-first-750"}


def test_mailbox_1000_transfer_protocols_are_single_checkpoint_fully_pinned_grids():
    for protocol_path in MAILBOX_1000_TRANSFER_PROTOCOLS:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        rows = module.build_manifest(protocol)
        assert len(rows) == 115
        assert len(protocol["source_checkpoints"]) == 1
        assert {row["checkpoint_steps"] for row in rows} == {1000}
        assert len({row["image_path"] for row in rows}) == 115
        assert len(protocol["source_checkpoints"][0]["model_sha256"]) == 64
        is_free = "_free_" in protocol_path.name
        assert all(("mailbox" not in row["prompt"].lower()) == is_free for row in rows)
