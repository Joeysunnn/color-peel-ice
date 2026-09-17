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
JOINT_RECONSTRUCTION_PROTOCOL = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_mailbox_orange_shared_kv_joint_1000_reconstruction_protocol_v1.json"
JOINT_COMPOSITION_PROTOCOL = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_mailbox_orange_shared_kv_joint_1000_composition_protocol_v1.json"
JOINT_SUBJECT_TRANSFER_PROTOCOL = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_mailbox_orange_shared_kv_joint_1000_subject_transfer_protocol_v1.json"
JOINT_COLOR_TRANSFER_PROTOCOL = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_mailbox_orange_shared_kv_joint_1000_color_transfer_protocol_v1.json"
JOINT_1500_PROTOCOLS = [
    ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / name
    for name in (
        "d1_mailbox_orange_shared_kv_joint_1500_reconstruction_protocol_v1.json",
        "d1_mailbox_orange_shared_kv_joint_1500_composition_protocol_v1.json",
        "d1_mailbox_orange_shared_kv_joint_1500_subject_transfer_protocol_v1.json",
        "d1_mailbox_orange_shared_kv_joint_1500_color_transfer_protocol_v1.json",
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


def test_required_joint_token_artifacts_reject_a_replaced_token(tmp_path):
    run_dir, model_dir = tmp_path / "run", tmp_path / "run" / "checkpoints"
    model_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("manifest", encoding="utf-8")
    for name in ("<S*>.bin", "<C*>.bin", module.WEIGHTS, "embedding_update_audit.json", "training_metrics.jsonl"):
        (model_dir / name).write_bytes(name.encode("utf-8"))
    protocol = {"source_checkpoints": [{
        "model_dir": str(model_dir), "run_dir": str(run_dir),
        "model_sha256": module.sha256(model_dir / module.WEIGHTS),
        "run_manifest_sha256": module.sha256(run_dir / "manifest.json"),
        "token_artifact_sha256": {"<S*>.bin": module.sha256(model_dir / "<S*>.bin"), "<C*>.bin": module.sha256(model_dir / "<C*>.bin")},
    }], "required_token_artifacts": ["<S*>.bin", "<C*>.bin"], "forbidden_token_artifacts": []}
    module.validate_model_dir(model_dir, protocol)
    (model_dir / "<C*>.bin").write_bytes(b"replaced")
    with pytest.raises(ValueError, match="token artifact"):
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


def test_shared_kv_joint_reconstruction_checks_both_single_token_branches_only():
    protocol = json.loads(JOINT_RECONSTRUCTION_PROTOCOL.read_text(encoding="utf-8"))
    rows = module.build_manifest(protocol)
    assert len(rows) == 40
    assert protocol["required_token_artifacts"] == ["<S*>.bin", "<C*>.bin"]
    assert set(protocol["source_checkpoints"][0]["token_artifact_sha256"]) == {"<S*>.bin", "<C*>.bin"}
    assert {row["color"] for row in rows} == {
        "subject_seen_red", "subject_seen_green", "subject_seen_cyan", "subject_seen_blue", "subject_seen_magenta",
        "color_cube", "color_sphere", "color_cylinder",
    }
    assert all(not ({"<S*>", "<C*>"} <= set(row["prompt"].split())) for row in rows)


def test_shared_kv_joint_composition_is_a_pinned_heldout_two_token_grid():
    protocol = json.loads(JOINT_COMPOSITION_PROTOCOL.read_text(encoding="utf-8"))
    rows = module.build_manifest(protocol)
    assert len(rows) == 25
    assert protocol["required_token_artifacts"] == ["<S*>.bin", "<C*>.bin"]
    assert set(protocol["source_checkpoints"][0]["token_artifact_sha256"]) == {"<S*>.bin", "<C*>.bin"}
    assert {row["checkpoint_id"] for row in rows} == {"shared-kv-1000"}
    assert {row["color"] for row in rows} == {"core", "closeup", "side_view", "white_background", "street"}
    assert all("<S*>" in row["prompt"] and "<C*>" in row["prompt"] for row in rows)
    assert all("orange" not in row["prompt"].lower() for row in rows)


def test_shared_kv_joint_single_token_transfer_grids_reuse_the_prior_prompt_sets():
    subject_protocol = json.loads(JOINT_SUBJECT_TRANSFER_PROTOCOL.read_text(encoding="utf-8"))
    subject_rows = module.build_manifest(subject_protocol)
    assert len(subject_rows) == 115
    assert {item["group"] for item in subject_protocol["prompts"]} == {"unseen_color", "context", "viewpoint", "composition", "hard_compositional"}
    assert all("<S*>" in row["prompt"] and "<C*>" not in row["prompt"] for row in subject_rows)

    color_protocol = json.loads(JOINT_COLOR_TRANSFER_PROTOCOL.read_text(encoding="utf-8"))
    color_rows = module.build_manifest(color_protocol)
    assert len(color_rows) == 100
    assert [item["color"] for item in color_protocol["prompts"]] == [f"transfer_{index:02d}_{name}" for index, name in enumerate(("bowl", "bowling_ball", "plate", "vase", "pants", "teddy_bear", "snooker_ball", "parrot", "sofa", "rose"))]
    assert all("<C*>" in row["prompt"] and "<S*>" not in row["prompt"] for row in color_rows)
    for protocol in (subject_protocol, color_protocol):
        assert protocol["required_token_artifacts"] == ["<S*>.bin", "<C*>.bin"]
        assert set(protocol["source_checkpoints"][0]["token_artifact_sha256"]) == {"<S*>.bin", "<C*>.bin"}


def test_shared_kv_joint_1500_protocols_bind_one_final_checkpoint_and_preserve_grid_roles():
    expected_counts = [40, 25, 115, 100]
    for protocol_path, expected_count in zip(JOINT_1500_PROTOCOLS, expected_counts):
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        rows = module.build_manifest(protocol)
        checkpoint = protocol["source_checkpoints"][0]
        assert len(rows) == expected_count
        assert {row["checkpoint_id"] for row in rows} == {"shared-kv-1500"}
        assert {row["checkpoint_steps"] for row in rows} == {1500}
        assert checkpoint["model_sha256"] == "55572e383028678d042304cbfbae5d0a3cd00187e42c899f957a285a4036f609"
        assert checkpoint["token_artifact_sha256"] == {"<S*>.bin": "f32b6eeac55fba2f7dff82f3537b330218d6f4ce84953ee9b77c00a610bfa688", "<C*>.bin": "0e9d002923f0cd080f9abd9e2de674bddcaa3ea938850494e4956b06dd192453"}
    reconstruction, composition, subject, color = [json.loads(path.read_text(encoding="utf-8")) for path in JOINT_1500_PROTOCOLS]
    assert all(not ({"<S*>", "<C*>"} <= set(row["prompt"].split())) for row in module.build_manifest(reconstruction))
    assert all("<S*>" in row["prompt"] and "<C*>" in row["prompt"] for row in module.build_manifest(composition))
    assert all("<S*>" in row["prompt"] and "<C*>" not in row["prompt"] for row in module.build_manifest(subject))
    assert all("<C*>" in row["prompt"] and "<S*>" not in row["prompt"] for row in module.build_manifest(color))
