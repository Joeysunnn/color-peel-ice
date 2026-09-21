from __future__ import annotations

import json
from pathlib import Path

from scripts.launch.colorpeel_run import read_config


ROOT = Path(__file__).parents[3]
CONFIGS = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs"


def test_step_dose_and_low_kv_arms_change_exactly_one_baseline_training_factor():
    step = json.loads((CONFIGS / "d1_subject_recolor_mailbox_category_exposure_matched_3000_step_dose_protocol_v1.json").read_text(encoding="utf-8"))
    low_kv = json.loads((CONFIGS / "d1_subject_recolor_mailbox_category_exposure_matched_kvlow_5000_protocol_v1.json").read_text(encoding="utf-8"))
    assert step["source_training_data"] == low_kv["source_training_data"]
    assert step["subject"] == low_kv["subject"]
    assert step["training"] == {"seed": 42, "max_train_steps": 3000, "embedding_learning_rate": 1.0e-5, "kv_learning_rate": 1.0e-5, "full_kv": True, "hflip": False, "from_scratch": True, "caa_cos_weight": 0.0}
    assert low_kv["training"] == {"seed": 42, "max_train_steps": 5000, "embedding_learning_rate": 1.0e-5, "kv_learning_rate": 1.0e-6, "full_kv": True, "hflip": False, "from_scratch": True, "caa_cos_weight": 0.0}


def test_followup_launch_configs_bind_the_same_frozen_concepts_and_declared_arm_values():
    step = read_config(CONFIGS / "d1_subject_recolor_mailbox_category_exposure_matched_3000_step_dose.yaml")
    low_kv = read_config(CONFIGS / "d1_subject_recolor_mailbox_category_exposure_matched_kvlow_5000.yaml")
    assert step["args"]["concepts_list"] == low_kv["args"]["concepts_list"]
    assert step["args"]["max_train_steps"] == 3000 and step["args"]["kv_learning_rate"] == 1.0e-5
    assert low_kv["args"]["max_train_steps"] == 5000 and low_kv["args"]["kv_learning_rate"] == 1.0e-6
    assert step["args"]["hflip"] is low_kv["args"]["hflip"] is False


def test_asymmetric_kv_arms_keep_the_base_five_image_protocol_and_split_only_kv_learning_rates():
    vonly = json.loads((CONFIGS / "d1_subject_recolor_mailbox_category_exposure_matched_vonly_5000_protocol_v1.json").read_text(encoding="utf-8"))
    k_low_v_full = json.loads((CONFIGS / "d1_subject_recolor_mailbox_category_exposure_matched_k_low_v_full_5000_protocol_v1.json").read_text(encoding="utf-8"))
    assert vonly["source_training_data"] == k_low_v_full["source_training_data"]
    assert vonly["subject"] == k_low_v_full["subject"]
    assert vonly["training"] == {"seed": 42, "max_train_steps": 5000, "embedding_learning_rate": 1.0e-5, "k_learning_rate": 0.0, "v_learning_rate": 1.0e-5, "hflip": False, "from_scratch": True, "caa_cos_weight": 0.0}
    assert k_low_v_full["training"] == {"seed": 42, "max_train_steps": 5000, "embedding_learning_rate": 1.0e-5, "k_learning_rate": 1.0e-6, "v_learning_rate": 1.0e-5, "hflip": False, "from_scratch": True, "caa_cos_weight": 0.0}


def test_asymmetric_kv_launch_configs_bind_the_declared_learning_rates():
    vonly = read_config(CONFIGS / "d1_subject_recolor_mailbox_category_exposure_matched_vonly_5000.yaml")
    k_low_v_full = read_config(CONFIGS / "d1_subject_recolor_mailbox_category_exposure_matched_k_low_v_full_5000.yaml")
    assert vonly["args"]["concepts_list"] == k_low_v_full["args"]["concepts_list"]
    assert vonly["args"]["k_learning_rate"] == 0.0 and vonly["args"]["v_learning_rate"] == 1.0e-5
    assert k_low_v_full["args"]["k_learning_rate"] == 1.0e-6 and k_low_v_full["args"]["v_learning_rate"] == 1.0e-5
    assert vonly["args"]["max_train_steps"] == k_low_v_full["args"]["max_train_steps"] == 5000


def test_context_prior_arm_changes_only_the_paired_generic_mailbox_prior_loss():
    value = json.loads((CONFIGS / "d1_subject_recolor_mailbox_category_context_prior_k_low_v_full_5000_protocol_v1.json").read_text(encoding="utf-8"))
    config = read_config(CONFIGS / "d1_subject_recolor_mailbox_category_context_prior_k_low_v_full_5000.yaml")
    assert value["source_instance_training_data"]["record_count"] == 5
    assert value["class_prior_assets"]["record_count"] == 25
    assert value["class_prior_assets"]["contains_modifier_token"] is False
    assert value["training"] == {"seed": 42, "max_train_steps": 5000, "embedding_learning_rate": 1.0e-5, "k_learning_rate": 1.0e-6, "v_learning_rate": 1.0e-5, "with_prior_preservation": True, "prior_loss_weight": 1.0, "hflip": False, "from_scratch": True, "caa_cos_weight": 0.0}
    assert config["args"]["with_prior_preservation"] is True
    assert config["args"]["prior_loss_weight"] == 1.0
    assert config["args"]["learning_rate"] == 1.0e-5


def test_followup_generation_protocol_binds_both_completed_arms_to_the_same_reconstruction_and_transfer_grid():
    value = json.loads((CONFIGS / "d1_subject_exposure_matched_followup_reconstruction_transfer_protocol_v1.json").read_text(encoding="utf-8"))
    assert len(value["source_checkpoints"]) == 2
    assert [item["id"] for item in value["source_checkpoints"]] == ["full-kv-3000", "low-kv-5000"]
    assert value["sampling"] == {"seeds": [42, 43, 44, 45, 46], "num_inference_steps": 100, "guidance_scale": 3.5, "expected_image_count": 280}
    assert len(value["prompts"]) == 28
    assert {item["group"] for item in value["prompts"]} == {"reconstruction", "unseen_color", "context", "viewpoint", "composition", "hard_compositional"}


def test_asymmetric_kv_generation_protocol_uses_the_same_fixed_reconstruction_and_transfer_grid():
    value = json.loads((CONFIGS / "d1_subject_asymmetric_kv_5000_reconstruction_transfer_protocol_v1.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in value["source_checkpoints"]] == ["v-only-5000", "k-low-v-full-5000"]
    assert value["sampling"] == {"seeds": [42, 43, 44, 45, 46], "num_inference_steps": 100, "guidance_scale": 3.5, "expected_image_count": 280}
    assert len(value["prompts"]) == 28
