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
