from pathlib import Path

import yaml


ROOT = Path(__file__).parents[3]


def test_statue_initializer_kv_ablation_configs_parse_and_lock_the_two_learning_rates():
    config_dir = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs"
    full = yaml.safe_load((config_dir / "d1_subject_recolor_statue_init_kvfull_500.yaml").read_text(encoding="utf-8"))
    low = yaml.safe_load((config_dir / "d1_subject_recolor_statue_init_kvlow_500.yaml").read_text(encoding="utf-8"))
    assert full["args"]["concepts_list"] == "${COLORPEEL_STATUE_INIT_ABLATION_CONCEPTS}"
    assert full["args"]["initializer_token"] == low["args"]["initializer_token"] == "statue"
    assert full["args"]["learning_rate"] == low["args"]["learning_rate"] == 1.0e-5
    assert "kv_learning_rate" not in full["args"]
    assert low["args"]["kv_learning_rate"] == 1.0e-6
    assert full["args"]["cos_weight"] == low["args"]["cos_weight"] == 0.0


def test_low_kv_step_dose_configs_vary_only_the_authorized_step_count():
    config_dir = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs"
    low_750 = yaml.safe_load((config_dir / "d1_subject_recolor_statue_init_kvlow_750.yaml").read_text(encoding="utf-8"))
    low_1000 = yaml.safe_load((config_dir / "d1_subject_recolor_statue_init_kvlow_1000.yaml").read_text(encoding="utf-8"))
    assert low_750["args"]["concepts_list"] == low_1000["args"]["concepts_list"] == "${COLORPEEL_STATUE_INIT_STEP_DOSE_CONCEPTS}"
    assert low_750["args"]["kv_learning_rate"] == low_1000["args"]["kv_learning_rate"] == 1.0e-6
    assert low_750["args"]["max_train_steps"] == low_750["args"]["checkpointing_steps"] == 750
    assert low_1000["args"]["max_train_steps"] == low_1000["args"]["checkpointing_steps"] == 1000
