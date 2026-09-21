import importlib.util
import json
from pathlib import Path

from scripts.launch.colorpeel_run import read_config


ROOT = Path(__file__).parents[3]
CONFIGS = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs"
SCRIPT_PATH = ROOT / "scripts" / "methods" / "colorpeel_ice" / "generate_d1_mailbox_context_priors.py"
SPEC = importlib.util.spec_from_file_location("mailbox_context_priors", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_context_prior_protocol_is_a_fixed_generic_mailbox_context_grid():
    protocol = json.loads((CONFIGS / "d1_subject_mailbox_context_prior_assets_protocol_v1.json").read_text(encoding="utf-8"))
    rows = MODULE.build_manifest(protocol)
    assert len(rows) == 25
    assert {row["prompt_id"] for row in rows} == {
        "context_white", "context_city_street", "context_house", "context_snow", "context_museum"
    }
    assert all("<S*>" not in row["prompt"] for row in rows)
    assert {row["seed"] for row in rows} == {42, 43, 44, 45, 46}


def test_context_prior_asset_launch_config_is_bound_to_the_new_stage():
    config = read_config(CONFIGS / "d1_subject_mailbox_context_prior_assets_generate.yaml")
    assert config["stage"] == "generate_mailbox_context_priors"
    assert config["args"]["device"] == "cuda:0"
