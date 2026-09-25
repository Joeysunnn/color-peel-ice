from pathlib import Path

import yaml

from scripts.methods.colorpeel_ice.render_clevr_multiview import validate_requests
from src.methods.colorpeel_ice.multiview_render_contract import EXPECTED_PROFILE_M_UNPAIRED
from src.methods.colorpeel_ice.prepare_unpaired_emission_material import (
    DEFAULT_PROTOCOL, protocol, requests, schedule,
)


def test_fresh_material_grid_uses_selected_scene_and_new_seeds():
    rows = requests(protocol(DEFAULT_PROTOCOL))
    assert validate_requests(rows, EXPECTED_PROFILE_M_UNPAIRED) == rows
    assert len(rows) == 72
    assert {row["view_index"] for row in rows[:12]} == {0}
    assert {row["color"] for row in rows[:12]} == {"red", "blue", "green", "yellow"}
    assert rows[0]["render_seed"] == 760000


def test_step_schedule_never_pairs_tokens_and_matches_prior_doses():
    color = [{"instance_prompt": ["a photo of a cube shape in <C*> color"]} for _ in range(9)]
    material = [{"instance_prompt": ["a photo of a cube in blue color with <M*> material"]} for _ in range(72)]
    rows = schedule(color, material)
    assert len(rows) == 5100
    assert sum("<C*>" in row["instance_prompt"][0] for row in rows) == 100
    assert sum("<M*>" in row["instance_prompt"][0] for row in rows) == 5000
    assert all(("<C*>" in row["instance_prompt"][0]) != ("<M*>" in row["instance_prompt"][0])
               for row in rows)
    assert all("<C*>" in rows[index]["instance_prompt"][0] for index in range(50, 5100, 51))


def test_unpaired_training_disables_caa_and_requires_strict_rows():
    root = Path(__file__).resolve().parents[3]
    value = yaml.safe_load((root / "experiments/color_material_composition_v1/configs/unpaired_emission_m_shared_kv_5100.yaml").read_text(encoding="utf-8"))
    assert value["args"]["cos_weight"] == 0.0
    assert value["args"]["strict_unpaired_modifier_updates"] is True
    assert value["args"]["adam_weight_decay"] == 0.0
    assert value["args"].get("token_local_kv", False) is False
