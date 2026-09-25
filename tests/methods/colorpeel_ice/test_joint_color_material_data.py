import copy
import json
from pathlib import Path
import pytest
import yaml

from scripts.methods.colorpeel_ice.render_clevr_multiview import RendererError, validate_requests
from src.methods.colorpeel_ice.multiview_render_contract import EXPECTED_PROFILE_CM_JOINT
from src.methods.colorpeel_ice.prepare_joint_color_material import (
    DEFAULT_PROTOCOL, build_requests, caption, validate_protocol,
)
from scripts.methods.colorpeel_ice.generate_joint_cm_evaluation import rows as evaluation_rows


def test_joint_grid_is_paired_and_caa_can_be_active():
    protocol = validate_protocol(DEFAULT_PROTOCOL)
    rows = build_requests(protocol)
    assert validate_requests(rows, EXPECTED_PROFILE_CM_JOINT) == rows
    assert len(rows) == 72
    assert [row["lighting_condition"] for row in rows[:12]] == ["soft_front"] * 12
    for shape in protocol["shapes"]:
        for view in range(6):
            paired = [row for row in rows if row["shape"] == shape and row["view_index"] == view]
            assert len(paired) == 4
            assert len({row["render_seed"] for row in paired}) == 1
    assert sum(bool(row["color_token"] and row["material_token"]) for row in rows) == 18
    assert caption("cube", "orange", "metal") == "a photo of a cube with <C*> color and <M*> material"


def test_joint_grid_rejects_changed_color_socket():
    rows = copy.deepcopy(build_requests(validate_protocol(DEFAULT_PROTOCOL)))
    rows[0]["material_socket_rgba"][0] = 0.5
    with pytest.raises(RendererError, match="color socket"):
        validate_requests(rows, EXPECTED_PROFILE_CM_JOINT)


def test_joint_training_configs_differ_only_in_caa_and_run_name():
    root = Path(__file__).resolve().parents[3] / "experiments/color_material_composition_v1"
    configs = [yaml.safe_load((root / "configs" / name).read_text(encoding="utf-8"))
               for name in ("joint_cm_caa0_1500.yaml", "joint_cm_caa02_1500.yaml")]
    assert [value["args"]["cos_weight"] for value in configs] == [0.0, 0.2]
    assert [value["run"]["variant"] for value in configs] == ["joint_cm_caa0_1500", "joint_cm_caa02_1500"]
    configs[1]["args"]["cos_weight"] = 0.0
    configs[1]["run"]["variant"] = "joint_cm_caa0_1500"
    assert configs[0] == configs[1]


def test_joint_evaluation_uses_matched_heldout_grid():
    root = Path(__file__).resolve().parents[3] / "experiments/color_material_composition_v1"
    protocol = json.loads((root / "protocols/joint_cm_evaluation_v1.json").read_text(encoding="utf-8"))
    samples = evaluation_rows(protocol)
    assert len(samples) == 36
    assert {row["seed"] for row in samples} == {42, 43, 44}
    assert len([row for row in samples if row["object"] == "mug"]) == 12
