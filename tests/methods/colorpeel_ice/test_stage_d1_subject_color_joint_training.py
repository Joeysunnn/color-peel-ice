from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts" / "methods" / "colorpeel_ice" / "stage_d1_subject_color_joint_training.py"
PROTOCOL = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_mailbox_orange_shared_kv_joint_training_protocol_v1.json"
STEP_1500_PROTOCOL = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_mailbox_orange_shared_kv_joint_1500_training_protocol_v1.json"
SPEC = importlib.util.spec_from_file_location("joint_stage", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_schedule_is_balanced_interleaved_and_never_pairs_the_tokens():
    subject = [{"instance_prompt": [f"a photo of <S*> mailbox in s{index} color"], "instance_data_dir": f"/s/{index}", "instance_mask_dir": f"/m/{index}"} for index in range(5)]
    color = [{"instance_prompt": [f"a photo of c{index} shape in <C*> color"], "instance_data_dir": f"/c/{index}"} for index in range(9)]
    rows = module.build_schedule(subject, color)
    assert len(rows) == 90
    assert all("<S*>" in rows[index]["instance_prompt"][0] for index in range(0, 90, 2))
    assert all("<C*>" in rows[index]["instance_prompt"][0] for index in range(1, 90, 2))
    assert sum("<S*>" in row["instance_prompt"][0] for row in rows) == 45
    assert sum("<C*>" in row["instance_prompt"][0] for row in rows) == 45
    assert all(not ({"<S*>", "<C*>"} <= set(row["instance_prompt"][0].split())) for row in rows)


def test_shipped_protocol_locks_unpaired_shared_kv_contract():
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert module.protocol(PROTOCOL) == value
    for branch in value["branches"].values():
        assert module.sha256(ROOT / branch["source_protocol_relative_path"]) == branch["source_protocol_sha256"]
    assert value["checkpoint_plan"] == {"max_train_steps": 1000, "state_checkpoint_steps": 250, "final_export_only_for_inference": True}
    assert value["training"]["cos_weight"] == 0.0


def test_1500_step_dose_protocol_reuses_the_hashed_unpaired_concepts_only():
    value = json.loads(STEP_1500_PROTOCOL.read_text(encoding="utf-8"))
    assert module.protocol(STEP_1500_PROTOCOL) == value
    for branch in value["branches"].values():
        assert module.sha256(ROOT / branch["source_protocol_relative_path"]) == branch["source_protocol_sha256"]
    assert value["staged_concepts"] == {
        "path": "/home/r12user5/Documents/Jiawei/colorpeel-runs/20260917-170639__natural_image_subject_color_pilot__mailbox_orange_shared_kv_joint_assets__bb81c73__42/concepts.json",
        "sha256": "eb492e21ddee37d5495edb8e40ee4a4b268c8aee75b4b763a18bc336a64c2c71",
        "row_count": 90,
    }
    assert value["checkpoint_plan"] == {"max_train_steps": 1500, "state_checkpoint_steps": 250, "final_export_only_for_inference": True}
    assert value["schedule"]["expected_exposures_per_token"] == 750
    assert value["training"]["initialization"] == "base_stable_diffusion_not_resume"
