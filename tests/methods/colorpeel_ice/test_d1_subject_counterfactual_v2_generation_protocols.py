from __future__ import annotations

import json
from pathlib import Path

from scripts.methods.colorpeel_ice import generate_d1_subject_recolor_statue_reconstruction as generate


ROOT = Path(__file__).parents[3]
CONFIGS = ROOT / "experiments/natural_image_subject_color_pilot/configs"


def test_v2_reconstruction_and_transfer_protocols_bind_the_completed_subject_only_checkpoint():
    reconstruction = json.loads((CONFIGS / "d1_subject_counterfactual_v2_mailbox_category_1000_reconstruction_protocol_v1.json").read_text(encoding="utf-8"))
    transfer = json.loads((CONFIGS / "d1_subject_counterfactual_v2_mailbox_category_1000_transfer_protocol_v1.json").read_text(encoding="utf-8"))
    assert len(generate.build_manifest(reconstruction)) == 25
    assert len(generate.build_manifest(transfer)) == 115
    assert reconstruction["source_checkpoints"] == transfer["source_checkpoints"]
    assert reconstruction["forbidden_token_artifacts"] == transfer["forbidden_token_artifacts"] == ["<C*>.bin"]
    assert "unseen_orange" in {row["color"] for row in transfer["prompts"]}


def test_exposure_matched_generation_protocols_use_the_same_sampling_grid_and_distinct_bound_weights():
    base_reconstruction = json.loads((CONFIGS / "d1_subject_recolor_mailbox_category_exposure_matched_5000_reconstruction_protocol_v1.json").read_text(encoding="utf-8"))
    base_transfer = json.loads((CONFIGS / "d1_subject_recolor_mailbox_category_exposure_matched_5000_transfer_protocol_v1.json").read_text(encoding="utf-8"))
    v2_reconstruction = json.loads((CONFIGS / "d1_subject_counterfactual_v2_mailbox_category_exposure_matched_5000_reconstruction_protocol_v1.json").read_text(encoding="utf-8"))
    v2_transfer = json.loads((CONFIGS / "d1_subject_counterfactual_v2_mailbox_category_exposure_matched_5000_transfer_protocol_v1.json").read_text(encoding="utf-8"))
    assert [len(generate.build_manifest(value)) for value in (base_reconstruction, v2_reconstruction, base_transfer, v2_transfer)] == [25, 25, 115, 115]
    assert base_reconstruction["sampling"] == v2_reconstruction["sampling"]
    assert base_transfer["sampling"] == v2_transfer["sampling"]
    assert base_reconstruction["source_checkpoints"][0]["model_sha256"] != v2_reconstruction["source_checkpoints"][0]["model_sha256"]


def test_exposure_matched_sampling_sweep_is_a_two_checkpoint_nine_setting_grid():
    sweep = json.loads((CONFIGS / "d1_subject_exposure_matched_5000_sampling_sweep_protocol_v1.json").read_text(encoding="utf-8"))
    rows = generate.build_manifest(sweep)
    assert len(rows) == 108
    assert {row["checkpoint_id"] for row in rows} == {"base5-exposure-matched-5000", "counterfactual-v2-exposure-matched-5000"}
    assert {row["sampling_id"] for row in rows} == {item["id"] for item in sweep["sampling_variants"]}
    assert {row["prompt"] for row in rows} == {item["prompt"] for item in sweep["prompts"]}
