from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from scripts.methods.colorpeel_ice import stage_d1_subject_recolor_training as stage


ROOT = Path(__file__).parents[3]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage_uses_only_five_auxiliary_images_and_repaired_masks(tmp_path):
    source, output = tmp_path / "source", tmp_path / "stage"
    (source / "images").mkdir(parents=True)
    (source / "masks").mkdir()
    names = ["red", "yellow", "green", "cyan", "blue"]
    records, hashes = [], {}
    for index, name in enumerate(names):
        image = source / "images" / f"{name}.png"
        Image.new("RGB", (8, 8), (index, 10, 20)).save(image)
        hashes[name] = digest(image)
        records.append({"name": name, "image_relative_path": f"images/{name}.png", "image_sha256": hashes[name]})
    mask = source / "masks" / "repaired_mask.png"
    Image.new("L", (8, 8), 255).save(mask)
    mask_hash = digest(mask)
    results = source / stage.RESULTS_NAME
    results.write_text(json.dumps({"records": records}), encoding="utf-8")
    analysis = source / stage.ANALYSIS_NAME
    analysis.write_text(json.dumps({"automatic_safety_pass": True, "added_bottom_region_recolored": True, "record_count": 5}), encoding="utf-8")
    (source / stage.REPAIR_NAME).write_text(json.dumps({"repaired_mask_sha256": mask_hash, "repaired_mask_relative_path": "masks/repaired_mask.png"}), encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    prompts = {name: f"a photo of <S*> in {name} color" for name in names}
    protocol.write_text(json.dumps({"schema": "natural_subject_recolor_training_protocol/v1", "source_pilot": {"analysis_sha256": digest(analysis), "results_sha256": digest(results), "repaired_mask_sha256": mask_hash, "required_automatic_safety_pass": True}, "subject": {"modifier_token": "<S*>", "initializer_token": "gorilla", "prompt": "a photo of <S*>"}, "training_data": {"image_names": names, "prompt_by_image": prompts, "expected_image_sha256": hashes, "use_repaired_binary_instance_mask": True, "original_subject_color_record": "forbidden", "color_modifier_token": "forbidden"}, "caa": {"enabled": False, "cos_weight": 0.0, "reason": "one learned modifier token cannot form a learned-token attention pair"}, "approval_state": {"subject_only_short_training_approved": True, "mixed_shared_checkpoint_training_approved": False}}, indent=2), encoding="utf-8")
    result = stage.stage(source, output, protocol)
    concepts = json.loads(Path(result["concepts"]).read_text(encoding="utf-8"))
    assert result["record_count"] == 5
    assert concepts == [{"instance_prompt": [prompts[name]], "instance_data_dir": str(output / name / "images"), "instance_mask_dir": str(output / name / "masks")} for name in names]
    assert {path.name for path in output.iterdir() if path.is_dir()} == set(names)


def test_shipped_statue_followup_protocol_locks_the_literal_statue_prompt():
    value = stage.protocol(
        ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_gorilla_statue_training_protocol_v2.json"
    )
    assert value["followup_training"]["authorized_steps"] == [250, 500, 750, 1000]
    assert set(value["training_data"]["prompt_by_image"].values()) == {
        f"a photo of <S*> statue in {color} color"
        for color in ("red", "yellow", "green", "cyan", "blue")
    }


def test_statue_initializer_ablation_protocol_locks_category_prompt():
    value = stage.protocol(
        ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_statue_initializer_kv_ablation_protocol_v3.json"
    )
    assert value["subject"] == {
        "modifier_token": "<S*>",
        "initializer_token": "statue",
        "training_prompt_template": "a photo of <S*> gorilla statue in {color} color",
    }
    assert set(value["training_data"]["prompt_by_image"].values()) == {
        f"a photo of <S*> gorilla statue in {color} color"
        for color in ("red", "yellow", "green", "cyan", "blue")
    }


def test_statue_initializer_low_kv_step_dose_protocol_locks_only_750_and_1000_steps():
    value = stage.protocol(
        ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_statue_init_kvlow_step_dose_protocol_v4.json"
    )
    assert value["followup_training"] == {
        "authorized_steps": [750, 1000],
        "kv_learning_rate": 1.0e-6,
        "from_scratch": True,
        "transfer": "forbidden",
    }


def test_statue_initializer_full_kv_step_dose_protocol_locks_750_and_1000_steps():
    value = stage.protocol(
        ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_statue_init_kvfull_step_dose_protocol_v1.json"
    )
    assert value["step_dose"] == {
        "authorized_steps": [750, 1000],
        "embedding_learning_rate": 1.0e-5,
        "kv_learning_rate": 1.0e-5,
        "from_scratch": True,
        "transfer_after_training": "authorized",
    }
    assert value["approval_state"]["initializer_kv_ablation_approved"] is True
    assert value["approval_state"]["full_kv_step_dose_approved"] is True
    assert value["approval_state"]["mixed_shared_checkpoint_training_approved"] is False
    assert value["approval_state"]["joint_training_approved"] is False


def test_statue_initializer_full_kv_transfer_protocol_binds_only_the_new_step_dose_runs():
    path = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_statue_init_kvfull_750_1000_transfer_protocol_v1.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert [item["steps"] for item in value["source_checkpoints"]] == [750, 1000]
    assert all("subject_recolor_statue_init_kvfull" in item["run_dir"] for item in value["source_checkpoints"])
    assert all(len(item["model_sha256"]) == 64 for item in value["source_checkpoints"])
    assert value["sampling"]["expected_image_count"] == 170
    assert len(value["prompts"]) == 17
    assert value["next_step"] == "human review only; no joint training authorized"


def test_mailbox_prompt_ablation_protocols_lock_full_kv_and_the_five_repaired_colors():
    configs = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs"
    category = stage.protocol(configs / "d1_subject_recolor_mailbox_category_training_protocol_v1.json")
    token_first = stage.protocol(configs / "d1_subject_recolor_mailbox_token_first_training_protocol_v1.json")
    assert category["training"]["authorized_steps"] == token_first["training"]["authorized_steps"] == [750]
    assert category["training"]["kv_learning_rate"] == token_first["training"]["kv_learning_rate"] == 1.0e-5
    assert category["training_data"]["image_names"] == token_first["training_data"]["image_names"] == ["red", "green", "cyan", "blue", "magenta"]
    assert category["subject"]["training_prompt_template"] == "a photo of <S*> mailbox in {color} color"
    assert token_first["subject"]["training_prompt_template"] == "a photo of <S*> in {color} color"
    assert category["caa"]["cos_weight"] == token_first["caa"]["cos_weight"] == 0.0


def test_mailbox_1000_step_protocols_keep_the_prompt_ablation_and_full_kv_contract():
    configs = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs"
    category = stage.protocol(configs / "d1_subject_recolor_mailbox_category_1000_training_protocol_v1.json")
    token_first = stage.protocol(configs / "d1_subject_recolor_mailbox_token_first_1000_training_protocol_v1.json")
    assert category["training"]["authorized_steps"] == token_first["training"]["authorized_steps"] == [1000]
    assert category["training"]["kv_learning_rate"] == token_first["training"]["kv_learning_rate"] == 1.0e-5
    assert category["training_data"] == token_first["training_data"] | {"prompt_by_image": category["training_data"]["prompt_by_image"]}
    assert category["subject"]["training_prompt_template"] == "a photo of <S*> mailbox in {color} color"
    assert token_first["subject"]["training_prompt_template"] == "a photo of <S*> in {color} color"
    assert category["caa"] == token_first["caa"] == {"enabled": False, "cos_weight": 0.0, "reason": "one learned modifier token cannot form a learned-token attention pair"}


def test_no_gorilla_initializer_screen_protocol_locks_three_single_token_candidates():
    value = stage.protocol(
        ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_no_gorilla_initializer_screen_protocol_v5.json"
    )
    assert value["subject"] == {
        "modifier_token": "<S*>",
        "initializer_tokens": ["statue", "gorilla", "sculpture"],
        "training_prompt_template": "a photo of <S*> statue in {color} color",
    }
    assert value["screen"] == {
        "authorized_steps": 500,
        "kv_learning_rate": 1.0e-5,
        "from_scratch": True,
        "transfer": "forbidden",
    }


def test_no_gorilla_initializer_step_screen_protocol_locks_the_two_step_values():
    value = stage.protocol(
        ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_no_gorilla_initializer_step_screen_protocol_v6.json"
    )
    assert value["screen"] == {
        "authorized_steps": [500, 750],
        "kv_learning_rate": 1.0e-5,
        "from_scratch": True,
        "transfer": "forbidden",
    }
