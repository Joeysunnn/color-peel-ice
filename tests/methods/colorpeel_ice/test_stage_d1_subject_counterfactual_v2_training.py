from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from scripts.methods.colorpeel_ice import stage_d1_subject_counterfactual_v2_training as stage


ROOT = Path(__file__).parents[3]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage_keeps_one_independent_image_mask_pair_for_every_v2_record(tmp_path):
    source, output = tmp_path / "source", tmp_path / "stage"
    (source / "images").mkdir(parents=True); (source / "masks").mkdir()
    source_protocol = ROOT / "experiments/natural_image_subject_color_pilot/configs/d1_subject_counterfactual_v2_protocol_v1.json"
    source_protocol_hash = digest(source_protocol)
    colors, variants, records = ["red", "green"], ["center", "mirror"], []
    for index, (color, variant) in enumerate((color, variant) for color in colors for variant in variants):
        record_id = f"{color}__{variant}"
        image, mask = source / "images" / f"{record_id}.png", source / "masks" / f"{record_id}.png"
        Image.new("RGB", (512, 512), (index, 2, 3)).save(image)
        Image.new("L", (512, 512), 255).save(mask)
        records.append({"record_id": record_id, "color_name": color, "variant": {"id": variant}, "hue_degrees": 0.0 if color == "red" else 120.0, "hue_distance_degrees": 40.0, "image_relative_path": image.relative_to(source).as_posix(), "image_sha256": digest(image), "mask_relative_path": mask.relative_to(source).as_posix(), "mask_sha256": digest(mask)})
    plan = source / "subject_counterfactual_v2_plan.json"
    plan.write_text(json.dumps({"protocol_sha256": source_protocol_hash}), encoding="utf-8")
    results = source / "subject_counterfactual_v2_results.json"
    results.write_text(json.dumps({"plan_sha256": digest(plan), "record_count": len(records), "records": records}), encoding="utf-8")
    analysis = source / "subject_counterfactual_v2_analysis.json"
    analysis.write_text(json.dumps({"automatic_safety_pass": True}), encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"schema": "natural_subject_counterfactual_v2_training_protocol/v1", "source_counterfactual": {"source_root": str(source), "source_protocol_relative_path": "experiments/natural_image_subject_color_pilot/configs/d1_subject_counterfactual_v2_protocol_v1.json", "source_protocol_sha256": source_protocol_hash, "plan_sha256": digest(plan), "results_sha256": digest(results), "analysis_sha256": digest(analysis)}, "subject": {"stable_id": "D1GT:81/130.png", "modifier_token": "<S*>", "initializer_token": "mailbox", "training_prompt_template": "a photo of <S*> mailbox in {color} color"}, "training_data": {"record_count": 4, "color_names": colors, "variant_ids": variants, "minimum_hue_separation_degrees": 30.0, "original_subject_color_record": "forbidden", "color_modifier_token": "forbidden", "uses_per_record_binary_instance_mask": True}, "caa": {"enabled": False, "cos_weight": 0.0, "reason": "one learned modifier token cannot form a learned-token attention pair"}, "training": {"seed": 42, "max_train_steps": 1000, "embedding_learning_rate": 1.0e-5, "kv_learning_rate": 1.0e-5, "full_kv": True, "hflip": False, "from_scratch": True}, "approval_state": {"subject_counterfactual_v2_training_approved": True, "joint_training_approved": False}}), encoding="utf-8")
    result = stage.stage(source, output, protocol)
    concepts = json.loads(Path(result["concepts"]).read_text(encoding="utf-8"))
    assert result["record_count"] == 4
    assert [Path(row["instance_data_dir"]).parent.name for row in concepts] == sorted(row["record_id"] for row in records)
    assert all(Path(row["instance_data_dir"]).joinpath("image.png").is_file() and Path(row["instance_mask_dir"]).joinpath("image.png").is_file() for row in concepts)
