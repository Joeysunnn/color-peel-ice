from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from scripts.methods.colorpeel_ice import stage_d1_subject_recolor_training as stage


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
