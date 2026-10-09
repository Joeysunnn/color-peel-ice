from copy import deepcopy
import json
from pathlib import Path

from experiments.lora_kv_subject_matte5_r64_v1 import prepare_staging
from scripts.launch import colorpeel_run


ROOT = Path(__file__).parents[3]
EXPERIMENT = ROOT / "experiments" / "lora_kv_subject_matte5_r64_v1"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_training_config_locks_rank64_three_checkpoints_and_corrected_captions():
    config = colorpeel_run.read_config(EXPERIMENT / "configs" / "train.json")
    args = config["args"]
    assert config["run"] == {
        "study": "lora_kv_subject_matte5_r64_v1",
        "variant": "mailbox_matte5_token_local_kv_r64_a64_kvlr5em5_3000",
        "seed": 42,
    }
    assert args["learning_rate"] == 1.0e-5
    assert args["kv_learning_rate"] == 5.0e-5
    assert args["subject_lora_mode"] == "token_local_kv"
    assert args["subject_lora_rank"] == args["subject_lora_alpha"] == 64
    assert args["max_train_steps"] == 3000
    assert args["checkpointing_steps"] == 1000
    protocol = prepare_staging.read_json(
        EXPERIMENT / "protocols" / "training_source_v1.json")
    prepare_staging.validate_protocol(protocol)
    prompts = {item["source_color"]: item["prompt"] for item in protocol["assets"]}
    assert prompts["red"] == "a photo of <S*> mailbox in red color"
    assert prompts["magenta"] == "a photo of <S*> mailbox in purple color"


def test_prepare_staging_copies_only_five_reviewed_matte_images(tmp_path):
    run_root = tmp_path / "runs"
    source = run_root / "source"
    protocol = deepcopy(prepare_staging.read_json(
        EXPERIMENT / "protocols" / "training_source_v1.json"))
    protocol["source_root_relative_to_COLORPEEL_RUN_ROOT"] = "source"
    preview_records = []
    source_assets = []
    for index, item in enumerate(protocol["assets"]):
        color = item["source_color"]
        image = source / "preview" / f"{color}_matte.png"
        mask = source / "caption_aligned_staged" / color / "matte" / "masks" / "image.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        mask.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"image-{index}".encode())
        mask.write_bytes(b"shared-mask")
        item["preview_image"] = str(image.relative_to(source)).replace("\\", "/")
        item["mask"] = str(mask.relative_to(source)).replace("\\", "/")
        item["image_sha256"] = prepare_staging.sha256(image)
        item["mask_sha256"] = prepare_staging.sha256(mask)
        preview_records.append({
            "color": color, "matte_image": image.name,
            "matte_image_sha256": item["image_sha256"],
        })
        source_assets.append({
            "color": color, "material": "matte", "image_sha256": item["image_sha256"],
            "mask": str(mask.resolve()), "mask_sha256": item["mask_sha256"],
        })
    preview_manifest = source / "preview" / "preview_manifest.json"
    write_json(preview_manifest, {"records": preview_records})
    protocol["preview_manifest_sha256"] = prepare_staging.sha256(preview_manifest)
    review = source / "preview_review.json"
    write_json(review, {"verdict": "pass",
                        "preview_manifest_sha256": protocol["preview_manifest_sha256"]})
    protocol["preview_review_sha256"] = prepare_staging.sha256(review)
    assets = source / "caption_aligned_staged" / "training_assets_manifest.jsonl"
    assets.write_text("".join(json.dumps(row) + "\n" for row in source_assets), encoding="utf-8")
    protocol["source_assets_manifest_sha256"] = prepare_staging.sha256(assets)
    source_provenance = source / "caption_aligned_staged" / "staging_provenance.json"
    write_json(source_provenance, {"fixture": True})
    protocol["source_staging_provenance_sha256"] = prepare_staging.sha256(source_provenance)
    protocol_path = tmp_path / "protocol.json"
    write_json(protocol_path, protocol)

    output = run_root / "lora_kv_subject_matte5_r64_v1" / "assets_v1" / "staging"
    result = prepare_staging.stage(protocol_path, run_root, output)
    concepts = prepare_staging.read_json(output / "concepts.json")
    rows = [json.loads(line) for line in
            (output / "training_assets_manifest.jsonl").read_text().splitlines()]
    assert result["row_count"] == len(concepts) == len(rows) == 5
    assert [row["source_color"] for row in rows] == list(prepare_staging.COLORS)
    assert concepts[0]["instance_prompt"] == ["a photo of <S*> mailbox in red color"]
    assert concepts[-1]["instance_prompt"] == ["a photo of <S*> mailbox in purple color"]
