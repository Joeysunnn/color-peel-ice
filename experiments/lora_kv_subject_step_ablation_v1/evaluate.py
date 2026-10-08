"""Evaluate 1000/2000/3000-step Subject LoRA snapshots with a fixed Material LoRA."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.lora_kv_subject_v1.evaluate import token_masks
from scripts.launch.colorpeel_run import read_config


WEIGHT_NAME = "pytorch_lora_kv_weights.bin"
ARMS = ("full_kv", "token_local_kv", "full_v", "token_local_v")
CONDITIONS = ("subject_only", "subject_literal_material", "subject_material_token")
PROTOCOL_SPECS = {
    "lora_kv_subject_step_ablation/v1": {
        "training_study": "lora_kv_subject_step_ablation_v1",
        "lora": {"rank": 4, "alpha": 4.0},
        "variant": "balanced_aligned_{mode}_r4_3000",
        "status": "authorized_lora_subject_step_ablation",
    },
    "lora_kv_subject_alpha8_step_ablation/v1": {
        "training_study": "lora_kv_subject_alpha8_step_ablation_v1",
        "lora": {"rank": 4, "alpha": 8.0},
        "variant": "balanced_aligned_{mode}_r4_a8_3000",
        "status": "authorized_lora_subject_alpha8_step_ablation",
    },
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol(protocol: dict) -> None:
    spec = PROTOCOL_SPECS.get(protocol.get("schema"))
    if spec is None:
        raise ValueError("unexpected Subject step-ablation protocol")
    if (protocol.get("base_model") != "CompVis/stable-diffusion-v1-4"
            or protocol.get("training_study") != spec["training_study"]
            or protocol.get("training_data") != {
                "cohort": "balanced_aligned",
                "concepts_sha256": "109e2b336bde5a465db4cf9a29cc578f83cda48899fe2f91b4956aafcc0010f3",
                "asset_manifest_sha256": "05bd9f15dbca82a33531dae0cd9620a14554200556c384e53fbc64bc2e782e90",
                "row_count": 10,
            }
            or protocol.get("lora") != spec["lora"]
            or protocol.get("arms") != list(ARMS)
            or protocol.get("snapshot_steps") != [1000, 2000, 3000]
            or protocol.get("material_checkpoint") != {
                "run_relative_to_COLORPEEL_RUN_ROOT": (
                    "lora_kv_material_v1/"
                    "20260930-120700__lora_kv_material_v1__"
                    "ground_reflection_token_local_kv_r4_5000__56dbb0e__42"
                ),
                "training_manifest_sha256": (
                    "a6bba5d3af0d9fd8fb45fe6f84a9d6afd1c846079b970f61bcb904ae4b58652a"
                ),
                "directory_relative_to_training_run": "checkpoints",
                "mode": "token_local_kv",
                "rank": 4,
                "alpha": 4.0,
                "weights_sha256": (
                    "9713ae80319aabb7ec08faa4349cfa2e03918bab915228bfa447afea6d7f170b"
                ),
                "adaptation_config_sha256": (
                    "cefde5e7bf5a623a7d682646e8d7f07de28c5339453ff399efe916a2fbe32280"
                ),
                "token_embedding_sha256": (
                    "ef012da735605359db80033052fff3b06710b1bb081bb59782ac039ce192c7c6"
                ),
            }
            or protocol.get("conditions") != list(CONDITIONS)
            or protocol.get("literal_material") != "metal"
            or protocol.get("sampling") != {
                "seeds": [42, 43, 44, 45, 46],
                "num_inference_steps": 100,
                "guidance_scale": 3.5,
            }
            or protocol.get("safety_checker") != "enabled"):
        raise ValueError("unexpected Subject step-ablation protocol")
    groups = protocol.get("groups", [])
    if groups != [
            {"id": "plain", "subject_prompt": "a photo of <S*> mailbox"},
            {"id": "red", "subject_prompt": "a photo of <S*> mailbox in red color"},
            {"id": "blue", "subject_prompt": "a photo of <S*> mailbox in blue color"},
    ]:
        raise ValueError("Subject prompt groups differ")


def comparison_rows(protocol: dict, mode: str) -> list[dict]:
    if mode not in ARMS:
        raise ValueError(f"unknown Subject arm: {mode}")
    rows = []
    for step in protocol["snapshot_steps"]:
        for group in protocol["groups"]:
            subject = group["subject_prompt"]
            prompts = {
                "subject_only": subject,
                "subject_literal_material": subject + " made of metal",
                "subject_material_token": subject + " made of <M*>",
            }
            for condition in CONDITIONS:
                prompt = prompts[condition]
                if (prompt.count("<S*>") != 1
                        or prompt.count("<M*>") != int(condition == "subject_material_token")):
                    raise ValueError(f"modifier token count differs: {condition}")
                for seed in protocol["sampling"]["seeds"]:
                    sample_id = f"{mode}__step{step}__{group['id']}__{condition}__seed{seed}"
                    rows.append({
                        "id": sample_id,
                        "mode": mode,
                        "step": step,
                        "group": group["id"],
                        "condition": condition,
                        "prompt": prompt,
                        "seed": seed,
                        "image_path": f"images/step{step}/{group['id']}/{condition}/seed{seed}.png",
                    })
    return rows


def verify_subject_run(run: Path, protocol: dict, mode: str, run_root: Path) -> tuple[Path, dict]:
    run = run.resolve()
    if run.parent != run_root / protocol["training_study"]:
        raise ValueError("Subject run belongs to a different study")
    manifest_path = run / "manifest.json"
    manifest = read_json(manifest_path)
    spec = PROTOCOL_SPECS[protocol["schema"]]
    expected_run = {
        "study": protocol["training_study"],
        "variant": spec["variant"].format(mode=mode),
        "seed": 42,
    }
    if (manifest.get("status") != "succeeded" or manifest.get("returncode") != 0
            or manifest.get("stage") != "train" or manifest.get("run") != expected_run
            or manifest.get("lora_training_data") != protocol["training_data"]):
        raise ValueError("Subject training manifest differs from the protocol")
    config = read_config(run / "config.yaml")
    if (config.get("run") != expected_run
            or config.get("status") != spec["status"]
            or config.get("args", {}).get("subject_lora_mode") != mode
            or config.get("args", {}).get("subject_lora_rank") != protocol["lora"]["rank"]
            or config.get("args", {}).get("subject_lora_alpha") != protocol["lora"]["alpha"]
            or config.get("args", {}).get("max_train_steps") != 3000
            or config.get("args", {}).get("checkpointing_steps") != 1000):
        raise ValueError("Subject training config differs from the step ablation")
    checkpoint = run / "checkpoints"
    adaptation = read_json(checkpoint / "adaptation_config.json")
    if (adaptation.get("adaptation_mode") != "lora_subject_kv"
            or adaptation.get("mode") != mode
            or adaptation.get("rank") != protocol["lora"]["rank"]
            or adaptation.get("alpha") != protocol["lora"]["alpha"]
            or adaptation.get("modifier_tokens") != ["<S*>"]
            or adaptation.get("weight_name") != WEIGHT_NAME):
        raise ValueError("Subject adaptation config differs")
    hashes = {
        "training_manifest_sha256": sha256(manifest_path),
        WEIGHT_NAME: sha256(checkpoint / WEIGHT_NAME),
        "adaptation_config.json": sha256(checkpoint / "adaptation_config.json"),
        "<S*>.bin": sha256(checkpoint / "<S*>.bin"),
    }
    return checkpoint, hashes


def verify_material_checkpoint(protocol: dict, run_root: Path) -> tuple[Path, dict]:
    lock = protocol["material_checkpoint"]
    run = (run_root / lock["run_relative_to_COLORPEEL_RUN_ROOT"]).resolve()
    checkpoint = run / lock["directory_relative_to_training_run"]
    if sha256(run / "manifest.json") != lock["training_manifest_sha256"]:
        raise ValueError("Material training manifest changed")
    if (sha256(checkpoint / WEIGHT_NAME) != lock["weights_sha256"]
            or sha256(checkpoint / "adaptation_config.json") != lock["adaptation_config_sha256"]
            or sha256(checkpoint / "<M*>.bin") != lock["token_embedding_sha256"]):
        raise ValueError("Material LoRA checkpoint changed")
    adaptation = read_json(checkpoint / "adaptation_config.json")
    if (adaptation.get("adaptation_mode") != "lora_material_kv"
            or adaptation.get("mode") != lock["mode"]
            or adaptation.get("rank") != lock["rank"]
            or adaptation.get("alpha") != lock["alpha"]
            or adaptation.get("modifier_tokens") != ["<M*>"]
            or adaptation.get("weight_name") != WEIGHT_NAME):
        raise ValueError("Material adaptation config differs")
    return checkpoint, {
        "training_manifest_sha256": sha256(run / "manifest.json"),
        WEIGHT_NAME: sha256(checkpoint / WEIGHT_NAME),
        "adaptation_config.json": sha256(checkpoint / "adaptation_config.json"),
        "<M*>.bin": sha256(checkpoint / "<M*>.bin"),
    }


def derive_snapshots(subject: Path, steps: list[int], output: Path | None) -> dict[str, dict]:
    import torch

    token_id = read_json(subject / "embedding_update_audit.json")["modifier_tokens"][0]["token_id"]
    final_adapter = torch.load(subject / WEIGHT_NAME, map_location="cpu")
    final_token = torch.load(subject / "<S*>.bin", map_location="cpu")["<S*>"]
    derived = {}
    for step in steps:
        snapshot = subject / f"checkpoint-{step}"
        adapter_path = snapshot / "pytorch_model.bin"
        encoder_path = snapshot / "pytorch_model_1.bin"
        if not adapter_path.is_file() or not encoder_path.is_file():
            raise FileNotFoundError(f"incomplete Subject snapshot: {snapshot}")
        adapter = torch.load(adapter_path, map_location="cpu")
        encoder = torch.load(encoder_path, map_location="cpu")
        token = encoder["text_model.embeddings.token_embedding.weight"][token_id].clone().contiguous()
        if (set(adapter) != set(final_adapter)
                or any(adapter[key].shape != final_adapter[key].shape for key in adapter)):
            raise ValueError(f"Subject snapshot {step} adapter format differs")
        if step == steps[-1] and (not torch.equal(token, final_token)
                or any(not torch.equal(adapter[key], final_adapter[key]) for key in adapter)):
            raise ValueError("final periodic snapshot differs from final Subject artifacts")
        record = {
            "source_adapter_sha256": sha256(adapter_path),
            "source_text_encoder_sha256": sha256(encoder_path),
        }
        if output is not None:
            checkpoint = output / "derived_checkpoints" / f"step{step}"
            checkpoint.mkdir(parents=True)
            torch.save(adapter, checkpoint / WEIGHT_NAME)
            torch.save({"<S*>": token}, checkpoint / "<S*>.bin")
            shutil.copyfile(subject / "adaptation_config.json", checkpoint / "adaptation_config.json")
            record.update({
                "path": str(checkpoint),
                "derived_adapter_sha256": sha256(checkpoint / WEIGHT_NAME),
                "derived_token_sha256": sha256(checkpoint / "<S*>.bin"),
            })
        derived[str(step)] = record
    return derived


def checked_output_dir(output: Path, protected_runs: list[Path]) -> Path:
    resolved = output.resolve()
    for run in protected_runs:
        protected = run.resolve()
        if resolved == protected or resolved.is_relative_to(protected):
            raise ValueError(f"output directory must not be inside a training run: {protected}")
    return resolved


def install_protocol_adapters(unet, subject_state: dict, material_state: dict,
                              protocol: dict, mode: str) -> None:
    from experiments.lora_kv_subject_step_ablation_v1.attention import (
        install_subject_material_lora_kv,
    )

    install_subject_material_lora_kv(
        unet, subject_state, material_state, mode,
        protocol["lora"]["rank"], protocol["lora"]["alpha"],
        material_rank=protocol["material_checkpoint"]["rank"],
        material_alpha=protocol["material_checkpoint"]["alpha"],
    )


def load_pipeline(protocol: dict, subject: Path, material: Path, mode: str, device: str):
    import torch
    from diffusers import DiffusionPipeline, PNDMScheduler

    train_root = str(REPO_ROOT / "src" / "train")
    if train_root not in sys.path:
        sys.path.insert(0, train_root)
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(
        protocol["base_model"], subfolder="unet", local_files_only=True,
        torch_dtype=torch.float16,
    )
    subject_state = torch.load(subject / WEIGHT_NAME, map_location="cpu")
    material_state = torch.load(material / WEIGHT_NAME, map_location="cpu")
    install_protocol_adapters(unet, subject_state, material_state, protocol, mode)
    pipe = DiffusionPipeline.from_pretrained(
        protocol["base_model"], unet=unet, low_cpu_mem_usage=False,
        torch_dtype=torch.float16, local_files_only=True,
    ).to(device)
    if pipe.safety_checker is None or not isinstance(pipe.scheduler, PNDMScheduler):
        raise ValueError("evaluation requires the base safety checker and PNDM scheduler")
    pipe.load_textual_inversion(str(subject), weight_name="<S*>.bin")
    pipe.load_textual_inversion(str(material), weight_name="<M*>.bin")
    return pipe


def generate(protocol: dict, mode: str, rows: list[dict], derived: dict[str, dict],
             material: Path, output: Path, device: str) -> None:
    import torch

    with (output / "generation_status.jsonl").open("w", encoding="utf-8") as ledger:
        for step in protocol["snapshot_steps"]:
            pipe = load_pipeline(protocol, Path(derived[str(step)]["path"]), material, mode, device)
            for row in rows:
                if row["step"] != step:
                    continue
                result = pipe(
                    row["prompt"],
                    num_inference_steps=protocol["sampling"]["num_inference_steps"],
                    guidance_scale=protocol["sampling"]["guidance_scale"],
                    generator=torch.Generator(device=device).manual_seed(row["seed"]),
                    cross_attention_kwargs=token_masks(
                        pipe, row["prompt"], protocol["sampling"]["guidance_scale"]),
                )
                path = output / row["image_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                result.images[0].save(path)
                flags = getattr(result, "nsfw_content_detected", None)
                if flags is not None and len(flags) != 1:
                    raise ValueError("expected one safety-checker result")
                filtered = bool(flags[0]) if flags is not None else False
                ledger.write(json.dumps({
                    **row,
                    "status": "safety_filtered" if filtered else "ok",
                    "image_sha256": sha256(path),
                    "nsfw_content_detected": filtered,
                }, sort_keys=True) + "\n")
                ledger.flush()
            del pipe
            gc.collect()
            torch.cuda.empty_cache()


def make_contact_sheets(protocol: dict, mode: str, rows: list[dict], output: Path) -> dict[str, str]:
    from PIL import Image, ImageDraw

    status = {item["id"]: item for item in (
        json.loads(line) for line in
        (output / "generation_status.jsonl").read_text(encoding="utf-8").splitlines())}
    if len(status) != len(rows):
        raise ValueError("contact sheets require one status per sample")
    lookup = {(row["step"], row["group"], row["condition"], row["seed"]): row for row in rows}
    tile, left, top, gap = 224, 105, 48, 6
    columns = [(step, condition) for step in protocol["snapshot_steps"] for condition in CONDITIONS]
    sheet_dir = output / "contact_sheets"
    sheet_dir.mkdir()
    hashes = {}
    for group in (item["id"] for item in protocol["groups"]):
        sheet = Image.new(
            "RGB",
            (left + len(columns) * (tile + gap),
             top + len(protocol["sampling"]["seeds"]) * (tile + gap)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for col, (step, condition) in enumerate(columns):
            draw.multiline_text(
                (left + col * (tile + gap), 4), f"{step}\n{condition}", fill="black")
        for row_index, seed in enumerate(protocol["sampling"]["seeds"]):
            y = top + row_index * (tile + gap)
            draw.text((8, y + 8), f"seed {seed}", fill="black")
            for col, (step, condition) in enumerate(columns):
                item = lookup[(step, group, condition, seed)]
                path = output / item["image_path"]
                if sha256(path) != status[item["id"]]["image_sha256"]:
                    raise ValueError(f"generated image changed: {path}")
                with Image.open(path) as image:
                    thumb = image.convert("RGB").resize((tile, tile))
                x = left + col * (tile + gap)
                sheet.paste(thumb, (x, y))
                if status[item["id"]]["status"] == "safety_filtered":
                    draw.text((x + 4, y + 4), "FILTERED", fill="red")
        path = sheet_dir / f"{mode}__{group}.jpg"
        sheet.save(path, quality=90)
        hashes[group] = sha256(path)
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--mode", choices=ARMS, required=True)
    parser.add_argument("--subject-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol = read_json(args.protocol)
    validate_protocol(protocol)
    run_root = Path(os.environ["COLORPEEL_RUN_ROOT"]).resolve()
    subject, subject_hashes = verify_subject_run(args.subject_run, protocol, args.mode, run_root)
    material, material_hashes = verify_material_checkpoint(protocol, run_root)
    rows = comparison_rows(protocol, args.mode)
    expected = (len(protocol["snapshot_steps"]) * len(protocol["groups"])
                * len(protocol["conditions"]) * len(protocol["sampling"]["seeds"]))
    if len(rows) != expected:
        raise ValueError("comparison row count differs")
    if args.dry_run:
        derive_snapshots(subject, protocol["snapshot_steps"], None)
        print(f"dry-run verified {len(rows)} matched {args.mode} samples")
        return
    output_dir = checked_output_dir(
        args.output_dir, [args.subject_run, material.parent])
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    derived = derive_snapshots(subject, protocol["snapshot_steps"], output_dir)
    provenance = {
        "schema": protocol["schema"],
        "status": "running",
        "protocol_sha256": sha256(args.protocol),
        "mode": args.mode,
        "subject_run": str(args.subject_run.resolve()),
        "subject_checkpoint_sha256": subject_hashes,
        "derived_subject_snapshots": derived,
        "material_checkpoint": str(material),
        "material_checkpoint_sha256": material_hashes,
    }
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "generation_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    generate(protocol, args.mode, rows, derived, material, output_dir, args.device)
    if (verify_subject_run(args.subject_run, protocol, args.mode, run_root)[1] != subject_hashes
            or verify_material_checkpoint(protocol, run_root)[1] != material_hashes):
        raise ValueError("a training checkpoint changed during evaluation")
    provenance["contact_sheet_sha256"] = make_contact_sheets(
        protocol, args.mode, rows, output_dir)
    provenance["status"] = "succeeded"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(rows)} matched {args.mode} samples: {output_dir}")


if __name__ == "__main__":
    main()
