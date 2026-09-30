"""Compare a fixed mailbox Subject with old and token-local LoRA Material."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.lora_kv_subject_v1.evaluate import (
    ARMS, MATERIAL_WEIGHT_NAME, WEIGHT_NAME, checked_file, comparison_rows,
    material_checkpoint, read_json, sha256, token_masks, verify_subject_checkpoint,
)


CONDITIONS = ("subject_only", "subject_old_material", "subject_new_material", "new_material_only")


def matched_rows(protocol: dict, subject_protocol: dict) -> list[dict]:
    reference = checked_file(REPO_ROOT / protocol["material_only_reference"],
                             protocol["material_only_reference_sha256"])
    if (reference.get("schema") != "subject_material_diagnostic/v1"
            or reference.get("sampling") != protocol["sampling"]):
        raise ValueError("Material-only prompt reference differs")
    old_rows = [row for row in comparison_rows(subject_protocol) if row["arm"] == "token_local_kv"]
    old_prompts = {(row["group"], row["condition"], row["seed"]): row["prompt"] for row in old_rows}
    if len(old_rows) != 18 or len(old_prompts) != 18:
        raise ValueError("expected 18 frozen Subject prompt-seed pairs")
    material_only = {"plain": "a photo of a mailbox made of <M*>"}
    for group in reference["groups"]:
        if group["id"] in {"red", "blue"}:
            material_only[group["id"]] = group["prompts"]["material_only"]
    if set(material_only) != set(protocol["groups"]):
        raise ValueError("Material-only groups differ")

    rows = []
    for group in protocol["groups"]:
        for seed in protocol["sampling"]["seeds"]:
            subject_only = old_prompts[(group, "subject_only", seed)]
            with_material = old_prompts[(group, "subject_material", seed)]
            for condition, prompt in (
                    ("subject_only", subject_only),
                    ("subject_old_material", with_material),
                    ("subject_new_material", with_material),
                    ("new_material_only", material_only[group])):
                if (prompt.count("<S*>") != int(condition != "new_material_only")
                        or prompt.count("<M*>") != int(condition != "subject_only")):
                    raise ValueError(f"modifier token count differs: {condition}")
                sample_id = f"{condition}__{group}__seed{seed}"
                rows.append({
                    "id": sample_id, "condition": condition, "group": group,
                    "prompt": prompt, "seed": seed,
                    "image_path": f"images/{condition}/{group}/{sample_id}.png",
                })
    if len(rows) != 36:
        raise ValueError("expected 36 matched samples")
    return rows


def verify_new_material_checkpoint(checkpoint: Path, protocol: dict, old_material: Path,
                                   run_root: Path) -> dict[str, str]:
    checkpoint = checkpoint.resolve()
    study = protocol["material_training"]["study"]
    run = checkpoint.parent
    if (checkpoint.name != "checkpoints" or run.parent != run_root / study):
        raise ValueError("new Material checkpoint must belong to the independent Material LoRA study")
    manifest_path = run / "manifest.json"
    manifest = read_json(manifest_path)
    expected = protocol["material_training"]
    if (manifest.get("status") != "succeeded" or manifest.get("returncode") != 0
            or manifest.get("stage") != "train"
            or manifest.get("run") != expected):
        raise ValueError("new Material training run differs from the locked protocol")
    old_manifest = read_json(old_material.parent / "manifest.json")
    if manifest.get("data_manifest") != old_manifest.get("data_manifest"):
        raise ValueError("new Material training data differs from selected old Material")
    adaptation = read_json(checkpoint / "adaptation_config.json")
    lora = protocol["material_lora"]
    if (adaptation.get("adaptation_mode") != "lora_material_kv"
            or adaptation.get("mode") != lora["mode"]
            or adaptation.get("rank") != lora["rank"]
            or adaptation.get("alpha") != lora["alpha"]
            or adaptation.get("modifier_tokens") != ["<M*>"]
            or adaptation.get("weight_name") != WEIGHT_NAME):
        raise ValueError("new Material checkpoint adaptation differs")
    return {name: sha256(checkpoint / name) for name in
            (WEIGHT_NAME, "adaptation_config.json", "<M*>.bin")} | {
                "training_manifest_sha256": sha256(manifest_path),
            }


def load_pipeline(protocol: dict, subject_dir: Path, material_dir: Path,
                  material_kind: str, device: str):
    import torch
    from diffusers import DiffusionPipeline, PNDMScheduler

    train_root = str(REPO_ROOT / "src" / "train")
    if train_root not in sys.path:
        sys.path.insert(0, train_root)
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel
    from experiments.lora_kv_subject_v1.attention import install_subject_lora_kv
    from experiments.lora_kv_material_v1.attention import install_dual_token_local_lora_kv

    unet = UNet2DConditionModel.from_pretrained(
        protocol["base_model"], subfolder="unet", local_files_only=True,
        torch_dtype=torch.float16,
    )
    subject_state = torch.load(subject_dir / WEIGHT_NAME, map_location="cpu")
    if material_kind == "old":
        material_state = torch.load(material_dir / MATERIAL_WEIGHT_NAME, map_location="cpu")
        install_subject_lora_kv(
            unet, subject_state, "token_local_kv", protocol["material_lora"]["rank"],
            protocol["material_lora"]["alpha"], material_state=material_state,
        )
    elif material_kind == "new":
        material_state = torch.load(material_dir / WEIGHT_NAME, map_location="cpu")
        install_dual_token_local_lora_kv(
            unet, subject_state, material_state,
            protocol["material_lora"]["rank"], protocol["material_lora"]["alpha"],
        )
    else:
        raise ValueError(f"unknown Material kind: {material_kind}")
    pipe = DiffusionPipeline.from_pretrained(
        protocol["base_model"], unet=unet, low_cpu_mem_usage=False,
        torch_dtype=torch.float16, local_files_only=True,
    ).to(device)
    if pipe.safety_checker is None or not isinstance(pipe.scheduler, PNDMScheduler):
        raise ValueError("evaluation requires base safety checker and PNDM scheduler")
    pipe.load_textual_inversion(str(subject_dir), weight_name="<S*>.bin")
    pipe.load_textual_inversion(str(material_dir), weight_name="<M*>.bin")
    return pipe


def generate(protocol: dict, subject_dir: Path, old_material: Path, new_material: Path,
             rows: list[dict], output_dir: Path, device: str) -> None:
    import torch

    for material_kind, material_dir, conditions in (
            ("old", old_material, {"subject_only", "subject_old_material"}),
            ("new", new_material, {"subject_new_material", "new_material_only"})):
        pipe = load_pipeline(protocol, subject_dir, material_dir, material_kind, device)
        with (output_dir / "generation_status.jsonl").open("a", encoding="utf-8") as ledger:
            for row in rows:
                if row["condition"] not in conditions:
                    continue
                result = pipe(
                    row["prompt"],
                    num_inference_steps=protocol["sampling"]["num_inference_steps"],
                    guidance_scale=protocol["sampling"]["guidance_scale"],
                    generator=torch.Generator(device=device).manual_seed(row["seed"]),
                    cross_attention_kwargs=token_masks(
                        pipe, row["prompt"], protocol["sampling"]["guidance_scale"]),
                )
                path = output_dir / row["image_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                result.images[0].save(path)
                flags = getattr(result, "nsfw_content_detected", None)
                if flags is not None and len(flags) != 1:
                    raise ValueError("expected one safety-checker result")
                filtered = bool(flags[0]) if flags is not None else False
                ledger.write(json.dumps({
                    **row, "status": "safety_filtered" if filtered else "ok",
                    "image_sha256": sha256(path), "nsfw_content_detected": filtered,
                }, sort_keys=True) + "\n")
                ledger.flush()
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


def make_contact_sheets(protocol: dict, rows: list[dict], output_dir: Path) -> dict[str, str]:
    from PIL import Image, ImageDraw

    statuses = [json.loads(line) for line in
                (output_dir / "generation_status.jsonl").read_text(encoding="utf-8").splitlines()]
    status = {item["id"]: item for item in statuses}
    if len(statuses) != len(rows) or len(status) != len(rows):
        raise ValueError("contact sheet requires one status for every sample")
    lookup = {(row["condition"], row["group"], row["seed"]): row for row in rows}
    tile, left, top, gap = 256, 170, 34, 8
    sheet_dir = output_dir / "contact_sheets"
    sheet_dir.mkdir()
    sheets = {}
    for group in protocol["groups"]:
        sheet = Image.new("RGB", (left + 4 * (tile + gap),
                                  top + len(protocol["sampling"]["seeds"]) * (tile + gap)), "white")
        draw = ImageDraw.Draw(sheet)
        for col, condition in enumerate(CONDITIONS):
            draw.text((left + col * (tile + gap), 8), condition, fill="black")
        for row_index, seed in enumerate(protocol["sampling"]["seeds"]):
            y = top + row_index * (tile + gap)
            draw.text((8, y + 8), f"{group}\nseed {seed}", fill="black")
            for col, condition in enumerate(CONDITIONS):
                item = lookup[(condition, group, seed)]
                with Image.open(output_dir / item["image_path"]) as sample:
                    thumbnail = sample.convert("RGB").resize((tile, tile))
                x = left + col * (tile + gap)
                sheet.paste(thumbnail, (x, y))
                if status[item["id"]]["status"] == "safety_filtered":
                    draw.text((x + 4, y + 4), "SAFETY FILTERED", fill="red")
        path = sheet_dir / f"{group}.jpg"
        sheet.save(path, quality=90)
        sheets[group] = sha256(path)
    return sheets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--subject-checkpoint", type=Path, required=True)
    parser.add_argument("--new-material-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    if (protocol.get("schema") != "lora_kv_material_comparison/v1"
            or protocol.get("base_model") != "CompVis/stable-diffusion-v1-4"
            or protocol.get("subject_arm") != "token_local_kv"
            or protocol.get("conditions") != list(CONDITIONS)
            or protocol.get("safety_checker") != "enabled"):
        raise ValueError("unexpected Material LoRA comparison protocol")
    subject_protocol = checked_file(REPO_ROOT / protocol["subject_comparison"],
                                    protocol["subject_comparison_sha256"])
    expected_material_lora = {**subject_protocol["lora"], "mode": "token_local_kv"}
    if (subject_protocol["cohort"] != "balanced_aligned"
            or subject_protocol["arms"] != list(ARMS)
            or subject_protocol["groups"] != protocol["groups"]
            or subject_protocol["sampling"] != protocol["sampling"]
            or subject_protocol["base_model"] != protocol["base_model"]
            or protocol["material_lora"] != expected_material_lora):
        raise ValueError("Subject or sampling protocol differs")
    rows = matched_rows(protocol, subject_protocol)
    run_root = Path(os.environ["COLORPEEL_RUN_ROOT"]).resolve()
    subject_dir = args.subject_checkpoint.resolve()
    if subject_dir.parent.parent != run_root / "lora_kv_subject_v1":
        raise ValueError("Subject checkpoint must belong to the Subject LoRA study")
    subject_hashes = verify_subject_checkpoint(subject_dir, "token_local_kv", subject_protocol)
    old_material = material_checkpoint(subject_protocol, run_root)
    old_material_hashes = {
        "training_manifest_sha256": sha256(old_material.parent / "manifest.json"),
        MATERIAL_WEIGHT_NAME: sha256(old_material / MATERIAL_WEIGHT_NAME),
        "adaptation_config.json": sha256(old_material / "adaptation_config.json"),
        "<M*>.bin": sha256(old_material / "<M*>.bin"),
    }
    new_material = args.new_material_checkpoint.resolve()
    new_material_hashes = verify_new_material_checkpoint(new_material, protocol, old_material, run_root)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    provenance = {
        "schema": protocol["schema"], "status": "dry_run" if args.dry_run else "running",
        "protocol_sha256": sha256(args.protocol),
        "subject_checkpoint": str(subject_dir), "subject_checkpoint_sha256": subject_hashes,
        "old_material_checkpoint": str(old_material),
        "old_material_checkpoint_sha256": old_material_hashes,
        "new_material_checkpoint": str(new_material),
        "new_material_checkpoint_sha256": new_material_hashes,
    }
    provenance_path = args.output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "generation_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if not args.dry_run:
        generate(protocol, subject_dir, old_material, new_material, rows, args.output_dir, args.device)
        if (verify_subject_checkpoint(subject_dir, "token_local_kv", subject_protocol) != subject_hashes
                or verify_new_material_checkpoint(new_material, protocol, old_material, run_root) != new_material_hashes
                or material_checkpoint(subject_protocol, run_root) != old_material):
            raise ValueError("a checkpoint changed during inference")
        provenance["contact_sheet_sha256"] = make_contact_sheets(protocol, rows, args.output_dir)
        provenance["status"] = "succeeded"
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(rows)} matched samples: {args.output_dir}")


if __name__ == "__main__":
    main()
