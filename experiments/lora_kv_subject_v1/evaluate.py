"""Generate a matched four-arm Subject LoRA comparison with frozen Material."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.perfusion_subject_pilot.evaluate import manifest_rows as reference_rows


ARMS = ("full_kv", "token_local_kv", "full_v", "token_local_v")
WEIGHT_NAME = "pytorch_lora_kv_weights.bin"
MATERIAL_WEIGHT_NAME = "pytorch_token_local_kv_weights.bin"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_file(path: Path, expected_sha256: str) -> dict:
    if sha256(path) != expected_sha256:
        raise ValueError(f"source file hash changed: {path}")
    return read_json(path)


def comparison_rows(protocol: dict) -> list[dict]:
    reference_path = REPO_ROOT / protocol["reference_protocol"]
    reference = checked_file(reference_path, protocol["reference_protocol_sha256"])
    if (reference.get("schema") != "perfusion_subject_pilot/v1"
            or reference.get("cohort") != protocol["cohort"]
            or reference.get("sampling") != protocol["sampling"]
            or reference.get("groups") != protocol["groups"]
            or reference.get("conditions") != protocol["conditions"]):
        raise ValueError("reference Subject comparison differs")
    baseline_path = REPO_ROOT / reference["baseline_protocol"]
    baseline = checked_file(baseline_path, reference["baseline_protocol_sha256"])
    if protocol["cohort"] == "balanced_aligned":
        baseline_path = REPO_ROOT / baseline["baseline_protocol"]
        baseline = checked_file(baseline_path, baseline["baseline_protocol_sha256"])
    if (baseline.get("schema") != "subject_material_diagnostic/v1"
            or baseline.get("sampling") != protocol["sampling"]):
        raise ValueError("frozen mailbox diagnostic differs")

    matched = [row for row in reference_rows(reference, baseline) if row["arm"] == "B0"]
    if len(matched) != 18:
        raise ValueError("expected 18 frozen Subject comparison prompts")
    rows = []
    for arm in ARMS:
        for item in matched:
            sample_id = f"{arm}__{item['group']}__{item['condition']}__seed{item['seed']}"
            rows.append({
                "id": sample_id, "arm": arm, "group": item["group"],
                "condition": item["condition"], "prompt": item["prompt"],
                "seed": item["seed"],
                "image_path": f"images/{arm}/{item['group']}/{sample_id}.png",
            })
    return rows


def material_checkpoint(protocol: dict, run_root: Path) -> Path:
    selection_path = REPO_ROOT / protocol["material_selection"]
    selection = checked_file(selection_path, protocol["material_selection_sha256"])
    run = run_root / selection["training_run_relative_to_COLORPEEL_RUN_ROOT"]
    if sha256(run / "manifest.json") != selection["training_manifest_sha256"]:
        raise ValueError("selected Material training manifest changed")
    checkpoint = run / selection["checkpoint"]["directory_relative_to_training_run"]
    expected = {
        MATERIAL_WEIGHT_NAME: selection["checkpoint"]["token_local_kv_weights_sha256"],
        "adaptation_config.json": selection["checkpoint"]["adaptation_config_sha256"],
        "<M*>.bin": selection["checkpoint"]["token_embedding_sha256"],
    }
    if any(sha256(checkpoint / name) != value for name, value in expected.items()):
        raise ValueError("selected Material checkpoint changed")
    return checkpoint


def verify_subject_checkpoint(checkpoint: Path, arm: str, protocol: dict) -> dict:
    manifest_path = checkpoint.parent / "manifest.json"
    manifest = read_json(manifest_path)
    expected_variant = f"{protocol['cohort']}_{arm}_r{protocol['lora']['rank']}_5000"
    if (manifest.get("status") != "succeeded"
            or manifest.get("stage") != "train"
            or manifest.get("run", {}).get("study") != "lora_kv_subject_v1"
            or manifest.get("run", {}).get("variant") != expected_variant):
        raise ValueError(f"Subject training run differs from {expected_variant}: {checkpoint}")
    if (manifest.get("lora_training_data") is not None
            and manifest["lora_training_data"] != protocol["training_data"]):
        raise ValueError(f"Subject training data differs from {protocol['cohort']}: {checkpoint}")
    adaptation = read_json(checkpoint / "adaptation_config.json")
    if (adaptation.get("adaptation_mode") != "lora_subject_kv"
            or adaptation.get("mode") != arm
            or adaptation.get("rank") != protocol["lora"]["rank"]
            or adaptation.get("alpha") != protocol["lora"]["alpha"]
            or adaptation.get("modifier_tokens") != ["<S*>"]
            or adaptation.get("weight_name") != WEIGHT_NAME):
        raise ValueError(f"Subject checkpoint does not match {arm}: {checkpoint}")
    return {name: sha256(checkpoint / name) for name in
            (WEIGHT_NAME, "adaptation_config.json", "<S*>.bin")} | {
                "training_manifest_sha256": sha256(manifest_path),
            }


def token_masks(pipe, prompt: str, guidance_scale: float) -> dict:
    import torch

    ids = pipe.tokenizer(
        prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(pipe.unet.device)
    masks = {}
    for label, token in (("subject", "<S*>"), ("material", "<M*>")):
        mask = ids == pipe.tokenizer.convert_tokens_to_ids(token)
        if int(mask.sum()) != prompt.count(token):
            raise ValueError(f"tokenization changed for {token}: {prompt}")
        if guidance_scale > 1:
            mask = torch.cat([torch.zeros_like(mask), mask], dim=0)
        masks[label] = mask
    return {"modifier_token_mask": masks}


def generate(protocol: dict, checkpoints: dict[str, Path], material_dir: Path,
             rows: list[dict], output_dir: Path, device: str) -> None:
    import torch
    from diffusers import DiffusionPipeline

    train_root = str(REPO_ROOT / "src" / "train")
    if train_root not in sys.path:
        sys.path.insert(0, train_root)
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel
    from experiments.lora_kv_subject_v1.attention import install_subject_lora_kv

    material_state = torch.load(material_dir / MATERIAL_WEIGHT_NAME, map_location="cpu")
    for arm in ARMS:
        checkpoint = checkpoints[arm]
        unet = UNet2DConditionModel.from_pretrained(
            protocol["base_model"], subfolder="unet", local_files_only=True,
            torch_dtype=torch.float16,
        )
        state = torch.load(checkpoint / WEIGHT_NAME, map_location="cpu")
        install_subject_lora_kv(
            unet, state, arm, protocol["lora"]["rank"], protocol["lora"]["alpha"],
            material_state=material_state,
        )
        pipe = DiffusionPipeline.from_pretrained(
            protocol["base_model"], unet=unet, low_cpu_mem_usage=False,
            torch_dtype=torch.float16, local_files_only=True,
        ).to(device)
        if pipe.safety_checker is None:
            raise ValueError("evaluation requires the base safety checker")
        pipe.load_textual_inversion(str(checkpoint), weight_name="<S*>.bin")
        pipe.load_textual_inversion(str(material_dir), weight_name="<M*>.bin")
        with (output_dir / "generation_status.jsonl").open("a", encoding="utf-8") as ledger:
            for row in rows:
                if row["arm"] != arm:
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
        del pipe, unet, state
        gc.collect()
        torch.cuda.empty_cache()


def make_contact_sheets(protocol: dict, rows: list[dict], output_dir: Path) -> dict[str, str]:
    from PIL import Image, ImageDraw

    status = {
        item["id"]: item
        for item in (json.loads(line) for line in
                     (output_dir / "generation_status.jsonl").read_text(encoding="utf-8").splitlines())
    }
    if len(status) != len(rows):
        raise ValueError("contact sheet requires one status for every generated image")
    lookup = {(row["arm"], row["group"], row["condition"], row["seed"]): row for row in rows}
    sheets = {}
    tile, left, top, gap = 256, 170, 34, 8
    conditions = protocol["conditions"]
    seeds = protocol["sampling"]["seeds"]
    sheet_dir = output_dir / "contact_sheets"
    sheet_dir.mkdir()
    for group in protocol["groups"]:
        sheet = Image.new("RGB", (left + 4 * (tile + gap),
                                  top + len(conditions) * len(seeds) * (tile + gap)), "white")
        draw = ImageDraw.Draw(sheet)
        for col, arm in enumerate(ARMS):
            draw.text((left + col * (tile + gap), 8), arm, fill="black")
        for row_index, (condition, seed) in enumerate(
                (condition, seed) for condition in conditions for seed in seeds):
            y = top + row_index * (tile + gap)
            draw.multiline_text((8, y + 8), f"{group}\n{condition}\nseed {seed}", fill="black")
            for col, arm in enumerate(ARMS):
                item = lookup[(arm, group, condition, seed)]
                image_path = output_dir / item["image_path"]
                with Image.open(image_path) as image:
                    thumbnail = image.convert("RGB").resize((tile, tile))
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
    parser.add_argument("--full-kv", type=Path, required=True)
    parser.add_argument("--token-local-kv", type=Path, required=True)
    parser.add_argument("--full-v", type=Path, required=True)
    parser.add_argument("--token-local-v", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    if (protocol.get("schema") != "lora_kv_subject_comparison/v1"
            or protocol.get("base_model") != "CompVis/stable-diffusion-v1-4"
            or protocol.get("arms") != list(ARMS)
            or protocol.get("safety_checker") != "enabled"):
        raise ValueError("unexpected Subject LoRA comparison protocol")
    rows = comparison_rows(protocol)
    if len(rows) != 72:
        raise ValueError("expected 72 four-arm comparison samples")
    checkpoints = {arm: getattr(args, arm) for arm in ARMS}
    run_root = Path(os.environ["COLORPEEL_RUN_ROOT"]).resolve()
    material_dir = material_checkpoint(protocol, run_root)
    checkpoint_hashes = {
        arm: verify_subject_checkpoint(path, arm, protocol)
        for arm, path in checkpoints.items()
    }
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    provenance = {
        "schema": protocol["schema"], "status": "dry_run" if args.dry_run else "running",
        "protocol_sha256": sha256(args.protocol), "cohort": protocol["cohort"],
        "subject_checkpoints": {arm: str(path.resolve()) for arm, path in checkpoints.items()},
        "subject_checkpoint_sha256": checkpoint_hashes,
        "material_checkpoint": str(material_dir.resolve()),
        "material_weights_sha256": sha256(material_dir / MATERIAL_WEIGHT_NAME),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "generation_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if not args.dry_run:
        generate(protocol, checkpoints, material_dir, rows, args.output_dir, args.device)
        provenance["contact_sheet_sha256"] = make_contact_sheets(protocol, rows, args.output_dir)
        provenance["status"] = "succeeded"
        (args.output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(rows)} matched samples: {args.output_dir}")


if __name__ == "__main__":
    main()
