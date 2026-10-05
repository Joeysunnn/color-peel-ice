"""Test the frozen ground-reflection Material LoRA on five object classes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

WEIGHTS = "pytorch_lora_kv_weights.bin"
OBJECTS = ("vase", "chair", "mug", "sofa", "mailbox")
ARMS = ("base", "literal", "token")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(protocol: dict) -> list[dict]:
    result = []
    for obj in OBJECTS:
        stem = f"a photo of a {obj}"
        for seed in protocol["sampling"]["seeds"]:
            for arm, prompt in (
                    ("base", stem),
                    ("literal", f'{stem} made of {protocol["literal_material"]}'),
                    ("token", protocol["prompt_template"].format(object=obj))):
                sample_id = f"{obj}__seed{seed}__{arm}"
                result.append({"id": sample_id, "object": obj, "seed": seed,
                               "arm": arm, "prompt": prompt,
                               "image_path": f"images/{obj}/{sample_id}.png"})
    return result


def token_mask(pipe, prompt: str, guidance_scale: float):
    import torch

    ids = pipe.tokenizer(prompt, padding="max_length",
                         max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids.to(pipe.unet.device)
    mask = ids == pipe.tokenizer.convert_tokens_to_ids("<M*>")
    if int(mask.sum()) != prompt.count("<M*>"):
        raise ValueError(f"Material tokenization changed: {prompt}")
    if guidance_scale > 1:
        mask = torch.cat([torch.zeros_like(mask), mask], dim=0)
    return {"modifier_token_mask": mask}


def make_sheets(rows: list[dict], output: Path) -> dict[str, str]:
    from PIL import Image, ImageDraw

    ledger = [json.loads(line) for line in (output / "status.jsonl").read_text().splitlines()]
    if len(ledger) != len(rows) or {item["id"] for item in ledger} != {row["id"] for row in rows}:
        raise ValueError("Transfer output is incomplete")
    status = {item["id"]: item for item in ledger}
    sheet_dir = output / "contact_sheets"
    sheet_dir.mkdir()
    hashes = {}
    for obj in OBJECTS:
        sheet = Image.new("RGB", (110 + 3 * 264, 42 + 3 * 264), "white")
        draw = ImageDraw.Draw(sheet)
        for col, arm in enumerate(ARMS):
            draw.text((110 + col * 264, 8), arm, fill="black")
        for row_no, seed in enumerate((42, 43, 44)):
            draw.text((8, 42 + row_no * 264), f"seed {seed}", fill="black")
            for col, arm in enumerate(ARMS):
                row = next(row for row in rows if row["object"] == obj and
                           row["seed"] == seed and row["arm"] == arm)
                path = output / row["image_path"]
                if sha(path) != status[row["id"]]["image_sha256"]:
                    raise ValueError(f"Image hash changed: {path}")
                x, y = 110 + col * 264, 42 + row_no * 264
                with Image.open(path) as image:
                    sheet.paste(image.convert("RGB").resize((256, 256)), (x, y))
                if status[row["id"]]["safety_filtered"]:
                    draw.text((x + 4, y + 4), "SAFETY FILTERED", fill="red")
        target = sheet_dir / f"{obj}.jpg"
        sheet.save(target, quality=90)
        hashes[obj] = sha(target)
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (protocol.get("schema") != "lora_kv_material_five_object_transfer/v1"
            or protocol.get("base_model") != "CompVis/stable-diffusion-v1-4"
            or protocol.get("objects") != list(OBJECTS)
            or protocol.get("arms") != list(ARMS)
            or protocol.get("prompt_template") != "a photo of a {object} made of <M*>"
            or protocol.get("literal_material") != "polished metal"
            or protocol.get("sampling") != {"seeds": [42, 43, 44],
                                            "num_inference_steps": 100,
                                            "guidance_scale": 3.5}):
        raise ValueError("Unexpected five-object transfer protocol")
    checkpoint = args.checkpoint.resolve()
    manifest = json.loads((checkpoint.parent / "manifest.json").read_text(encoding="utf-8"))
    adaptation = json.loads((checkpoint / "adaptation_config.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "succeeded" or manifest.get("returncode") != 0
            or manifest.get("run") != {"study": "lora_kv_material_v1",
                                       "variant": "ground_reflection_token_local_kv_r4_5000",
                                       "seed": 42}
            or adaptation.get("adaptation_mode") != "lora_material_kv"
            or adaptation.get("mode") != "token_local_kv"
            or adaptation.get("rank") != 4 or adaptation.get("alpha") != 4
            or adaptation.get("modifier_tokens") != ["<M*>"]
            or adaptation.get("weight_name") != WEIGHTS
            or sha(checkpoint / WEIGHTS) != protocol["checkpoint_weights_sha256"]):
        raise ValueError("Checkpoint differs from the frozen Material LoRA")
    token_hash = sha(checkpoint / "<M*>.bin")
    if token_hash != "ef012da735605359db80033052fff3b06710b1bb081bb59782ac039ce192c7c6":
        raise ValueError("Material token differs from the frozen checkpoint")
    if args.output.exists():
        raise FileExistsError(args.output)
    entries = rows(protocol)
    if len(entries) != 45:
        raise ValueError("Expected 45 transfer rows")
    if args.dry_run:
        print(f"Verified checkpoint and {len(entries)} transfer rows")
        return

    args.output.mkdir(parents=True)
    (args.output / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in entries), encoding="utf-8")
    provenance = {"status": "running", "protocol_sha256": sha(args.protocol),
                  "training_manifest_sha256": sha(checkpoint.parent / "manifest.json"),
                  "checkpoint": str(checkpoint), "weights_sha256": sha(checkpoint / WEIGHTS),
                  "token_sha256": token_hash, "row_count": len(entries)}
    provenance_path = args.output / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    import torch
    from diffusers import DiffusionPipeline, PNDMScheduler

    train_root = str(ROOT / "src" / "train")
    if train_root not in sys.path:
        sys.path.insert(0, train_root)
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel
    from experiments.lora_kv_subject_v1.attention import install_subject_lora_kv

    unet = UNet2DConditionModel.from_pretrained(
        protocol["base_model"], subfolder="unet", local_files_only=True,
        torch_dtype=torch.float16)
    state = torch.load(checkpoint / WEIGHTS, map_location="cpu")
    install_subject_lora_kv(unet, state, "token_local_kv", 4, 4)
    pipe = DiffusionPipeline.from_pretrained(
        protocol["base_model"], unet=unet, low_cpu_mem_usage=False,
        torch_dtype=torch.float16, local_files_only=True).to(args.device)
    if pipe.safety_checker is None or not isinstance(pipe.scheduler, PNDMScheduler):
        raise ValueError("Transfer requires the base safety checker and PNDM scheduler")
    pipe.load_textual_inversion(str(checkpoint), weight_name="<M*>.bin")
    with (args.output / "status.jsonl").open("w", encoding="utf-8") as ledger:
        for row in entries:
            result = pipe(
                row["prompt"],
                num_inference_steps=protocol["sampling"]["num_inference_steps"],
                guidance_scale=protocol["sampling"]["guidance_scale"],
                generator=torch.Generator(device=args.device).manual_seed(row["seed"]),
                cross_attention_kwargs=token_mask(
                    pipe, row["prompt"], protocol["sampling"]["guidance_scale"]),
            )
            path = args.output / row["image_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            result.images[0].save(path)
            flags = getattr(result, "nsfw_content_detected", None)
            if flags is not None and len(flags) != 1:
                raise ValueError("Expected one safety-checker result")
            ledger.write(json.dumps({**row, "image_sha256": sha(path),
                                     "safety_filtered": bool(flags[0]) if flags else False},
                                    sort_keys=True) + "\n")
            ledger.flush()
    if sha(checkpoint / WEIGHTS) != provenance["weights_sha256"] or sha(checkpoint / "<M*>.bin") != token_hash:
        raise ValueError("Material checkpoint changed during inference")
    provenance["contact_sheet_sha256"] = make_sheets(entries, args.output)
    provenance["status"] = "succeeded"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(entries)} images: {args.output}")


if __name__ == "__main__":
    main()
