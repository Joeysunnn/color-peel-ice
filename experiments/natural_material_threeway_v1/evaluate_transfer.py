"""Matched material-only and fixed-subject transfer tests for one learned M token."""

import argparse
import gc
import hashlib
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/train"))

WEIGHTS = "pytorch_token_local_kv_weights.bin"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(protocol, material):
    result = []
    for obj, color, seed in itertools.product(
            protocol["objects"], protocol["colors"], protocol["sampling"]["seeds"]):
        stem = f"a photo of a {color} {obj}"
        for arm, prompt in (("base", stem),
                            ("literal", f'{stem} made of {protocol["literal_material"][material]}'),
                            ("token", f"{stem} made of <M*>")):
            result.append({"arm": arm, "object": obj, "color": color,
                           "seed": seed, "prompt": prompt,
                           "id": f"{arm}__{obj}__{color}__{seed}"})
    for color, seed in itertools.product(protocol["colors"], protocol["sampling"]["seeds"]):
        stem = f"a photo of <S*> mailbox in {color} color"
        for arm, prompt in (("subject_only", stem),
                            ("subject_material", f"{stem} made of <M*>")):
            result.append({"arm": arm, "object": "subject_mailbox", "color": color,
                           "seed": seed, "prompt": prompt,
                           "id": f"{arm}__subject_mailbox__{color}__{seed}"})
    return result


def masks(pipe, prompt, guidance, dual):
    import torch
    ids = pipe.tokenizer(prompt, padding="max_length",
                         max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids.to(pipe.unet.device)
    names = (("subject", "<S*>"), ("material", "<M*>")) if dual else (("material", "<M*>"),)
    values = {}
    for label, token in names:
        value = ids == pipe.tokenizer.convert_tokens_to_ids(token)
        if int(value.sum()) != prompt.count(token):
            raise ValueError(f"Tokenization differs: {prompt}")
        if guidance > 1:
            value = torch.cat([torch.zeros_like(value), value], dim=0)
        values[label] = value
    return {"modifier_token_mask": values if dual else values["material"]}


def pipeline(protocol, material_checkpoint, subject_checkpoint, dual, device):
    import torch
    from diffusers import DiffusionPipeline, PNDMScheduler
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel
    from src.methods.colorpeel_ice.dual_token_local_kv import install_dual_token_local_kv

    unet = UNet2DConditionModel.from_pretrained(
        protocol["base_model"], subfolder="unet", local_files_only=True,
        torch_dtype=torch.float16)
    if dual:
        unet.load_attn_procs(str(subject_checkpoint), weight_name=WEIGHTS,
                             adaptation_mode="token_local_kv")
        material_state = torch.load(material_checkpoint / WEIGHTS, map_location="cpu")
        install_dual_token_local_kv(unet, material_state, primary_label="subject")
    else:
        unet.load_attn_procs(str(material_checkpoint), weight_name=WEIGHTS,
                             adaptation_mode="token_local_kv")
    pipe = DiffusionPipeline.from_pretrained(
        protocol["base_model"], unet=unet, low_cpu_mem_usage=False,
        torch_dtype=torch.float16, local_files_only=True).to(device)
    if pipe.safety_checker is None or not isinstance(pipe.scheduler, PNDMScheduler):
        raise ValueError("Transfer evaluation requires enabled safety checker and PNDM")
    pipe.load_textual_inversion(str(material_checkpoint), weight_name="<M*>.bin")
    if dual:
        pipe.load_textual_inversion(str(subject_checkpoint), weight_name="<S*>.bin")
    return pipe


def contact_sheets(all_rows, output):
    from PIL import Image, ImageDraw
    for obj in ("cone", "mailbox", "subject_mailbox"):
        for color in ("gray", "red", "blue"):
            group = [row for row in all_rows if row["object"] == obj and row["color"] == color]
            arms = ("subject_only", "subject_material") if obj == "subject_mailbox" else ("base", "literal", "token")
            sheet = Image.new("RGB", (len(arms) * 264 + 120, 3 * 264 + 45), "white")
            draw = ImageDraw.Draw(sheet)
            for col, arm in enumerate(arms):
                draw.text((120 + col * 264, 8), arm, fill="black")
            for row_no, seed in enumerate((42, 43, 44)):
                draw.text((8, 45 + row_no * 264), f"seed {seed}", fill="black")
                for col, arm in enumerate(arms):
                    row = next(row for row in group if row["arm"] == arm and row["seed"] == seed)
                    with Image.open(output / "images" / f'{row["id"]}.png') as image:
                        sheet.paste(image.convert("RGB").resize((256, 256)),
                                    (120 + col * 264, 45 + row_no * 264))
            target = output / "contact_sheets" / f"{obj}__{color}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(target, quality=90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--material-id", choices=("mailbox", "metal_spoon", "wood_spoon"), required=True)
    parser.add_argument("--material-checkpoint", type=Path, required=True)
    parser.add_argument("--subject-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    for path, name in ((args.material_checkpoint, "<M*>.bin"),
                       (args.subject_checkpoint, "<S*>.bin")):
        for file in (name, WEIGHTS, "adaptation_config.json"):
            if not (path / file).is_file():
                raise FileNotFoundError(path / file)
    if sha(args.subject_checkpoint / WEIGHTS) != protocol["subject_weights_sha256"]:
        raise ValueError("Fixed Subject checkpoint hash differs")
    training = json.loads((args.material_checkpoint.parent / "manifest.json").read_text())
    if (training.get("status") != "succeeded" or training.get("returncode") != 0
            or training.get("run", {}).get("study") != "natural_material_threeway_v1"
            or training.get("run", {}).get("variant") != f"{args.material_id}_token_local_kv_5000"):
        raise ValueError("Material checkpoint is not a completed three-way training run")
    adaptation = json.loads((args.material_checkpoint / "adaptation_config.json").read_text())
    if adaptation["adaptation_mode"] != "token_local_kv" or adaptation["modifier_tokens"] != ["<M*>"]:
        raise ValueError("Material checkpoint is not the requested original token-local K/V")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    entries = rows(protocol, args.material_id)
    (args.output / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in entries))
    provenance = {"status": "running", "protocol_sha256": sha(args.protocol),
                  "material_id": args.material_id,
                  "material_checkpoint": str(args.material_checkpoint.resolve()),
                  "material_weights_sha256": sha(args.material_checkpoint / WEIGHTS),
                  "training_manifest_sha256": sha(args.material_checkpoint.parent / "manifest.json"),
                  "subject_checkpoint": str(args.subject_checkpoint.resolve()),
                  "subject_weights_sha256": sha(args.subject_checkpoint / WEIGHTS),
                  "row_count": len(entries)}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    import torch
    with (args.output / "status.jsonl").open("w") as ledger:
        for dual in (False, True):
            pipe = pipeline(protocol, args.material_checkpoint, args.subject_checkpoint, dual, args.device)
            selected = [row for row in entries if (row["object"] == "subject_mailbox") == dual]
            for row in selected:
                result = pipe(row["prompt"],
                              num_inference_steps=protocol["sampling"]["num_inference_steps"],
                              guidance_scale=protocol["sampling"]["guidance_scale"],
                              generator=torch.Generator(device=args.device).manual_seed(row["seed"]),
                              cross_attention_kwargs=masks(
                                  pipe, row["prompt"], protocol["sampling"]["guidance_scale"], dual))
                target = args.output / "images" / f'{row["id"]}.png'
                target.parent.mkdir(parents=True, exist_ok=True)
                result.images[0].save(target)
                flags = getattr(result, "nsfw_content_detected", None)
                ledger.write(json.dumps({**row, "image_sha256": sha(target),
                                         "safety_filtered": bool(flags[0]) if flags else False}) + "\n")
                ledger.flush()
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
    contact_sheets(entries, args.output)
    provenance["status"] = "succeeded"
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
