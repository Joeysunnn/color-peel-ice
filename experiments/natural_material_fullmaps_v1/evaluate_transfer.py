"""Matched shape, source-color, and fixed-subject tests for full-map M tokens."""

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("COLORPEEL_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
from experiments.natural_material_threeway_v1.evaluate_transfer import masks, pipeline


WEIGHTS = "pytorch_token_local_kv_weights.bin"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(protocol, material):
    result = []
    for obj in protocol["objects"]:
        stem = f"a photo of a {obj}"
        for seed in protocol["sampling"]["seeds"]:
            for arm, prompt in (("base", stem),
                                ("literal", f'{stem} made of {protocol["literal_material"][material]}'),
                                ("token", f"{stem} made of <M*>")):
                result.append({"arm": arm, "object": obj, "seed": seed,
                               "prompt": prompt, "id": f"{arm}__{obj}__{seed}"})
    stem = "a photo of <S*> mailbox"
    for seed in protocol["sampling"]["seeds"]:
        for arm, prompt in (("subject_only", stem),
                            ("subject_material", f"{stem} made of <M*>")):
            result.append({"arm": arm, "object": "subject_mailbox", "seed": seed,
                           "prompt": prompt, "id": f"{arm}__subject_mailbox__{seed}"})
    return result


def contact_sheets(entries, output):
    from PIL import Image, ImageDraw
    for obj in ("cone", "mailbox", "subject_mailbox"):
        arms = ("subject_only", "subject_material") if obj == "subject_mailbox" else ("base", "literal", "token")
        sheet = Image.new("RGB", (len(arms) * 264 + 120, 3 * 264 + 45), "white")
        draw = ImageDraw.Draw(sheet)
        for col, arm in enumerate(arms):
            draw.text((120 + col * 264, 8), arm, fill="black")
        for row_no, seed in enumerate((42, 43, 44)):
            draw.text((8, 45 + row_no * 264), f"seed {seed}", fill="black")
            for col, arm in enumerate(arms):
                row = next(row for row in entries if row["object"] == obj
                           and row["arm"] == arm and row["seed"] == seed)
                with Image.open(output / "images" / f'{row["id"]}.png') as image:
                    sheet.paste(image.convert("RGB").resize((256, 256)),
                                (120 + col * 264, 45 + row_no * 264))
        target = output / "contact_sheets" / f"{obj}.jpg"
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
            or training.get("run", {}).get("study") != "natural_material_fullmaps_v1"
            or training.get("run", {}).get("variant") !=
            f"{args.material_id}_fullmaps_token_local_kv_5000"):
        raise ValueError("Material checkpoint is not a completed full-map training run")
    adaptation = json.loads((args.material_checkpoint / "adaptation_config.json").read_text())
    if adaptation["adaptation_mode"] != "token_local_kv" or adaptation["modifier_tokens"] != ["<M*>"]:
        raise ValueError("Material checkpoint is not original token-local K/V")
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
