"""Matched CLEVR-pilot versus natural-material token-local K/V transfer inference."""

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(os.environ.get("COLORPEEL_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT))
from experiments.natural_material_threeway_v1.evaluate_transfer import masks, pipeline


WEIGHTS = "pytorch_token_local_kv_weights.bin"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompts():
    for obj in ("mailbox", "cone"):
        for prefix in ("plain", "photo"):
            stem = f"a {obj}" if prefix == "plain" else f"a photo of a {obj}"
            yield obj, prefix, stem


def rows(literal):
    result = []
    for obj, prefix, stem in prompts():
        for seed in range(42, 47):
            for arm, prompt in (
                ("base", stem),
                ("literal", f"{stem} made of {literal}"),
                ("old_token", f"{stem} made of <M*>"),
                ("new_token", f"{stem} made of <M*>"),
            ):
                result.append({"id": f"{obj}__{prefix}__{seed}__{arm}",
                               "object": obj, "prefix": prefix, "seed": seed,
                               "arm": arm, "prompt": prompt})
    return result


def sheet(entries, out):
    arms = ("base", "literal", "old_token", "new_token")
    for obj, prefix, _ in prompts():
        canvas = Image.new("RGB", (4 * 264 + 110, 5 * 264 + 50), "white")
        draw = ImageDraw.Draw(canvas)
        for col, arm in enumerate(arms):
            draw.text((110 + 264 * col, 10), arm, fill="black")
        for row_no, seed in enumerate(range(42, 47)):
            draw.text((8, 50 + row_no * 264), str(seed), fill="black")
            for col, arm in enumerate(arms):
                row = next(r for r in entries if r["object"] == obj and
                           r["prefix"] == prefix and r["seed"] == seed and
                           r["arm"] == arm)
                with Image.open(out / "images" / f'{row["id"]}.png') as im:
                    canvas.paste(im.convert("RGB").resize((256, 256)),
                                 (110 + col * 264, 50 + row_no * 264))
        target = out / "contact_sheets" / f"{obj}__{prefix}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(target, quality=92)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-checkpoint", type=Path, required=True)
    parser.add_argument("--new-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--literal", default="polished stainless steel")
    args = parser.parse_args()
    for checkpoint in (args.old_checkpoint, args.new_checkpoint):
        for name in ("<M*>.bin", WEIGHTS, "adaptation_config.json"):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(checkpoint / name)
        adaptation = json.loads((checkpoint / "adaptation_config.json").read_text())
        if adaptation["adaptation_mode"] != "token_local_kv" or adaptation["modifier_tokens"] != ["<M*>"]:
            raise ValueError(f"Unexpected adaptation: {checkpoint}")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    entries = rows(args.literal)
    (args.output / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in entries))
    protocol = {"base_model": "CompVis/stable-diffusion-v1-4", "sampling": {
        "num_inference_steps": 100, "guidance_scale": 3.5, "seeds": list(range(42, 47))},
        "objects": ["mailbox", "cone"], "prefixes": ["plain", "photo"],
        "literal": args.literal}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    provenance = {"status": "running", "script_sha256": sha(Path(__file__)),
                  "old_checkpoint": str(args.old_checkpoint.resolve()),
                  "new_checkpoint": str(args.new_checkpoint.resolve()),
                  "old_weights_sha256": sha(args.old_checkpoint / WEIGHTS),
                  "new_weights_sha256": sha(args.new_checkpoint / WEIGHTS),
                  "protocol": protocol, "expected_images": len(entries)}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    import numpy as np
    import torch
    from diffusers import DiffusionPipeline, PNDMScheduler
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel

    def render(pipe, row):
        result = pipe(row["prompt"], num_inference_steps=100,
                      guidance_scale=3.5,
                      generator=torch.Generator(device=args.device).manual_seed(row["seed"]),
                      cross_attention_kwargs=masks(pipe, row["prompt"], 3.5, False))
        flags = getattr(result, "nsfw_content_detected", None)
        return result.images[0], bool(flags[0]) if flags else False

    with (args.output / "status.jsonl").open("w") as ledger:
        for arm, checkpoint in (("old", args.old_checkpoint), ("new", args.new_checkpoint)):
            pipe = pipeline(protocol, checkpoint, checkpoint, False, args.device)
            selected = [row for row in entries if row["arm"] in
                        (("base", "literal", "old_token") if arm == "old" else ("new_token",))]
            if arm == "new":
                # Zero modifier masks should leave ordinary prompts exactly unchanged.
                probe = next(row for row in entries if row["arm"] == "base")
                probe_image, _ = render(pipe, probe)
                with Image.open(args.output / "images" / f'{probe["id"]}.png') as old_image:
                    if not np.array_equal(np.asarray(probe_image), np.asarray(old_image)):
                        raise ValueError("Base prompt changed between material checkpoints")
                provenance["base_control_equal_across_checkpoints"] = True
                (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
            for row in selected:
                image, filtered = render(pipe, row)
                target = args.output / "images" / f'{row["id"]}.png'
                target.parent.mkdir(parents=True, exist_ok=True)
                image.save(target)
                ledger.write(json.dumps({**row, "checkpoint": arm,
                                         "weights_sha256": provenance[f"{arm}_weights_sha256"],
                                         "image_sha256": sha(target),
                                         "safety_filtered": filtered}) + "\n")
                ledger.flush()
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
    sheet(entries, args.output)
    provenance["status"] = "succeeded"
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
