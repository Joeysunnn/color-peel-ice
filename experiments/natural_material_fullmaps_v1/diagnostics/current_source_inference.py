"""Matched transfer protocol for additional natural material sources."""

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from matched_inference import WEIGHTS, masks, pipeline, prompts, sha


def rows(literal):
    for obj, prefix, stem in prompts():
        for seed in range(42, 47):
            for arm, prompt in (("base", stem),
                                ("literal", f"{stem} made of {literal}"),
                                ("token", f"{stem} made of <M*>")):
                yield {"id": f"{obj}__{prefix}__{seed}__{arm}",
                       "object": obj, "prefix": prefix, "seed": seed,
                       "arm": arm, "prompt": prompt}


def sheet(entries, out):
    arms = ("base", "literal", "token")
    for obj, prefix, _ in prompts():
        canvas = Image.new("RGB", (3 * 264 + 110, 5 * 264 + 50), "white")
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
    parser.add_argument("--material-id", choices=("mailbox", "wood_spoon"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    literal = {"mailbox": "red painted metal", "wood_spoon": "light wood"}[args.material_id]
    for name in ("<M*>.bin", WEIGHTS, "adaptation_config.json"):
        if not (args.checkpoint / name).is_file():
            raise FileNotFoundError(args.checkpoint / name)
    adaptation = json.loads((args.checkpoint / "adaptation_config.json").read_text())
    if adaptation["adaptation_mode"] != "token_local_kv" or adaptation["modifier_tokens"] != ["<M*>"]:
        raise ValueError("Checkpoint is not original token-local K/V")
    old_provenance = json.loads((args.base_run / "provenance.json").read_text())
    if old_provenance["status"] != "succeeded" or not old_provenance["base_control_equal_across_checkpoints"]:
        raise ValueError("Shared base controls were not verified")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    (args.output / "images").mkdir()
    entries = list(rows(literal))
    (args.output / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in entries))
    provenance = {"status": "running", "script_sha256": sha(Path(__file__)),
                  "checkpoint": str(args.checkpoint.resolve()),
                  "weights_sha256": sha(args.checkpoint / WEIGHTS),
                  "base_control_source": str(args.base_run.resolve()),
                  "base_control_source_provenance_sha256": sha(args.base_run / "provenance.json"),
                  "material_id": args.material_id, "literal": literal,
                  "expected_rendered_images": 40, "expected_reused_base_images": 20}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    import torch

    protocol = old_provenance["protocol"]
    pipe = pipeline(protocol, args.checkpoint, args.checkpoint, False, args.device)

    def render(row):
        result = pipe(row["prompt"], num_inference_steps=100,
                      guidance_scale=3.5,
                      generator=torch.Generator(device=args.device).manual_seed(row["seed"]),
                      cross_attention_kwargs=masks(pipe, row["prompt"], 3.5, False))
        flags = getattr(result, "nsfw_content_detected", None)
        return result.images[0], bool(flags[0]) if flags else False

    probe = next(row for row in entries if row["arm"] == "base")
    probe_image, _ = render(probe)
    with Image.open(args.base_run / "images" / f'{probe["id"]}.png') as old_image:
        if not np.array_equal(np.asarray(probe_image), np.asarray(old_image)):
            raise ValueError("Base prompt differs from shared controls")
    provenance["base_control_equal_across_checkpoints"] = True
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with (args.output / "status.jsonl").open("w") as ledger:
        for row in entries:
            target = args.output / "images" / f'{row["id"]}.png'
            if row["arm"] == "base":
                source = args.base_run / "images" / target.name
                shutil.copy2(source, target)
                filtered = False
                source_status = next(json.loads(line) for line in
                                     (args.base_run / "status.jsonl").read_text().splitlines()
                                     if json.loads(line)["id"] == row["id"])
                filtered = source_status["safety_filtered"]
                origin = str(source)
            else:
                image, filtered = render(row)
                image.save(target)
                origin = None
            ledger.write(json.dumps({**row, "checkpoint": args.material_id,
                                     "weights_sha256": provenance["weights_sha256"],
                                     "image_sha256": sha(target),
                                     "safety_filtered": filtered,
                                     "base_control_origin": origin}) + "\n")
            ledger.flush()
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    sheet(entries, args.output)
    provenance["status"] = "succeeded"
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
