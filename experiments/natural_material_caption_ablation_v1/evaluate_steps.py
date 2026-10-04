"""Evaluate saved material token-local K/V snapshots on fixed shape prompts."""

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(os.environ.get("COLORPEEL_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT))
from experiments.natural_material_threeway_v1.evaluate_transfer import masks, pipeline

WEIGHTS = "pytorch_token_local_kv_weights.bin"
STEPS = (1000, 2000, 3000, 4000, 5000)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def entries(protocol):
    return [{"id": f"step{step}__{obj}__{seed}", "step": step,
             "object": obj, "seed": seed,
             "prompt": protocol["prompt_template"].format(object=obj),
             "image_path": f"images/step{step}/{obj}/seed{seed}.png"}
            for step in STEPS for obj in protocol["objects"]
            for seed in protocol["sampling"]["seeds"]]


def make_sheets(output, rows):
    from PIL import Image, ImageDraw

    ledger = {row["id"]: row for row in
              (json.loads(line) for line in (output / "status.jsonl").read_text().splitlines())}
    if len(ledger) != len(rows):
        raise ValueError("Checkpoint progression output is incomplete")
    hashes = {}
    folder = output / "contact_sheets"
    folder.mkdir()
    tile, left, top, gap = 256, 100, 34, 8
    for obj in ("cone", "mailbox"):
        sheet = Image.new("RGB", (left + len(STEPS) * (tile + gap),
                                  top + 3 * (tile + gap)), "white")
        draw = ImageDraw.Draw(sheet)
        for col, step in enumerate(STEPS):
            draw.text((left + col * (tile + gap), 8), f"step {step}", fill="black")
        for row_index, seed in enumerate((42, 43, 44)):
            y = top + row_index * (tile + gap)
            draw.text((8, y + 8), f"seed {seed}", fill="black")
            for col, step in enumerate(STEPS):
                row = ledger[f"step{step}__{obj}__{seed}"]
                source = output / row["image_path"]
                if sha(source) != row["image_sha256"]:
                    raise ValueError(f"Generated image hash changed: {source}")
                with Image.open(source) as image:
                    sheet.paste(image.convert("RGB").resize((tile, tile)),
                                (left + col * (tile + gap), y))
                if row["safety_filtered"]:
                    draw.text((left + col * (tile + gap) + 4, y + 4),
                              "SAFETY FILTERED", fill="red")
        target = folder / f"{obj}.jpg"
        sheet.save(target, quality=90)
        hashes[obj] = sha(target)
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-snapshots-only", action="store_true")
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    if (protocol.get("schema_version") != 1 or protocol.get("material_id") != "metal_spoon"
            or protocol.get("steps") != list(STEPS)
            or protocol.get("objects") != ["cone", "mailbox"]
            or protocol.get("sampling") != {"seeds": [42, 43, 44],
                                            "num_inference_steps": 100,
                                            "guidance_scale": 3.5}
            or protocol.get("prompt_template") != "a photo of a {object} made of <M*>"
            or protocol.get("base_model") != "CompVis/stable-diffusion-v1-4"
            or protocol.get("reuse_step_5000_transfer") is not False):
        raise ValueError("Unexpected checkpoint progression protocol")
    training = read_json(args.checkpoint_run / "manifest.json")
    if (training.get("status") != "succeeded" or training.get("returncode") != 0
            or training.get("run", {}).get("study") != "natural_material_caption_ablation_v1"
            or training.get("run", {}).get("variant") != "metal_spoon_generic_caption_token_local_kv_5000"):
        raise ValueError("Incorrect completed material training run")
    final = args.checkpoint_run / "checkpoints"
    audit = read_json(final / "embedding_update_audit.json")
    token = audit["modifier_tokens"][0]
    if token["token"] != "<M*>" or token["exposure_steps"] != 5000:
        raise ValueError("Unexpected embedding audit")
    adaptation = read_json(final / "adaptation_config.json")
    if adaptation["adaptation_mode"] != "token_local_kv" or adaptation["modifier_tokens"] != ["<M*>"]:
        raise ValueError("Unexpected final adaptation")
    for step in STEPS:
        snapshot = final / f"checkpoint-{step}"
        for name in ("pytorch_model.bin", "pytorch_model_1.bin"):
            if not (snapshot / name).is_file():
                raise FileNotFoundError(snapshot / name)
    rows = entries(protocol)
    if len(rows) != 30 or args.output.exists():
        raise ValueError("Expected 30 rows and a fresh output directory")
    if args.dry_run:
        print("Verified 30 rows and five snapshots")
        return

    import torch

    final_token = torch.load(final / "<M*>.bin", map_location="cpu", weights_only=True)["<M*>"]
    final_adapter = torch.load(final / WEIGHTS, map_location="cpu", weights_only=True)
    final_tensor_keys = {key for key, value in final_adapter.items() if isinstance(value, torch.Tensor)}
    non_tensor_entries = {key: value for key, value in final_adapter.items()
                          if not isinstance(value, torch.Tensor)}
    if (len(final_tensor_keys) != 32 or len(non_tensor_entries) != 16
            or any(value != {} or not key.endswith("attn1.processor")
                   for key, value in non_tensor_entries.items())):
        raise ValueError("Unexpected final attention processor schema")
    token_id = token["token_id"]
    snapshots = {}
    for step in STEPS:
        snapshot = final / f"checkpoint-{step}"
        adapter_path, encoder_path = snapshot / "pytorch_model.bin", snapshot / "pytorch_model_1.bin"
        adapter = torch.load(adapter_path, map_location="cpu", weights_only=True)
        encoder = torch.load(encoder_path, map_location="cpu", weights_only=True)
        row = encoder["text_model.embeddings.token_embedding.weight"][token_id].clone().contiguous()
        if set(adapter) != final_tensor_keys or any(
                adapter[key].shape != final_adapter[key].shape for key in adapter):
            raise ValueError(f"Adapter schema mismatch at step {step}")
        if step == 5000 and (not torch.equal(row, final_token) or any(
                not torch.equal(adapter[key], final_adapter[key]) for key in adapter)):
            raise ValueError("Final snapshot does not match the exported checkpoint")
        adapter.update(non_tensor_entries)
        snapshots[step] = (adapter, row, sha(adapter_path), sha(encoder_path))
        del encoder
    if args.verify_snapshots_only:
        print("Verified five paired K/V and token snapshots; step 5000 exactly matches the final export")
        return

    args.output.mkdir(parents=True)
    (args.output / "manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    provenance = {"status": "running", "protocol_sha256": sha(args.protocol),
                  "training_manifest_sha256": sha(args.checkpoint_run / "manifest.json"),
                  "checkpoint_run": str(args.checkpoint_run.resolve()),
                  "final_adapter_sha256": sha(final / WEIGHTS),
                  "source_snapshots": {}, "device": args.device,
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "row_count": len(rows)}
    provenance_path = args.output / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    with (args.output / "status.jsonl").open("w") as ledger:
        for step in STEPS:
            adapter, row, adapter_sha, encoder_sha = snapshots.pop(step)
            provenance["source_snapshots"][str(step)] = {
                "adapter_sha256": adapter_sha, "text_encoder_sha256": encoder_sha}
            checkpoint = args.output / "derived_checkpoints" / f"step{step}"
            checkpoint.mkdir(parents=True)
            torch.save(adapter, checkpoint / WEIGHTS)
            torch.save({"<M*>": row}, checkpoint / "<M*>.bin")
            shutil.copy2(final / "adaptation_config.json", checkpoint / "adaptation_config.json")
            provenance["source_snapshots"][str(step)].update({
                "derived_adapter_sha256": sha(checkpoint / WEIGHTS),
                "derived_token_sha256": sha(checkpoint / "<M*>.bin")})
            del adapter, row
            pipe = pipeline(protocol, checkpoint, None, False, args.device)
            for item in rows:
                if item["step"] != step:
                    continue
                result = pipe(item["prompt"],
                              num_inference_steps=protocol["sampling"]["num_inference_steps"],
                              guidance_scale=protocol["sampling"]["guidance_scale"],
                              generator=torch.Generator(device=args.device).manual_seed(item["seed"]),
                              cross_attention_kwargs=masks(
                                  pipe, item["prompt"], protocol["sampling"]["guidance_scale"], False))
                target = args.output / item["image_path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                result.images[0].save(target)
                flags = getattr(result, "nsfw_content_detected", None)
                filtered = bool(flags[0]) if flags else False
                ledger.write(json.dumps({**item, "image_sha256": sha(target),
                                         "safety_filtered": filtered}, sort_keys=True) + "\n")
                ledger.flush()
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    provenance["contact_sheet_sha256"] = make_sheets(args.output, rows)
    provenance["status"] = "succeeded"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Completed {len(rows)} checkpoint rows: {args.output}")


if __name__ == "__main__":
    main()
