"""Transfer each material token with its matching early K/V and embedding snapshot."""

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(os.environ.get("COLORPEEL_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
from experiments.natural_material_threeway_v1.evaluate_transfer import masks, pipeline

STEPS = (1000, 2000, 3000)
OBJECTS = ("vase", "sofa", "mug", "mailbox", "chair")
SEEDS = (42, 43, 44, 45, 46)
WEIGHTS = "pytorch_token_local_kv_weights.bin"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def make_sheets(output):
    from PIL import Image, ImageDraw

    folder = output / "contact_sheets"
    folder.mkdir()
    filtered = {(row["step"], row["object"], row["seed"]): row["safety_filtered"]
                for row in (json.loads(line) for line in (output / "status.jsonl").read_text().splitlines())}
    for obj in OBJECTS:
        image = Image.new("RGB", (3 * 264 + 100, 5 * 264 + 35), "white")
        draw = ImageDraw.Draw(image)
        for col, step in enumerate(STEPS):
            draw.text((100 + col * 264, 8), f"step {step}", fill="black")
        for row, seed in enumerate(SEEDS):
            draw.text((8, 35 + row * 264), f"seed {seed}", fill="black")
            for col, step in enumerate(STEPS):
                with Image.open(output / "images" / f"step{step}" / obj / f"seed{seed}.png") as tile:
                    image.paste(tile.convert("RGB").resize((256, 256)),
                                (100 + col * 264, 35 + row * 264))
                if filtered[(step, obj, seed)]:
                    draw.text((104 + col * 264, 39 + row * 264), "SAFETY FILTERED", fill="red")
        image.save(folder / f"{obj}.jpg", quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-run", required=True, type=Path)
    parser.add_argument("--study", required=True,
                        choices=("natural_material_fullmaps_v1", "natural_material_caption_ablation_v1"))
    parser.add_argument("--material", required=True,
                        choices=("mailbox", "metal_spoon", "wood_spoon"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    run = args.checkpoint_run.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    expected_variant = (f"{args.material}_fullmaps_token_local_kv_5000"
                        if args.study == "natural_material_fullmaps_v1" else
                        f"{args.material}_generic_caption_token_local_kv_5000")
    if (manifest.get("status") != "succeeded" or manifest.get("returncode") != 0
            or manifest.get("run") != {"study": args.study, "variant": expected_variant, "seed": 42}):
        raise ValueError("Training manifest does not match the requested completed run")
    final = run / "checkpoints"
    adaptation = json.loads((final / "adaptation_config.json").read_text())
    audit = json.loads((final / "embedding_update_audit.json").read_text())
    if (adaptation["adaptation_mode"] != "token_local_kv"
            or adaptation["modifier_tokens"] != ["<M*>"]
            or audit["modifier_tokens"][0]["token"] != "<M*>"):
        raise ValueError("Unexpected material adaptation or token audit")
    for step in STEPS:
        for name in ("pytorch_model.bin", "pytorch_model_1.bin"):
            if not (final / f"checkpoint-{step}" / name).is_file():
                raise FileNotFoundError(final / f"checkpoint-{step}" / name)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.verify_only:
        print(f"Snapshot files present for {args.study}/{args.material} at {STEPS}")
        return

    import torch

    final_adapter = torch.load(final / WEIGHTS, map_location="cpu", weights_only=True)
    tensor_keys = {key for key, value in final_adapter.items() if isinstance(value, torch.Tensor)}
    empty = {key: value for key, value in final_adapter.items() if not isinstance(value, torch.Tensor)}
    if (len(tensor_keys) != 32 or len(empty) != 16 or
            any(value != {} or not key.endswith("attn1.processor") for key, value in empty.items())):
        raise ValueError("Unexpected final adapter schema")
    token_id = audit["modifier_tokens"][0]["token_id"]
    final_snapshot = final / "checkpoint-5000"
    final_snapshot_adapter = torch.load(final_snapshot / "pytorch_model.bin", map_location="cpu", weights_only=True)
    final_snapshot_encoder = torch.load(final_snapshot / "pytorch_model_1.bin", map_location="cpu", weights_only=True)
    final_token = torch.load(final / "<M*>.bin", map_location="cpu", weights_only=True)["<M*>"]
    if (set(final_snapshot_adapter) != tensor_keys or
            any(not torch.equal(final_snapshot_adapter[key], final_adapter[key]) for key in tensor_keys) or
            not torch.equal(final_snapshot_encoder["text_model.embeddings.token_embedding.weight"][token_id],
                            final_token)):
        raise ValueError("Step 5000 snapshot differs from the final adapter or embedding export")
    del final_snapshot_adapter, final_snapshot_encoder, final_token
    protocol = {"base_model": "CompVis/stable-diffusion-v1-4",
                "sampling": {"seeds": SEEDS, "num_inference_steps": 100, "guidance_scale": 3.5},
                "prompt_template": "a photo of a {object} made of <M*>"}
    output = args.output.resolve()
    output.mkdir(parents=True)
    rows = [{"step": step, "object": obj, "seed": seed,
             "prompt": protocol["prompt_template"].format(object=obj),
             "image_path": f"images/step{step}/{obj}/seed{seed}.png"}
            for step in STEPS for obj in OBJECTS for seed in SEEDS]
    (output / "manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    provenance = {"status": "running", "study": args.study, "material": args.material,
                  "training_run": str(run), "training_manifest_sha256": sha(run / "manifest.json"),
                  "training_git_commit": manifest["git"]["commit"], "protocol": protocol,
                  "source_snapshots": {}, "device": args.device,
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "row_count": len(rows)}
    write_json(output / "provenance.json", provenance)
    with (output / "status.jsonl").open("w") as ledger:
        for step in STEPS:
            source = final / f"checkpoint-{step}"
            adapter_path = source / "pytorch_model.bin"
            encoder_path = source / "pytorch_model_1.bin"
            adapter = torch.load(adapter_path, map_location="cpu", weights_only=True)
            encoder = torch.load(encoder_path, map_location="cpu", weights_only=True)
            if set(adapter) != tensor_keys or any(adapter[key].shape != final_adapter[key].shape for key in adapter):
                raise ValueError(f"Adapter schema mismatch at step {step}")
            token = encoder["text_model.embeddings.token_embedding.weight"][token_id].clone().contiguous()
            checkpoint = output / "derived_checkpoints" / f"step{step}"
            checkpoint.mkdir(parents=True)
            adapter.update(empty)
            torch.save(adapter, checkpoint / WEIGHTS)
            torch.save({"<M*>": token}, checkpoint / "<M*>.bin")
            shutil.copy2(final / "adaptation_config.json", checkpoint / "adaptation_config.json")
            provenance["source_snapshots"][str(step)] = {
                "adapter_sha256": sha(adapter_path), "text_encoder_sha256": sha(encoder_path),
                "derived_adapter_sha256": sha(checkpoint / WEIGHTS),
                "derived_embedding_sha256": sha(checkpoint / "<M*>.bin")}
            del adapter, encoder, token
            pipe = pipeline(protocol, checkpoint, None, False, args.device)
            for row in (row for row in rows if row["step"] == step):
                result = pipe(row["prompt"], num_inference_steps=100, guidance_scale=3.5,
                              generator=torch.Generator(device=args.device).manual_seed(row["seed"]),
                              cross_attention_kwargs=masks(pipe, row["prompt"], 3.5, False))
                target = output / row["image_path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                result.images[0].save(target)
                flags = getattr(result, "nsfw_content_detected", None)
                ledger.write(json.dumps({**row, "image_sha256": sha(target),
                                         "safety_filtered": bool(flags[0]) if flags else False},
                                        sort_keys=True) + "\n")
                ledger.flush()
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            write_json(output / "provenance.json", provenance)
    make_sheets(output)
    provenance["contact_sheet_sha256"] = {obj: sha(output / "contact_sheets" / f"{obj}.jpg")
                                          for obj in OBJECTS}
    provenance["status"] = "succeeded"
    write_json(output / "provenance.json", provenance)
    print(f"Completed {len(rows)} images: {output}")


if __name__ == "__main__":
    main()
