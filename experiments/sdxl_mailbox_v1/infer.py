"""Matched single-branch SDXL transfers with paired embeddings and KV snapshots."""

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import PNDMScheduler
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from PIL import Image
from transformers import CLIPImageProcessor

from .attention import load_adapter_state_dict
from .conditioning import encode_conditioning
from .contract import transfer_cases
from .train import load_model, tensor_hash, write_json


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_snapshot(snapshot, config, step, embeddings):
    if snapshot.get("schema") != "sdxl_token_local_v1" or snapshot["step"] != step:
        raise ValueError("wrong snapshot schema or step")
    if any(snapshot["config"].get(key) != value for key, value in config.items()):
        raise ValueError("snapshot recipe differs from inference config")
    vectors = snapshot["embeddings"]
    if len(vectors) != 2:
        raise ValueError("both SDXL embeddings are required")
    for vector, embedding in zip(vectors, embeddings):
        if vector.shape != embedding.vector.shape or not torch.isfinite(vector).all():
            raise ValueError("invalid SDXL embedding")
    if snapshot["ordinary_embedding_hashes"] != [tensor_hash(e.base.weight) for e in embeddings]:
        raise ValueError("frozen vocabulary differs from training")


def review(output, rows):
    content = ['<!doctype html><meta charset="utf-8"><title>SDXL single branch transfer</title>',
               '<style>body{font:16px sans-serif;background:#eee}section{display:inline-block;width:320px;vertical-align:top;margin:8px}img{width:320px}p{overflow-wrap:anywhere}</style>',
               '<h1>SDXL single branch transfer</h1>']
    for row in rows:
        content.append(f'<section><img loading="lazy" src="{html.escape(row["image"])}"><p>'
                       f'{html.escape(row["prompt"])}<br>step {row["step"]}, seed {row["seed"]}'
                       f'<br>safety flagged: {row["nsfw"]}</p></section>')
    (output / "index.html").write_text("\n".join(content), encoding="utf8")


def run(config, training, output):
    training, output = Path(training), Path(output)
    if json.loads((training / "status.json").read_text())["status"] != "succeeded":
        raise ValueError("training must finish successfully before transfer")
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    status = {"status": "running", "generated_images": 0, "completed_steps": []}
    write_json(output / "status.json", status)
    records = []
    try:
        pipe, tokenizers, encoders, embeddings = load_model(config, training=False)
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        pipe.vae.to(device="cuda", dtype=torch.float32)
        pipe.vae.enable_slicing()
        checker = StableDiffusionSafetyChecker.from_pretrained(
            config["model_dir"], subfolder="safety_checker", local_files_only=True,
            torch_dtype=torch.float32).to("cuda").eval()
        extractor = CLIPImageProcessor.from_pretrained(
            config["model_dir"], subfolder="feature_extractor", local_files_only=True)
        cases = transfer_cases(config["branch"], config["checkpoints"], smoke=config["smoke"])
        status["expected_images"] = len(cases)
        with torch.inference_mode(), (output / "images.jsonl").open("w", encoding="utf8", buffering=1) as log:
            for step in config["checkpoints"]:
                path = training / f"checkpoint-{step}.pt"
                snapshot = torch.load(path, map_location="cpu", weights_only=False)
                check_snapshot(snapshot, config, step, embeddings)
                load_adapter_state_dict(pipe.unet, snapshot["adapter"])
                for vector, embedding in zip(snapshot["embeddings"], embeddings):
                    embedding.vector.copy_(vector.to(embedding.vector.device))
                snapshot_hash = sha256(path)
                del snapshot
                prompt_cache = {}
                for case in (item for item in cases if item["step"] == step):
                    prompt = case["prompt"]
                    if prompt not in prompt_cache:
                        hidden, pooled, gate = encode_conditioning(tokenizers, encoders, prompt, config["token"])
                        prompt_cache[prompt] = (hidden.half(), pooled.half(), gate)
                    hidden, pooled, gate = prompt_cache[prompt]
                    cfg_gate = torch.cat((torch.zeros_like(gate), gate))
                    generator = torch.Generator(device="cuda").manual_seed(case["seed"])
                    with torch.autocast("cuda", dtype=torch.float16):
                        latent = pipe(prompt_embeds=hidden, pooled_prompt_embeds=pooled,
                            negative_prompt_embeds=torch.zeros_like(hidden),
                            negative_pooled_prompt_embeds=torch.zeros_like(pooled),
                            width=1024, height=1024, original_size=(512, 512), target_size=(1024, 1024),
                            negative_original_size=(512, 512), negative_target_size=(1024, 1024),
                            crops_coords_top_left=(0, 0), num_inference_steps=100,
                            guidance_scale=config["cfg"], generator=generator, output_type="latent",
                            cross_attention_kwargs={"modifier_token_mask": cfg_gate}).images
                    # Decode outside autocast: official SDXL VAE overflows in fp16.
                    decoded = pipe.vae.decode(latent.float() / pipe.vae.config.scaling_factor, return_dict=False)[0]
                    if not torch.isfinite(decoded).all():
                        raise FloatingPointError("SDXL VAE produced nonfinite pixels")
                    pixels = pipe.image_processor.postprocess(decoded, output_type="np")
                    pil = pipe.image_processor.numpy_to_pil(pixels)
                    clip_input = extractor(images=pil, return_tensors="pt").pixel_values.to("cuda")
                    checked, flags = checker(images=pixels, clip_input=clip_input)
                    image = Image.fromarray((checked[0] * 255).round().astype(np.uint8))
                    relative = f'images/{case["id"]}.png'
                    image.save(output / relative)
                    record = dict(case, image=relative, image_sha256=sha256(output / relative),
                                  checkpoint_sha256=snapshot_hash, nsfw=bool(flags[0]), status="succeeded")
                    log.write(json.dumps(record) + "\n")
                    records.append(record)
                    status["generated_images"] += 1
                    write_json(output / "status.json", status)
                    print(json.dumps({"generated_images": len(records), "id": case["id"]}), flush=True)
                status["completed_steps"].append(step)
                review(output, records)
        if len(records) != len(cases):
            raise ValueError("transfer matrix is incomplete")
        status["status"] = "succeeded"
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "status.json", status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text(encoding="utf8")), args.training, args.output)


if __name__ == "__main__":
    main()
