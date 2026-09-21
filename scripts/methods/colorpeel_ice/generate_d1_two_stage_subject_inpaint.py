"""Generate scene-first, latent-constrained subject-inpainting transfer images."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


MODEL_ID = "CompVis/stable-diffusion-v1-4"
WEIGHTS = "pytorch_custom_diffusion_weights.bin"
SCHEMA = "d1_two_stage_subject_inpaint/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def read_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema") != SCHEMA:
        raise ValueError("unexpected two-stage inpainting protocol schema")
    sampling = protocol.get("sampling", {})
    required_sampling = {"seeds", "background_num_inference_steps", "background_guidance_scale", "subject_num_inference_steps", "subject_guidance_scale", "expected_image_count"}
    if not required_sampling <= set(sampling):
        raise ValueError("protocol sampling is incomplete")
    prompts = protocol.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("protocol requires prompts")
    if any("<S*>" in item.get("background_prompt", "") or "<S*>" not in item.get("subject_prompt", "") for item in prompts):
        raise ValueError("background prompts must omit and subject prompts must include <S*>")
    if len(prompts) * len(sampling["seeds"]) != sampling["expected_image_count"]:
        raise ValueError("expected_image_count does not match prompt-seed grid")
    return protocol


def build_manifest(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    sampling = protocol["sampling"]
    rows = []
    for item in protocol["prompts"]:
        for seed in sampling["seeds"]:
            row_id = f"{item['id']}-seed-{seed}"
            rows.append({
                "id": row_id,
                "prompt_id": item["id"],
                "group": item["group"],
                "seed": seed,
                "background_prompt": item["background_prompt"],
                "subject_prompt": item["subject_prompt"],
                "background_num_inference_steps": sampling["background_num_inference_steps"],
                "background_guidance_scale": sampling["background_guidance_scale"],
                "subject_num_inference_steps": sampling["subject_num_inference_steps"],
                "subject_guidance_scale": sampling["subject_guidance_scale"],
                "background_path": f"backgrounds/{row_id}.png",
                "mask_path": f"masks/{row_id}.png",
                "image_path": f"images/{row_id}.png",
            })
    return rows


def validate_checkpoint(protocol: dict[str, Any]) -> dict[str, str]:
    checkpoint = protocol["source_checkpoint"]
    model_dir = Path(checkpoint["model_dir"])
    run_dir = Path(checkpoint["run_dir"])
    required = (WEIGHTS, "<S*>.bin", "embedding_update_audit.json", "training_metrics.jsonl")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("missing checkpoint artifacts: " + ", ".join(missing))
    hashes = {name: sha256(model_dir / name) for name in required}
    if hashes[WEIGHTS] != checkpoint["model_sha256"]:
        raise ValueError("checkpoint weights do not match the protocol hash")
    if hashes["<S*>.bin"] != checkpoint["token_artifact_sha256"]["<S*>.bin"]:
        raise ValueError("checkpoint subject token does not match the protocol hash")
    manifest = run_dir / "manifest.json"
    if model_dir.resolve() != (run_dir / "checkpoints").resolve() or not manifest.is_file() or sha256(manifest) != checkpoint["run_manifest_sha256"]:
        raise ValueError("checkpoint run manifest does not match the protocol hash")
    return hashes


def make_mask() -> Image.Image:
    mask = Image.new("L", (512, 512), 0)
    ImageDraw.Draw(mask).rounded_rectangle((64, 220, 448, 480), radius=48, fill=255)
    return mask


def all_black(image: Image.Image) -> bool:
    return image.convert("RGB").getextrema() == ((0, 0), (0, 0), (0, 0))


def encode_prompt(pipe: Any, prompt: str, device: str, guidance_scale: float) -> Any:
    import torch

    tokens = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    embeds = pipe.text_encoder(tokens.input_ids.to(device))[0]
    if guidance_scale <= 1.0:
        return embeds
    negatives = pipe.tokenizer("", padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
    negative_embeds = pipe.text_encoder(negatives.input_ids.to(device))[0]
    return torch.cat([negative_embeds, embeds])


def scheduler_step(scheduler: Any, prediction: Any, timestep: Any, latents: Any, generator: Any) -> Any:
    kwargs = {"generator": generator} if "generator" in inspect.signature(scheduler.step).parameters else {}
    return scheduler.step(prediction, timestep, latents, **kwargs).prev_sample


def latent_constrained_inpaint(pipe: Any, background: Image.Image, mask: Image.Image, prompt: str, seed: int, steps: int, guidance_scale: float, device: str) -> Image.Image:
    import torch

    vae_scale = getattr(pipe.vae.config, "scaling_factor", 0.18215)
    image_tensor = torch.from_numpy(__import__("numpy").array(background.convert("RGB"))).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=pipe.unet.dtype) / 127.5 - 1.0
    with torch.no_grad():
        original_latents = pipe.vae.encode(image_tensor).latent_dist.mean * vae_scale
    resize = getattr(Image, "Resampling", Image).NEAREST
    mask_array = __import__("numpy").array(mask.resize((original_latents.shape[-1], original_latents.shape[-2]), resize)) / 255.0
    mask_latents = torch.from_numpy(mask_array).to(device=device, dtype=original_latents.dtype)[None, None]
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(original_latents.shape, generator=generator, device=device, dtype=original_latents.dtype)
    latents = noise * pipe.scheduler.init_noise_sigma
    prompt_embeds = encode_prompt(pipe, prompt, device, guidance_scale)
    pipe.scheduler.set_timesteps(steps, device=device)
    for index, timestep in enumerate(pipe.scheduler.timesteps):
        model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        model_input = pipe.scheduler.scale_model_input(model_input, timestep)
        with torch.no_grad():
            prediction = pipe.unet(model_input, timestep, encoder_hidden_states=prompt_embeds).sample
        if guidance_scale > 1.0:
            unconditional, conditional = prediction.chunk(2)
            prediction = unconditional + guidance_scale * (conditional - unconditional)
        latents = scheduler_step(pipe.scheduler, prediction, timestep, latents, generator)
        if index + 1 < len(pipe.scheduler.timesteps):
            preserved = pipe.scheduler.add_noise(original_latents, noise, pipe.scheduler.timesteps[index + 1])
        else:
            preserved = original_latents
        latents = mask_latents * latents + (1.0 - mask_latents) * preserved
    with torch.no_grad():
        decoded = pipe.vae.decode(latents / vae_scale).sample
    output = (decoded / 2 + 0.5).clamp(0, 1)
    array = (output[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def generate(rows: list[dict[str, Any]], protocol: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch
    from diffusers import DiffusionPipeline

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    base = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, low_cpu_mem_usage=False, torch_dtype=dtype, local_files_only=True).to(args.device)
    checkpoint_dir = Path(protocol["source_checkpoint"]["model_dir"])
    subject = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, low_cpu_mem_usage=False, torch_dtype=dtype, local_files_only=True).to(args.device)
    subject.unet.load_attn_procs(str(checkpoint_dir), weight_name=WEIGHTS)
    subject.load_textual_inversion(str(checkpoint_dir), weight_name="<S*>.bin")
    statuses = []
    for row in rows:
        background_path = output_dir / row["background_path"]
        mask_path = output_dir / row["mask_path"]
        image_path = output_dir / row["image_path"]
        status = {"id": row["id"], "status": None, "failure_reason": None, "background_sha256": None, "mask_sha256": None, "image_sha256": None}
        try:
            result = base(row["background_prompt"], num_inference_steps=row["background_num_inference_steps"], guidance_scale=row["background_guidance_scale"], generator=torch.Generator(device=args.device).manual_seed(row["seed"]))
            background = result.images[0]
            if not isinstance(background, Image.Image) or background.mode != "RGB" or background.size != (512, 512) or all_black(background):
                raise ValueError("stage-one background must be a non-black 512x512 RGB PIL image")
            mask = make_mask()
            background_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            background.save(background_path)
            mask.save(mask_path)
            image = latent_constrained_inpaint(subject, background, mask, row["subject_prompt"], row["seed"], row["subject_num_inference_steps"], row["subject_guidance_scale"], args.device)
            image.save(image_path)
            status.update(background_sha256=sha256(background_path), mask_sha256=sha256(mask_path), image_sha256=sha256(image_path))
            if all_black(image):
                status.update(status="failure", failure_reason="all_black_output")
            else:
                status["status"] = "ok"
        except Exception as error:
            status.update(status="failure", failure_reason=f"generation_error:{type(error).__name__}:{error}")
        statuses.append(status)
    del base, subject
    torch.cuda.empty_cache()
    return statuses


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--pretrained-model-name-or-path", default=MODEL_ID)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = read_protocol(args.protocol)
    rows = build_manifest(protocol)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("output directory must be new or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "generation_manifest.jsonl", rows)
    provenance = {"protocol_sha256": sha256(args.protocol), "model_artifact_sha256": None, "method": "stage-one base scene followed by mask-constrained subject diffusion"}
    if args.dry_run:
        write_json(args.output_dir / "provenance.json", provenance)
        return 0
    provenance["model_artifact_sha256"] = validate_checkpoint(protocol)
    statuses = generate(rows, protocol, args.output_dir, args)
    write_json(args.output_dir / "provenance.json", provenance)
    write_jsonl(args.output_dir / "generation_status.jsonl", statuses)
    return 0 if all(row["status"] == "ok" for row in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
