"""Single-branch SDXL training; frozen base plus token-local KV and two vectors."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler, StableDiffusionXLPipeline

from .attention import adapter_parameters, adapter_state_dict, install_token_local_kv
from .conditioning import encode_conditioning, install_learned_tokens, make_time_ids
from .data import load_rows, load_training_image


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf8")


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_model(config, training):
    from .data import sha256
    if sha256(Path(config["model_dir"]) / "model_source.json") != config["model_source_sha256"]:
        raise ValueError("model source receipt differs from the selected training base")
    if sha256(config["dataset"]) != config["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest differs from the selected training input")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config["model_dir"], torch_dtype=torch.float32, local_files_only=True,
        use_safetensors=True, add_watermarker=False,
    )
    pipe.vae.requires_grad_(False).eval()
    pipe.unet.to(device="cuda", dtype=torch.float16)
    encoders = [pipe.text_encoder, pipe.text_encoder_2]
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    for encoder in encoders:
        encoder.to(device="cuda", dtype=torch.float32).eval()
    embeddings = install_learned_tokens(tokenizers, encoders, config["token"], config["initializer"])
    install_token_local_kv(pipe.unet, config["method"])
    if training:
        pipe.unet.train()
        pipe.unet.enable_gradient_checkpointing()
        for encoder in encoders:
            encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            encoder.train()
    else:
        pipe.unet.eval()
    return pipe, tokenizers, encoders, embeddings


def masked_loss(prediction, target, mask):
    # Preserve the historical channel-sum / spatial-mask normalization (no /4).
    error = (prediction.float() - target.float()).square()
    return ((error * mask).sum((1, 2, 3)) / mask.sum((1, 2, 3))).mean()


def cache_posteriors(pipe, rows):
    """Cache deterministic VAE distributions, sampling a fresh latent each update."""
    pipe.vae.to(device="cuda", dtype=torch.float32)
    pipe.vae.enable_slicing()
    cached = []
    with torch.no_grad():
        for row in rows:
            image, mask = load_training_image(row)
            pixels = torch.from_numpy(np.array(image, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None].cuda()
            posterior = pipe.vae.encode(pixels).latent_dist
            mask_tensor = torch.from_numpy(np.array(mask, dtype=np.float32) / 255)[None, None]
            if mask_tensor.sum() <= 0:
                raise ValueError("empty training mask")
            cached.append((posterior.mean.cpu(), posterior.std.cpu(), mask_tensor))
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    return cached


def checkpoint(path, config, step, pipe, embeddings, optimizer, scaler, initial_hashes):
    payload = {
        "schema": "sdxl_token_local_v1", "config": config, "step": step,
        "adapter": adapter_state_dict(pipe.unet),
        "embeddings": [embedding.vector.detach().cpu().clone() for embedding in embeddings],
        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
        "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(), "python": random.getstate()},
        "ordinary_embedding_hashes": initial_hashes,
        "next_row": step % config["image_count"],
        "frozen_model": config["model_dir"],
    }
    torch.save(payload, path)


def run(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    status = {"status": "running", "completed_steps": 0, "max_steps": config["steps"]}
    write_json(output / "status.json", status)
    try:
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        rows = load_rows(config["dataset"])
        config = dict(config, image_count=len(rows))
        pipe, tokenizers, encoders, embeddings = load_model(config, training=True)
        original_hashes = [tensor_hash(embedding.base.weight) for embedding in embeddings]
        initial_vectors = [embedding.vector.detach().cpu().clone() for embedding in embeddings]
        params = adapter_parameters(pipe.unet)
        vector_params = [embedding.vector for embedding in embeddings]
        optimizer = torch.optim.AdamW([
            {"params": vector_params, "lr": config["embedding_lr"]},
            {"params": params, "lr": config["kv_lr"]},
        ], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        scaler = torch.amp.GradScaler("cuda", init_scale=1024)
        scheduler = DDPMScheduler.from_pretrained(config["model_dir"], subfolder="scheduler", local_files_only=True)
        if scheduler.config.prediction_type != "epsilon":
            raise ValueError("expected SDXL epsilon prediction")
        cache = cache_posteriors(pipe, rows)
        # Source files are 512px; both training and inference use this same conditioning.
        time_ids = make_time_ids((512, 512), (1024, 1024), device="cuda", dtype=torch.float16)
        trainable = {name: tuple(p.shape) for name, p in pipe.unet.named_parameters() if p.requires_grad}
        if any(".processor." not in name for name in trainable):
            raise ValueError("a frozen UNet parameter became trainable")
        write_json(output / "trainable.json", {"unet": trainable, "vectors": [list(p.shape) for p in vector_params],
            "precision": "frozen UNet fp16; text encoders/VAE/master parameters fp32; autocast fp16+GradScaler",
            "time_ids": [512, 512, 0, 0, 1024, 1024], "cached": "posterior mean/std; fresh sample each update"})
        started = time.monotonic()
        with (output / "metrics.jsonl").open("w", encoding="utf8", buffering=1) as log:
            for step in range(1, config["steps"] + 1):
                index = (step - 1) % len(rows)
                row = rows[index]
                mean, std, mask = [value.cuda() for value in cache[index]]
                latent = ((mean + std * torch.randn_like(std)) * pipe.vae.config.scaling_factor).half()
                noise = torch.randn_like(latent)
                timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device="cuda")
                noisy = scheduler.add_noise(latent, noise, timestep)
                optimizer.zero_grad(set_to_none=True)
                # Frozen CLIP computations stay FP32 so 1e-5 vector updates are retained.
                hidden, pooled, gate = encode_conditioning(tokenizers, encoders, row["prompt"], config["token"])
                if gate.sum().item() != 1:
                    raise ValueError("each training caption must contain the target token exactly once")
                with torch.autocast("cuda", dtype=torch.float16):
                    prediction = pipe.unet(noisy, timestep, encoder_hidden_states=hidden.half(),
                        added_cond_kwargs={"text_embeds": pooled.half(), "time_ids": time_ids},
                        cross_attention_kwargs={"modifier_token_mask": gate}).sample
                    loss = masked_loss(prediction, noise, mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at step {step}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if any(p.grad is None for p in vector_params):
                    raise ValueError("both token vectors must receive gradients")
                norm = torch.nn.utils.clip_grad_norm_(vector_params + params, 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError(f"nonfinite gradients at step {step}")
                scaler.step(optimizer)
                scaler.update()
                metric = {"step": step, "row": index, "id": row["id"], "loss": float(loss.detach()),
                          "gradient_norm": float(norm), "seconds": time.monotonic() - started,
                          "cuda_peak_bytes": torch.cuda.max_memory_allocated()}
                log.write(json.dumps(metric) + "\n")
                print(json.dumps(metric), flush=True)
                status["completed_steps"] = step
                if step in config["checkpoints"]:
                    hashes = [tensor_hash(embedding.base.weight) for embedding in embeddings]
                    if hashes != original_hashes:
                        raise ValueError("ordinary vocabulary changed")
                    checkpoint(output / f"checkpoint-{step}.pt", config, step, pipe, embeddings, optimizer, scaler, hashes)
                write_json(output / "status.json", status)
        audit = {"ordinary_embedding_drift": [0.0, 0.0], "ordinary_embedding_hashes": original_hashes,
                 "vector_l2_change": [float((e.vector.detach().cpu() - v).norm()) for e, v in zip(embeddings, initial_vectors)],
                 "completed_steps": config["steps"], "optimizer_updates": config["steps"],
                 "fixed_order": True, "rows": len(rows), "checkpoints": config["checkpoints"]}
        if any(change <= 0 for change in audit["vector_l2_change"]):
            raise ValueError("one of the two learned vectors did not update")
        write_json(output / "audit.json", audit)
        status.update(status="succeeded", stop_reason="max_steps_reached")
    except BaseException as exc:
        status.update(status="failed", stop_reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "status.json", status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text(encoding="utf8")), args.output)


if __name__ == "__main__":
    main()
