"""Train one paper-style Perfusion concept on a frozen ColorPeel source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "train"))
sys.path.insert(0, str(REPO_ROOT))

from experiments.perfusion_full.attention import FullPerfusionAttnProcessor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_config(path: Path) -> dict:
    import yaml

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"cohort", "token", "superclass", "reference_prompt", "soft_mask_root",
                "covariance_root", "base_model", "seed", "steps", "batch_size",
                "microbatch", "beta", "tau", "lr_value", "lr_embedding",
                "checkpoint_every", "hflip"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("full Perfusion config keys differ")
    if (config["base_model"] != "CompVis/stable-diffusion-v1-4"
            or config["seed"] != 42 or config["steps"] != 400
            or config["batch_size"] != 16 or config["batch_size"] % config["microbatch"]
            or config["beta"] != .75 or config["tau"] != .1
            or config["lr_value"] != .03 or config["lr_embedding"] != .006
            or config["checkpoint_every"] != 25):
        raise ValueError("full Perfusion training settings differ from the paper protocol")
    expected = {
        "original": ("<S*>", "mailbox", 25),
        "balanced_aligned": ("<S*>", "mailbox", 10),
        "material": ("<M*>", "metal", 72),
    }
    if (config["cohort"] not in expected
            or (config["token"], config["superclass"]) != expected[config["cohort"]][:2]
            or config["hflip"] not in ("none", "symmetric_shapes")):
        raise ValueError("concept identity or flip policy differs")
    return config


def load_records(config: dict) -> tuple[list[dict], dict]:
    root = Path(config["soft_mask_root"]).resolve()
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    derived = root / "derived_concepts.json"
    manifest = root / "soft_mask_manifest.jsonl"
    concepts = json.loads(derived.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    expected = {"original": 25, "balanced_aligned": 10, "material": 72}[config["cohort"]]
    if (provenance.get("schema") != "perfusion_full_clipseg_soft_masks/v1"
            or provenance.get("images") != expected or len(rows) != expected
            or provenance.get("concept_rows") != len(concepts)
            or sha256(derived) != provenance["derived_concepts_sha256"]
            or sha256(manifest) != provenance["soft_mask_manifest_sha256"]
            or provenance.get("query") != ("mailbox" if config["token"] == "<S*>" else "metal object")):
        raise ValueError("CLIPSeg source contract differs")
    records = []
    for row in rows:
        concept = concepts[row["row_index"]]
        prompt = concept["instance_prompt"]
        if (not isinstance(prompt, list) or len(prompt) != 1
                or prompt[0].count(config["token"]) != 1):
            raise ValueError("training prompt differs from one-concept source")
        image, mask = Path(row["source_image"]), Path(row["soft_mask"])
        if (sha256(image) != row["source_image_sha256"]
                or sha256(mask) != row["soft_mask_sha256"]
                or mask.parent != Path(concept["instance_mask_dir"])):
            raise ValueError("training image or CLIPSeg mask hash differs")
        records.append({"image": image, "mask": mask, "prompt": prompt[0]})
    return records, provenance


def sample_batch(records: list[dict], rng: random.Random, batch_size: int,
                 tokenizer, hflip: str, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images, masks, prompts = [], [], []
    for _ in range(batch_size):
        row = rng.choice(records)
        flip = (hflip == "symmetric_shapes" and row["image"].stem.split("_")[0] in {"cube", "sphere"}
                and rng.random() < .5)
        with Image.open(row["image"]) as source, Image.open(row["mask"]) as soft:
            image = source.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
            mask = soft.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
            if flip:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1
            weights = np.asarray(mask, dtype=np.float32) / 255
        maximum = float(weights.max())
        if maximum <= 0 or not np.isfinite(weights).all():
            raise ValueError(f"empty/nonfinite soft mask: {row['mask']}")
        images.append(torch.from_numpy(pixels).permute(2, 0, 1))
        masks.append(torch.from_numpy(weights / maximum).unsqueeze(0))
        prompts.append(row["prompt"])
    ids = tokenizer(prompts, padding="max_length", truncation=True,
                    max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
    return torch.stack(images).to(device), torch.stack(masks).to(device), ids.to(device)


def superclass_encoding(text_encoder, tokenizer, prompt: str, superclass: str,
                        device: str) -> torch.Tensor:
    word_ids = tokenizer.encode(superclass, add_special_tokens=False)
    if len(word_ids) != 1:
        raise ValueError("superclass must be exactly one CLIP token")
    ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").input_ids.to(device)
    positions = (ids[0] == word_ids[0]).nonzero(as_tuple=False).flatten()
    if positions.numel() != 1:
        raise ValueError("superclass prompt must contain exactly one superclass token")
    with torch.no_grad():
        return text_encoder(ids)[0][0, positions.item()].detach().float().clone()


def install_processors(unet, reference: torch.Tensor) -> list[FullPerfusionAttnProcessor]:
    processors = {}
    trainable = []
    for name in unet.attn_processors:
        if name.endswith("attn1.processor"):
            processor = FullPerfusionAttnProcessor()
        else:
            layer = name.removesuffix(".processor")
            attention = unet.get_submodule(layer)
            with torch.no_grad():
                key = attention.to_k(reference).view(1, -1)
                value = attention.to_v(reference).view(1, -1)
            processor = FullPerfusionAttnProcessor(
                hidden_size=key.shape[-1], cross_attention_dim=reference.numel(),
                key_outputs=key, value_outputs=value,
            )
            trainable.append(processor)
        processors[name] = processor
    unet.set_attn_processor(processors)
    if len(trainable) != 16:
        raise ValueError(f"expected 16 cross-attention processors, found {len(trainable)}")
    return trainable


def save_checkpoint(path: Path, step: int, token: str, modifier: torch.Tensor,
                    anchor: torch.Tensor, unet, optimizer, scaler, config_sha: str) -> None:
    state = {
        "schema": "perfusion_full_concept/v1", "step": step, "token": token,
        "modifier_embedding": modifier.detach().float().cpu().clone(),
        "anchor_ema": anchor.detach().float().cpu().clone(),
        "attention": {name: {key: value.detach().float().cpu().clone()
                             for key, value in processor.state_dict().items()}
                      for name, processor in unet.attn_processors.items()},
        "optimizer": optimizer.state_dict(), "grad_scaler": scaler.state_dict(),
        "config_sha256": config_sha,
    }
    torch.save(state, path)


def train(config_path: Path, output_dir: Path, device: str) -> None:
    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    config = read_config(config_path)
    records, _ = load_records(config)
    covariance_root = Path(config["covariance_root"]).resolve()
    covariance_provenance = json.loads((covariance_root / "provenance.json").read_text())
    inverse_path = covariance_root / "covariance_inverse.pt"
    if (covariance_provenance.get("schema") != "perfusion_full_covariance/v1"
            or covariance_provenance.get("selected_distinct_count") != 100000
            or sha256(inverse_path) != covariance_provenance["inverse_sha256"]):
        raise ValueError("paper covariance artifact differs")
    inverse = torch.load(inverse_path, map_location="cpu").float().to(device)
    if inverse.shape != (768, 768) or not torch.isfinite(inverse).all():
        raise ValueError("invalid covariance inverse")

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    rng = random.Random(config["seed"])
    model = config["base_model"]
    tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder", local_files_only=True).to(device)
    text_encoder.eval().requires_grad_(False)
    reference = superclass_encoding(text_encoder, tokenizer, config["reference_prompt"],
                                    config["superclass"], device)
    superclass_id = tokenizer.encode(config["superclass"], add_special_tokens=False)[0]
    if tokenizer.add_tokens(config["token"]) != 1:
        raise ValueError("modifier token already exists in base tokenizer")
    text_encoder.resize_token_embeddings(len(tokenizer))
    embedding_layer = text_encoder.get_input_embeddings()
    embedding_layer.requires_grad_(False)
    with torch.no_grad():
        embedding_layer.weight[-1].copy_(embedding_layer.weight[superclass_id])
    modifier_id = tokenizer.convert_tokens_to_ids(config["token"])
    modifier = torch.nn.Parameter(embedding_layer.weight[modifier_id].detach().float().clone())

    def replace_modifier(_module, inputs, output):
        return torch.where((inputs[0] == modifier_id).unsqueeze(-1),
                           modifier.view(1, 1, -1).to(output.dtype), output)

    hook = embedding_layer.register_forward_hook(replace_modifier)
    vae = AutoencoderKL.from_pretrained(model, subfolder="vae", local_files_only=True).to(device)
    vae.eval().requires_grad_(False)
    scheduler = DDPMScheduler.from_pretrained(model, subfolder="scheduler", local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet", local_files_only=True).to(device)
    unet.requires_grad_(False)
    processors = install_processors(unet, reference)
    unet.train()
    unet.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW([
        {"params": [p.value_outputs for p in processors], "lr": config["lr_value"]},
        {"params": [modifier], "lr": config["lr_embedding"]},
    ], weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))
    anchor = reference.detach().float().clone().view(1, -1)
    output_dir.mkdir(parents=True)
    (output_dir / "checkpoints").mkdir()
    config_hash = sha256(config_path)
    (output_dir / "config.yaml").write_bytes(config_path.read_bytes())
    provenance = {
        "schema": "perfusion_full_train/v1", "status": "running", "cohort": config["cohort"],
        "token": config["token"], "config_sha256": config_hash,
        "soft_mask_provenance_sha256": sha256(Path(config["soft_mask_root"]) / "provenance.json"),
        "covariance_provenance_sha256": sha256(covariance_root / "provenance.json"),
        "training_images": len(records), "device": device,
        "started_at_unix": time.time(),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    metrics = (output_dir / "training_metrics.jsonl").open("w", encoding="utf-8")
    for step in range(1, config["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        concept_vectors = []
        step_loss = 0.0
        for _ in range(config["batch_size"] // config["microbatch"]):
            pixels, mask, ids = sample_batch(records, rng, config["microbatch"],
                                             tokenizer, config["hflip"], device)
            positions = (ids == modifier_id).nonzero(as_tuple=False)
            if positions.shape[0] != ids.shape[0] or not torch.equal(
                    positions[:, 0], torch.arange(ids.shape[0], device=ids.device)):
                raise ValueError("each training prompt must contain exactly one concept token")
            with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                with torch.no_grad():
                    latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                          (latents.shape[0],), device=device).long()
                noisy = scheduler.add_noise(latents, noise, timesteps)
                text_states = text_encoder(ids)[0]
                concept_vectors.append(text_states[positions[:, 0], positions[:, 1]].detach().float())
                prediction = unet(
                    noisy, timesteps, text_states,
                    cross_attention_kwargs={"perfusion_inv_cov": inverse,
                                            "perfusion_anchors": anchor,
                                            "perfusion_beta": config["beta"],
                                            "perfusion_tau": config["tau"]},
                ).sample
                weighted = (prediction.float() - noise.float()).square().mean(dim=1, keepdim=True)
                if weighted.shape[-2:] != mask.shape[-2:]:
                    raise ValueError("CLIPSeg mask does not match latent resolution")
                loss = ((weighted * mask).sum(dim=(1, 2, 3)) /
                        mask.sum(dim=(1, 2, 3))).mean()
                loss = loss / (config["batch_size"] // config["microbatch"])
            scaler.scale(loss).backward()
            step_loss += float(loss.detach())
        scaler.unscale_(optimizer)
        for parameter in [modifier, *(processor.value_outputs for processor in processors)]:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise ValueError(f"missing/nonfinite Perfusion gradient at step {step}")
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            observed = torch.cat(concept_vectors).mean(dim=0, keepdim=True)
            anchor.mul_(.99).add_(observed, alpha=.01)
        if not np.isfinite(step_loss) or not torch.isfinite(anchor).all():
            raise ValueError(f"nonfinite Perfusion training state at step {step}")
        metrics.write(json.dumps({"step": step, "loss": step_loss,
                                  "anchor_norm": float(anchor.norm())}) + "\n")
        metrics.flush()
        if step % config["checkpoint_every"] == 0:
            save_checkpoint(output_dir / "checkpoints" / f"step_{step:04d}.pt", step,
                            config["token"], modifier, anchor, unet, optimizer, scaler, config_hash)
            print(f"step {step}/{config['steps']} loss={step_loss:.5f}", flush=True)
    metrics.close()
    hook.remove()
    provenance["status"] = "succeeded"
    provenance["finished_at_unix"] = time.time()
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    train(arguments.config.resolve(), arguments.output_dir.resolve(), arguments.device)
