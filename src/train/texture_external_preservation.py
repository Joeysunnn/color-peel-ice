"""Opt-in beach-image adapter regularization; inference remains token-local."""

from contextlib import contextmanager
import hashlib
import io
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch

if __package__:
    from .texture_background_preservation import _frozen_forward
else:
    from texture_background_preservation import _frozen_forward


def validate_texture_external_preservation(args):
    weight = getattr(args, "texture_external_preservation_weight", 0.0)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("external preservation weight must be finite and nonnegative")
    if weight == 0:
        return False
    tokens = args.modifier_token.split("+") if isinstance(args.modifier_token, str) else args.modifier_token
    initializers = args.initializer_token.split("+") if isinstance(args.initializer_token, str) else args.initializer_token
    fullrank = args.token_local_kv and not args.subject_lora_mode
    lora = args.subject_lora_mode == "token_local_kv" and not args.token_local_kv
    if (tokens != ["<T*>"] or initializers != ["texture"] or not (fullrank or lora)
            or args.freeze_model != "crossattn_kv" or args.mixed_precision != "no"
            or args.with_prior_preservation or args.texture_background_preservation_weight != 0
            or args.joint_two_object_binding or args.perfusion_subject or args.strict_unpaired_modifier_updates
            or args.enable_xformers_memory_efficient_attention or args.cos_weight != 0 or args.lambda_attention != 0
            or args.train_batch_size != 1 or args.gradient_accumulation_steps != 1
            or not args.texture_external_preservation_manifest):
        raise ValueError("external preservation requires single Texture token-local K/V, FP32, batch1, no other preservation/binding modes")
    if (args.max_train_steps != 600 or not args.resume_from_checkpoint
            or Path(args.resume_from_checkpoint).name != "checkpoint-400"):
        raise ValueError("external preservation is pinned to the complete T400-to600 continuation")
    return True


@contextmanager
def preservation_rng(step, device):
    """Isolate the extra forward AND backward, including checkpoint recomputation."""
    if type(step) is not int or not 401 <= step <= 600:
        raise ValueError("external preservation only has records for steps401 through600")
    device = torch.device(device)
    devices = [] if device.type == "cpu" else [device.index if device.index is not None else torch.cuda.current_device()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            seed = 42 + 100000 + step
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            yield seed
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class ExternalTexturePreservation:
    def __init__(self, manifest_path, tokenizer, base_text_encoder, modifier_token_id, initializer_token_id):
        self.manifest_path = Path(manifest_path).resolve()
        payload = self.manifest_path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(payload).hexdigest()
        manifest = json.loads(payload)
        if (manifest.get("schema") != 1 or manifest.get("prompt") != "a photo at the beach"
                or manifest.get("seed") != 42 or not isinstance(manifest.get("records"), list)
                or len(manifest["records"]) != 200):
            raise ValueError("require the fixed schema1 beach manifest with seed42 and200 ordered records")
        self.prompt, self.records, self.images = manifest["prompt"], manifest["records"], []
        paths = set()
        for record in self.records:
            path = Path(record["path"])
            if not path.is_absolute() or path.resolve() in paths:
                raise ValueError("preservation image paths must be absolute and unique")
            paths.add(path.resolve())
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("preservation image SHA differs from the manifest")
            with Image.open(io.BytesIO(data)) as image:
                if image.mode != "RGB" or image.size != (512, 512):
                    raise ValueError("preservation images must already be512px RGB; no crop or conversion")
                self.images.append(np.asarray(image, dtype=np.uint8).copy())
        embedding = base_text_encoder.get_input_embeddings().weight
        if embedding.dtype != torch.float32 or not torch.equal(embedding[modifier_token_id], embedding[initializer_token_id]):
            raise ValueError("cache external conditioning in FP32 before loading the learned Texture state")
        full_ids = tokenizer.encode(self.prompt, add_special_tokens=True)
        if tokenizer.model_max_length != 77 or len(full_ids) > 77 or modifier_token_id in full_ids:
            raise ValueError("require an untruncated ordinary CLIP77 caption without Texture")
        encoded = tokenizer(self.prompt, padding="max_length", max_length=77, truncation=False, return_tensors="pt")
        self.input_ids = encoded.input_ids.detach().cpu().clone()
        if tuple(self.input_ids.shape) != (1, 77) or self.input_ids[0, :len(full_ids)].tolist() != full_ids:
            raise ValueError("ordinary caption IDs differ from the untruncated encoding")
        kwargs = ({"attention_mask": encoded.attention_mask.to(embedding.device)}
                  if bool(getattr(base_text_encoder.config, "use_attention_mask", False)) else {})
        with _frozen_forward(base_text_encoder):
            self.hidden_states = base_text_encoder(self.input_ids.to(embedding.device), **kwargs)[0].detach().cpu().clone()
        if tuple(self.hidden_states.shape) != (1, 77, 1024) or not bool(torch.isfinite(self.hidden_states).all()):
            raise ValueError("require finite SD2.1 ordinary hidden states")

    def audit(self):
        return {"manifest": str(self.manifest_path), "manifest_sha256": self.manifest_sha256,
                "prompt": self.prompt, "input_ids": self.input_ids.tolist(), "record_count": len(self.records),
                "hidden_sha256": hashlib.sha256(self.hidden_states.numpy().tobytes()).hexdigest(),
                "hidden_requires_grad": self.hidden_states.requires_grad,
                "teacher": "frozen base, ordinary detached conditioning, all-zero gate",
                "student": "same detached ordinary conditioning, all77 gate positions enabled",
                "region": "whole latent frame; no foreground/background mask",
                "reduction": "channel sum divided by pixel area", "step_seed": "42+100000+optimization_step",
                "first_record_step": 401, "last_record_step": 600}

    @contextmanager
    def loss_context(self, unet, vae, scheduler, step, device, dtype=torch.float32):
        """Caller must perform backward inside this context to preserve the foreground RNG stream."""
        if dtype != torch.float32:
            raise ValueError("external preservation is FP32 only")
        with preservation_rng(step, device) as seed:
            record = self.records[step - 401]
            pixels = torch.from_numpy(self.images[step - 401]).permute(2, 0, 1).unsqueeze(0)
            pixels = pixels.to(device=device, dtype=dtype).div(127.5).sub(1)
            with _frozen_forward(vae):
                posterior = vae.encode(pixels).latent_dist
            with torch.no_grad():
                latents = posterior.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device).long()
                noisy = scheduler.add_noise(latents, noise, timesteps)
            hidden = self.hidden_states.to(device=device, dtype=dtype)
            gate = torch.zeros_like(self.input_ids, device=device, dtype=torch.bool)
            with _frozen_forward(unet):
                teacher = unet(noisy, timesteps, encoder_hidden_states=hidden,
                               cross_attention_kwargs={"modifier_token_mask": gate}).sample.detach()
            # Reentrant checkpointing needs a grad-bearing input even though text/VAE/base are frozen.
            student_noisy = noisy.detach().requires_grad_(True)
            student = unet(student_noisy, timesteps, encoder_hidden_states=hidden,
                           cross_attention_kwargs={"modifier_token_mask": torch.ones_like(gate)}).sample
            if student.shape != teacher.shape or student.shape != latents.shape:
                raise ValueError("external teacher/student/latent prediction shapes differ")
            loss = (student.float() - teacher.float()).square().sum((1, 2, 3)).div(student.shape[-2] * student.shape[-1]).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("external preservation loss is nonfinite")
            metrics = {"texture_external_preservation_loss": float(loss.detach()),
                       "texture_external_preservation_image_path": record["path"],
                       "texture_external_preservation_image_sha256": record["sha256"],
                       "texture_external_preservation_step": step, "texture_external_preservation_seed": seed,
                       "texture_external_preservation_timestep": int(timesteps.item()),
                       "teacher_requires_grad": teacher.requires_grad, "hidden_requires_grad": hidden.requires_grad}
            yield loss, metrics
