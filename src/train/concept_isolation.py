"""TokenVerse-inspired right-panel consistency for one SD2.1 token-local LoRA."""

from contextlib import contextmanager
import hashlib
import io
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image


def combine_prompt(input_prompt, generated_prompt):
    left, right = (text.strip().rstrip(".").rstrip() for text in (input_prompt, generated_prompt))
    if not left or not right:
        raise ValueError("both isolation captions must be nonempty")
    return f"A picture divided into two parts. The left part is {left}. The right part is {right}."


def validate_concept_isolation(args):
    weight = getattr(args, "concept_isolation_weight", 0.0)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("concept isolation weight must be finite and nonnegative")
    if weight == 0:
        return False
    tokens = args.modifier_token.split("+") if isinstance(args.modifier_token, str) else args.modifier_token
    initializers = args.initializer_token.split("+") if isinstance(args.initializer_token, str) else args.initializer_token
    allowed = {"<S*>": ("mailbox", 1000), "<C*>": ("orange", 150), "<T*>": ("texture", 1000)}
    if (not isinstance(tokens, list) or len(tokens) != 1 or tokens[0] not in allowed
            or initializers != [allowed[tokens[0]][0]] or args.max_train_steps != allowed[tokens[0]][1]
            or args.subject_lora_mode != "token_local_kv" or args.token_local_kv
            or args.freeze_model != "crossattn_kv" or args.mixed_precision != "no"
            or args.train_batch_size != 1 or args.gradient_accumulation_steps != 1
            or args.with_prior_preservation or args.texture_background_preservation_weight != 0
            or args.texture_external_preservation_weight != 0 or args.joint_two_object_binding
            or args.perfusion_subject or args.strict_unpaired_modifier_updates
            or args.enable_xformers_memory_efficient_attention or args.lambda_attention != 0 or args.cos_weight != 0
            or args.adam_weight_decay != 0 or args.resume_from_checkpoint is not None
            or not args.gradient_checkpointing or not args.concept_isolation_manifest or args.seed != 42):
        raise ValueError("isolation requires fresh single S/C/T token-local LoRA, FP32, checkpointing, batch1, seed42, decay0, no other auxiliary objectives")
    return True


def clear_attention_caches(unet):
    for module in unet.modules():
        if hasattr(module, "attn_probs"):
            module.attn_probs = None


@contextmanager
def isolation_rng(seed, device):
    import torch

    device = torch.device(device)
    devices = [] if device.type == "cpu" else [device.index if device.index is not None else torch.cuda.current_device()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _ordinary_hash(embedding, modifier_id):
    digest = hashlib.sha256()
    for start, end in ((0, modifier_id), (modifier_id + 1, embedding.shape[0])):
        for index in range(start, end, 4096):
            digest.update(embedding[index:min(index + 4096, end)].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class ConceptIsolation:
    def __init__(self, manifest_path, tokenizer, modifier_token_id, initializer_token_id, concept_prompts, max_steps, seed=42):
        import torch

        if type(max_steps) is not int or max_steps <= 0 or max_steps % 2 or seed != 42:
            raise ValueError("require a positive even step count and the fixed seed42")
        self.max_steps, self.seed = max_steps, seed
        self.steps = sorted(random.Random(seed + 9000).sample(range(1, max_steps + 1), max_steps // 2))
        self._steps = set(self.steps)
        self.modifier_id, self.initializer_id = modifier_token_id, initializer_token_id
        self.modifier_token = tokenizer.convert_ids_to_tokens(modifier_token_id)
        if self.modifier_token not in ("<S*>", "<C*>", "<T*>") or initializer_token_id == modifier_token_id:
            raise ValueError("require one explicit S/C/T token and a different ordinary initializer ID")
        self.manifest_path = Path(manifest_path).resolve()
        payload = self.manifest_path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(payload).hexdigest()
        manifest = json.loads(payload)
        if manifest.get("schema") != 1 or not isinstance(manifest.get("records"), list) or len(manifest["records"]) != 25:
            raise ValueError("require schema1 and exactly25 audited base-generated images")
        self.records, self.images, self.encodings = manifest["records"], [], {}
        seen_ids, seen_paths = set(), set()
        for record in self.records:
            path = Path(record["path"])
            if (record["id"] in seen_ids or not path.is_absolute() or path.resolve() in seen_paths
                    or record.get("size") != [512, 512] or record.get("mode") != "RGB"
                    or not isinstance(record.get("caption"), str) or not record["caption"].strip()
                    or any(token in record["caption"] for token in ("<S*>", "<C*>", "<T*>", "<M*>"))):
                raise ValueError("invalid or repeated bank record, dimensions, or ordinary caption")
            seen_ids.add(record["id"])
            seen_paths.add(path.resolve())
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("isolation bank image SHA differs")
            with Image.open(io.BytesIO(data)) as image:
                if image.mode != "RGB" or image.size != (512, 512):
                    raise ValueError("bank images must already be512px RGB; no crop or resize")
                self.images.append(np.asarray(image, dtype=np.uint8).copy())
        self.concept_prompts = list(dict.fromkeys(concept_prompts))
        if not self.concept_prompts or tokenizer.model_max_length != 77:
            raise ValueError("require original concept captions and CLIP77")
        for prompt in self.concept_prompts:
            if prompt.count(self.modifier_token) != 1:
                raise ValueError("the original caption must contain the modifier exactly once")
            for index, record in enumerate(self.records):
                combined = combine_prompt(prompt, record["caption"])
                full = tokenizer.encode(combined, add_special_tokens=True)
                if len(full) > 77 or full.count(modifier_token_id) != 1:
                    raise ValueError("combined caption must fit77 tokens and retain its modifier exactly once")
                encoded = tokenizer(combined, padding="max_length", max_length=77, truncation=False, return_tensors="pt")
                ids = encoded.input_ids.detach().cpu().clone()
                if tuple(ids.shape) != (1, 77) or ids[0, :len(full)].tolist() != full:
                    raise ValueError("combined padded encoding differs from the untruncated caption")
                teacher_ids = ids.clone()
                teacher_ids[teacher_ids == modifier_token_id] = initializer_token_id
                self.encodings[prompt, index] = {"prompt": combined, "token_count": len(full), "ids": ids,
                    "teacher_ids": teacher_ids, "attention_mask": encoded.attention_mask.detach().cpu().clone()}
        self.teacher_reference = None

    def enabled(self, step):
        if type(step) is not int or not 1 <= step <= self.max_steps:
            raise ValueError("isolation step is outside the registered schedule")
        return step in self._steps

    def bind_text_encoder(self, text_encoder):
        import torch

        embedding = text_encoder.get_input_embeddings().weight
        if (embedding.dtype != torch.float32 or not embedding.requires_grad
                or any(parameter.requires_grad for parameter in text_encoder.parameters() if parameter is not embedding)):
            raise ValueError("bind the FP32 encoder with only its embedding table trainable")
        self.teacher_reference = {"ordinary_embedding_sha256": _ordinary_hash(embedding, self.modifier_id),
            "non_embedding_parameters_frozen": True,
            "modifier_initially_equals_initializer": torch.equal(embedding[self.modifier_id], embedding[self.initializer_id]),
            "reference_source": "caller supplies base ordinary rows; modifier may be learned for a gradient probe"}

    def verify_teacher(self, text_encoder):
        if self.teacher_reference is None:
            raise ValueError("isolation teacher premise was not recorded before training")
        embedding = text_encoder.get_input_embeddings().weight
        if (any(parameter.requires_grad for parameter in text_encoder.parameters() if parameter is not embedding)
                or _ordinary_hash(embedding, self.modifier_id) != self.teacher_reference["ordinary_embedding_sha256"]):
            raise ValueError("ordinary encoder parameters changed; initializer teacher is no longer the base reference")
        return {**self.teacher_reference, "ordinary_embedding_unchanged": True}

    def audit(self):
        return {"manifest": str(self.manifest_path), "manifest_sha256": self.manifest_sha256,
                "modifier_token_id": self.modifier_id, "initializer_token_id": self.initializer_id,
                "max_steps": self.max_steps, "isolation_steps": self.steps,
                "schedule": "Random(seed+9000).sample(range(1,max_steps+1),max_steps//2)",
                "step_seed": "42+200000+optimization_step", "teacher_reference": self.teacher_reference,
                "caption_normalization": "strip outer whitespace and trailing periods before the fixed Fig4 template",
                "right_region": "latent x=64:128, all64 rows; channel sum divided by4096 pixels",
                "encodings": [{"concept_prompt": prompt, "bank_id": self.records[index]["id"],
                               "combined_prompt": value["prompt"], "token_count": value["token_count"],
                               "input_ids": value["ids"].tolist(), "teacher_input_ids": value["teacher_ids"].tolist()}
                              for (prompt, index), value in self.encodings.items()]}

    @contextmanager
    def loss_context(self, unet, vae, text_encoder, scheduler, pixel_values, prompt, step, device, dtype=None):
        import torch
        if __package__:
            from .texture_background_preservation import _frozen_forward
        else:
            from texture_background_preservation import _frozen_forward

        dtype = torch.float32 if dtype is None else dtype
        if not self.enabled(step) or self.teacher_reference is None or dtype != torch.float32:
            raise ValueError("require an enabled step, recorded base teacher premise, and FP32")
        if tuple(pixel_values.shape) != (1, 3, 512, 512):
            raise ValueError("the actual preprocessed concept RGB must remain512x512")
        seed = self.seed + 200000 + step
        index = random.Random(seed).randrange(len(self.records))
        if (prompt, index) not in self.encodings:
            raise ValueError("actual concept caption was not included in the tokenization audit")
        encoded, record = self.encodings[prompt, index], self.records[index]
        with isolation_rng(seed, device):
            clear_attention_caches(unet)
            try:
                right = torch.from_numpy(self.images[index]).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
                canvas = torch.cat((pixel_values.detach().to(device=device, dtype=dtype), right.div(127.5).sub(1)), dim=-1)
                with _frozen_forward(vae):
                    posterior = vae.encode(canvas).latent_dist
                with torch.no_grad():
                    latents = posterior.sample() * vae.config.scaling_factor
                    if tuple(latents.shape) != (1, 4, 64, 128):
                        raise ValueError("whole-canvas VAE must yield1x4x64x128")
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device).long()
                    noisy = scheduler.add_noise(latents, noise, timesteps)
                ids, teacher_ids = encoded["ids"].to(device), encoded["teacher_ids"].to(device)
                kwargs = ({"attention_mask": encoded["attention_mask"].to(device)}
                          if bool(getattr(text_encoder.config, "use_attention_mask", False)) else {})
                with _frozen_forward(text_encoder):
                    teacher_hidden = text_encoder(teacher_ids, **kwargs)[0].detach()
                with _frozen_forward(unet):
                    teacher = unet(noisy, timesteps, encoder_hidden_states=teacher_hidden,
                        cross_attention_kwargs={"modifier_token_mask": torch.zeros_like(ids, dtype=torch.bool)}).sample.detach()
                clear_attention_caches(unet)
                student_hidden = text_encoder(ids, **kwargs)[0]
                # Preserve adapter gradients with Torch1.13 reentrant checkpointing.
                student = unet(noisy.detach().requires_grad_(True), timesteps, encoder_hidden_states=student_hidden,
                    cross_attention_kwargs={"modifier_token_mask": ids == self.modifier_id}).sample
                if student.shape != teacher.shape or student.shape != latents.shape:
                    raise ValueError("isolation predictions must match the full rectangular latent")
                loss = (student[..., 64:].float() - teacher[..., 64:].float()).square().sum((1, 2, 3)).div(64 * 64).mean()
                if not bool(torch.isfinite(loss)):
                    raise ValueError("isolation loss is nonfinite")
                metrics = {"concept_isolation_applied": True, "concept_isolation_loss": float(loss.detach()),
                    "concept_isolation_step": step, "concept_isolation_seed": seed,
                    "concept_isolation_bank_id": record["id"], "concept_isolation_image_sha256": record["sha256"],
                    "concept_isolation_concept_prompt": prompt, "concept_isolation_combined_prompt": encoded["prompt"],
                    "concept_isolation_timestep": int(timesteps.item()), "concept_isolation_right_pixels": 4096,
                    "teacher_requires_grad": teacher.requires_grad, "teacher_hidden_requires_grad": teacher_hidden.requires_grad,
                    "student_hidden_requires_grad": student_hidden.requires_grad}
                yield loss, metrics
            finally:
                clear_attention_caches(unet)
