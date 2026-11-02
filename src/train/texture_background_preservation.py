"""Opt-in frozen-base background consistency for the single Texture branch."""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import itertools
import math
import random

import numpy as np
import torch
import torch.nn.functional as F


def validate_texture_background_preservation(args):
    weight = getattr(args, "texture_background_preservation_weight", 0.0)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("texture background preservation weight must be finite and nonnegative")
    if weight == 0:
        return False
    tokens = args.modifier_token.split("+") if isinstance(args.modifier_token, str) else args.modifier_token
    initializers = args.initializer_token.split("+") if isinstance(args.initializer_token, str) else args.initializer_token
    fullrank = args.token_local_kv and not args.subject_lora_mode
    lora = args.subject_lora_mode == "token_local_kv" and not args.token_local_kv
    if (tokens != ["<T*>"] or initializers != ["texture"] or not (fullrank or lora)
            or args.freeze_model != "crossattn_kv" or args.with_prior_preservation
            or args.joint_two_object_binding or args.perfusion_subject):
        raise ValueError("texture background preservation requires one texture-initialized <T*>, token-local K/V, no prior or joint binding")
    return True


@contextmanager
def _frozen_forward(module):
    """Use deterministic evaluation without changing student modes or RNG."""
    modes = [(child, child.training) for child in module.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    # Do not initialize unrelated visible GPUs merely to save their RNG states.
    devices = sorted({value.device.index for value in itertools.chain(module.parameters(), module.buffers())
                      if value.is_cuda})
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            module.eval()
            yield
    finally:
        for child, training in modes:
            child.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@dataclass
class ReferenceConditioning:
    hidden_states: dict
    modifier_token_id: int
    initializer_token_id: int

    def lookup(self, input_ids, device):
        keys = [tuple(row) for row in input_ids.detach().cpu().tolist()]
        if any(key not in self.hidden_states for key in keys):
            raise ValueError("teacher conditioning was not cached before state restore")
        return torch.cat([self.hidden_states[key] for key in keys]).to(device=device)

    def audit(self):
        digest = hashlib.sha256()
        for key, hidden in sorted(self.hidden_states.items()):
            digest.update(np.asarray(key, dtype=np.int64).tobytes())
            digest.update(hidden.float().numpy().tobytes())
        return {"caption_count": len(self.hidden_states), "modifier_token_id": self.modifier_token_id,
                "initializer_token_id": self.initializer_token_id, "conditioning_sha256": digest.hexdigest(),
                "reference": "initial texture embedding; frozen base UNet with zero modifier gate",
                "background_mask": "valid interior excluding 3x3-dilated foreground", "channel_reduction": "sum"}


def cache_reference_conditioning(text_encoder, input_ids, modifier_token_id, initializer_token_id):
    """Cache initial conditioning BEFORE loading any learned token checkpoint."""
    if input_ids.ndim != 2 or not bool(((input_ids == modifier_token_id).sum(1) == 1).all()):
        raise ValueError("teacher captions must each contain the Texture token exactly once")
    embeddings = text_encoder.get_input_embeddings().weight
    if not torch.equal(embeddings[modifier_token_id], embeddings[initializer_token_id]):
        raise ValueError("cache teacher conditioning before restoring the learned Texture embedding")
    cached = {}
    with _frozen_forward(text_encoder):
        for row in input_ids:
            key = tuple(row.detach().cpu().tolist())
            if key not in cached:
                hidden = text_encoder(row[None].to(embeddings.device))[0]
                cached[key] = hidden.detach().cpu().clone()
    return ReferenceConditioning(cached, modifier_token_id, initializer_token_id)


def reference_prediction(unet, noisy_latents, timesteps, input_ids, reference):
    hidden = reference.lookup(input_ids, noisy_latents.device)
    gate = torch.zeros_like(input_ids, device=noisy_latents.device, dtype=torch.bool)
    with _frozen_forward(unet):
        prediction = unet(noisy_latents, timesteps, encoder_hidden_states=hidden,
                          cross_attention_kwargs={"modifier_token_mask": gate}).sample
    return prediction.detach()


def background_mask(foreground_mask):
    if (foreground_mask.ndim != 4 or foreground_mask.shape[1] != 1
            or min(foreground_mask.shape[-2:]) < 3
            or not bool(((foreground_mask == 0) | (foreground_mask == 1)).all())):
        raise ValueError("background preservation requires a binary Bx1xHxW foreground mask")
    foreground_mask = foreground_mask.float()
    valid = torch.zeros_like(foreground_mask)
    valid[:, :, 1:-1, 1:-1] = 1
    mask = valid * (1 - F.max_pool2d(foreground_mask, kernel_size=3, stride=1, padding=1))
    if not bool((mask.sum((1, 2, 3)) > 0).all()):
        raise ValueError("background preservation region is empty")
    return mask


def background_preservation_loss(student_prediction, teacher_prediction, foreground_mask):
    if teacher_prediction.requires_grad:
        raise ValueError("teacher output must be detached")
    if (student_prediction.shape != teacher_prediction.shape
            or student_prediction.ndim != 4
            or student_prediction.shape[0] != foreground_mask.shape[0]
            or student_prediction.shape[-2:] != foreground_mask.shape[-2:]):
        raise ValueError("student, teacher and foreground dimensions differ")
    mask = background_mask(foreground_mask).to(student_prediction.device)
    error = (student_prediction.float() - teacher_prediction.float()).square()
    loss = ((error * mask).sum((1, 2, 3)) / mask.sum((1, 2, 3))).mean()
    return loss, mask


def background_training_metrics(loss, weight, mask):
    value = float(loss.detach())
    return {"texture_background_loss": value, "texture_background_weight": float(weight),
            "texture_background_weighted_loss": value * weight,
            "texture_background_pixels": mask.sum((1, 2, 3)).detach().cpu().tolist()}


def gradient_report(loss, parameter_groups, embedding_parameter, modifier_token_id, *, retain_graph=True):
    """Report effective trainable gradients; supports sequential low-memory probes."""
    named = [(name, parameter) for name, group in parameter_groups.items() for parameter in group]
    parameters = [parameter for _, parameter in named] + [embedding_parameter]
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("gradient audit parameter groups must be disjoint")
    gradients = (torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
                 if loss.requires_grad else [None] * len(parameters))
    groups = {name: 0.0 for name in parameter_groups}
    finite = True
    for (name, _), gradient in zip(named, gradients[:-1]):
        if gradient is not None:
            finite = finite and bool(torch.isfinite(gradient).all())
            groups[name] += float(gradient.detach().double().square().sum())
    groups = {name: math.sqrt(value) for name, value in groups.items()}
    embedding_gradient = gradients[-1]
    modifier_norm = ordinary_before = ordinary_after = 0.0
    if embedding_gradient is not None:
        finite = finite and bool(torch.isfinite(embedding_gradient).all())
        modifier_norm = float(embedding_gradient[modifier_token_id].detach().norm())
        ordinary = torch.ones(embedding_gradient.shape[0], dtype=torch.bool, device=embedding_gradient.device)
        ordinary[modifier_token_id] = False
        ordinary_before = float(embedding_gradient[ordinary].detach().norm())
        # Exactly the trainer's gradient-row mask, on this probe's local gradient only.
        embedding_gradient = embedding_gradient.detach().clone()
        embedding_gradient[ordinary] = 0
        ordinary_after = float(embedding_gradient[ordinary].norm())
    return {"adapter_group_norms": groups, "modifier_embedding_norm": modifier_norm,
            "ordinary_embedding_norm_before_mask": ordinary_before,
            "ordinary_embedding_norm_after_mask": ordinary_after, "finite": finite}


def audit_background_preservation_gradients(*, student_prediction, teacher_prediction, foreground_mask,
                                          parameter_groups, embedding_parameter, modifier_token_id,
                                          class_only_loss, class_only_input_ids, unconditional_loss,
                                          unconditional_input_ids, require_nonzero=True):
    """Audit a nonzero trained checkpoint; no optimizer step and no .grad mutation.

    Caller computes both controls with current student text encoder and zero
    adapter gates, and supplies losses from their actual UNet predictions.
    """
    if (bool((class_only_input_ids == modifier_token_id).any())
            or bool((unconditional_input_ids == modifier_token_id).any())):
        raise ValueError("negative controls must not contain the learned Texture token")
    parameter_groups = {name: list(group) for name, group in parameter_groups.items()}
    loss, mask = background_preservation_loss(student_prediction, teacher_prediction, foreground_mask)
    reports = {name: gradient_report(value, parameter_groups, embedding_parameter, modifier_token_id)
               for name, value in (("background", loss), ("class_only", class_only_loss),
                                   ("unconditional", unconditional_loss))}
    background = reports["background"]
    controls_zero = all(all(value == 0 for value in reports[name]["adapter_group_norms"].values())
                        and reports[name]["modifier_embedding_norm"] == 0
                        for name in ("class_only", "unconditional"))
    nonzero = (any(value > 0 for value in background["adapter_group_norms"].values())
               and background["modifier_embedding_norm"] > 0)
    passed = (all(report["finite"] and report["ordinary_embedding_norm_after_mask"] == 0
                  for report in reports.values()) and controls_zero and (nonzero or not require_nonzero))
    return {"passed": passed, "requires_nonzero_trained_checkpoint": require_nonzero,
            "teacher_requires_grad": teacher_prediction.requires_grad,
            "background_loss": float(loss.detach()), "background_pixels": mask.sum((1, 2, 3)).tolist(),
            "negative_controls_zero": controls_zero, "background_gradients_nonzero": nonzero,
            "gradients": reports}
