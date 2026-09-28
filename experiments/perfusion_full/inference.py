"""Load one or two independently trained full Perfusion concepts for sampling."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "train"))
sys.path.insert(0, str(REPO_ROOT))

from experiments.perfusion_full.attention import FullPerfusionAttnProcessor


def load_pipeline(checkpoints: list[Path], covariance_root: Path, device: str,
                  ddim: bool = False):
    """Return a pipeline and kwargs implementing Eq. (3), or Eq. (4) for two concepts."""
    from diffusers import DDIMScheduler, DiffusionPipeline
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel

    if not 1 <= len(checkpoints) <= 2:
        raise ValueError("full Perfusion inference needs one or two concepts")
    states = [torch.load(path, map_location="cpu") for path in checkpoints]
    if any(state.get("schema") != "perfusion_full_concept/v1" for state in states):
        raise ValueError("checkpoint schema differs")
    tokens = [state["token"] for state in states]
    if len(set(tokens)) != len(tokens):
        raise ValueError("concept tokens must be distinct")
    inverse = torch.load(covariance_root / "covariance_inverse.pt", map_location="cpu").float().to(device)
    anchors = torch.stack([state["anchor_ema"] for state in states]).reshape(len(states), -1).to(device)
    if inverse.shape != (768, 768) or anchors.shape != (len(states), 768):
        raise ValueError("Perfusion metric shapes differ")

    model = "CompVis/stable-diffusion-v1-4"
    unet = UNet2DConditionModel.from_pretrained(
        model, subfolder="unet", local_files_only=True, torch_dtype=torch.float16)
    processors = {}
    for name in unet.attn_processors:
        if name.endswith("attn1.processor"):
            processors[name] = FullPerfusionAttnProcessor()
            continue
        layer_states = [state["attention"][name] for state in states]
        keys = torch.cat([item["key_outputs"] for item in layer_states])
        values = torch.cat([item["value_outputs"] for item in layer_states])
        processor = FullPerfusionAttnProcessor(
            hidden_size=keys.shape[-1], cross_attention_dim=768,
            key_outputs=keys, value_outputs=values)
        processors[name] = processor.to(dtype=torch.float16)
    unet.set_attn_processor(processors)
    pipe = DiffusionPipeline.from_pretrained(
        model, unet=unet, low_cpu_mem_usage=False,
        torch_dtype=torch.float16, local_files_only=True).to(device)
    if ddim:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    for token, state in zip(tokens, states):
        if pipe.tokenizer.add_tokens(token) != 1:
            raise ValueError(f"concept token already exists: {token}")
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
        token_id = pipe.tokenizer.convert_tokens_to_ids(token)
        with torch.no_grad():
            pipe.text_encoder.get_input_embeddings().weight[token_id].copy_(
                state["modifier_embedding"].to(device=device, dtype=torch.float16))
    pipe.text_encoder.eval()
    pipe.unet.eval()
    kwargs = {"perfusion_inv_cov": inverse, "perfusion_anchors": anchors,
              "perfusion_beta": .75, "perfusion_tau": .15}
    return pipe, kwargs


def generate(pipe, kwargs: dict, prompt: str, seed: int, steps: int,
             cfg: float, device: str):
    result = pipe(prompt, num_inference_steps=steps, guidance_scale=cfg,
                  generator=torch.Generator(device=device).manual_seed(seed),
                  cross_attention_kwargs=kwargs)
    filtered = bool(result.nsfw_content_detected[0]) if result.nsfw_content_detected is not None else False
    return result.images[0], filtered
