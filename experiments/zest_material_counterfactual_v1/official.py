"""Official ZeST/Tencent inference; optional Canny is an explicit adaptation."""

import inspect
from pathlib import Path
import sys

from PIL import Image, ImageChops, ImageEnhance

from experiments.zest_material_counterfactual_v1 import assets

PROMPT = "best quality, high quality"
NEGATIVE_PROMPT = "monochrome, lowres, bad anatomy, worst quality, low quality"


def compatibility(adapter=None):
    result = {
        "official_sources": {"ZeST": assets.OFFICIAL_COMMIT, "Tencent_IPAdapter": assets.IP_ADAPTER_COMMIT},
        "baseline": "Official SDXL base + Depth ControlNet + Tencent IPAdapterXL on all cross-attention layers; no MARBLE InstantStyle or attribute MLP.",
        "defaults": {"strength": 0.9999, "guidance_scale": 5.0, "depth_scale": 0.9,
                     "ip_adapter_scale": 1.0, "num_inference_steps": 30, "seed": 42},
        "edge_adaptation": "Additional Canny SDXL ControlNet; Tencent IPAdapter explicitly installs CNAttnProcessor on each MultiControlNet member. This is not native ZeST texture preservation.",
        "initialization": "Official foreground grayscale + untouched RGB outside effective edit mask unless an explicit alternative init_image is supplied.",
        "interpretation": "No exact albedo/texture/identity or PBR-parameter guarantee. Self reference remains diffusion regeneration.",
    }
    if adapter is not None:
        pipe = adapter.pipe
        signature = inspect.signature(pipe.__call__).parameters
        controls = pipe.controlnet.nets if hasattr(pipe.controlnet, "nets") else [pipe.controlnet]
        result["runtime"] = {"pipeline": type(pipe).__name__, "adapter_module": type(adapter).__module__,
            "scheduler": type(pipe.scheduler).__name__, "controlnets": len(controls),
            "pipeline_default_strength": signature["strength"].default,
            "pipeline_default_guidance_scale": signature["guidance_scale"].default,
            "unet_processors": sorted({type(value).__name__ for value in pipe.unet.attn_processors.values()}),
            "controlnet_processors": [sorted({type(value).__name__ for value in net.attn_processors.values()}) for net in controls]}
    return result


def setup(receipt, with_edges=False, device="cuda", cpu_offload=True):
    assets.verify(receipt)
    if with_edges and not receipt["with_canny"]:
        raise ValueError("edge adaptation requires a separate asset receipt with pinned Canny weights")
    adapter_root = Path(receipt["ip_adapter_official"]).resolve()
    existing = sys.modules.get("ip_adapter")
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(adapter_root):
        raise ValueError("another IP-Adapter implementation is already imported; use an isolated process")
    sys.path.insert(0, str(adapter_root))
    import torch
    from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline
    from ip_adapter import IPAdapterXL
    from ip_adapter.utils import register_cross_attention_hook

    paths = receipt["paths"]
    controls = [ControlNetModel.from_pretrained(paths["controlnet"], variant="fp16", use_safetensors=True,
                torch_dtype=torch.float16, local_files_only=True).to(device)]
    if with_edges:
        controls.append(ControlNetModel.from_pretrained(paths["canny"], variant="fp16", use_safetensors=True,
            torch_dtype=torch.float16, local_files_only=True).to(device))
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(paths["base"],
        controlnet=controls if with_edges else controls[0], use_safetensors=True,
        torch_dtype=torch.float16, add_watermarker=False, local_files_only=True).to(device)
    pipe.unet = register_cross_attention_hook(pipe.unet)
    adapter = IPAdapterXL(pipe, paths["image_encoder"], paths["ip_checkpoint"], device)
    for name, processor in pipe.unet.attn_processors.items():
        if name.endswith("attn2.processor") and not type(processor).__name__.startswith("IPAttnProcessor"):
            raise ValueError("Tencent reference features must reach all UNet cross-attention layers")
    for control in controls:
        if any(not type(processor).__name__.startswith("CNAttnProcessor") for processor in control.attn_processors.values()):
            raise ValueError("IPAdapter must remove reference tokens from every ControlNet")
    if cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=0)
    return adapter


def grayscale_init(target, mask):
    gray = ImageEnhance.Brightness(target.convert("L").convert("RGB")).enhance(1.0)
    mask = mask.convert("RGB")
    return ImageChops.lighter(ImageChops.darker(target, ImageChops.invert(mask)),
                              ImageChops.darker(gray, mask))


def generate(adapter, target, mask, depth, reference, init_image=None, edge_image=None, *,
             strength=0.9999, guidance_scale=5.0, depth_scale=0.9, edge_scale=0.3, scale=1.0,
             num_inference_steps=30, seed=42, prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT):
    import torch

    images = [target, mask, depth, reference]
    images += [image for image in (init_image, edge_image) if image is not None]
    if any(not isinstance(image, Image.Image) or image.size != (1024, 1024) for image in images):
        raise ValueError("freeze all aligned inputs/reference at1024 before inference")
    if not 0 < strength <= 1:
        raise ValueError("strength must be in(0,1]")
    has_edges = hasattr(adapter.pipe.controlnet, "nets")
    if has_edges != (edge_image is not None):
        raise ValueError("edge condition and loaded MultiControlNet must agree")
    init = grayscale_init(target.convert("RGB"), mask) if init_image is None else init_image.convert("RGB")
    control = [depth.convert("RGB"), edge_image.convert("RGB")] if has_edges else depth.convert("RGB")
    control_scale = [depth_scale, edge_scale] if has_edges else depth_scale
    with torch.inference_mode():
        result = adapter.generate(pil_image=reference.convert("RGB"), image=init,
            control_image=control, mask_image=mask.convert("RGB"), controlnet_conditioning_scale=control_scale,
            num_samples=1, num_inference_steps=num_inference_steps, seed=seed,
            strength=strength, guidance_scale=guidance_scale, scale=scale,
            prompt=prompt, negative_prompt=negative_prompt)[0]
    if result.size != (1024, 1024) or result.mode != "RGB":
        raise ValueError("require the raw1024 RGB output without subject/background compositing")
    return result
