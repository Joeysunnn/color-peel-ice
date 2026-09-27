"""One-step SD pipeline smoke for Subject Key, Subject Value, and selected Material masks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    from diffusers import DiffusionPipeline
    from scripts.methods.colorpeel_ice.generate_subject_material_diagnostic import (
        attention_masks, verify_sources,
    )
    from experiments.perfusion_subject_pilot.perfusion_attention import (
        install_subject_material, mailbox_key_reference,
    )

    train_root = str(REPO_ROOT / "src" / "train")
    if train_root not in sys.path:
        sys.path.insert(0, train_root)
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel

    baseline_path = REPO_ROOT / "experiments/subject_material_composition_v1/protocols/mailbox_metal_diagnostic_v1.json"
    protocol, subject_dir, material_dir = verify_sources(baseline_path, args.run_root.resolve())
    unet = UNet2DConditionModel.from_pretrained(
        protocol["base_model"], subfolder="unet", local_files_only=True, torch_dtype=torch.float16)
    material_state = torch.load(material_dir / "pytorch_token_local_kv_weights.bin", map_location="cpu")
    subject_state = {}
    for name in unet.attn_processors:
        if name.endswith("attn1.processor"):
            continue
        weight = material_state[f"{name}.delta_k.weight"]
        subject_state[f"{name}.value_a"] = torch.zeros(weight.shape[0])
        subject_state[f"{name}.value_b"] = torch.ones(weight.shape[1]) * 0.01
    install_subject_material(unet, subject_state, material_state)
    pipe = DiffusionPipeline.from_pretrained(
        protocol["base_model"], unet=unet, low_cpu_mem_usage=False,
        torch_dtype=torch.float16, local_files_only=True).to(args.device)
    pipe.load_textual_inversion(str(subject_dir), weight_name="<S*>.bin")
    pipe.load_textual_inversion(str(material_dir), weight_name="<M*>.bin")
    reference = mailbox_key_reference(pipe.text_encoder, pipe.tokenizer)
    for prompt in ("a photo of <S*> mailbox in blue color",
                   "a photo of <S*> mailbox in blue color made of <M*>"):
        kwargs = attention_masks(pipe, prompt, 3.5)
        kwargs["key_reference"] = reference
        result = pipe(prompt, num_inference_steps=1, guidance_scale=3.5,
                      generator=torch.Generator(device=args.device).manual_seed(42),
                      cross_attention_kwargs=kwargs)
        if len(result.images) != 1:
            raise RuntimeError("pipeline smoke did not produce one image")
    print("Perfusion Subject and selected Material pipeline smoke: passed")


if __name__ == "__main__":
    main()
