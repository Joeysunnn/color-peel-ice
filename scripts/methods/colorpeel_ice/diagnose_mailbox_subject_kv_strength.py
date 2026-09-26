"""Test whether learned S K/V residuals suppress red/blue prompt control."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.methods.colorpeel_ice.generate_color_material_diagnostic import sha256
from scripts.methods.colorpeel_ice.generate_mailbox_matte_subject_comparison import (
    load_pipeline, verify_sources,
)
from scripts.methods.colorpeel_ice.generate_subject_material_diagnostic import attention_masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-kv", action="store_true")
    args = parser.parse_args()
    protocol, baseline, subjects, material_dir = verify_sources(
        args.protocol, Path(os.environ["COLORPEEL_RUN_ROOT"]).resolve())
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    rows = []
    prompts = {group["id"]: group["prompts"]["subject_only"] for group in baseline["groups"]}
    for arm in ("matte_only", "balanced"):
        pipe = load_pipeline(protocol, subjects[arm], material_dir, "cuda:0")
        from custom_attention.attention_processor_custom import DualTokenLocalKVAttnProcessor
        processors = [proc for proc in pipe.unet.attn_processors.values()
                      if getattr(proc, "cross_attention_dim", None) is not None]
        if len(processors) != 16 or not all(
                isinstance(proc, DualTokenLocalKVAttnProcessor) and proc.primary_label == "subject"
                for proc in processors):
            raise ValueError("inference cross-attention is not dual token-local K/V")
        original = [(proc, proc.delta_k.weight.detach().clone(), proc.delta_v.weight.detach().clone())
                    for proc in processors]
        settings = ((0.0, 1.0), (0.5, 1.0), (1.0, 0.5)) if args.split_kv else ((0.0, 0.0), (0.5, 0.5))
        for key_scale, value_scale in settings:
            with torch.no_grad():
                for proc, key, value in original:
                    proc.delta_k.weight.copy_(key * key_scale)
                    proc.delta_v.weight.copy_(value * value_scale)
            for group in ("red", "blue"):
                prompt = prompts[group]
                masks = attention_masks(pipe, prompt, 3.5)["modifier_token_mask"]
                if (masks["subject"].shape[0] != 2 or masks["subject"].sum().item() != 1
                        or masks["material"].sum().item() != 0):
                    raise ValueError("subject/material token masks differ")
                for seed in ((42,) if args.split_kv else (42, 43)):
                    result = pipe(prompt, num_inference_steps=100, guidance_scale=3.5,
                                  generator=torch.Generator(device="cuda:0").manual_seed(seed),
                                  cross_attention_kwargs={"modifier_token_mask": masks})
                    name = f"{arm}__k{key_scale:g}__v{value_scale:g}__{group}__seed{seed}.png"
                    path = args.output_dir / name
                    result.images[0].save(path)
                    flags = getattr(result, "nsfw_content_detected", None)
                    rows.append({"arm": arm, "subject_k_scale": key_scale,
                                 "subject_v_scale": value_scale, "group": group,
                                 "seed": seed, "prompt": prompt, "image": name,
                                 "image_sha256": sha256(path),
                                 "safety_filtered": bool(flags[0]) if flags else False,
                                 "cross_attention_processors": len(processors),
                                 "subject_mask_position": int(masks["subject"][1].nonzero()[0][0])})
                    with (args.output_dir / "status.jsonl").open("a", encoding="utf-8") as status:
                        status.write(json.dumps(rows[-1], sort_keys=True) + "\n")
        del pipe
        torch.cuda.empty_cache()
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "protocol_sha256": sha256(args.protocol), "status": "succeeded", "image_count": len(rows),
        "note": "Only the S K/V residual is scaled; the learned S embedding and selected M adapter are unchanged."
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
