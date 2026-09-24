"""Compare archived full-K/V emission C with current token-local C and frozen M."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "train"))

from scripts.methods.colorpeel_ice.generate_color_material_diagnostic import (
    read_json, sha256, verify_sources,
)


def verify(protocol_path: Path, run_root: Path):
    protocol = read_json(protocol_path)
    if (protocol.get("schema") != "archived_emission_color_comparison/v1"
            or protocol.get("base_model") != "CompVis/stable-diffusion-v1-4"
            or protocol.get("objects") != ["cube", "sphere", "mug"]
            or protocol.get("seeds") != [42, 43, 44]
            or protocol.get("num_inference_steps") != 100
            or protocol.get("guidance_scale") != 6.0
            or protocol.get("safety_checker") != "enabled"
            or protocol.get("prompts") != {
                "color_only": "a photo of a {object} in <C*> color",
                "color_material": "a photo of a {object} in <C*> color and <M*> material",
            }):
        raise ValueError("comparison protocol differs")
    old_run = run_root / protocol["old_color_run"]
    old_dir = old_run / "checkpoints"
    for path, expected in (
        (old_run / "manifest.json", protocol["old_color_manifest_sha256"]),
        (old_dir / "pytorch_custom_diffusion_weights.bin", protocol["old_color_weights_sha256"]),
        (old_dir / "<C*>.bin", protocol["old_color_token_sha256"]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"archived color source differs: {path}")
    if read_json(old_run / "manifest.json").get("status") != "succeeded":
        raise ValueError("archived color training failed")
    new_protocol, new_dir, material_dir = verify_sources(REPO_ROOT / protocol["new_color_protocol"], run_root)
    if protocol["material_selection"] != new_protocol["material_selection_path"]:
        raise ValueError("material selection differs")
    return protocol, old_dir, new_dir, material_dir


def rows(protocol: dict) -> list[dict]:
    return [
        {"branch": branch, "condition": condition, "object": obj, "seed": seed,
         "prompt": template.format(object=obj),
         "image_path": f"images/{obj}/{branch}__{condition}__seed{seed}.png"}
        for obj in protocol["objects"]
        for branch in ("archived_full_kv", "current_token_local")
        for condition, template in protocol["prompts"].items()
        for seed in protocol["seeds"]
    ]


def mask(pipe, prompt: str, token: str, guidance: float):
    import torch
    ids = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids.to(pipe.unet.device)
    result = ids == pipe.tokenizer.convert_tokens_to_ids(token)
    if int(result.sum()) != int(token in prompt):
        raise ValueError(f"tokenization differs: {token}, {prompt}")
    if guidance > 1:
        result = torch.cat((torch.zeros_like(result), result), dim=0)
    return result


def pipeline(protocol: dict, color_dir: Path, material_dir: Path, branch: str, device: str):
    import torch
    from diffusers import DiffusionPipeline
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel
    from src.methods.colorpeel_ice.dual_token_local_kv import install_dual_token_local_kv
    from src.methods.colorpeel_ice.full_color_material_kv import install_full_color_material_kv

    unet = UNet2DConditionModel.from_pretrained(
        protocol["base_model"], subfolder="unet", local_files_only=True, torch_dtype=torch.float16)
    if branch == "archived_full_kv":
        unet.load_attn_procs(str(color_dir), weight_name="pytorch_custom_diffusion_weights.bin")
        install_full_color_material_kv(unet, torch.load(
            material_dir / "pytorch_token_local_kv_weights.bin", map_location="cpu"))
    else:
        unet.load_attn_procs(str(color_dir), weight_name="pytorch_token_local_kv_weights.bin",
                             adaptation_mode="token_local_kv")
        install_dual_token_local_kv(unet, torch.load(
            material_dir / "pytorch_token_local_kv_weights.bin", map_location="cpu"))
    pipe = DiffusionPipeline.from_pretrained(
        protocol["base_model"], unet=unet, low_cpu_mem_usage=False,
        torch_dtype=torch.float16, local_files_only=True).to(device)
    pipe.load_textual_inversion(str(color_dir), weight_name="<C*>.bin")
    pipe.load_textual_inversion(str(material_dir), weight_name="<M*>.bin")
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol, old_dir, new_dir, material_dir = verify(args.protocol, Path(os.environ["COLORPEEL_RUN_ROOT"]))
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    plan = rows(protocol)
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "protocol_sha256": sha256(args.protocol), "old_color_dir": str(old_dir),
        "new_color_dir": str(new_dir), "material_dir": str(material_dir),
        "status": "dry_run" if args.dry_run else "running",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "generation_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in plan), encoding="utf-8")
    if args.dry_run:
        print(f"planned {len(plan)} images")
        return
    import torch
    with (args.output_dir / "generation_status.jsonl").open("w", encoding="utf-8") as ledger:
        for branch, directory in (("archived_full_kv", old_dir), ("current_token_local", new_dir)):
            pipe = pipeline(protocol, directory, material_dir, branch, args.device)
            for row in (item for item in plan if item["branch"] == branch):
                prompt = row["prompt"]
                material_mask = mask(pipe, prompt, "<M*>", protocol["guidance_scale"])
                masks = material_mask if branch == "archived_full_kv" else {
                    "color": mask(pipe, prompt, "<C*>", protocol["guidance_scale"]),
                    "material": material_mask,
                }
                output = pipe(prompt, num_inference_steps=protocol["num_inference_steps"],
                              guidance_scale=protocol["guidance_scale"],
                              generator=torch.Generator(device=args.device).manual_seed(row["seed"]),
                              cross_attention_kwargs={"modifier_token_mask": masks})
                image_path = args.output_dir / row["image_path"]
                image_path.parent.mkdir(parents=True, exist_ok=True)
                output.images[0].save(image_path)
                flags = getattr(output, "nsfw_content_detected", None)
                filtered = bool(flags[0]) if flags is not None else False
                ledger.write(json.dumps({**row, "status": "safety_filtered" if filtered else "ok",
                                         "image_sha256": sha256(image_path)}, sort_keys=True) + "\n")
                ledger.flush()
            del pipe
            torch.cuda.empty_cache()
    provenance = read_json(args.output_dir / "provenance.json")
    provenance["status"] = "succeeded"
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"generated {len(plan)} images")


if __name__ == "__main__":
    main()
