"""Generate fixed base/C/M/C+M comparisons from two frozen token-local adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WEIGHTS = "pytorch_token_local_kv_weights.bin"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_checkpoint(directory: Path, expected: dict, token: str) -> None:
    artifacts = {
        "weights": directory / WEIGHTS,
        "adaptation": directory / "adaptation_config.json",
        "token": directory / f"{token}.bin",
    }
    if {key: sha256(path) for key, path in artifacts.items()} != expected:
        raise ValueError(f"{token} checkpoint hash differs")
    adaptation = read_json(artifacts["adaptation"])
    if adaptation.get("adaptation_mode") != "token_local_kv" or adaptation.get("modifier_tokens") != [token]:
        raise ValueError(f"{token} adaptation metadata differs")


def verify_sources(protocol_path: Path, run_root: Path) -> tuple[dict, Path, Path]:
    protocol = read_json(protocol_path)
    if (protocol.get("schema") != "color_material_diagnostic/v1"
            or protocol.get("base_model") != "CompVis/stable-diffusion-v1-4"
            or protocol.get("safety_checker") != "enabled"
            or protocol.get("sampling") != {"seeds": [42, 43, 44], "num_inference_steps": 100, "guidance_scale": 3.5}
            or protocol.get("objects") != ["cube", "sphere", "mug"]):
        raise ValueError("color/material diagnostic protocol differs")
    color_run = (run_root / protocol["color_training_run_relative_to_COLORPEEL_RUN_ROOT"]).resolve()
    manifest = color_run / "manifest.json"
    if sha256(manifest) != protocol["color_training_manifest_sha256"]:
        raise ValueError("color training manifest hash differs")
    completed = read_json(manifest)
    if (completed.get("status") != "succeeded" or completed.get("stage") != "train"
            or completed.get("run", {}).get("variant") != "orange_token_local_color_short100"):
        raise ValueError("color training did not complete as specified")
    color_dir = color_run / "checkpoints"
    verify_checkpoint(color_dir, protocol["color_checkpoint_sha256"], "<C*>")

    selection_path = REPO_ROOT / protocol["material_selection_path"]
    if sha256(selection_path) != protocol["material_selection_sha256"]:
        raise ValueError("material selection hash differs")
    selection = read_json(selection_path)
    material_run = (run_root / selection["training_run_relative_to_COLORPEEL_RUN_ROOT"]).resolve()
    if sha256(material_run / "manifest.json") != selection["training_manifest_sha256"]:
        raise ValueError("selected material training manifest differs")
    material_dir = material_run / selection["checkpoint"]["directory_relative_to_training_run"]
    verify_checkpoint(material_dir, {
        "weights": selection["checkpoint"]["token_local_kv_weights_sha256"],
        "adaptation": selection["checkpoint"]["adaptation_config_sha256"],
        "token": selection["checkpoint"]["token_embedding_sha256"],
    }, "<M*>")
    return protocol, color_dir, material_dir


def manifest_rows(protocol: dict) -> list[dict]:
    prompts = protocol["prompts"]
    if set(prompts) != {"base", "color_only", "material_only", "color_material"}:
        raise ValueError("diagnostic comparison conditions differ")
    rows = []
    for object_name in protocol["objects"]:
        for condition, template in prompts.items():
            prompt = template.format(object=object_name)
            if ("<C*>" in prompt) != (condition in {"color_only", "color_material"}):
                raise ValueError("color token differs from condition")
            if ("<M*>" in prompt) != (condition in {"material_only", "color_material"}):
                raise ValueError("material token differs from condition")
            for seed in protocol["sampling"]["seeds"]:
                sample_id = f"{object_name}__{condition}__seed{seed}"
                rows.append({"id": sample_id, "object": object_name, "condition": condition,
                             "prompt": prompt, "seed": seed, "image_path": f"images/{object_name}/{sample_id}.png"})
    if len(rows) != 36:
        raise ValueError("expected 36 diagnostic samples")
    return rows


def attention_masks(pipe, prompt: str, guidance: float) -> dict:
    import torch

    ids = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids.to(pipe.unet.device)
    masks = {}
    for label, token in (("color", "<C*>"), ("material", "<M*>")):
        mask = ids == pipe.tokenizer.convert_tokens_to_ids(token)
        if int(mask.sum()) != int(token in prompt):
            raise ValueError(f"tokenization differs for {label} in {prompt}")
        if guidance > 1:
            mask = torch.cat([torch.zeros_like(mask), mask], dim=0)
        masks[label] = mask
    return {"modifier_token_mask": masks}


def generate(protocol: dict, color_dir: Path, material_dir: Path, output_dir: Path, device: str) -> None:
    import torch
    from diffusers import DiffusionPipeline

    train_root = str(REPO_ROOT / "src" / "train")
    if train_root not in sys.path:
        sys.path.insert(0, train_root)
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel
    from src.methods.colorpeel_ice.dual_token_local_kv import install_dual_token_local_kv

    unet = UNet2DConditionModel.from_pretrained(
        protocol["base_model"], subfolder="unet", local_files_only=True, torch_dtype=torch.float16)
    unet.load_attn_procs(str(color_dir), weight_name=WEIGHTS, adaptation_mode="token_local_kv")
    material_state = torch.load(material_dir / WEIGHTS, map_location="cpu")
    install_dual_token_local_kv(unet, material_state)
    pipe = DiffusionPipeline.from_pretrained(
        protocol["base_model"], unet=unet, low_cpu_mem_usage=False,
        torch_dtype=torch.float16, local_files_only=True).to(device)
    pipe.load_textual_inversion(str(color_dir), weight_name="<C*>.bin")
    pipe.load_textual_inversion(str(material_dir), weight_name="<M*>.bin")
    sampling = protocol["sampling"]
    status_path = output_dir / "generation_status.jsonl"
    with status_path.open("w", encoding="utf-8") as ledger:
        for row in manifest_rows(protocol):
            result = pipe(row["prompt"], num_inference_steps=sampling["num_inference_steps"],
                          guidance_scale=sampling["guidance_scale"],
                          generator=torch.Generator(device=device).manual_seed(row["seed"]),
                          cross_attention_kwargs=attention_masks(pipe, row["prompt"], sampling["guidance_scale"]))
            image_path = output_dir / row["image_path"]
            image_path.parent.mkdir(parents=True, exist_ok=True)
            result.images[0].save(image_path)
            flags = getattr(result, "nsfw_content_detected", None)
            if flags is not None and len(flags) != 1:
                raise ValueError("expected one safety-checker result")
            filtered = bool(flags[0]) if flags is not None else False
            ledger.write(json.dumps({**row, "status": "safety_filtered" if filtered else "ok",
                                     "image_sha256": sha256(image_path), "nsfw_content_detected": filtered}, sort_keys=True) + "\n")
            ledger.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_root = Path(os.environ["COLORPEEL_RUN_ROOT"]).resolve()
    protocol, color_dir, material_dir = verify_sources(args.protocol, run_root)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    rows = manifest_rows(protocol)
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "protocol_sha256": sha256(args.protocol), "color_dir": str(color_dir),
        "material_dir": str(material_dir), "status": "dry_run" if args.dry_run else "running",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "generation_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if not args.dry_run:
        generate(protocol, color_dir, material_dir, args.output_dir, args.device)
        provenance = read_json(args.output_dir / "provenance.json")
        provenance["status"] = "succeeded"
        (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(rows)} samples: {args.output_dir}")


if __name__ == "__main__":
    main()
