"""Generate the standalone material token-local pilot evaluation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


TOKEN_LOCAL_WEIGHTS = "pytorch_token_local_kv_weights.bin"
ADAPTATION_CONFIG = "adaptation_config.json"
TOKEN = "<M*>"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    if protocol.get("schema") != "material_token_local_pilot_evaluation/v1":
        raise ValueError("Unexpected material evaluation schema")
    if protocol.get("protocol_id") != "material_token_local_pilot_v1_evaluation":
        raise ValueError("Unexpected material evaluation protocol")
    if protocol.get("checkpoint") != {"adaptation_mode": "token_local_kv", "required_token": TOKEN}:
        raise ValueError("Evaluation must require only the token-local material checkpoint")
    sampling = protocol.get("sampling")
    if sampling != {"seeds": [42, 43, 44, 45, 46], "num_inference_steps": 100, "guidance_scale": 3.5}:
        raise ValueError("Evaluation sampling contract changed")
    groups = protocol.get("groups")
    expected_ids = ["seen_reconstruction", "color_invariance", "unseen_object_transfer", "lighting_robustness"]
    if not isinstance(groups, list) or [group.get("id") for group in groups] != expected_ids:
        raise ValueError("Evaluation group contract changed")
    if any(not isinstance(prompt, str) or TOKEN not in prompt for group in groups for prompt in group.get("prompts", [])):
        raise ValueError("Every evaluation prompt must contain the material token")
    return protocol


def build_manifest(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    protocol = validate_protocol(protocol)
    sampling = protocol["sampling"]
    rows: list[dict[str, Any]] = []
    for group in protocol["groups"]:
        for prompt_index, prompt in enumerate(group["prompts"]):
            for seed in sampling["seeds"]:
                item_id = f"{group['id']}-prompt-{prompt_index:02d}-seed-{seed}"
                rows.append({
                    "id": item_id,
                    "group": group["id"],
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "seed": seed,
                    "num_inference_steps": sampling["num_inference_steps"],
                    "guidance_scale": sampling["guidance_scale"],
                    "adaptation_mode": "token_local_kv",
                    "modifier_token": TOKEN,
                    "image_path": f"images/{group['id']}/{item_id}.png",
                })
    if len(rows) != 60 or len({row["id"] for row in rows}) != len(rows):
        raise AssertionError("Material evaluation manifest is incomplete")
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def validate_model_dir(model_dir: Path) -> dict[str, str]:
    paths = {
        "weights": model_dir / TOKEN_LOCAL_WEIGHTS,
        "adaptation": model_dir / ADAPTATION_CONFIG,
        "token": model_dir / f"{TOKEN}.bin",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing material checkpoint artifacts:\n" + "\n".join(missing))
    adaptation = read_json(paths["adaptation"])
    if adaptation.get("adaptation_mode") != "token_local_kv" or adaptation.get("modifier_tokens") != [TOKEN]:
        raise ValueError("Checkpoint is not a single-token material token-local checkpoint")
    return {name: sha256(path) for name, path in paths.items()}


def validate_checkpoint_lock(lock_path: Path, protocol_path: Path, model_dir: Path,
                             checkpoint_hashes: dict[str, str], base_model: str) -> str:
    lock = read_json(lock_path)
    manifest_path = Path(lock.get("run_manifest_path", "")).resolve()
    if (lock.get("schema") != "material_token_local_checkpoint_lock/v1"
            or lock.get("evaluation_protocol_sha256") != sha256(protocol_path)
            or Path(lock.get("model_dir", "")).resolve() != model_dir.resolve()
            or model_dir.resolve() != manifest_path.parent / "checkpoints"
            or lock.get("checkpoint_sha256") != checkpoint_hashes
            or lock.get("base_model") != base_model
            or lock.get("run_manifest_sha256") != sha256(manifest_path)):
        raise ValueError("Material checkpoint lock does not match the selected run and artifacts")
    manifest = read_json(manifest_path)
    if (manifest.get("status") != "succeeded" or manifest.get("stage") != "train"
            or manifest.get("run", {}).get("study") != "material_token_local_pilot_v1"
            or manifest.get("run", {}).get("variant") != "standalone_metal_token_local_kv_5000"):
        raise ValueError("Checkpoint source is not a completed standalone material training run")
    return sha256(lock_path)


def token_local_cross_attention_kwargs(pipe: Any, prompt: str, guidance_scale: float) -> dict[str, Any]:
    import torch

    input_ids = pipe.tokenizer(
        prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(pipe.unet.device)
    token_id = pipe.tokenizer.convert_tokens_to_ids(TOKEN)
    conditional_mask = input_ids == token_id
    if conditional_mask.sum().item() != 1:
        raise ValueError("Every evaluation prompt must contain exactly one material token")
    if guidance_scale > 1.0:
        conditional_mask = torch.cat([torch.zeros_like(conditional_mask), conditional_mask], dim=0)
    return {"modifier_token_mask": conditional_mask}


def generate(rows: Iterable[dict[str, Any]], args: argparse.Namespace, checkpoint_hashes: dict[str, str]) -> None:
    import torch
    from diffusers import DiffusionPipeline

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    train_root = str(Path(__file__).resolve().parents[3] / "src" / "train")
    if train_root not in sys.path:
        sys.path.insert(0, train_root)
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", local_files_only=True, torch_dtype=dtype,
    )
    unet.load_attn_procs(
        str(args.model_dir), weight_name=TOKEN_LOCAL_WEIGHTS, adaptation_mode="token_local_kv",
    )
    pipe = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, unet=unet, low_cpu_mem_usage=False,
        torch_dtype=dtype, local_files_only=True,
    ).to(args.device)
    if args.disable_safety_checker:
        pipe.safety_checker = None
        if hasattr(pipe, "requires_safety_checker"):
            pipe.requires_safety_checker = False
    pipe.load_textual_inversion(str(args.model_dir), weight_name=f"{TOKEN}.bin")

    for row in rows:
        image_path = args.output_dir / row["image_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator(device=args.device).manual_seed(row["seed"])
        result = pipe(
            row["prompt"], num_inference_steps=row["num_inference_steps"], guidance_scale=row["guidance_scale"],
            generator=generator,
            cross_attention_kwargs=token_local_cross_attention_kwargs(pipe, row["prompt"], row["guidance_scale"]),
        )
        nsfw_flags = getattr(result, "nsfw_content_detected", None)
        if nsfw_flags is not None and len(nsfw_flags) != 1:
            raise ValueError("Expected one safety-checker result per evaluation image")
        safety_filtered = bool(nsfw_flags[0]) if nsfw_flags is not None else False
        result.images[0].save(image_path)
        append_jsonl(args.output_dir / "generation_status.jsonl", {
            **row, "status": "safety_filtered" if safety_filtered else "ok",
            "nsfw_content_detected": safety_filtered, "image_sha256": sha256(image_path),
            "checkpoint_sha256": checkpoint_hashes,
        })


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--checkpoint-lock", type=Path)
    parser.add_argument("--pretrained-model-name-or-path", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--disable-safety-checker", action="store_true")
    parser.add_argument("--acknowledge-safety-risk", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.disable_safety_checker != args.acknowledge_safety_risk:
        parser.error("safety-checker disabling requires explicit acknowledgement and vice versa")
    if not args.dry_run and (args.model_dir is None or args.checkpoint_lock is None):
        parser.error("--model-dir and --checkpoint-lock are required unless --dry-run is used")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = validate_protocol(read_json(args.protocol))
    rows = build_manifest(protocol)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {args.output_dir}")
    checkpoint_hashes = None if args.dry_run else validate_model_dir(args.model_dir)
    lock_hash = None if args.dry_run else validate_checkpoint_lock(
        args.checkpoint_lock, args.protocol, args.model_dir, checkpoint_hashes,
        args.pretrained_model_name_or_path,
    )
    write_jsonl(args.output_dir / "generation_manifest.jsonl", rows)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True,
    ).strip()
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "git_commit": commit, "protocol_sha256": sha256(args.protocol),
        "generation_manifest_sha256": sha256(args.output_dir / "generation_manifest.jsonl"),
        "checkpoint_sha256": checkpoint_hashes,
        "checkpoint_lock_sha256": lock_hash,
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.dry_run:
        return 0
    generate(rows, args, checkpoint_hashes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
