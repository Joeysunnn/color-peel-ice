#!/usr/bin/env python3
"""Generate the authorized 18-image natural-noun check for one Emission color token."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
CUSTOM_DIFFUSION_WEIGHTS = "pytorch_custom_diffusion_weights.bin"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if value.get("schema") != "emission_color_transfer_generation_protocol/v1":
        raise ValueError("unexpected generation protocol")
    if value.get("expected_image_count") != 18 or value.get("target", {}).get("modifier_token") != "<C*>":
        raise ValueError("generation protocol identity differs")
    return value


def build_manifest(value: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for noun, template in value["nouns"]:
        for seed in value["seeds"]:
            pair_id = f"{noun}-seed-{seed}"
            for condition in value["conditions"]:
                item_id = f"{pair_id}-{condition['id']}"
                rows.append({
                    "id": item_id, "pair_id": pair_id, "noun": noun, "seed": seed,
                    "condition": condition["id"], "prompt": template.format(color=condition["color_expression"]),
                    "uses_trained_kv": condition["uses_trained_kv"],
                    "num_inference_steps": value["inference"]["num_inference_steps"],
                    "guidance_scale": value["inference"]["guidance_scale"],
                    "image_path": f"images/{condition['id']}/{item_id}.png",
                })
    if len(rows) != value["expected_image_count"] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("generation matrix differs")
    return rows


def verify_parent(parent: Path, model_dir: Path, value: dict[str, Any]) -> dict[str, Any]:
    parent, model_dir = parent.resolve(), model_dir.resolve()
    if model_dir != parent / "checkpoints":
        raise ValueError("model-dir must equal parent-training-run/checkpoints")
    manifest_path = parent / "manifest.json"
    manifest = read_json(manifest_path)
    expected = value["parent_training"]
    if manifest.get("status") != "succeeded" or manifest.get("returncode") != 0:
        raise ValueError("parent training did not succeed")
    if manifest.get("git", {}).get("commit") != expected["git_commit"] or sha256(manifest_path) != expected["training_manifest_sha256"]:
        raise ValueError("parent training manifest differs")
    required = (CUSTOM_DIFFUSION_WEIGHTS, "<C*>.bin")
    if any(not (model_dir / name).is_file() for name in required):
        raise FileNotFoundError("missing trained model artifact")
    if sha256(model_dir / "<C*>.bin") != expected["embedding_sha256"]:
        raise ValueError("learned embedding differs")
    hashes = {name: sha256(model_dir / name) for name in required}
    fingerprint = hashlib.sha256("".join(f"{name}:{hashes[name]}\n" for name in required).encode()).hexdigest()
    return {"parent_training_run": str(parent), "parent_training_manifest_sha256": sha256(manifest_path),
            "model_artifact_sha256": hashes, "model_fingerprint_sha256": fingerprint}


def load_pipeline(args: argparse.Namespace, trained: bool) -> Any:
    import torch
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, low_cpu_mem_usage=False,
                                             torch_dtype=torch.float16, local_files_only=True).to(args.device)
    if trained:
        pipe.unet.load_attn_procs(str(args.model_dir), weight_name=CUSTOM_DIFFUSION_WEIGHTS)
        pipe.load_textual_inversion(str(args.model_dir), weight_name="<C*>.bin")
    return pipe


def generate(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch

    statuses = []
    for trained in (False, True):
        pipe = load_pipeline(args, trained)
        for row in (item for item in rows if item["uses_trained_kv"] is trained):
            path = args.output_dir / row["image_path"]
            status = {"id": row["id"], "image_path": str(path), "status": None, "failure_reason": None}
            try:
                image = pipe(row["prompt"], num_inference_steps=row["num_inference_steps"],
                             guidance_scale=row["guidance_scale"], generator=torch.Generator(args.device).manual_seed(row["seed"])).images[0]
                if not isinstance(image, Image.Image) or image.size != (512, 512) or image.mode != "RGB":
                    raise ValueError("pipeline did not return 512x512 RGB")
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                status.update(status="ok", image_sha256=sha256(path))
            except Exception as exc:
                status.update(status="failure", failure_reason=f"{type(exc).__name__}:{exc}")
            statuses.append(status)
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return statuses


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-training-run", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16",), default="float16")
    parser.add_argument("--pretrained-model-name-or-path", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    value, rows = protocol(args.protocol), None
    rows = build_manifest(value)
    write_jsonl(rows, args.output_dir / "generation_manifest.jsonl")
    if args.dry_run:
        return 0
    if args.parent_training_run is None or args.model_dir is None:
        parser.error("--parent-training-run and --model-dir are required unless --dry-run")
    provenance = verify_parent(args.parent_training_run, args.model_dir, value)
    (args.output_dir / "generation_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    statuses = generate(rows, args)
    write_jsonl(statuses, args.output_dir / "generation_status.jsonl")
    return 0 if all(row["status"] == "ok" for row in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
