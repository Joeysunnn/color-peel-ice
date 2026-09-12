#!/usr/bin/env python3
"""Generate the user-authorized, one-off 5-template by 8-seed transfer check."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from PIL import Image


CUSTOM_DIFFUSION_WEIGHTS = "pytorch_custom_diffusion_weights.bin"
OLD_COLOR_TOKEN = re.compile(r"<c[123]\*>")


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
    if value.get("schema") != "emission_color_transfer_quick40_protocol/v1":
        raise ValueError("unexpected quick-transfer protocol")
    if value.get("expected_image_count") != 40 or value.get("purpose") != "one_off_quick_transfer_test_not_a_future_evaluation_default":
        raise ValueError("quick-transfer scope differs")
    return value


def source_templates(source_manifest: Path, value: dict[str, Any]) -> dict[int, str]:
    source = value["source_manifest"]
    if sha256(source_manifest) != source["sha256"]:
        raise ValueError("source manifest hash differs")
    rows = [json.loads(line) for line in source_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    transfer = [row for row in rows if row.get("category") == source["category"]]
    if len(transfer) != source["required_transfer_row_count"]:
        raise ValueError("source transfer row count differs")
    templates: dict[int, set[str]] = {}
    for row in transfer:
        index, prompt = row.get("transfer_template_index"), row.get("prompt")
        if not isinstance(index, int) or not isinstance(prompt, str) or OLD_COLOR_TOKEN.search(prompt) is None:
            raise ValueError("source transfer row differs")
        templates.setdefault(index, set()).add(OLD_COLOR_TOKEN.sub("{color}", prompt))
    canonical = {index: next(iter(prompts)) for index, prompts in templates.items() if len(prompts) == 1}
    if sorted(canonical) != value["selection"]["candidate_transfer_template_indices"]:
        raise ValueError("source transfer templates differ")
    return canonical


def build_manifest(source_manifest: Path, value: dict[str, Any]) -> list[dict[str, Any]]:
    templates = source_templates(source_manifest, value)
    selection = value["selection"]
    sampled = random.Random(selection["random_seed"]).sample(selection["candidate_transfer_template_indices"], 5)
    if sampled != selection["selected_transfer_template_indices"]:
        raise ValueError("recorded random selection differs")
    rows = []
    for index in sampled:
        for seed in selection["inference_seeds"]:
            item_id = f"transfer-{index:02d}-seed-{seed}"
            rows.append({"id": item_id, "category": "transfer", "transfer_template_index": index,
                         "prompt": templates[index].format(color=selection["replacement_token"]), "seed": seed,
                         "num_inference_steps": value["inference"]["num_inference_steps"],
                         "guidance_scale": value["inference"]["guidance_scale"],
                         "image_path": f"images/transfer/{item_id}.png"})
    if len(rows) != value["expected_image_count"]:
        raise ValueError("quick-transfer image count differs")
    return rows


def verify_parent(parent: Path, model_dir: Path, value: dict[str, Any]) -> dict[str, Any]:
    parent, model_dir = parent.resolve(), model_dir.resolve()
    if model_dir != parent / "checkpoints":
        raise ValueError("model-dir must equal parent-training-run/checkpoints")
    manifest_path = parent / "manifest.json"
    manifest, expected = read_json(manifest_path), value["parent_training"]
    if manifest.get("status") != "succeeded" or manifest.get("returncode") != 0:
        raise ValueError("parent training did not succeed")
    if manifest.get("git", {}).get("commit") != expected["git_commit"] or sha256(manifest_path) != expected["training_manifest_sha256"]:
        raise ValueError("parent training manifest differs")
    required = (CUSTOM_DIFFUSION_WEIGHTS, "<C*>.bin")
    if any(not (model_dir / name).is_file() for name in required) or sha256(model_dir / "<C*>.bin") != expected["embedding_sha256"]:
        raise ValueError("parent model artifact differs")
    hashes = {name: sha256(model_dir / name) for name in required}
    fingerprint = hashlib.sha256("".join(f"{name}:{hashes[name]}\n" for name in required).encode()).hexdigest()
    return {"parent_training_run": str(parent), "parent_training_manifest_sha256": sha256(manifest_path),
            "model_artifact_sha256": hashes, "model_fingerprint_sha256": fingerprint}


def load_pipeline(args: argparse.Namespace) -> Any:
    import torch
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, low_cpu_mem_usage=False,
                                             torch_dtype=torch.float16, local_files_only=True).to(args.device)
    pipe.unet.load_attn_procs(str(args.model_dir), weight_name=CUSTOM_DIFFUSION_WEIGHTS)
    pipe.load_textual_inversion(str(args.model_dir), weight_name="<C*>.bin")
    return pipe


def generate(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch

    pipe, statuses = load_pipeline(args), []
    for row in rows:
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
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16",), default="float16")
    parser.add_argument("--pretrained-model-name-or-path", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    value = protocol(args.protocol)
    rows = build_manifest(args.source_manifest, value)
    write_jsonl(rows, args.output_dir / "generation_manifest.jsonl")
    if args.dry_run:
        return 0
    if args.parent_training_run is None or args.model_dir is None:
        parser.error("--parent-training-run and --model-dir are required unless --dry-run")
    provenance = verify_parent(args.parent_training_run, args.model_dir, value)
    provenance["source_manifest_sha256"] = sha256(args.source_manifest)
    (args.output_dir / "generation_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    statuses = generate(rows, args)
    write_jsonl(statuses, args.output_dir / "generation_status.jsonl")
    return 0 if all(row["status"] == "ok" for row in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
