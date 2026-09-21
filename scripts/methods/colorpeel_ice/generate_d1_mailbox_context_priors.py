"""Generate frozen generic-mailbox context priors for subject-only training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


MODEL_ID = "CompVis/stable-diffusion-v1-4"
SCHEMA = "d1_mailbox_context_prior_assets_protocol/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def read_protocol(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError("unexpected context-prior protocol schema")
    source = Path(value["source_instance_concepts"]["path"])
    if not source.is_file() or sha256(source) != value["source_instance_concepts"]["sha256"]:
        raise ValueError("frozen instance concepts do not match the protocol")
    prompts = value.get("prompts")
    sampling = value.get("sampling")
    if not isinstance(prompts, list) or not prompts or not isinstance(sampling, dict):
        raise ValueError("protocol requires prompts and sampling")
    if any("<S*>" in item.get("prompt", "") for item in prompts):
        raise ValueError("class prior prompts must not contain <S*>")
    expected = len(prompts) * len(sampling.get("seeds", []))
    if expected != sampling.get("expected_image_count"):
        raise ValueError("expected_image_count does not match the prompt-seed grid")
    return value


def build_manifest(protocol: dict) -> list[dict]:
    rows = []
    sampling = protocol["sampling"]
    for item in protocol["prompts"]:
        for seed in sampling["seeds"]:
            item_id = f"{item['id']}-seed-{seed}"
            rows.append(
                {
                    "id": item_id,
                    "prompt_id": item["id"],
                    "prompt": item["prompt"],
                    "seed": seed,
                    "num_inference_steps": sampling["num_inference_steps"],
                    "guidance_scale": sampling["guidance_scale"],
                    "image_path": f"images/{item_id}.png",
                }
            )
    return rows


def all_black(image: Image.Image) -> bool:
    return image.convert("RGB").getextrema() == ((0, 0), (0, 0), (0, 0))


def generate(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    import torch
    from diffusers import DiffusionPipeline

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, low_cpu_mem_usage=False, torch_dtype=dtype, local_files_only=True
    ).to(args.device)
    statuses = []
    for row in rows:
        image_path = args.output_dir / row["image_path"]
        status = {"id": row["id"], "status": None, "failure_reason": None, "image_sha256": None}
        try:
            result = pipe(
                row["prompt"],
                num_inference_steps=row["num_inference_steps"],
                guidance_scale=row["guidance_scale"],
                generator=torch.Generator(device=args.device).manual_seed(row["seed"]),
            )
            image = result.images[0]
            if not isinstance(image, Image.Image) or image.mode != "RGB" or image.size != (512, 512):
                raise ValueError("pipeline must return a 512x512 RGB PIL image")
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path)
            status["image_sha256"] = sha256(image_path)
            if all_black(image):
                status.update(status="failure", failure_reason="all_black_output")
            elif any(getattr(result, "nsfw_content_detected", ()) or ()):
                status.update(status="failure", failure_reason="safety_checker_filtered")
            else:
                status["status"] = "ok"
        except Exception as error:
            status.update(status="failure", failure_reason=f"generation_error:{type(error).__name__}:{error}")
        statuses.append(status)
    return statuses


def write_training_concepts(protocol: dict, output_dir: Path, rows: list[dict], statuses: list[dict]) -> None:
    if any(row["status"] != "ok" for row in statuses):
        raise RuntimeError("refusing to create training concepts from failed class-prior images")
    source = Path(protocol["source_instance_concepts"]["path"])
    concepts = json.loads(source.read_text(encoding="utf-8"))
    class_manifest = output_dir / "class_prior_manifest.jsonl"
    records = []
    for row, status in zip(rows, statuses):
        records.append(
            {
                "id": row["id"],
                "image_path": str((output_dir / row["image_path"]).resolve()),
                "prompt": row["prompt"],
                "prompt_id": row["prompt_id"],
                "seed": row["seed"],
                "image_sha256": status["image_sha256"],
            }
        )
    write_jsonl(class_manifest, records)
    for concept in concepts:
        concept["class_data_manifest"] = str(class_manifest.resolve())
    write_json(output_dir / "concepts_with_context_prior.json", concepts)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--pretrained-model-name-or-path", default=MODEL_ID)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    protocol = read_protocol(args.protocol)
    rows = build_manifest(protocol)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "generation_manifest.jsonl", rows)
    write_json(args.output_dir / "provenance.json", {"protocol_sha256": sha256(args.protocol)})
    if args.dry_run:
        return 0
    statuses = generate(rows, args)
    write_jsonl(args.output_dir / "generation_status.jsonl", statuses)
    write_training_concepts(protocol, args.output_dir, rows, statuses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
