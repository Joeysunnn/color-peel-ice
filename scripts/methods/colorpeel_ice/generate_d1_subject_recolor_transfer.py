"""Generate the fixed D1 subject-only transfer diagnostic without retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


MODEL_ID = "CompVis/stable-diffusion-v1-4"
CUSTOM_DIFFUSION_WEIGHTS = "pytorch_custom_diffusion_weights.bin"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_manifest(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    sampling = protocol["sampling"]
    rows = []
    for group in protocol["prompt_groups"]:
        for item in group["items"]:
            for seed in sampling["seeds"]:
                rows.append(
                    {
                        "id": f"{item['id']}-seed-{seed}",
                        "group": group["group"],
                        "prompt_id": item["id"],
                        "prompt": item["prompt"],
                        "seed": seed,
                        "num_inference_steps": sampling["num_inference_steps"],
                        "guidance_scale": sampling["guidance_scale"],
                        "image_path": f"images/{group['group']}/{item['id']}-seed-{seed}.png",
                    }
                )
    if len(rows) != sampling["expected_image_count"]:
        raise ValueError("protocol expected_image_count does not match its prompt/seed grid")
    return rows


def validate_model_dir(model_dir: Path, protocol: dict[str, Any]) -> dict[str, str]:
    required = set(protocol["source_training"]["required_artifacts"])
    required.add(CUSTOM_DIFFUSION_WEIGHTS)
    missing = [name for name in sorted(required) if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("missing source artifacts: " + ", ".join(missing))
    forbidden = [name for name in protocol["source_training"]["forbidden_artifacts"] if (model_dir / name).exists()]
    if forbidden:
        raise ValueError("forbidden color artifacts are present: " + ", ".join(forbidden))
    return {name: sha256(model_dir / name) for name in sorted(required)}


def load_pipeline(args: argparse.Namespace) -> Any:
    import torch
    from diffusers import DiffusionPipeline

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        low_cpu_mem_usage=False,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(args.device)
    pipe.unet.load_attn_procs(str(args.model_dir), weight_name=CUSTOM_DIFFUSION_WEIGHTS)
    pipe.load_textual_inversion(str(args.model_dir), weight_name="<S*>.bin")
    return pipe


def generate(rows: Iterable[dict[str, Any]], pipe: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch

    statuses = []
    for row in rows:
        output_path = args.output_dir / row["image_path"]
        status = {"id": row["id"], "image_path": str(output_path), "status": None, "failure_reason": None,
                  "image_sha256": None, "nsfw_content_detected": None}
        try:
            image = pipe(
                row["prompt"],
                num_inference_steps=row["num_inference_steps"],
                guidance_scale=row["guidance_scale"],
                generator=torch.Generator(device=args.device).manual_seed(row["seed"]),
            ).images[0]
            if not isinstance(image, Image.Image) or image.mode != "RGB" or image.size != (512, 512):
                raise ValueError("pipeline must return a 512x512 RGB PIL image")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            status.update(status="ok", image_sha256=sha256(output_path))
        except Exception as error:
            status.update(status="failure", failure_reason=f"generation_error:{type(error).__name__}:{error}")
        statuses.append(status)
    return statuses


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--pretrained-model-name-or-path", default=MODEL_ID)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run and args.model_dir is None:
        parser.error("--model-dir is required unless --dry-run is used")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = read_json(args.protocol)
    rows = build_manifest(protocol)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(rows, args.output_dir / "generation_manifest.jsonl")
    provenance = {"protocol_path": str(args.protocol.resolve()), "protocol_sha256": sha256(args.protocol),
                  "source_training": protocol["source_training"], "model_artifact_sha256": None}
    if args.dry_run:
        write_json(args.output_dir / "provenance.json", provenance)
        return 0
    artifact_hashes = validate_model_dir(args.model_dir, protocol)
    provenance["model_dir"] = str(args.model_dir.resolve())
    provenance["model_artifact_sha256"] = artifact_hashes
    write_json(args.output_dir / "provenance.json", provenance)
    statuses = generate(rows, load_pipeline(args), args)
    write_jsonl(statuses, args.output_dir / "generation_status.jsonl")
    return 0 if all(status["status"] == "ok" for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
