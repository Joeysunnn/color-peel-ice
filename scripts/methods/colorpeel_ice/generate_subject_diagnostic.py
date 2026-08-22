"""Generate the fixed, small subject-token diagnostic without retraining.

All 75 images use the trained Custom Diffusion K/V and learned embeddings.
For each shape and seed 42--46, the diagnostic compares learned versus natural
subject-only prompts and the learned subject paired with literal red/cyan/gray.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


MODEL_ID = "CompVis/stable-diffusion-v1-4"
CUSTOM_DIFFUSION_WEIGHTS = "pytorch_custom_diffusion_weights.bin"
SUBJECTS = (
    ("s1", "<s1*>", "cube"),
    ("s2", "<s2*>", "sphere"),
    ("s3", "<s3*>", "cylinder"),
)
COLORS = ("red", "cyan", "gray")
TOKENS = ("<s1*>", "<s2*>", "<s3*>", "<c1*>", "<c2*>", "<c3*>")
SEEDS = tuple(range(42, 47))
STEPS = 100
GUIDANCE_SCALE = 6.0


def build_manifest() -> list[dict[str, Any]]:
    rows = []
    for subject_id, subject_token, subject_label in SUBJECTS:
        conditions = (
            (
                "learned_subject_only",
                f"a photo of {subject_token} shape",
                "learned",
                None,
            ),
            (
                "natural_subject_only",
                f"a photo of {subject_label} shape",
                "natural",
                None,
            ),
            *(
                (
                    f"learned_subject_literal_{color}",
                    f"a photo of {subject_token} shape in {color} color",
                    "learned",
                    color,
                )
                for color in COLORS
            ),
        )
        for seed in SEEDS:
            pair_id = f"subject-{subject_id}-seed-{seed}"
            for condition, prompt, subject_expression, literal_color in conditions:
                item_id = f"{pair_id}-{condition}"
                rows.append(
                    {
                        "id": item_id,
                        "pair_id": pair_id,
                        "category": "subject_diagnostic",
                        "condition": condition,
                        "subject_id": subject_id,
                        "subject_token": subject_token,
                        "subject_label": subject_label,
                        "subject_expression": subject_expression,
                        "literal_color": literal_color,
                        "prompt": prompt,
                        "seed": seed,
                        "num_inference_steps": STEPS,
                        "guidance_scale": GUIDANCE_SCALE,
                        "uses_trained_kv": True,
                        "image_path": f"images/{condition}/{item_id}.png",
                    }
                )
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def validate_model_dir(model_dir: Path) -> None:
    required = [model_dir / CUSTOM_DIFFUSION_WEIGHTS]
    required.extend(model_dir / f"{token}.bin" for token in TOKENS)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing trained weights:\n" + "\n".join(missing))


def load_pipeline(args: argparse.Namespace) -> Any:
    import torch
    from diffusers import DiffusionPipeline

    model_dir = args.model_dir.resolve()
    validate_model_dir(model_dir)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        low_cpu_mem_usage=False,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(args.device)
    pipe.unet.load_attn_procs(
        str(model_dir), weight_name=CUSTOM_DIFFUSION_WEIGHTS
    )
    for token in TOKENS:
        pipe.load_textual_inversion(str(model_dir), weight_name=f"{token}.bin")
    if args.disable_safety_checker:
        pipe.safety_checker = None
        if hasattr(pipe, "requires_safety_checker"):
            pipe.requires_safety_checker = False
    return pipe


def generate(
    rows: Iterable[dict[str, Any]], pipe: Any, args: argparse.Namespace
) -> list[dict[str, Any]]:
    import torch

    statuses = []
    for row in rows:
        output_path = args.output_dir / row["image_path"]
        status = {
            "id": row["id"],
            "pair_id": row["pair_id"],
            "condition": row["condition"],
            "image_path": str(output_path),
            "status": None,
            "failure_reason": None,
            "nsfw_content_detected": None,
            "safety_checker_disabled": bool(args.disable_safety_checker),
            "safety_risk_acknowledged": bool(args.acknowledge_safety_risk),
        }
        try:
            generator = torch.Generator(device=args.device).manual_seed(row["seed"])
            result = pipe(
                row["prompt"],
                num_inference_steps=row["num_inference_steps"],
                guidance_scale=row["guidance_scale"],
                generator=generator,
            )
            image = result.images[0]
            if not isinstance(image, Image.Image):
                raise TypeError(f"pipeline returned {type(image).__name__}, not PIL.Image")
            if image.size != (512, 512) or image.mode != "RGB":
                raise ValueError(f"unexpected image mode/size: {image.mode} {image.size}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            detected = getattr(result, "nsfw_content_detected", None)
            status["nsfw_content_detected"] = (
                bool(detected[0])
                if isinstance(detected, (list, tuple)) and detected
                else None
            )
            status["status"] = "ok"
        except Exception as error:
            status.update(
                status="failure",
                failure_reason=f"generation_error:{type(error).__name__}:{error}",
            )
        statuses.append(status)
    return statuses


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--status-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--pretrained-model-name-or-path", default=MODEL_ID)
    parser.add_argument("--disable-safety-checker", action="store_true")
    parser.add_argument("--acknowledge-safety-risk", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.disable_safety_checker and not args.acknowledge_safety_risk:
        parser.error("disabling the safety checker requires --acknowledge-safety-risk")
    if not args.disable_safety_checker and args.acknowledge_safety_risk:
        parser.error("--acknowledge-safety-risk requires --disable-safety-checker")
    if not args.dry_run and args.model_dir is None:
        parser.error("--model-dir is required unless --dry-run is used")
    if args.manifest_path is None:
        args.manifest_path = args.output_dir / "subject_diagnostic_manifest.jsonl"
    if args.status_path is None:
        args.status_path = args.output_dir / "subject_diagnostic_status.jsonl"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = build_manifest()
    write_jsonl(rows, args.manifest_path)
    if args.dry_run:
        return 0
    pipe = load_pipeline(args)
    statuses = generate(rows, pipe, args)
    write_jsonl(statuses, args.status_path)
    return 0 if all(row["status"] == "ok" for row in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
