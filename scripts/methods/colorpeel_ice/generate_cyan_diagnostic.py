"""Generate paired cyan controls without retraining ColorPeel.

The locked design crosses ten nouns, seeds 42--44, two prompt families, and
literal cyan/aqua/teal/turquoise candidates. Literal candidates run on both
vanilla SD1.4 and trained Custom Diffusion K/V; learned ``<c2*>`` is valid only
with the trained K/V and learned embeddings. This yields 540 images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


MODEL_ID = "CompVis/stable-diffusion-v1-4"
CUSTOM_DIFFUSION_WEIGHTS = "pytorch_custom_diffusion_weights.bin"
TOKENS = ("<s1*>", "<s2*>", "<s3*>", "<c1*>", "<c2*>", "<c3*>")
CURRENT_CYAN_TOKEN = "<c2*>"
SEEDS = (42, 43, 44)
STEPS = 100
GUIDANCE_SCALE = 6.0
TRANSFER_CASES = (
    ("bowl", "a bowl", "a {color} bowl on the table"),
    ("bowling_ball", "a bowling ball", "a {color} bowling ball in a bowling alley"),
    ("plate", "a plate", "a {color} plate on the table"),
    ("vase", "a vase", "a {color} vase on the shelf"),
    ("pants", "a pair of pants", "a women wearing {color} pants"),
    ("teddy_bear", "a teddy-bear", "a {color} teddy-bear in Time Square"),
    ("snooker_ball", "a snooker ball", "a {color} snooker ball on the table"),
    ("parrot", "a parrot", "a {color} parrot perched on a tree"),
    ("sofa", "a sofa", "a {color} sofa in living room"),
    ("rose", "a rose", "a {color} rose blooming in a wooden pot"),
)
LITERAL_CANDIDATES = ("cyan", "aqua", "teal", "turquoise")
COLOR_CANDIDATES = (
    ("learned_token", CURRENT_CYAN_TOKEN, ("trained",)),
    *((color, color, ("vanilla", "trained")) for color in LITERAL_CANDIDATES),
)
TEMPLATE_FAMILIES = ("adjective_transfer", "training_suffix")


def build_manifest() -> list[dict[str, Any]]:
    rows = []
    for template_index, (noun, noun_phrase, transfer_template) in enumerate(TRANSFER_CASES):
        for seed in SEEDS:
            pair_id = f"cyan-{template_index:02d}-seed-{seed}"
            for template_family in TEMPLATE_FAMILIES:
                template = (
                    transfer_template
                    if template_family == "adjective_transfer"
                    else f"a photo of {noun_phrase} in {{color}} color"
                )
                for color_candidate, color_text, model_variants in COLOR_CANDIDATES:
                    for model_variant in model_variants:
                        condition = f"{model_variant}_{color_candidate}"
                        item_id = f"{pair_id}-{template_family}-{condition}"
                        rows.append(
                            {
                                "id": item_id,
                                "pair_id": pair_id,
                                "comparison_id": (
                                    f"{pair_id}-{template_family}-{color_candidate}"
                                ),
                                "category": "cyan_diagnostic",
                                "noun": noun,
                                "noun_phrase": noun_phrase,
                                "template_index": template_index,
                                "template_family": template_family,
                                "template": template,
                                "seed": seed,
                                "condition": condition,
                                "model_variant": model_variant,
                                "uses_trained_kv": model_variant == "trained",
                                "color_candidate": color_candidate,
                                "color_expression": color_text,
                                "prompt": template.format(color=color_text),
                                "num_inference_steps": STEPS,
                                "guidance_scale": GUIDANCE_SCALE,
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


def load_pipeline(args: argparse.Namespace, trained: bool) -> Any:
    import torch
    from diffusers import DiffusionPipeline

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        low_cpu_mem_usage=False,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(args.device)
    if args.disable_safety_checker:
        pipe.safety_checker = None
        if hasattr(pipe, "requires_safety_checker"):
            pipe.requires_safety_checker = False
    if trained:
        model_dir = args.model_dir.resolve()
        validate_model_dir(model_dir)
        pipe.unet.load_attn_procs(
            str(model_dir), weight_name=CUSTOM_DIFFUSION_WEIGHTS
        )
        for token in TOKENS:
            pipe.load_textual_inversion(str(model_dir), weight_name=f"{token}.bin")
    return pipe


def generate_condition(
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
                bool(detected[0]) if isinstance(detected, (list, tuple)) and detected else None
            )
            status["status"] = "ok"
        except Exception as error:
            status.update(
                status="failure",
                failure_reason=f"generation_error:{type(error).__name__}:{error}",
            )
        statuses.append(status)
    return statuses


def generate(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch

    statuses = []
    for trained in (False, True):
        selected = [row for row in rows if row["uses_trained_kv"] is trained]
        pipe = load_pipeline(args, trained=trained)
        statuses.extend(generate_condition(selected, pipe, args))
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    status_by_id = {row["id"]: row for row in statuses}
    return [status_by_id[row["id"]] for row in rows]


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
        args.manifest_path = args.output_dir / "cyan_diagnostic_manifest.jsonl"
    if args.status_path is None:
        args.status_path = args.output_dir / "cyan_diagnostic_status.jsonl"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = build_manifest()
    write_jsonl(rows, args.manifest_path)
    if args.dry_run:
        return 0
    statuses = generate(rows, args)
    write_jsonl(statuses, args.status_path)
    return 0 if all(row["status"] == "ok" for row in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
