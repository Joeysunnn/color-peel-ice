"""Generate the fixed ColorPeel-on-CLEVR evaluation protocol.

The manifest can be created without importing torch or diffusers by passing
``--dry-run``. This makes the protocol independently auditable on machines
without the training environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


SUBJECTS = (
    ("s1", "<s1*>", "cube"),
    ("s2", "<s2*>", "sphere"),
    ("s3", "<s3*>", "cylinder"),
)
COLORS = (
    ("c1", "<c1*>", "red"),
    ("c2", "<c2*>", "cyan"),
    ("c3", "<c3*>", "gray"),
)

# Appendix C.4, "Evaluation prompt templates", in the ColorPeel paper.
TRANSFER_PROMPTS = (
    "a {color} bowl on the table",
    "a {color} bowling ball in a bowling alley",
    "a {color} plate on the table",
    "a {color} vase on the shelf",
    "a women wearing {color} pants",
    "a {color} teddy-bear in Time Square",
    "a {color} snooker ball on the table",
    "a {color} parrot perched on a tree",
    "a {color} sofa in living room",
    "a {color} rose blooming in a wooden pot",
)

DEFAULT_SEEDS = tuple(range(42, 62))
DEFAULT_STEPS = 100
DEFAULT_GUIDANCE_SCALE = 6.0
CUSTOM_DIFFUSION_WEIGHTS = "pytorch_custom_diffusion_weights.bin"
EXPECTED_IMAGE_SIZE = (512, 512)


def _entry(
    *,
    category: str,
    prompt_index: int,
    seed: int,
    prompt: str,
    subject: tuple[str, str, str] | None = None,
    color: tuple[str, str, str] | None = None,
    transfer_template_index: int | None = None,
) -> dict[str, Any]:
    subject_id, subject_token, subject_label = subject or (None, None, None)
    color_id, color_token, color_label = color or (None, None, None)
    item_id = f"{category}-{prompt_index:03d}-seed-{seed:02d}"
    return {
        "id": item_id,
        "category": category,
        "prompt_index": prompt_index,
        "prompt": prompt,
        "seed": seed,
        "num_inference_steps": DEFAULT_STEPS,
        "guidance_scale": DEFAULT_GUIDANCE_SCALE,
        "subject_id": subject_id,
        "subject_token": subject_token,
        "subject_label": subject_label,
        "color_id": color_id,
        "color_token": color_token,
        "color_label": color_label,
        "transfer_template_index": transfer_template_index,
        "image_path": f"images/{category}/{item_id}.png",
    }


def build_manifest(seeds: Iterable[int] = DEFAULT_SEEDS) -> list[dict[str, Any]]:
    """Build all 900 protocol items in a stable order."""

    seeds = tuple(seeds)
    items: list[dict[str, Any]] = []

    prompt_index = 0
    for subject in SUBJECTS:
        for color in COLORS:
            prompt = f"a photo of {subject[1]} shape in {color[1]} color"
            for seed in seeds:
                items.append(
                    _entry(
                        category="grid",
                        prompt_index=prompt_index,
                        seed=seed,
                        prompt=prompt,
                        subject=subject,
                        color=color,
                    )
                )
            prompt_index += 1

    for prompt_index, subject in enumerate(SUBJECTS):
        prompt = f"a photo of {subject[1]} shape"
        for seed in seeds:
            items.append(
                _entry(
                    category="subject_only",
                    prompt_index=prompt_index,
                    seed=seed,
                    prompt=prompt,
                    subject=subject,
                )
            )

    for prompt_index, color in enumerate(COLORS):
        prompt = f"a photo in {color[1]} color"
        for seed in seeds:
            items.append(
                _entry(
                    category="color_only",
                    prompt_index=prompt_index,
                    seed=seed,
                    prompt=prompt,
                    color=color,
                )
            )

    prompt_index = 0
    for color in COLORS:
        for transfer_template_index, template in enumerate(TRANSFER_PROMPTS):
            prompt = template.format(color=color[1])
            for seed in seeds:
                items.append(
                    _entry(
                        category="transfer",
                        prompt_index=prompt_index,
                        seed=seed,
                        prompt=prompt,
                        color=color,
                        transfer_template_index=transfer_template_index,
                    )
                )
            prompt_index += 1

    return items


def write_manifest(items: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def validate_model_dir(model_dir: Path) -> None:
    required = [model_dir / CUSTOM_DIFFUSION_WEIGHTS]
    required.extend(model_dir / f"{token}.bin" for _, token, _ in (*SUBJECTS, *COLORS))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing trained weights:\n" + "\n".join(missing))


def is_decodable_image(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            if image.size != EXPECTED_IMAGE_SIZE or image.mode != "RGB":
                return False
            image.verify()
        with Image.open(path) as image:
            image.load()
    except (OSError, ValueError):
        return False
    return True


def pending_items(
    items: Iterable[dict[str, Any]], output_dir: Path, skip_existing: bool
) -> list[dict[str, Any]]:
    if not skip_existing:
        return list(items)
    return [
        item
        for item in items
        if not is_decodable_image(output_dir / item["image_path"])
    ]


def generate(items: Iterable[dict[str, Any]], args: argparse.Namespace) -> None:
    items = pending_items(items, args.output_dir, args.skip_existing)
    if not items:
        return
    # Lazy imports are intentional: --dry-run must work without the ML stack.
    import torch
    from diffusers import DiffusionPipeline

    model_dir = args.model_dir.resolve()
    validate_model_dir(model_dir)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        low_cpu_mem_usage=False,
        torch_dtype=dtype,
    ).to(args.device)
    if args.disable_safety_checker:
        pipe.safety_checker = None
        if hasattr(pipe, "requires_safety_checker"):
            pipe.requires_safety_checker = False
    pipe.unet.load_attn_procs(
        str(model_dir), weight_name=CUSTOM_DIFFUSION_WEIGHTS
    )
    for _, token, _ in (*SUBJECTS, *COLORS):
        pipe.load_textual_inversion(str(model_dir), weight_name=f"{token}.bin")

    for item in items:
        image_path = args.output_dir / item["image_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator(device=args.device).manual_seed(item["seed"])
        result = pipe(
            item["prompt"],
            num_inference_steps=item["num_inference_steps"],
            guidance_scale=item["guidance_scale"],
            generator=generator,
        )
        result.images[0].save(image_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Training output containing six token .bin files and Custom Diffusion weights.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        help="Defaults to OUTPUT_DIR/generation_manifest.jsonl.",
    )
    parser.add_argument(
        "--pretrained-model-name-or-path",
        default="CompVis/stable-diffusion-v1-4",
    )
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--disable-safety-checker", action="store_true")
    parser.add_argument("--acknowledge-safety-risk", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip only a decodable, native-RGB 512x512 output image.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the complete manifest without importing or loading ML models.",
    )
    args = parser.parse_args(argv)
    if args.disable_safety_checker and not args.acknowledge_safety_risk:
        parser.error("disabling the safety checker requires --acknowledge-safety-risk")
    if not args.disable_safety_checker and args.acknowledge_safety_risk:
        parser.error("--acknowledge-safety-risk requires --disable-safety-checker")
    if not args.dry_run and args.model_dir is None:
        parser.error("--model-dir is required unless --dry-run is used")
    if args.manifest_path is None:
        args.manifest_path = args.output_dir / "generation_manifest.jsonl"
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    items = [
        {
            **item,
            "safety_checker_disabled": bool(args.disable_safety_checker),
            "safety_risk_acknowledged": bool(args.acknowledge_safety_risk),
        }
        for item in build_manifest()
    ]
    write_manifest(items, args.manifest_path)
    if not args.dry_run:
        generate(items, args)


if __name__ == "__main__":
    main()
