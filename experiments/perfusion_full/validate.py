"""Select a full Perfusion checkpoint from 8-prompt CLIP-I/CLIP-T validation."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from PIL import Image

from experiments.perfusion_full.inference import generate, load_pipeline
from experiments.perfusion_full.train import load_records, read_config, sha256


PROMPTS = {
    "original": [
        "a photo of <S*> mailbox", "a photo of <S*> mailbox in red color",
        "a photo of <S*> mailbox in blue color", "a photo of <S*> mailbox on a city street",
        "a close-up photo of <S*> mailbox", "a photo of <S*> mailbox in a garden",
        "a photo of <S*> mailbox at night", "a photo of <S*> mailbox in sunlight",
    ],
    "balanced_aligned": [
        "a photo of <S*> mailbox", "a photo of <S*> mailbox in red color",
        "a photo of <S*> mailbox in blue color", "a photo of <S*> mailbox on a city street",
        "a close-up photo of <S*> mailbox", "a photo of <S*> mailbox in a garden",
        "a photo of <S*> mailbox at night", "a photo of <S*> mailbox in sunlight",
    ],
    "material": [
        "a photo of an object made of <M*>", "a photo of a cube made of <M*>",
        "a photo of a sphere made of <M*>", "a photo of a cylinder made of <M*>",
        "a photo of a red object made of <M*>", "a photo of a blue object made of <M*>",
        "a photo of a mailbox made of <M*>", "a photo of a mug made of <M*>",
    ],
}


def clip_text(prompt: str) -> str:
    return prompt.replace("<S*> ", "").replace("<M*>", "metal")


def features(model, processor, images, device: str):
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        vectors = model.get_image_features(**inputs).float()
    return torch.nn.functional.normalize(vectors, dim=-1)


def validate(config_path: Path, train_run: Path, output_dir: Path, device: str,
             steps: list[int] | None) -> None:
    from transformers import CLIPModel, CLIPProcessor

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    config = read_config(config_path)
    records, _ = load_records(config)
    all_steps = list(range(25, 401, 25))
    steps = steps or all_steps
    if not steps or any(step not in all_steps for step in steps):
        raise ValueError("only paper 25-step checkpoints can be validated")
    checkpoint_paths = [train_run / "checkpoints" / f"step_{step:04d}.pt" for step in steps]
    if any(not path.is_file() for path in checkpoint_paths):
        raise FileNotFoundError("requested training checkpoints are incomplete")
    output_dir.mkdir(parents=True)
    clip_name = "openai/clip-vit-base-patch32"
    clip = CLIPModel.from_pretrained(clip_name, local_files_only=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(clip_name, local_files_only=True)
    reference_features = []
    for offset in range(0, len(records), 8):
        images = []
        for row in records[offset:offset + 8]:
            with Image.open(row["image"]) as source:
                images.append(source.convert("RGB"))
        reference_features.append(features(clip, processor, images, device))
    reference_features = torch.cat(reference_features)
    text_inputs = processor(text=[clip_text(prompt) for prompt in PROMPTS[config["cohort"]]],
                            padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = torch.nn.functional.normalize(
            clip.get_text_features(**text_inputs).float(), dim=-1)
    ledger = (output_dir / "validation.jsonl").open("w", encoding="utf-8")
    summary = []
    for step, checkpoint in zip(steps, checkpoint_paths):
        pipe, kwargs = load_pipeline([checkpoint], Path(config["covariance_root"]), device, ddim=True)
        kwargs["perfusion_tau"] = .1
        image_scores, text_scores, harmonics = [], [], []
        for index, prompt in enumerate(PROMPTS[config["cohort"]]):
            image, filtered = generate(pipe, kwargs, prompt, 42 + index, 50, 6.0, device)
            path = output_dir / "images" / f"step_{step:04d}" / f"prompt_{index:02d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            if filtered:
                image_score = text_score = harmonic = None
            else:
                vector = features(clip, processor, [image], device)[0]
                image_score = float((reference_features @ vector).mean())
                text_score = float(text_features[index] @ vector)
                harmonic = 2 * image_score * text_score / (image_score + text_score)
                image_scores.append(image_score)
                text_scores.append(text_score)
                harmonics.append(harmonic)
            ledger.write(json.dumps({"step": step, "prompt_index": index, "prompt": prompt,
                                     "seed": 42 + index, "image": str(path),
                                     "image_sha256": sha256(path), "safety_filtered": filtered,
                                     "clip_i": image_score, "clip_t": text_score,
                                     "harmonic": harmonic}) + "\n")
            ledger.flush()
        mean_i = sum(image_scores) / len(image_scores) if harmonics else None
        mean_t = sum(text_scores) / len(text_scores) if harmonics else None
        row = {"step": step, "valid_count": len(harmonics),
               "clip_i": mean_i, "clip_t": mean_t,
               "harmonic": 2 * mean_i * mean_t / (mean_i + mean_t) if harmonics else None}
        summary.append(row)
        print(json.dumps(row), flush=True)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    ledger.close()
    eligible = [row for row in summary if row["valid_count"] == 8]
    selected = max(eligible, key=lambda row: row["harmonic"]) if eligible else None
    result = {"schema": "perfusion_full_validation/v1", "cohort": config["cohort"],
              "train_run": str(train_run), "config_sha256": sha256(config_path),
              "clip_model": clip_name, "prompt_count": 8, "seeds": list(range(42, 50)),
              "sampler": "DDIM", "steps": 50, "cfg": 6.0, "beta": .75, "tau": .1,
              "reference_images": len(records), "results": summary,
              "selected_step": selected["step"] if selected else None,
              "complete_validation": steps == all_steps}
    (output_dir / "selection.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--step", type=int, action="append", help="Smoke one or more checkpoints")
    arguments = parser.parse_args()
    validate(arguments.config, arguments.train_run, arguments.output_dir,
             arguments.device, arguments.step)
