"""Probe whether full Perfusion Material transfers metal to ordinary objects."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from experiments.perfusion_full.compare import selected_checkpoint
from experiments.perfusion_full.inference import generate, load_pipeline
from experiments.perfusion_full.train import sha256


OBJECTS = ("cube", "sphere", "mug")
SEEDS = (42, 43, 44)


def probe(validation: Path, covariance_root: Path, output_dir: Path, device: str) -> None:
    from diffusers import DiffusionPipeline

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    checkpoint = selected_checkpoint(validation)
    output_dir.mkdir(parents=True)
    rows = []
    for obj in OBJECTS:
        for condition, prompt in (
                ("plain", f"a photo of a {obj}"),
                ("literal_metal", f"a photo of a {obj} made of metal"),
                ("perfusion_material", f"a photo of a {obj} made of <M*>")):
            for seed in SEEDS:
                rows.append({"object": obj, "condition": condition,
                             "prompt": prompt, "seed": seed,
                             "id": f"{obj}__{condition}__seed{seed}"})
    (output_dir / "generation_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    provenance = {"schema": "perfusion_full_material_probe/v1", "status": "running",
                  "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                  "steps": 100, "cfg": 3.5, "seeds": list(SEEDS),
                  "sampler": "base SD pipeline scheduler", "beta": .75, "tau": .15}
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with (output_dir / "generation_status.jsonl").open("w") as ledger:
        for adapter in (False, True):
            if adapter:
                pipe, kwargs = load_pipeline([checkpoint], covariance_root, device)
            else:
                pipe = DiffusionPipeline.from_pretrained(
                    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16,
                    local_files_only=True).to(device)
                kwargs = {}
            for row in rows:
                if (row["condition"] == "perfusion_material") != adapter:
                    continue
                image, filtered = generate(pipe, kwargs, row["prompt"],
                                           row["seed"], 100, 3.5, device)
                path = output_dir / "images" / f"{row['id']}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                ledger.write(json.dumps({**row, "image": str(path),
                                         "image_sha256": sha256(path),
                                         "safety_filtered": filtered}) + "\n")
                ledger.flush()
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
    provenance["status"] = "succeeded"
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material-validation", type=Path, required=True)
    parser.add_argument("--covariance-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    probe(args.material_validation, args.covariance_root, args.output_dir, args.device)
