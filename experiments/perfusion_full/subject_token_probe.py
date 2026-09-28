"""Probe full Perfusion Subject identity without an explicit ordinary mailbox word."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from experiments.perfusion_full.compare import selected_checkpoint
from experiments.perfusion_full.inference import generate, load_pipeline
from experiments.perfusion_full.train import sha256


PROMPTS = {
    "plain": ("a photo of <S*>", "a photo of <S*> made of <M*>"),
    "red": ("a photo of a red <S*>", "a photo of a red <S*> made of <M*>"),
    "blue": ("a photo of a blue <S*>", "a photo of a blue <S*> made of <M*>"),
}


def probe(subject_validation: Path, material_validation: Path,
          covariance_root: Path, output_dir: Path, device: str) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    subject = selected_checkpoint(subject_validation)
    material = selected_checkpoint(material_validation)
    output_dir.mkdir(parents=True)
    rows = [{"id": f"{group}__{condition}__seed{seed}",
             "group": group, "condition": condition, "prompt": prompt,
             "seed": seed}
            for group, prompts in PROMPTS.items()
            for condition, prompt in zip(("subject_only", "subject_material"), prompts)
            for seed in (42, 43, 44)]
    (output_dir / "generation_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    provenance = {"schema": "perfusion_full_subject_token_probe/v1",
                  "status": "running", "subject_checkpoint": str(subject),
                  "subject_sha256": sha256(subject), "material_checkpoint": str(material),
                  "material_sha256": sha256(material), "seeds": [42, 43, 44],
                  "steps": 100, "cfg": 3.5, "beta": .75, "tau": .15,
                  "sampler": "base SD pipeline scheduler"}
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with (output_dir / "generation_status.jsonl").open("w") as ledger:
        for condition, checkpoints in (("subject_only", [subject]),
                                       ("subject_material", [subject, material])):
            pipe, kwargs = load_pipeline(checkpoints, covariance_root, device)
            for row in rows:
                if row["condition"] != condition:
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
    parser.add_argument("--subject-validation", type=Path, required=True)
    parser.add_argument("--material-validation", type=Path, required=True)
    parser.add_argument("--covariance-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    probe(args.subject_validation, args.material_validation, args.covariance_root,
          args.output_dir, args.device)
