"""Generate matched mailbox, color, and material diagnostics for full Perfusion."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from experiments.perfusion_full.inference import generate, load_pipeline
from experiments.perfusion_full.train import sha256


def selected_checkpoint(validation: Path) -> Path:
    result = json.loads((validation / "selection.json").read_text())
    if (result.get("schema") != "perfusion_full_validation/v1"
            or not result.get("complete_validation") or result.get("selected_step") is None
            or len(result.get("results", [])) != 16):
        raise ValueError("full 25-step checkpoint validation is required")
    checkpoint = Path(result["train_run"]) / "checkpoints" / f"step_{result['selected_step']:04d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def comparison_rows(protocol_path: Path) -> list[dict]:
    protocol = json.loads(protocol_path.read_text())
    if (protocol["schema"] != "subject_material_diagnostic/v1"
            or protocol["sampling"] != {"seeds": [42, 43, 44],
                                        "num_inference_steps": 100,
                                        "guidance_scale": 3.5}):
        raise ValueError("existing mailbox diagnostic protocol changed")
    groups = [{"id": "plain", "prompts": {
        "subject_only": "a photo of <S*> mailbox",
        "material_only": "a photo of a mailbox made of <M*>",
        "subject_material": "a photo of <S*> mailbox made of <M*>",
    }}] + protocol["groups"][:2]
    if [group["id"] for group in groups] != ["plain", "red", "blue"]:
        raise ValueError("comparison groups differ")
    return [{"id": f"{group['id']}__{condition}__seed{seed}",
             "group": group["id"], "condition": condition,
             "prompt": group["prompts"][condition], "seed": seed}
            for group in groups
            for condition in ("subject_only", "material_only", "subject_material")
            for seed in protocol["sampling"]["seeds"]]


def compare(subject_validation: Path, material_validation: Path,
            covariance_root: Path, protocol_path: Path, output_dir: Path,
            device: str) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    subject = selected_checkpoint(subject_validation)
    material = selected_checkpoint(material_validation)
    rows = comparison_rows(protocol_path)
    output_dir.mkdir(parents=True)
    provenance = {"schema": "perfusion_full_comparison/v1", "status": "running",
                  "subject_checkpoint": str(subject), "subject_sha256": sha256(subject),
                  "material_checkpoint": str(material), "material_sha256": sha256(material),
                  "subject_validation": str(subject_validation),
                  "material_validation": str(material_validation),
                  "baseline_protocol": str(protocol_path),
                  "baseline_protocol_sha256": sha256(protocol_path),
                  "seeds": [42, 43, 44], "steps": 100, "cfg": 3.5,
                  "sampler": "base SD pipeline scheduler", "beta": .75, "tau": .15}
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with (output_dir / "generation_manifest.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with (output_dir / "generation_status.jsonl").open("w") as ledger:
        for condition, checkpoints in (
                ("subject_only", [subject]),
                ("material_only", [material]),
                ("subject_material", [subject, material])):
            pipe, kwargs = load_pipeline(checkpoints, covariance_root, device)
            for row in rows:
                if row["condition"] != condition:
                    continue
                image, filtered = generate(pipe, kwargs, row["prompt"],
                                           row["seed"], 100, 3.5, device)
                path = output_dir / "images" / condition / f"{row['id']}.png"
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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    compare(args.subject_validation, args.material_validation, args.covariance_root,
            args.protocol, args.output_dir, args.device)
