"""Run the released Material Palette stages in an isolated experiment folder."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def record(root, stage, sample_id, inputs, outputs, args):
    path = root / f"{stage}_manifest.json"
    if path.exists():
        raise FileExistsError(path)
    data = {"stage": stage, "sample_id": sample_id,
            "project_git_commit": git_commit(args.project_root),
            "official_git_commit": git_commit(args.official_root),
            "config_sha256": sha(args.config), "script_sha256": sha(__file__),
            "inputs": {str(p): sha(p) for p in inputs},
            "outputs": {str(p): sha(p) for p in outputs}}
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"stage": stage, "sample": sample_id,
                      "outputs": list(data["outputs"])}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("invert", "generate", "decompose"), required=True)
    parser.add_argument("--sample-id", choices=("mailbox", "spoon_positive_control"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--sd15-root", type=Path, required=True)
    parser.add_argument("--decomposer", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    sample = args.run_root / "samples" / args.sample_id
    if not (args.run_root / "input_manifest.json").is_file() or not sample.is_dir():
        raise FileNotFoundError("Run prepare_and_crop.py first")
    if len(list((sample / "crops/material").glob("*.png"))) < 2:
        raise ValueError("Material concept requires at least two crops")
    if not args.sd15_root.joinpath("download_manifest.json").is_file():
        raise FileNotFoundError("Pinned SD 1.5 mirror snapshot is missing")
    if not args.decomposer.is_file():
        raise FileNotFoundError(args.decomposer)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("Select a GPU with CUDA_VISIBLE_DEVICES")
    sys.path.insert(0, str(args.official_root.resolve()))
    os.chdir(args.official_root)
    import concept
    import capture

    input_manifest = args.run_root / "input_manifest.json"
    sd_manifest = args.sd15_root / "download_manifest.json"
    lora_dir = sample / "weights/material/an_object_with_azertyuiop_texture" / f"checkpoint-{config['max_train_steps']}"
    if args.stage == "invert":
        if (sample / "invert_manifest.json").exists():
            raise FileExistsError("Inversion already recorded")
        checkpoint = concept.invert(sample / "crops/material",
                                    pretrained_model_name_or_path=str(args.sd15_root.resolve()),
                                    seed=config["seed"], max_train_steps=config["max_train_steps"])
        if Path(checkpoint).resolve() != lora_dir.resolve():
            raise ValueError(f"Unexpected LoRA checkpoint: {checkpoint}")
        weights = sorted(p for p in lora_dir.rglob("*") if p.is_file())
        if not weights:
            raise RuntimeError("No LoRA weights saved")
        record(sample, "invert", args.sample_id,
               [input_manifest, sd_manifest, *sorted((sample / "crops/material").glob("*.png"))],
               weights, args)
    elif args.stage == "generate":
        if not (sample / "invert_manifest.json").is_file():
            raise FileNotFoundError("Inversion manifest missing")
        if (sample / "generate_manifest.json").exists():
            raise FileExistsError("Generation already recorded")
        texture = Path(concept.infer(lora_dir, renorm=True,
                                    resolution=config["generation_resolution"],
                                    num_inference_steps=config["generation_steps"],
                                    seed=config["seed"], prompt=config["generation_prompt"]))
        if not texture.is_file():
            raise FileNotFoundError(texture)
        record(sample, "generate", args.sample_id,
               [sample / "invert_manifest.json", sd_manifest], [texture], args)
    else:
        if not (sample / "generate_manifest.json").is_file():
            raise FileNotFoundError("Generation manifest missing")
        if (sample / "decompose_manifest.json").exists():
            raise FileExistsError("Decomposition already recorded")
        from pytorch_lightning import Trainer
        data = capture.get_data(predict_dir=sample, predict_ds="sd")
        module = capture.get_inference_module(pt=args.decomposer)
        trainer = Trainer(default_root_dir=sample, accelerator="gpu", devices=1,
                          precision=16, logger=False, enable_checkpointing=False)
        trainer.predict(module, data)
        outputs = sorted(p for p in lora_dir.rglob("*.png")
                         if p.stem.endswith(("_albedo", "_normals", "_roughness")))
        if len(outputs) != 3:
            raise RuntimeError(f"Expected one A/N/R triplet, found {len(outputs)}: {outputs}")
        record(sample, "decompose", args.sample_id,
               [sample / "generate_manifest.json", args.decomposer], outputs, args)


if __name__ == "__main__":
    main()
