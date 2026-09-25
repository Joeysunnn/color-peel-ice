"""Evaluate one completed shared-K/V C/M run on the fixed 36-image grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


WEIGHTS = "pytorch_custom_diffusion_weights.bin"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(protocol: dict):
    if (protocol.get("schema") != "joint_cm_evaluation/v1"
            or protocol.get("base_model") != "CompVis/stable-diffusion-v1-4"
            or protocol.get("objects") != ["cube", "sphere", "mug"]
            or protocol.get("seeds") != [42, 43, 44]
            or protocol.get("sampling") != {"num_inference_steps": 100, "guidance_scale": 3.5}
            or protocol.get("conditions") != {
                "base": "a photo of a {object}",
                "color_only": "a photo of a {object} with <C*> color",
                "material_only": "a photo of a {object} with <M*> material",
                "color_material": "a photo of a {object} with <C*> color and <M*> material",
            }
            or protocol.get("safety_checker") != "enabled"):
        raise ValueError("joint C/M evaluation protocol differs")
    result = []
    for obj in protocol["objects"]:
        for condition, template in protocol["conditions"].items():
            for seed in protocol["seeds"]:
                sample_id = f"{obj}__{condition}__seed{seed}"
                result.append({"id": sample_id, "object": obj, "condition": condition,
                               "seed": seed, "prompt": template.format(object=obj),
                               "image_path": f"images/{obj}/{sample_id}.png"})
    return result


def verify_training(run: Path, variant: str):
    manifest_path = run / "manifest.json"
    manifest = read_json(manifest_path)
    if (manifest.get("status") != "succeeded" or manifest.get("returncode") != 0
            or manifest.get("stage") != "train"
            or manifest.get("run", {}).get("study") != "color_material_composition_v1"
            or manifest.get("run", {}).get("variant") != variant
            or variant not in {"joint_cm_caa0_1500", "joint_cm_caa02_1500"}):
        raise ValueError("joint C/M training run is not complete")
    model_dir = run / "checkpoints"
    hashes = {name: sha256(model_dir / name) for name in (WEIGHTS, "<C*>.bin", "<M*>.bin")}
    return model_dir, {"training_run": str(run.resolve()), "training_manifest_sha256": sha256(manifest_path),
                       "model_artifact_sha256": hashes, "variant": variant}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--variant", choices=("joint_cm_caa0_1500", "joint_cm_caa02_1500"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    samples = rows(protocol)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    model_dir, provenance = verify_training(args.training_run, args.variant) if not args.dry_run else (None, {})
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "generation_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in samples), encoding="utf-8")
    provenance.update({"protocol_sha256": sha256(args.protocol), "status": "dry_run" if args.dry_run else "running"})
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.dry_run:
        print(f"{len(samples)} planned samples: {args.output_dir}")
        return

    import torch
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(protocol["base_model"], low_cpu_mem_usage=False,
                                             torch_dtype=torch.float16, local_files_only=True).to(args.device)
    pipe.unet.load_attn_procs(str(model_dir), weight_name=WEIGHTS)
    for token in ("<C*>", "<M*>"):
        pipe.load_textual_inversion(str(model_dir), weight_name=f"{token}.bin")
    with (args.output_dir / "generation_status.jsonl").open("w", encoding="utf-8") as ledger:
        for row in samples:
            result = pipe(row["prompt"], num_inference_steps=100, guidance_scale=3.5,
                          generator=torch.Generator(device=args.device).manual_seed(row["seed"]))
            image_path = args.output_dir / row["image_path"]
            image_path.parent.mkdir(parents=True, exist_ok=True)
            result.images[0].save(image_path)
            flags = getattr(result, "nsfw_content_detected", None)
            if flags is not None and len(flags) != 1:
                raise ValueError("expected one safety-checker result")
            filtered = bool(flags[0]) if flags is not None else False
            ledger.write(json.dumps({**row, "status": "safety_filtered" if filtered else "ok",
                                     "image_sha256": sha256(image_path), "nsfw_content_detected": filtered},
                                    sort_keys=True) + "\n")
            ledger.flush()
    provenance["status"] = "succeeded"
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(samples)} generated samples: {args.output_dir}")


if __name__ == "__main__":
    main()
