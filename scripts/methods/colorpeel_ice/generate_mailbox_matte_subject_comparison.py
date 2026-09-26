"""Infer the same mailbox prompts with the old, matte-only, and balanced S adapters."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.methods.colorpeel_ice.generate_color_material_diagnostic import (
    read_json, sha256, verify_checkpoint,
)
from scripts.methods.colorpeel_ice.generate_subject_material_diagnostic import (
    WEIGHTS, attention_masks, verify_sources as verify_baseline,
)

CONDITIONS = ("subject_only", "subject_literal_matte", "subject_material", "subject_literal_metal")
ARMS = ("baseline", "matte_only", "balanced")


def verify_sources(protocol_path: Path, run_root: Path):
    protocol = read_json(protocol_path)
    baseline_path = REPO_ROOT / protocol["baseline_protocol"]
    if (protocol.get("schema") != "mailbox_matte_subject_inference/v1"
            or protocol.get("base_model") != "CompVis/stable-diffusion-v1-4"
            or protocol.get("safety_checker") != "enabled"
            or protocol.get("sampling") != {"seeds": [42, 43, 44], "num_inference_steps": 100, "guidance_scale": 3.5}
            or protocol.get("conditions") != list(CONDITIONS)
            or sha256(baseline_path) != protocol["baseline_protocol_sha256"]):
        raise ValueError("mailbox matte inference protocol differs")
    baseline, baseline_dir, material_dir = verify_baseline(baseline_path, run_root)
    if protocol["sampling"] != baseline["sampling"]:
        raise ValueError("mailbox inference sampling differs from baseline")
    subjects = {"baseline": baseline_dir}
    if [item.get("id") for item in protocol.get("new_subject_arms", [])] != list(ARMS[1:]):
        raise ValueError("mailbox inference arms differ")
    for item in protocol["new_subject_arms"]:
        run = (run_root / item["training_run_relative_to_COLORPEEL_RUN_ROOT"]).resolve()
        manifest_path = run / "manifest.json"
        if sha256(manifest_path) != item["training_manifest_sha256"]:
            raise ValueError(f"{item['id']} training manifest hash differs")
        manifest = read_json(manifest_path)
        if (manifest.get("status") != "succeeded" or manifest.get("returncode") != 0
                or manifest.get("stage") != "train"
                or manifest.get("run", {}).get("variant") != item["training_variant"]):
            raise ValueError(f"{item['id']} training did not finish as specified")
        checkpoint = run / "checkpoints"
        verify_checkpoint(checkpoint, item["checkpoint_sha256"], "<S*>")
        subjects[item["id"]] = checkpoint
    return protocol, baseline, subjects, material_dir


def manifest_rows(protocol: dict, baseline: dict) -> list[dict]:
    matte_prompts = protocol["subject_literal_matte_prompts"]
    groups = baseline["groups"]
    if ([group["id"] for group in groups] != ["red", "blue", "city_street"]
            or set(matte_prompts) != {group["id"] for group in groups}):
        raise ValueError("mailbox comparison groups differ")
    rows = []
    for arm in ARMS:
        for group in groups:
            group_id = group["id"]
            original = group["prompts"]
            prompts = {
                "subject_only": original["subject_only"],
                "subject_literal_matte": matte_prompts[group_id],
                "subject_material": original["subject_material"],
                "subject_literal_metal": original["literal_metal"],
            }
            for condition in CONDITIONS:
                prompt = prompts[condition]
                if (prompt.count("<S*>") != 1
                        or prompt.count("<M*>") != int(condition == "subject_material")):
                    raise ValueError(f"mailbox token position differs: {condition}")
                for seed in protocol["sampling"]["seeds"]:
                    sample_id = f"{arm}__{group_id}__{condition}__seed{seed}"
                    rows.append({"id": sample_id, "arm": arm, "group": group_id,
                                 "condition": condition, "prompt": prompt, "seed": seed,
                                 "image_path": f"images/{arm}/{group_id}/{sample_id}.png"})
    if len(rows) != 108:
        raise ValueError("expected 108 mailbox comparison samples")
    return rows


def load_pipeline(protocol: dict, subject_dir: Path, material_dir: Path, device: str):
    import torch
    from diffusers import DiffusionPipeline

    train_root = str(REPO_ROOT / "src" / "train")
    if train_root not in sys.path:
        sys.path.insert(0, train_root)
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel
    from src.methods.colorpeel_ice.dual_token_local_kv import install_dual_token_local_kv

    unet = UNet2DConditionModel.from_pretrained(
        protocol["base_model"], subfolder="unet", local_files_only=True, torch_dtype=torch.float16)
    unet.load_attn_procs(str(subject_dir), weight_name=WEIGHTS, adaptation_mode="token_local_kv")
    material_state = torch.load(material_dir / WEIGHTS, map_location="cpu")
    install_dual_token_local_kv(unet, material_state, primary_label="subject")
    pipe = DiffusionPipeline.from_pretrained(
        protocol["base_model"], unet=unet, low_cpu_mem_usage=False,
        torch_dtype=torch.float16, local_files_only=True).to(device)
    pipe.load_textual_inversion(str(subject_dir), weight_name="<S*>.bin")
    pipe.load_textual_inversion(str(material_dir), weight_name="<M*>.bin")
    return pipe


def generate(protocol: dict, subjects: dict, material_dir: Path,
             rows: list[dict], output_dir: Path, device: str):
    import torch

    sampling = protocol["sampling"]
    with (output_dir / "generation_status.jsonl").open("w", encoding="utf-8") as ledger:
        for arm in ARMS:
            pipe = load_pipeline(protocol, subjects[arm], material_dir, device)
            for row in rows:
                if row["arm"] != arm:
                    continue
                result = pipe(
                    row["prompt"], num_inference_steps=sampling["num_inference_steps"],
                    guidance_scale=sampling["guidance_scale"],
                    generator=torch.Generator(device=device).manual_seed(row["seed"]),
                    cross_attention_kwargs=attention_masks(pipe, row["prompt"], sampling["guidance_scale"]),
                )
                path = output_dir / row["image_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                result.images[0].save(path)
                flags = getattr(result, "nsfw_content_detected", None)
                if flags is not None and len(flags) != 1:
                    raise ValueError("expected one safety-checker result")
                filtered = bool(flags[0]) if flags is not None else False
                ledger.write(json.dumps({**row, "status": "safety_filtered" if filtered else "ok",
                                         "image_sha256": sha256(path), "nsfw_content_detected": filtered},
                                        sort_keys=True) + "\n")
                ledger.flush()
            del pipe
            gc.collect()
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_root = Path(os.environ["COLORPEEL_RUN_ROOT"]).resolve()
    protocol, baseline, subjects, material_dir = verify_sources(args.protocol, run_root)
    rows = manifest_rows(protocol, baseline)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "protocol_sha256": sha256(args.protocol),
        "subject_dirs": {key: str(path) for key, path in subjects.items()},
        "material_dir": str(material_dir),
        "status": "dry_run" if args.dry_run else "running",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "generation_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if not args.dry_run:
        generate(protocol, subjects, material_dir, rows, args.output_dir, args.device)
        provenance = read_json(args.output_dir / "provenance.json")
        provenance["status"] = "succeeded"
        (args.output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(rows)} samples: {args.output_dir}")


if __name__ == "__main__":
    main()
