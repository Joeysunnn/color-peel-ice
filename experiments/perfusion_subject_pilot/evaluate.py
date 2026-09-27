"""Generate a matched B0/P1 mailbox Subject and Subject+Material comparison."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.methods.colorpeel_ice.generate_color_material_diagnostic import read_json, sha256
from scripts.methods.colorpeel_ice.generate_subject_material_diagnostic import (
    attention_masks, verify_sources as verify_baseline,
)
from scripts.methods.colorpeel_ice.generate_mailbox_matte_subject_comparison import (
    load_pipeline as load_baseline_pipeline, manifest_rows as aligned_manifest_rows,
    verify_sources as verify_aligned,
)


def verify_sources(protocol_path: Path, run_root: Path, subject_run: Path):
    protocol = read_json(protocol_path)
    baseline_path = REPO_ROOT / protocol["baseline_protocol"]
    if (protocol.get("schema") != "perfusion_subject_pilot/v1"
            or protocol.get("base_model") != "CompVis/stable-diffusion-v1-4"
            or protocol.get("safety_checker") != "enabled"
            or sha256(baseline_path) != protocol["baseline_protocol_sha256"]):
        raise ValueError("Perfusion comparison protocol differs")
    cohort = protocol.get("cohort")
    if cohort == "original":
        baseline, baseline_dir, material_dir = verify_baseline(baseline_path, run_root)
        expected_variant = "mailbox_keylocked_rank1_value_5000"
    elif cohort == "balanced_aligned":
        aligned, baseline, subjects, material_dir = verify_aligned(baseline_path, run_root)
        baseline_dir = subjects["balanced_aligned"]
        expected_variant = "mailbox_balanced_aligned_keylocked_rank1_value_5000"
        aligned_prompts = {
            (row["group"], row["condition"]): row["prompt"]
            for row in aligned_manifest_rows(aligned, baseline)
            if row["arm"] == "balanced_aligned" and row["seed"] == 42
            and row["group"] in {"red", "blue"}
            and row["condition"] in {"subject_only", "subject_material"}
        }
        for group in baseline["groups"][:2]:
            for condition in ("subject_only", "subject_material"):
                if aligned_prompts.get((group["id"], condition)) != group["prompts"][condition]:
                    raise ValueError("balanced B0 prompt differs from frozen red/blue diagnostic")
    else:
        raise ValueError("unknown Perfusion comparison cohort")
    if protocol["sampling"] != baseline["sampling"]:
        raise ValueError("sampling differs from frozen B0")
    subject_run = subject_run.resolve()
    if not subject_run.is_relative_to(run_root.resolve()) or subject_run.parent != run_root.resolve() / "perfusion_subject_pilot":
        raise ValueError("P1 run must be under the independent Perfusion study")
    manifest = read_json(subject_run / "manifest.json")
    if (manifest.get("status") != "succeeded" or manifest.get("returncode") != 0
            or manifest.get("stage") != "train"
            or manifest.get("run", {}).get("variant") != expected_variant):
        raise ValueError("P1 training did not complete under the selected protocol")
    checkpoint = subject_run / "checkpoints"
    adaptation = read_json(checkpoint / "adaptation_config.json")
    if (adaptation.get("adaptation_mode") != "perfusion_subject_rank1"
            or adaptation.get("modifier_tokens") != ["<S*>"]
            or adaptation.get("key_reference_prompt") != "a photo of a mailbox"
            or adaptation.get("value_rank") != 1 or adaptation.get("value_alpha") != 1.0):
        raise ValueError("P1 checkpoint adaptation differs")
    for name in (adaptation["weight_name"], adaptation["key_reference_name"], "<S*>.bin"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    return protocol, baseline, baseline_dir, checkpoint, material_dir


def manifest_rows(protocol: dict, baseline: dict) -> list[dict]:
    red, blue = baseline["groups"][:2]
    groups = [
        ("plain", "a photo of <S*> mailbox", "a photo of <S*> mailbox made of <M*>"),
        ("red", red["prompts"]["subject_only"], red["prompts"]["subject_material"]),
        ("blue", blue["prompts"]["subject_only"], blue["prompts"]["subject_material"]),
    ]
    if [group["id"] for group in baseline["groups"][:2]] != ["red", "blue"]:
        raise ValueError("B0 red/blue groups changed")
    rows = []
    for arm in ("B0", "P1"):
        for group, only, with_material in groups:
            for condition, prompt in (("subject_only", only), ("subject_material", with_material)):
                if prompt.count("<S*>") != 1 or prompt.count("<M*>") != int(condition == "subject_material"):
                    raise ValueError("comparison token positions changed")
                for seed in protocol["sampling"]["seeds"]:
                    item_id = f"{arm}__{group}__{condition}__seed{seed}"
                    rows.append({"id": item_id, "arm": arm, "group": group,
                                 "condition": condition, "prompt": prompt, "seed": seed,
                                 "image_path": f"images/{arm}/{group}/{item_id}.png"})
    if len(rows) != 36:
        raise ValueError("expected 36 matched samples")
    return rows


def load_p1_pipeline(protocol: dict, checkpoint: Path, material_dir: Path, device: str):
    import torch
    from diffusers import DiffusionPipeline

    train_root = str(REPO_ROOT / "src" / "train")
    if train_root not in sys.path:
        sys.path.insert(0, train_root)
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel
    from experiments.perfusion_subject_pilot.perfusion_attention import (
        REFERENCE_NAME, WEIGHT_NAME, install_subject_material,
    )

    unet = UNet2DConditionModel.from_pretrained(
        protocol["base_model"], subfolder="unet", local_files_only=True, torch_dtype=torch.float16)
    subject_state = torch.load(checkpoint / WEIGHT_NAME, map_location="cpu")
    material_state = torch.load(material_dir / "pytorch_token_local_kv_weights.bin", map_location="cpu")
    install_subject_material(unet, subject_state, material_state)
    pipe = DiffusionPipeline.from_pretrained(
        protocol["base_model"], unet=unet, low_cpu_mem_usage=False,
        torch_dtype=torch.float16, local_files_only=True).to(device)
    pipe.load_textual_inversion(str(checkpoint), weight_name="<S*>.bin")
    pipe.load_textual_inversion(str(material_dir), weight_name="<M*>.bin")
    reference = torch.load(checkpoint / REFERENCE_NAME, map_location="cpu").to(
        device=pipe.unet.device, dtype=pipe.unet.dtype)
    return pipe, reference


def generate(protocol: dict, baseline: dict, baseline_dir: Path, checkpoint: Path,
             material_dir: Path, rows: list[dict], output_dir: Path, device: str) -> None:
    import torch

    with (output_dir / "generation_status.jsonl").open("w", encoding="utf-8") as ledger:
        for arm in ("B0", "P1"):
            if arm == "B0":
                pipe = load_baseline_pipeline(baseline, baseline_dir, material_dir, device)
                reference = None
            else:
                pipe, reference = load_p1_pipeline(protocol, checkpoint, material_dir, device)
            for row in rows:
                if row["arm"] != arm:
                    continue
                kwargs = attention_masks(pipe, row["prompt"], protocol["sampling"]["guidance_scale"])
                if reference is not None:
                    kwargs["key_reference"] = reference
                result = pipe(
                    row["prompt"], num_inference_steps=protocol["sampling"]["num_inference_steps"],
                    guidance_scale=protocol["sampling"]["guidance_scale"],
                    generator=torch.Generator(device=device).manual_seed(row["seed"]),
                    cross_attention_kwargs=kwargs,
                )
                image_path = output_dir / row["image_path"]
                image_path.parent.mkdir(parents=True, exist_ok=True)
                result.images[0].save(image_path)
                flags = getattr(result, "nsfw_content_detected", None)
                if flags is not None and len(flags) != 1:
                    raise ValueError("expected one safety-checker result")
                filtered = bool(flags[0]) if flags is not None else False
                ledger.write(json.dumps({**row, "status": "safety_filtered" if filtered else "ok",
                                         "image_sha256": sha256(image_path),
                                         "nsfw_content_detected": filtered}, sort_keys=True) + "\n")
                ledger.flush()
            del pipe
            gc.collect()
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--subject-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_root = Path(os.environ["COLORPEEL_RUN_ROOT"]).resolve()
    protocol, baseline, baseline_dir, checkpoint, material_dir = verify_sources(
        args.protocol, run_root, args.subject_run)
    rows = manifest_rows(protocol, baseline)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "status": "dry_run" if args.dry_run else "running",
        "cohort": protocol["cohort"],
        "protocol_sha256": sha256(args.protocol),
        "baseline_checkpoint": str(baseline_dir),
        "perfusion_checkpoint": str(checkpoint),
        "perfusion_manifest_sha256": sha256(args.subject_run / "manifest.json"),
        "perfusion_weights_sha256": sha256(checkpoint / "pytorch_perfusion_subject_weights.bin"),
        "perfusion_key_reference_sha256": sha256(checkpoint / "mailbox_key_reference.pt"),
        "material_checkpoint": str(material_dir),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "generation_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if not args.dry_run:
        generate(protocol, baseline, baseline_dir, checkpoint, material_dir, rows, args.output_dir, args.device)
        provenance["status"] = "succeeded"
        (args.output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(rows)} matched samples: {args.output_dir}")


if __name__ == "__main__":
    main()
