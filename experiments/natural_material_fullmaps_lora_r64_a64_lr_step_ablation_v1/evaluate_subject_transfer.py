"""Replay the fixed 140-row Subject transfer manifest for one rank-64 snapshot."""

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

from experiments.natural_material_fullmaps_lora_r64_a64_lr_step_ablation_v1.evaluate import (
    SUBJECT_STEPS,
    WEIGHT_NAME,
    derive_snapshots,
    sha256,
    verify_subject_run,
)


SOURCE_FIELDS = {
    "checkpoint_id", "checkpoint_steps", "color", "guidance_scale", "id",
    "image_path", "model_dir", "num_inference_steps", "prompt", "sampling_id", "seed",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_protocol(protocol: dict) -> None:
    expected = {
        "schema": "lora_kv_subject_r64_a64_manifest_transfer/v1",
        "base_model": "CompVis/stable-diffusion-v1-4",
        "subject": {
            "run_relative_to_COLORPEEL_RUN_ROOT": (
                "lora_kv_subject_r64_a64_lr_step_ablation_v1/"
                "20261008-115005__lora_kv_subject_r64_a64_lr_step_ablation_v1__"
                "balanced_aligned_token_local_kv_r64_a64_kvlr5em5_3000__4bf787f__42"
            ),
            "training_study": "lora_kv_subject_r64_a64_lr_step_ablation_v1",
            "variant": "balanced_aligned_token_local_kv_r64_a64_kvlr5em5_3000",
            "mode": "token_local_kv", "rank": 64, "alpha": 64.0,
            "kv_learning_rate": 5.0e-5, "snapshot_steps": list(SUBJECT_STEPS),
        },
        "prompt_manifest": {
            "path_relative_to_COLORPEEL_RUN_ROOT": (
                "20260921-170000__natural_image_subject_color_pilot__"
                "subject_token_local_kv_5000_reconstruction_transfer__7eb2319__42/"
                "inference/generation_manifest.jsonl"
            ),
            "sha256": "0e22585e08d44c12499a59efb35a9124164a01ca1167637257bfbf42e1e21353",
            "row_count": 140, "prompt_count": 28,
            "source_checkpoint_id": "token-local-kv-5000",
            "source_checkpoint_steps": 5000,
        },
        "sampling": {"seeds": [42, 43, 44, 45, 46],
                     "num_inference_steps": 100, "guidance_scale": 3.5},
        "safety_checker": "enabled",
    }
    if protocol != expected:
        raise ValueError("Subject transfer protocol differs from the locked experiment")


def validate_source_rows(rows: list[dict], protocol: dict) -> None:
    source = protocol["prompt_manifest"]
    sampling = protocol["sampling"]
    if len(rows) != source["row_count"] or any(set(row) != SOURCE_FIELDS for row in rows):
        raise ValueError("source prompt manifest schema or row count differs")
    if (len({row["id"] for row in rows}) != len(rows)
            or len({row["image_path"] for row in rows}) != len(rows)):
        raise ValueError("source prompt manifest ids or paths are not unique")
    pairs: dict[tuple[str, str], set[int]] = {}
    for row in rows:
        prompt = row["prompt"]
        if (row["checkpoint_id"] != source["source_checkpoint_id"]
                or row["checkpoint_steps"] != source["source_checkpoint_steps"]
                or row["sampling_id"] is not None
                or row["seed"] not in sampling["seeds"]
                or row["num_inference_steps"] != sampling["num_inference_steps"]
                or row["guidance_scale"] != sampling["guidance_scale"]
                or prompt.count("<S*>") != 1
                or "<M*>" in prompt or "<C*>" in prompt):
            raise ValueError(f"source transfer row differs: {row.get('id')}")
        key = (row["color"], prompt)
        pairs.setdefault(key, set())
        if row["seed"] in pairs[key]:
            raise ValueError(f"duplicate source prompt seed: {row['id']}")
        pairs[key].add(row["seed"])
    if (len(pairs) != source["prompt_count"]
            or any(seeds != set(sampling["seeds"]) for seeds in pairs.values())
            or len({color for color, _ in pairs}) != source["prompt_count"]):
        raise ValueError("source prompt/seed grid differs")


def load_source_rows(protocol: dict, run_root: Path) -> tuple[Path, list[dict], dict]:
    source = protocol["prompt_manifest"]
    path = (run_root / source["path_relative_to_COLORPEEL_RUN_ROOT"]).resolve()
    if sha256(path) != source["sha256"]:
        raise ValueError("source generation manifest hash differs")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    validate_source_rows(rows, protocol)
    source_run = path.parent.parent
    manifest_path = source_run / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "succeeded" or manifest.get("returncode") != 0:
        raise ValueError("source prompt run did not succeed")
    provenance_path = path.parent / "provenance.json"
    return path, rows, {
        "generation_manifest_sha256": sha256(path),
        "run_manifest_sha256": sha256(manifest_path),
        "provenance_sha256": sha256(provenance_path),
    }


def replay_rows(source_rows: list[dict], subject_step: int) -> list[dict]:
    if subject_step not in SUBJECT_STEPS:
        raise ValueError(f"unsupported Subject snapshot: {subject_step}")
    rows = []
    for source in source_rows:
        sample_id = f"s{subject_step}__{source['color']}__seed{source['seed']}"
        rows.append({
            "id": sample_id, "subject_step": subject_step,
            "source_id": source["id"], "color": source["color"],
            "prompt": source["prompt"], "seed": source["seed"],
            "num_inference_steps": source["num_inference_steps"],
            "guidance_scale": source["guidance_scale"],
            "image_path": f"images/{source['color']}/seed{source['seed']}.png",
        })
    if len(rows) != 140 or len({row["id"] for row in rows}) != 140:
        raise ValueError("Subject transfer replay matrix differs")
    return rows


def subject_token_mask(pipe, prompt: str, guidance_scale: float) -> dict:
    import torch

    ids = pipe.tokenizer(
        prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(pipe.unet.device)
    mask = ids == pipe.tokenizer.convert_tokens_to_ids("<S*>")
    if int(mask.sum()) != prompt.count("<S*>"):
        raise ValueError(f"Subject tokenization changed: {prompt}")
    if guidance_scale > 1:
        mask = torch.cat([torch.zeros_like(mask), mask], dim=0)
    return {"modifier_token_mask": mask}


def load_pipeline(base_model: str, checkpoint: Path, device: str):
    import torch
    from diffusers import DiffusionPipeline, PNDMScheduler
    from experiments.lora_kv_subject_v1.attention import install_subject_lora_kv

    train_root = str(REPO_ROOT / "src" / "train")
    if train_root not in sys.path:
        sys.path.insert(0, train_root)
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(
        base_model, subfolder="unet", local_files_only=True, torch_dtype=torch.float16)
    state = torch.load(checkpoint / WEIGHT_NAME, map_location="cpu")
    install_subject_lora_kv(unet, state, "token_local_kv", 64, 64)
    pipe = DiffusionPipeline.from_pretrained(
        base_model, unet=unet, low_cpu_mem_usage=False,
        torch_dtype=torch.float16, local_files_only=True).to(device)
    if pipe.safety_checker is None or not isinstance(pipe.scheduler, PNDMScheduler):
        raise ValueError("Subject transfer requires the base safety checker and PNDM scheduler")
    pipe.load_textual_inversion(str(checkpoint), weight_name="<S*>.bin")
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--subject-step", type=int, choices=SUBJECT_STEPS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol = read_json(args.protocol)
    validate_protocol(protocol)
    run_root = Path(os.environ["COLORPEEL_RUN_ROOT"]).resolve()
    source_path, source_rows, source_hashes = load_source_rows(protocol, run_root)
    subject_run, subject_final, subject_hashes = verify_subject_run(protocol, run_root)
    source_snapshot_audit = derive_snapshots(subject_final, "<S*>", SUBJECT_STEPS, None)
    rows = replay_rows(source_rows, args.subject_step)
    if args.dry_run:
        print(f"verified Subject step {args.subject_step}: {len(rows)} manifest-replay rows")
        return

    output = args.output.resolve()
    source_run = source_path.parent.parent
    for protected in (subject_run, source_run):
        if output == protected or output.is_relative_to(protected):
            raise ValueError(f"output must not be inside source run: {protected}")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    derived = derive_snapshots(
        subject_final, "<S*>", SUBJECT_STEPS, output / "derived_subject")
    with (output / "generation_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    provenance = {
        "schema": protocol["schema"], "status": "running",
        "protocol_sha256": sha256(args.protocol),
        "source_prompt_manifest": str(source_path),
        "source_prompt_sha256": source_hashes,
        "subject_run": str(subject_run), "subject_checkpoint_sha256": subject_hashes,
        "subject_snapshots": derived, "subject_step": args.subject_step,
        "row_count": len(rows), "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    import torch

    checkpoint = Path(derived[str(args.subject_step)]["path"])
    pipe = load_pipeline(protocol["base_model"], checkpoint, args.device)
    with (output / "generation_status.jsonl").open("w", encoding="utf-8") as ledger:
        for row in rows:
            result = pipe(
                row["prompt"], num_inference_steps=row["num_inference_steps"],
                guidance_scale=row["guidance_scale"],
                generator=torch.Generator(device=args.device).manual_seed(row["seed"]),
                cross_attention_kwargs=subject_token_mask(
                    pipe, row["prompt"], row["guidance_scale"]),
            )
            target = output / row["image_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            result.images[0].save(target)
            flags = getattr(result, "nsfw_content_detected", None)
            if flags is not None and len(flags) != 1:
                raise ValueError("expected one safety-checker result")
            filtered = bool(flags[0]) if flags is not None else False
            ledger.write(json.dumps({
                **row, "status": "safety_filtered" if filtered else "ok",
                "image_sha256": sha256(target), "nsfw_content_detected": filtered,
            }, sort_keys=True) + "\n")
            ledger.flush()
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    if (load_source_rows(protocol, run_root)[2] != source_hashes
            or verify_subject_run(protocol, run_root)[2] != subject_hashes
            or derive_snapshots(subject_final, "<S*>", SUBJECT_STEPS, None)
            != source_snapshot_audit):
        raise ValueError("source manifest or Subject checkpoints changed during inference")
    provenance["status"] = "succeeded"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"generated {len(rows)} Subject transfer images: {output}")


if __name__ == "__main__":
    main()
