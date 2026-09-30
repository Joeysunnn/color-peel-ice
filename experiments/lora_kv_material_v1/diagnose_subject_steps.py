"""Compare saved Subject LoRA steps on the fixed Subject-only prompts."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.lora_kv_material_v1.evaluate import (
    checked_file, load_pipeline, matched_rows, material_checkpoint, read_json,
    sha256, token_masks, verify_subject_checkpoint,
)


STEPS = (1000, 2000, 3000, 4000, 5000)


def make_sheets(protocol: dict, rows: list[dict], output: Path) -> dict[str, str]:
    from PIL import Image, ImageDraw

    ledger = {row["id"]: row for row in
              (json.loads(line) for line in (output / "generation_status.jsonl").read_text().splitlines())}
    if len(ledger) != len(rows):
        raise ValueError("step sweep is incomplete")
    sheets = {}
    sheet_dir = output / "contact_sheets"
    sheet_dir.mkdir()
    tile, left, top, gap = 256, 100, 34, 8
    for group in protocol["groups"]:
        sheet = Image.new("RGB", (left + len(STEPS) * (tile + gap),
                                  top + len(protocol["sampling"]["seeds"]) * (tile + gap)), "white")
        draw = ImageDraw.Draw(sheet)
        for col, step in enumerate(STEPS):
            draw.text((left + col * (tile + gap), 8), f"step {step}", fill="black")
        for row_index, seed in enumerate(protocol["sampling"]["seeds"]):
            y = top + row_index * (tile + gap)
            draw.text((8, y + 8), f"{group}\nseed {seed}", fill="black")
            for col, step in enumerate(STEPS):
                record = ledger[f"step{step}__{group}__seed{seed}"]
                path = output / record["image_path"]
                if sha256(path) != record["image_sha256"]:
                    raise ValueError(f"generated image hash changed: {path}")
                with Image.open(path) as image:
                    sheet.paste(image.convert("RGB").resize((tile, tile)),
                                (left + col * (tile + gap), y))
                if record["status"] == "safety_filtered":
                    draw.text((left + col * (tile + gap) + 4, y + 4), "SAFETY FILTERED", fill="red")
        path = sheet_dir / f"{group}.jpg"
        sheet.save(path, quality=90)
        sheets[group] = sha256(path)
    return sheets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--subject-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    if (protocol.get("schema") != "lora_kv_material_comparison/v1"
            or protocol.get("groups") != ["plain", "red", "blue"]
            or protocol.get("sampling") != {"seeds": [42, 43, 44],
                                             "num_inference_steps": 100, "guidance_scale": 3.5}):
        raise ValueError("unexpected fixed S/M comparison protocol")
    subject_protocol = checked_file(REPO_ROOT / protocol["subject_comparison"],
                                    protocol["subject_comparison_sha256"])
    run_root = Path(os.environ["COLORPEEL_RUN_ROOT"]).resolve()
    subject = args.subject_checkpoint.resolve()
    subject_hashes = verify_subject_checkpoint(subject, "token_local_kv", subject_protocol)
    old_material = material_checkpoint(subject_protocol, run_root)
    source_rows = [row for row in matched_rows(protocol, subject_protocol)
                   if row["condition"] == "subject_only"]
    if len(source_rows) != 9 or args.output_dir.exists():
        raise ValueError("expected nine fixed Subject prompts and a fresh output directory")
    for step in STEPS:
        snapshot = subject / f"checkpoint-{step}"
        if not (snapshot / "pytorch_model.bin").is_file() or not (snapshot / "pytorch_model_1.bin").is_file():
            raise FileNotFoundError(f"incomplete Subject step {step}: {snapshot}")
    args.output_dir.mkdir(parents=True)
    rows = [{"id": f"step{step}__{row['group']}__seed{row['seed']}",
             "step": step, "group": row["group"], "seed": row["seed"],
             "prompt": row["prompt"],
             "image_path": f"images/step{step}/{row['group']}/seed{row['seed']}.png"}
            for step in STEPS for row in source_rows]
    provenance = {"status": "dry_run" if args.dry_run else "running",
                  "protocol_sha256": sha256(args.protocol),
                  "subject_checkpoint": str(subject),
                  "subject_checkpoint_sha256": subject_hashes,
                  "old_material_checkpoint": str(old_material),
                  "source_snapshots": {}}
    provenance_path = args.output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "generation_manifest.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if args.dry_run:
        print(f"{len(rows)} Subject-only step controls: {args.output_dir}")
        return

    import torch

    token_id = read_json(subject / "embedding_update_audit.json")["modifier_tokens"][0]["token_id"]
    final_token = torch.load(subject / "<S*>.bin", map_location="cpu")["<S*>"]
    final_adapter = torch.load(subject / "pytorch_lora_kv_weights.bin", map_location="cpu")
    with (args.output_dir / "generation_status.jsonl").open("w") as ledger:
        for step in STEPS:
            snapshot = subject / f"checkpoint-{step}"
            adapter_path = snapshot / "pytorch_model.bin"
            encoder_path = snapshot / "pytorch_model_1.bin"
            adapter = torch.load(adapter_path, map_location="cpu")
            encoder = torch.load(encoder_path, map_location="cpu")
            token = encoder["text_model.embeddings.token_embedding.weight"][token_id].clone().contiguous()
            if (set(adapter) != set(final_adapter)
                    or any(adapter[key].shape != final_adapter[key].shape for key in adapter)
                    or (step == 5000 and (not torch.equal(token, final_token)
                                          or any(not torch.equal(adapter[key], final_adapter[key])
                                                 for key in adapter)))):
                raise ValueError(f"Subject snapshot {step} differs from the final adapter format")
            checkpoint = args.output_dir / "derived_checkpoints" / f"step{step}"
            checkpoint.mkdir(parents=True)
            torch.save(adapter, checkpoint / "pytorch_lora_kv_weights.bin")
            torch.save({"<S*>": token}, checkpoint / "<S*>.bin")
            shutil.copyfile(subject / "adaptation_config.json", checkpoint / "adaptation_config.json")
            provenance["source_snapshots"][str(step)] = {
                "adapter_sha256": sha256(adapter_path), "text_encoder_sha256": sha256(encoder_path),
                "derived_adapter_sha256": sha256(checkpoint / "pytorch_lora_kv_weights.bin"),
                "derived_token_sha256": sha256(checkpoint / "<S*>.bin"),
            }
            del adapter, encoder, token
            pipe = load_pipeline(protocol, checkpoint, old_material, "old", args.device)
            for row in rows:
                if row["step"] != step:
                    continue
                result = pipe(row["prompt"],
                              num_inference_steps=protocol["sampling"]["num_inference_steps"],
                              guidance_scale=protocol["sampling"]["guidance_scale"],
                              generator=torch.Generator(device=args.device).manual_seed(row["seed"]),
                              cross_attention_kwargs=token_masks(
                                  pipe, row["prompt"], protocol["sampling"]["guidance_scale"]))
                path = args.output_dir / row["image_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                result.images[0].save(path)
                flags = getattr(result, "nsfw_content_detected", None)
                if flags is not None and len(flags) != 1:
                    raise ValueError("expected one safety-checker result")
                filtered = bool(flags[0]) if flags is not None else False
                ledger.write(json.dumps({**row, "status": "safety_filtered" if filtered else "ok",
                                         "image_sha256": sha256(path),
                                         "nsfw_content_detected": filtered}, sort_keys=True) + "\n")
                ledger.flush()
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
    provenance["contact_sheet_sha256"] = make_sheets(protocol, rows, args.output_dir)
    provenance["status"] = "succeeded"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} Subject-only step controls: {args.output_dir}")


if __name__ == "__main__":
    main()
