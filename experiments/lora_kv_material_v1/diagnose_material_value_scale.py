"""Sweep only the new Material Value residual at fixed Subject and Material Keys."""

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

from experiments.lora_kv_material_v1.evaluate import (
    checked_file, load_pipeline, matched_rows, material_checkpoint, read_json,
    sha256, token_masks, verify_new_material_checkpoint, verify_subject_checkpoint,
)


SCALES = {"v0": 0.0, "v1": 1.0, "v1p5": 1.5, "v2": 2.0}
COLUMNS = ("subject_only", *SCALES, "new_material_only")


def make_sheets(protocol: dict, source: Path, output: Path, rows: list[dict]) -> dict[str, str]:
    from PIL import Image, ImageDraw

    old = {row["id"]: row for row in
           (json.loads(line) for line in (source / "generation_status.jsonl").read_text().splitlines())}
    new = {row["id"]: row for row in
           (json.loads(line) for line in (output / "generation_status.jsonl").read_text().splitlines())}
    if len(old) != 36 or len(new) != len(rows):
        raise ValueError("source comparison or Value sweep is incomplete")
    sheet_dir = output / "contact_sheets"
    sheet_dir.mkdir()
    tile, left, top, gap = 256, 135, 34, 8
    sheets = {}
    for group in protocol["groups"]:
        sheet = Image.new("RGB", (left + len(COLUMNS) * (tile + gap),
                                  top + len(protocol["sampling"]["seeds"]) * (tile + gap)), "white")
        draw = ImageDraw.Draw(sheet)
        for col, condition in enumerate(COLUMNS):
            draw.text((left + col * (tile + gap), 8), condition, fill="black")
        for row_index, seed in enumerate(protocol["sampling"]["seeds"]):
            y = top + row_index * (tile + gap)
            draw.text((8, y + 8), f"{group}\nseed {seed}", fill="black")
            for col, condition in enumerate(COLUMNS):
                sample_id = f"{condition}__{group}__seed{seed}"
                record = new[sample_id] if condition in SCALES else old[sample_id]
                root = output if condition in SCALES else source
                path = root / record["image_path"]
                if sha256(path) != record["image_sha256"]:
                    raise ValueError(f"source or generated image hash changed: {path}")
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
    parser.add_argument("--new-material-checkpoint", type=Path, required=True)
    parser.add_argument("--source-comparison", type=Path, required=True)
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
    new_material = args.new_material_checkpoint.resolve()
    source = args.source_comparison.resolve()
    old_material = material_checkpoint(subject_protocol, run_root)
    subject_hashes = verify_subject_checkpoint(subject, "token_local_kv", subject_protocol)
    material_hashes = verify_new_material_checkpoint(new_material, protocol, old_material, run_root)
    source_provenance = read_json(source / "provenance.json")
    if (source_provenance.get("status") != "succeeded"
            or source_provenance.get("protocol_sha256") != sha256(args.protocol)
            or source_provenance.get("subject_checkpoint_sha256") != subject_hashes
            or source_provenance.get("new_material_checkpoint_sha256") != material_hashes):
        raise ValueError("source S/M comparison differs from the fixed checkpoints")
    base_rows = [row for row in matched_rows(protocol, subject_protocol)
                 if row["condition"] == "subject_new_material"]
    if len(base_rows) != 9 or args.output_dir.exists():
        raise ValueError("expected nine fixed S+M prompts and a fresh output directory")
    rows = [{"id": f"{label}__{row['group']}__seed{row['seed']}",
             "value_scale": scale, "group": row["group"], "seed": row["seed"],
             "prompt": row["prompt"],
             "image_path": f"images/{label}/{row['group']}/seed{row['seed']}.png"}
            for label, scale in SCALES.items() for row in base_rows]
    args.output_dir.mkdir(parents=True)
    provenance = {"status": "dry_run" if args.dry_run else "running",
                  "protocol_sha256": sha256(args.protocol),
                  "source_comparison": str(source),
                  "source_provenance_sha256": sha256(source / "provenance.json"),
                  "subject_checkpoint_sha256": subject_hashes,
                  "new_material_checkpoint_sha256": material_hashes,
                  "fixed_key_scale": 1.0, "value_scales": SCALES}
    provenance_path = args.output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "generation_manifest.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if args.dry_run:
        print(f"{len(rows)} Value-scale controls: {args.output_dir}")
        return
    import torch
    from experiments.lora_kv_material_v1.attention import DualTokenLocalLoraKVAttnProcessor

    previous = {row["id"]: row for row in
                (json.loads(line) for line in (source / "generation_status.jsonl").read_text().splitlines())}
    with (args.output_dir / "generation_status.jsonl").open("w") as ledger:
        for label, scale in SCALES.items():
            pipe = load_pipeline(protocol, subject, new_material, "new", args.device)
            for processor in pipe.unet.attn_processors.values():
                if not isinstance(processor, DualTokenLocalLoraKVAttnProcessor):
                    raise ValueError("unexpected attention processor in Material Value sweep")
                processor.material_value_scale = scale
            for row in rows:
                if row["value_scale"] != scale:
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
                record = {**row, "status": "safety_filtered" if filtered else "ok",
                          "image_sha256": sha256(path), "nsfw_content_detected": filtered}
                if label == "v1":
                    original = previous[f"subject_new_material__{row['group']}__seed{row['seed']}"]
                    if (record["image_sha256"] != original["image_sha256"]
                            or record["status"] != original["status"]):
                        raise ValueError("unit Material Value scale did not reproduce the source image")
                ledger.write(json.dumps(record, sort_keys=True) + "\n")
                ledger.flush()
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
    provenance["contact_sheet_sha256"] = make_sheets(protocol, source, args.output_dir, rows)
    provenance["status"] = "succeeded"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} Value-scale controls: {args.output_dir}")


if __name__ == "__main__":
    main()
