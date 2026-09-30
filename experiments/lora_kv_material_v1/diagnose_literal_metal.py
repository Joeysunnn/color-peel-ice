"""Add a literal-metal control to the completed matched S/M LoRA comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.lora_kv_material_v1.evaluate import (
    checked_file, load_pipeline, matched_rows, material_checkpoint,
    read_json, sha256, token_masks, verify_subject_checkpoint,
)


CONDITIONS = (
    "subject_only", "subject_old_material", "subject_new_material",
    "new_material_only", "subject_literal_metal",
)


def contact_sheets(protocol: dict, source: Path, output: Path, rows: list[dict]) -> dict[str, str]:
    from PIL import Image, ImageDraw

    old = {item["id"]: item for item in
           (json.loads(line) for line in (source / "generation_status.jsonl").read_text().splitlines())}
    new = {item["id"]: item for item in
           (json.loads(line) for line in (output / "generation_status.jsonl").read_text().splitlines())}
    if len(old) != 36 or len(new) != 9 or set(new) != {row["id"] for row in rows}:
        raise ValueError("matched source or literal-metal generation is incomplete")
    sheet_dir = output / "contact_sheets"
    sheet_dir.mkdir()
    sheets = {}
    tile, left, top, gap = 256, 170, 34, 8
    for group in protocol["groups"]:
        sheet = Image.new("RGB", (left + len(CONDITIONS) * (tile + gap),
                                  top + len(protocol["sampling"]["seeds"]) * (tile + gap)), "white")
        draw = ImageDraw.Draw(sheet)
        for col, condition in enumerate(CONDITIONS):
            draw.text((left + col * (tile + gap), 8), condition, fill="black")
        for row_index, seed in enumerate(protocol["sampling"]["seeds"]):
            y = top + row_index * (tile + gap)
            draw.text((8, y + 8), f"{group}\nseed {seed}", fill="black")
            for col, condition in enumerate(CONDITIONS):
                sample_id = f"{condition}__{group}__seed{seed}"
                record = new[sample_id] if condition == "subject_literal_metal" else old[sample_id]
                root = output if condition == "subject_literal_metal" else source
                path = root / record["image_path"]
                if sha256(path) != record["image_sha256"]:
                    raise ValueError(f"source image hash changed: {path}")
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
    parser.add_argument("--source-comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    if (protocol.get("schema") != "lora_kv_material_comparison/v1"
            or protocol.get("sampling") != {"seeds": [42, 43, 44],
                                             "num_inference_steps": 100, "guidance_scale": 3.5}):
        raise ValueError("unexpected fixed S/M comparison protocol")
    subject_protocol = checked_file(REPO_ROOT / protocol["subject_comparison"],
                                    protocol["subject_comparison_sha256"])
    run_root = Path(os.environ["COLORPEEL_RUN_ROOT"]).resolve()
    subject = args.subject_checkpoint.resolve()
    source = args.source_comparison.resolve()
    old_material = material_checkpoint(subject_protocol, run_root)
    source_provenance = read_json(source / "provenance.json")
    subject_hashes = verify_subject_checkpoint(subject, "token_local_kv", subject_protocol)
    if (source_provenance.get("status") != "succeeded"
            or source_provenance.get("protocol_sha256") != sha256(args.protocol)
            or source_provenance.get("subject_checkpoint") != str(subject)
            or source_provenance.get("subject_checkpoint_sha256") != subject_hashes
            or source_provenance.get("old_material_checkpoint") != str(old_material)):
        raise ValueError("source S/M comparison differs from the fixed checkpoints or protocol")
    rows = []
    for original in matched_rows(protocol, subject_protocol):
        if original["condition"] != "subject_old_material":
            continue
        prompt = original["prompt"].replace("<M*>", "metal")
        if prompt.count("<S*>") != 1 or "<M*>" in prompt or prompt.count("metal") != 1:
            raise ValueError("literal-metal prompt differs from its paired S+M prompt")
        sample_id = f"subject_literal_metal__{original['group']}__seed{original['seed']}"
        rows.append({"id": sample_id, "condition": "subject_literal_metal",
                     "group": original["group"], "seed": original["seed"], "prompt": prompt,
                     "image_path": f"images/{original['group']}/{sample_id}.png"})
    if len(rows) != 9 or args.output_dir.exists():
        raise ValueError("expected nine new samples and a fresh output directory")
    args.output_dir.mkdir(parents=True)
    provenance = {"status": "dry_run" if args.dry_run else "running",
                  "protocol_sha256": sha256(args.protocol),
                  "source_comparison": str(source),
                  "source_provenance_sha256": sha256(source / "provenance.json"),
                  "subject_checkpoint_sha256": subject_hashes,
                  "old_material_weights_sha256": sha256(old_material / "pytorch_token_local_kv_weights.bin")}
    provenance_path = args.output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "generation_manifest.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if args.dry_run:
        print(f"{len(rows)} literal-metal controls: {args.output_dir}")
        return
    import torch

    pipe = load_pipeline(protocol, subject, old_material, "old", args.device)
    with (args.output_dir / "generation_status.jsonl").open("w") as ledger:
        for row in rows:
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
    provenance["contact_sheet_sha256"] = contact_sheets(protocol, source, args.output_dir, rows)
    provenance["status"] = "succeeded"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} literal-metal controls: {args.output_dir}")


if __name__ == "__main__":
    main()
