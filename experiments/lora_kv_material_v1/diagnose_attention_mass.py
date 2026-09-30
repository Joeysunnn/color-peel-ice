"""Measure per-token cross-attention in matched new-M-only and S+new-M samples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.lora_kv_material_v1.evaluate import (
    checked_file, load_pipeline, matched_rows, material_checkpoint, read_json,
    sha256, token_masks, verify_new_material_checkpoint, verify_subject_checkpoint,
)


CONDITIONS = ("new_material_only", "subject_new_material")


def attention_measurements(pipe, masks: dict, prompt: str, seed: int,
                           sampling: dict, device: str):
    import torch

    token_masks_by_name = masks["modifier_token_mask"]
    positions = {name: torch.where(mask[1])[0].tolist()
                 for name, mask in token_masks_by_name.items()}
    if len(positions["material"]) != 1 or len(positions["subject"]) > 1:
        raise ValueError("expected one Material token and at most one Subject token")
    layer_values = {}
    handles = []
    for name in pipe.unet.attn_processors:
        if not name.endswith("attn2.processor"):
            continue
        attn = pipe.unet.get_submodule(name.removesuffix(".processor"))
        layer_values[name] = {"material": [], "subject": []}

        def capture(module, _inputs, _output, layer=name):
            probs = module.attn_probs.detach()
            if probs.shape[0] != 2 * module.heads or probs.shape[-1] != pipe.tokenizer.model_max_length:
                raise ValueError(f"unexpected CFG attention shape: {layer}")
            conditional = probs.reshape(2, module.heads, probs.shape[1], probs.shape[2])[1]
            layer_values[layer]["material"].append(
                conditional[..., positions["material"][0]].float().mean())
            if positions["subject"]:
                layer_values[layer]["subject"].append(
                    conditional[..., positions["subject"][0]].float().mean())

        handles.append(attn.register_forward_hook(capture))
    try:
        result = pipe(prompt, num_inference_steps=sampling["num_inference_steps"],
                      guidance_scale=sampling["guidance_scale"],
                      generator=torch.Generator(device=device).manual_seed(seed),
                      cross_attention_kwargs=masks)
    finally:
        for handle in handles:
            handle.remove()
    by_layer = {}
    for name, values in layer_values.items():
        if len(values["material"]) != sampling["num_inference_steps"]:
            raise ValueError(f"unexpected attention calls at {name}: {len(values['material'])}")
        by_layer[name] = {}
        for token_name, sequence in values.items():
            if not sequence:
                continue
            tensor = torch.stack(sequence)
            by_layer[name][token_name] = {
                "all": tensor.mean().item(),
                "early20": tensor[:20].mean().item(),
                "late20": tensor[-20:].mean().item(),
            }
    summary = {token_name: {
        period: statistics.mean(values[token_name][period] for values in by_layer.values()
                                if token_name in values)
        for period in ("all", "early20", "late20")}
        for token_name in ("material", "subject") if positions[token_name]}
    return result, summary, by_layer


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
        raise ValueError("source comparison differs from the fixed checkpoints")
    rows = [row for row in matched_rows(protocol, subject_protocol)
            if row["condition"] in CONDITIONS]
    if len(rows) != 18 or args.output_dir.exists():
        raise ValueError("expected 18 matched Material controls and a fresh output directory")
    args.output_dir.mkdir(parents=True)
    provenance = {"status": "dry_run" if args.dry_run else "running",
                  "protocol_sha256": sha256(args.protocol),
                  "source_comparison": str(source),
                  "source_provenance_sha256": sha256(source / "provenance.json"),
                  "subject_checkpoint_sha256": subject_hashes,
                  "new_material_checkpoint_sha256": material_hashes,
                  "measurement": "conditional token attention probability averaged over heads and spatial queries; equal weight per cross-attention layer"}
    provenance_path = args.output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    if args.dry_run:
        print(f"{len(rows)} paired attention measurements: {args.output_dir}")
        return
    pipe = load_pipeline(protocol, subject, new_material, "new", args.device)
    prior = {row["id"]: row for row in
             (json.loads(line) for line in (source / "generation_status.jsonl").read_text().splitlines())}
    records = []
    with (args.output_dir / "attention_status.jsonl").open("w") as ledger:
        for row in rows:
            masks = token_masks(pipe, row["prompt"], protocol["sampling"]["guidance_scale"])
            result, summary, by_layer = attention_measurements(
                pipe, masks, row["prompt"], row["seed"], protocol["sampling"], args.device)
            path = args.output_dir / row["image_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            result.images[0].save(path)
            flags = getattr(result, "nsfw_content_detected", None)
            if flags is not None and len(flags) != 1:
                raise ValueError("expected one safety-checker result")
            filtered = bool(flags[0]) if flags is not None else False
            image_sha = sha256(path)
            original = prior[row["id"]]
            if image_sha != original["image_sha256"] or filtered != original["nsfw_content_detected"]:
                raise ValueError("attention hook changed a matched source image")
            record = {**row, "image_sha256": image_sha,
                      "status": "safety_filtered" if filtered else "ok",
                      "attention": summary, "layers": by_layer}
            ledger.write(json.dumps(record, sort_keys=True) + "\n")
            ledger.flush()
            records.append(record)
    summary = {condition: {
        "material_all": statistics.mean(row["attention"]["material"]["all"] for row in records
                                        if row["condition"] == condition),
        "material_early20": statistics.mean(row["attention"]["material"]["early20"] for row in records
                                            if row["condition"] == condition),
        "material_late20": statistics.mean(row["attention"]["material"]["late20"] for row in records
                                           if row["condition"] == condition),
    } for condition in CONDITIONS}
    summary["subject_new_material"]["subject_all"] = statistics.mean(
        row["attention"]["subject"]["all"] for row in records
        if row["condition"] == "subject_new_material")
    (args.output_dir / "attention_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    provenance["status"] = "succeeded"
    provenance["attention_status_sha256"] = sha256(args.output_dir / "attention_status.jsonl")
    provenance["attention_summary_sha256"] = sha256(args.output_dir / "attention_summary.json")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
