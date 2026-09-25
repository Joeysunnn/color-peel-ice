"""Generate literal-orange + M pairs for the existing C + M diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.methods.colorpeel_ice.generate_color_material_diagnostic import (
    WEIGHTS, attention_masks, read_json, sha256, verify_sources,
)


def comparison_rows(protocol_path: Path, run_root: Path):
    protocol = read_json(protocol_path)
    if (protocol.get("schema") != "literal_orange_vs_color_token_with_m/v1"
            or protocol.get("objects") != ["cube", "sphere", "mug"]
            or protocol.get("seeds") != [42, 43, 44]
            or protocol.get("learned_prompt") != "a photo of a {object} with <C*> color and <M*> material"
            or protocol.get("literal_prompt") != "a photo of a {object} with orange color and <M*> material"
            or protocol.get("sampling") != {"num_inference_steps": 100, "guidance_scale": 3.5}
            or protocol.get("safety_checker") != "enabled"):
        raise ValueError("literal orange comparison protocol differs")
    source_protocol_path = REPO_ROOT / protocol["source_protocol"]
    if sha256(source_protocol_path) != protocol["source_protocol_sha256"]:
        raise ValueError("source protocol differs")
    source, color_dir, material_dir = verify_sources(source_protocol_path, run_root)
    if (source["sampling"] != {"seeds": protocol["seeds"], **protocol["sampling"]}
            or source["prompts"]["color_material"] != protocol["learned_prompt"]
            or source["objects"] != protocol["objects"]):
        raise ValueError("source generation settings differ")
    reference = run_root / protocol["reference_run_relative_to_COLORPEEL_RUN_ROOT"]
    if (sha256(reference / "generation_status.jsonl") != protocol["reference_status_sha256"]
            or read_json(reference / "provenance.json").get("protocol_sha256") != protocol["source_protocol_sha256"]):
        raise ValueError("reference generation differs")
    statuses = [json.loads(line) for line in (reference / "generation_status.jsonl").read_text(encoding="utf-8").splitlines()]
    paired = {(row["object"], row["seed"]): row for row in statuses if row["condition"] == "color_material"}
    if len(paired) != 9:
        raise ValueError("expected nine reference C+M images")
    rows = []
    for obj in protocol["objects"]:
        for seed in protocol["seeds"]:
            learned = paired[obj, seed]
            expected_prompt = protocol["learned_prompt"].format(object=obj)
            learned_path = reference / learned["image_path"]
            if (learned["prompt"] != expected_prompt or learned["status"] != "ok"
                    or sha256(learned_path) != learned["image_sha256"]):
                raise ValueError(f"reference C+M image differs: {obj}, {seed}")
            rows.append({"object": obj, "seed": seed,
                         "literal_prompt": protocol["literal_prompt"].format(object=obj),
                         "literal_image_path": f"images/{obj}/literal_orange_material__seed{seed}.png",
                         "learned_prompt": expected_prompt,
                         "learned_image_path": str(learned_path),
                         "learned_image_sha256": learned["image_sha256"]})
    return protocol, color_dir, material_dir, rows


def generate(protocol: dict, color_dir: Path, material_dir: Path, rows: list[dict], output_dir: Path, device: str):
    import torch
    from diffusers import DiffusionPipeline

    train_root = str(REPO_ROOT / "src" / "train")
    if train_root not in sys.path:
        sys.path.insert(0, train_root)
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel
    from src.methods.colorpeel_ice.dual_token_local_kv import install_dual_token_local_kv

    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet", local_files_only=True, torch_dtype=torch.float16)
    unet.load_attn_procs(str(color_dir), weight_name=WEIGHTS, adaptation_mode="token_local_kv")
    install_dual_token_local_kv(unet, torch.load(material_dir / WEIGHTS, map_location="cpu"))
    pipe = DiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", unet=unet, low_cpu_mem_usage=False,
        torch_dtype=torch.float16, local_files_only=True).to(device)
    pipe.load_textual_inversion(str(color_dir), weight_name="<C*>.bin")
    pipe.load_textual_inversion(str(material_dir), weight_name="<M*>.bin")
    with (output_dir / "generation_status.jsonl").open("w", encoding="utf-8") as ledger:
        for row in rows:
            prompt = row["literal_prompt"]
            result = pipe(prompt, num_inference_steps=protocol["sampling"]["num_inference_steps"],
                          guidance_scale=protocol["sampling"]["guidance_scale"],
                          generator=torch.Generator(device=device).manual_seed(row["seed"]),
                          cross_attention_kwargs=attention_masks(pipe, prompt, protocol["sampling"]["guidance_scale"]))
            image_path = output_dir / row["literal_image_path"]
            image_path.parent.mkdir(parents=True, exist_ok=True)
            result.images[0].save(image_path)
            flags = getattr(result, "nsfw_content_detected", None)
            if flags is not None and len(flags) != 1:
                raise ValueError("expected one safety-checker result")
            filtered = bool(flags[0]) if flags is not None else False
            ledger.write(json.dumps({"object": row["object"], "seed": row["seed"],
                                     "status": "safety_filtered" if filtered else "ok",
                                     "image_path": row["literal_image_path"],
                                     "image_sha256": sha256(image_path),
                                     "nsfw_content_detected": filtered}, sort_keys=True) + "\n")
            ledger.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol, color_dir, material_dir, rows = comparison_rows(
        args.protocol, Path(os.environ["COLORPEEL_RUN_ROOT"]))
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "comparison_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    provenance = {"protocol_sha256": sha256(args.protocol), "color_dir": str(color_dir),
                  "material_dir": str(material_dir), "status": "dry_run" if args.dry_run else "running"}
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.dry_run:
        generate(protocol, color_dir, material_dir, rows, args.output_dir, args.device)
        provenance["status"] = "succeeded"
        (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(rows)} literal/reference pairs: {args.output_dir}")


if __name__ == "__main__":
    main()
