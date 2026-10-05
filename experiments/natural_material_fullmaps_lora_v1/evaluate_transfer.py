"""Five-object Material-only transfer for one full-map Material LoRA."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.lora_kv_material_v1.transfer_five_objects import (
    ARMS, OBJECTS, WEIGHTS, make_sheets, rows, sha, token_mask,
)


MATERIALS = ("mailbox", "metal_spoon", "wood_spoon")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--material-id", choices=MATERIALS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if (protocol.get("schema") != "natural_material_fullmaps_lora_five_object_transfer/v1"
            or protocol.get("base_model") != "CompVis/stable-diffusion-v1-4"
            or protocol.get("objects") != list(OBJECTS)
            or protocol.get("arms") != list(ARMS)
            or protocol.get("prompt_template") != "a photo of a {object} made of <M*>"
            or protocol.get("sampling") != {"seeds": [42, 43, 44],
                                            "num_inference_steps": 100,
                                            "guidance_scale": 3.5}
            or set(protocol.get("literal_material", {})) != set(MATERIALS)):
        raise ValueError("Unexpected five-object transfer protocol")
    checkpoint = args.checkpoint.resolve()
    manifest_path = checkpoint.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    adaptation = json.loads((checkpoint / "adaptation_config.json").read_text())
    expected_run = {"study": "natural_material_fullmaps_lora_v1",
                    "variant": f"{args.material_id}_fullmaps_token_local_kv_lora_r4_5000",
                    "seed": 42}
    if (checkpoint.name != "checkpoints" or manifest.get("status") != "succeeded"
            or manifest.get("returncode") != 0 or manifest.get("run") != expected_run
            or adaptation.get("adaptation_mode") != "lora_material_kv"
            or adaptation.get("mode") != "token_local_kv"
            or adaptation.get("rank") != 4 or adaptation.get("alpha") != 4
            or adaptation.get("modifier_tokens") != ["<M*>"]
            or adaptation.get("weight_name") != WEIGHTS):
        raise ValueError("Checkpoint is not the completed full-map Material LoRA")
    weight_hash = sha(checkpoint / WEIGHTS)
    token_hash = sha(checkpoint / "<M*>.bin")
    selected = dict(protocol, literal_material=protocol["literal_material"][args.material_id])
    entries = rows(selected)
    if len(entries) != 45:
        raise ValueError("Expected 45 transfer rows")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.dry_run:
        print(f"Verified {args.material_id} LoRA checkpoint; {len(entries)} transfer rows")
        return
    args.output.mkdir(parents=True)
    (args.output / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in entries))
    provenance = {"status": "running", "material_id": args.material_id,
                  "protocol_sha256": sha(args.protocol),
                  "training_manifest_sha256": sha(manifest_path),
                  "checkpoint": str(checkpoint), "weights_sha256": weight_hash,
                  "token_sha256": token_hash, "row_count": len(entries)}
    provenance_path = args.output / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    import torch
    from diffusers import DiffusionPipeline, PNDMScheduler
    from experiments.lora_kv_subject_v1.attention import install_subject_lora_kv

    train_root = str(ROOT / "src" / "train")
    if train_root not in sys.path:
        sys.path.insert(0, train_root)
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(
        protocol["base_model"], subfolder="unet", local_files_only=True,
        torch_dtype=torch.float16)
    state = torch.load(checkpoint / WEIGHTS, map_location="cpu")
    install_subject_lora_kv(unet, state, "token_local_kv", 4, 4)
    pipe = DiffusionPipeline.from_pretrained(
        protocol["base_model"], unet=unet, low_cpu_mem_usage=False,
        torch_dtype=torch.float16, local_files_only=True).to(args.device)
    if pipe.safety_checker is None or not isinstance(pipe.scheduler, PNDMScheduler):
        raise ValueError("Transfer requires the base safety checker and PNDM scheduler")
    pipe.load_textual_inversion(str(checkpoint), weight_name="<M*>.bin")
    with (args.output / "status.jsonl").open("w") as ledger:
        for row in entries:
            result = pipe(
                row["prompt"],
                num_inference_steps=protocol["sampling"]["num_inference_steps"],
                guidance_scale=protocol["sampling"]["guidance_scale"],
                generator=torch.Generator(device=args.device).manual_seed(row["seed"]),
                cross_attention_kwargs=token_mask(
                    pipe, row["prompt"], protocol["sampling"]["guidance_scale"]),
            )
            path = args.output / row["image_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            result.images[0].save(path)
            flags = getattr(result, "nsfw_content_detected", None)
            if flags is not None and len(flags) != 1:
                raise ValueError("Expected one safety-checker result")
            ledger.write(json.dumps({**row, "image_sha256": sha(path),
                                     "safety_filtered": bool(flags[0]) if flags else False},
                                    sort_keys=True) + "\n")
            ledger.flush()
    if sha(checkpoint / WEIGHTS) != weight_hash or sha(checkpoint / "<M*>.bin") != token_hash:
        raise ValueError("Material checkpoint changed during inference")
    provenance["contact_sheet_sha256"] = make_sheets(entries, args.output)
    provenance["status"] = "succeeded"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Generated {len(entries)} images: {args.output}")


if __name__ == "__main__":
    main()
