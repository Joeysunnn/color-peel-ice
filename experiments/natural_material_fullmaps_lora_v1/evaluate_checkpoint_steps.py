"""Transfer paired Material LoRA and token snapshots at steps 1000-3000."""

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.lora_kv_material_v1.transfer_five_objects import OBJECTS, sha, token_mask


STEPS = (1000, 2000, 3000)
MATERIALS = ("mailbox", "metal_spoon", "wood_spoon")
WEIGHTS = "pytorch_lora_kv_weights.bin"


def read_json(path):
    return json.loads(path.read_text())


def rows(protocol):
    return [{"id": f"step{step}__{obj}__{seed}", "step": step,
             "object": obj, "seed": seed,
             "prompt": protocol["prompt_template"].format(object=obj),
             "image_path": f"images/step{step}/{obj}/seed{seed}.png"}
            for step in STEPS for obj in OBJECTS
            for seed in protocol["sampling"]["seeds"]]


def pipeline(protocol, checkpoint, device):
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
        torch_dtype=torch.float16, local_files_only=True).to(device)
    if pipe.safety_checker is None or not isinstance(pipe.scheduler, PNDMScheduler):
        raise ValueError("Transfer requires the base safety checker and PNDM scheduler")
    pipe.load_textual_inversion(str(checkpoint), weight_name="<M*>.bin")
    return pipe


def contact_sheets(output, entries, final_transfer, final_rows):
    from PIL import Image, ImageDraw

    status = {row["id"]: row for row in
              (json.loads(line) for line in (output / "status.jsonl").read_text().splitlines())}
    if len(status) != len(entries):
        raise ValueError("Step transfer output is incomplete")
    sheets = {}
    folder = output / "contact_sheets"
    folder.mkdir()
    tile, left, top, gap = 256, 100, 34, 8
    for obj in OBJECTS:
        sheet = Image.new("RGB", (left + 4 * (tile + gap), top + 3 * (tile + gap)), "white")
        draw = ImageDraw.Draw(sheet)
        for col, step in enumerate((*STEPS, 5000)):
            draw.text((left + col * (tile + gap), 8), f"step {step}", fill="black")
        for row_index, seed in enumerate((42, 43, 44)):
            y = top + row_index * (tile + gap)
            draw.text((8, y + 8), f"seed {seed}", fill="black")
            for col, step in enumerate((*STEPS, 5000)):
                if step == 5000:
                    item = final_rows[f"{obj}__seed{seed}__token"]
                    path = final_transfer / item["image_path"]
                else:
                    item = status[f"step{step}__{obj}__{seed}"]
                    path = output / item["image_path"]
                if sha(path) != item["image_sha256"]:
                    raise ValueError(f"Image hash changed: {path}")
                x = left + col * (tile + gap)
                with Image.open(path) as image:
                    sheet.paste(image.convert("RGB").resize((tile, tile)), (x, y))
                if item["safety_filtered"]:
                    draw.text((x + 4, y + 4), "SAFETY FILTERED", fill="red")
        target = folder / f"{obj}.jpg"
        sheet.save(target, quality=90)
        sheets[obj] = sha(target)
    return sheets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--material-id", choices=MATERIALS, required=True)
    parser.add_argument("--checkpoint-run", type=Path, required=True)
    parser.add_argument("--final-transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol = read_json(args.protocol)
    if (protocol.get("schema") != "natural_material_fullmaps_lora_five_object_transfer/v1"
            or protocol.get("base_model") != "CompVis/stable-diffusion-v1-4"
            or protocol.get("objects") != list(OBJECTS)
            or protocol.get("prompt_template") != "a photo of a {object} made of <M*>"
            or protocol.get("sampling") != {"seeds": [42, 43, 44],
                                            "num_inference_steps": 100,
                                            "guidance_scale": 3.5}):
        raise ValueError("Unexpected five-object transfer protocol")
    run = args.checkpoint_run.resolve()
    manifest = read_json(run / "manifest.json")
    expected_run = {"study": "natural_material_fullmaps_lora_v1",
                    "variant": f"{args.material_id}_fullmaps_token_local_kv_lora_r4_5000",
                    "seed": 42}
    if (manifest.get("status") != "succeeded" or manifest.get("returncode") != 0
            or manifest.get("run") != expected_run):
        raise ValueError("Training run is not the completed Material LoRA")
    final = run / "checkpoints"
    adaptation = read_json(final / "adaptation_config.json")
    audit = read_json(final / "embedding_update_audit.json")
    if (adaptation.get("adaptation_mode") != "lora_material_kv"
            or adaptation.get("mode") != "token_local_kv"
            or adaptation.get("rank") != 4 or adaptation.get("alpha") != 4
            or adaptation.get("modifier_tokens") != ["<M*>"]
            or adaptation.get("weight_name") != WEIGHTS
            or audit["modifier_tokens"][0]["token"] != "<M*>"):
        raise ValueError("Unexpected Material LoRA or token audit")
    for step in (*STEPS, 5000):
        snapshot = final / f"checkpoint-{step}"
        for name in ("pytorch_model.bin", "pytorch_model_1.bin"):
            if not (snapshot / name).is_file():
                raise FileNotFoundError(snapshot / name)

    final_transfer = args.final_transfer.resolve()
    final_provenance = read_json(final_transfer / "provenance.json")
    if (final_provenance.get("status") != "succeeded"
            or final_provenance.get("material_id") != args.material_id
            or Path(final_provenance.get("checkpoint", "")).resolve() != final
            or final_provenance.get("weights_sha256") != sha(final / WEIGHTS)
            or final_provenance.get("token_sha256") != sha(final / "<M*>.bin")
            or final_provenance.get("protocol_sha256") != sha(args.protocol)):
        raise ValueError("Step-5000 transfer does not match this training run")
    final_rows = {row["id"]: row for row in
                  (json.loads(line) for line in (final_transfer / "status.jsonl").read_text().splitlines())}
    if len(final_rows) != 45:
        raise ValueError("Step-5000 transfer is incomplete")
    entries = rows(protocol)
    output = args.output.resolve()
    if (len(entries) != 45 or output.exists() or output.is_relative_to(run)
            or output.is_relative_to(final_transfer)):
        raise ValueError("Expected 45 new rows and a fresh output directory")
    for item in entries:
        source = final_rows[f'{item["object"]}__seed{item["seed"]}__token']
        if (source["prompt"] != item["prompt"] or source["seed"] != item["seed"]
                or sha(final_transfer / source["image_path"]) != source["image_sha256"]):
            raise ValueError("Step-5000 prompt or image differs")

    import torch

    final_adapter = torch.load(final / WEIGHTS, map_location="cpu")
    final_token = torch.load(final / "<M*>.bin", map_location="cpu")["<M*>"]
    token_id = audit["modifier_tokens"][0]["token_id"]
    if len(final_adapter) != 64 or not all(isinstance(v, torch.Tensor) for v in final_adapter.values()):
        raise ValueError("Unexpected final Material LoRA state")
    for step in (*STEPS, 5000):
        snapshot = final / f"checkpoint-{step}"
        adapter = torch.load(snapshot / "pytorch_model.bin", map_location="cpu")
        encoder = torch.load(snapshot / "pytorch_model_1.bin", map_location="cpu")
        token = encoder["text_model.embeddings.token_embedding.weight"][token_id]
        if (set(adapter) != set(final_adapter)
                or any(adapter[key].shape != final_adapter[key].shape for key in adapter)
                or token.shape != final_token.shape):
            raise ValueError(f"Snapshot schema differs at step {step}")
        if step == 5000 and (not torch.equal(token, final_token) or any(
                not torch.equal(adapter[key], final_adapter[key]) for key in adapter)):
            raise ValueError("Step-5000 snapshot does not match the final export")
        del adapter, encoder, token
    if args.dry_run:
        print(f"Verified {args.material_id}: three paired snapshots, 45 new rows, 15 final controls")
        return

    args.output.mkdir(parents=True)
    (args.output / "manifest.jsonl").write_text("".join(
        json.dumps(item, sort_keys=True) + "\n" for item in entries))
    provenance = {"status": "running", "material_id": args.material_id,
                  "protocol_sha256": sha(args.protocol),
                  "training_run": str(run), "training_manifest_sha256": sha(run / "manifest.json"),
                  "final_transfer": str(final_transfer),
                  "final_transfer_provenance_sha256": sha(final_transfer / "provenance.json"),
                  "final_adapter_sha256": sha(final / WEIGHTS),
                  "source_snapshots": {}, "row_count": len(entries)}
    provenance_path = args.output / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    with (args.output / "status.jsonl").open("w") as ledger:
        for step in STEPS:
            snapshot = final / f"checkpoint-{step}"
            adapter_path = snapshot / "pytorch_model.bin"
            encoder_path = snapshot / "pytorch_model_1.bin"
            adapter = torch.load(adapter_path, map_location="cpu")
            encoder = torch.load(encoder_path, map_location="cpu")
            token = encoder["text_model.embeddings.token_embedding.weight"][token_id].clone().contiguous()
            checkpoint = args.output / "derived_checkpoints" / f"step{step}"
            checkpoint.mkdir(parents=True)
            torch.save(adapter, checkpoint / WEIGHTS)
            torch.save({"<M*>": token}, checkpoint / "<M*>.bin")
            shutil.copy2(final / "adaptation_config.json", checkpoint / "adaptation_config.json")
            provenance["source_snapshots"][str(step)] = {
                "adapter_sha256": sha(adapter_path), "text_encoder_sha256": sha(encoder_path),
                "derived_adapter_sha256": sha(checkpoint / WEIGHTS),
                "derived_token_sha256": sha(checkpoint / "<M*>.bin")}
            del adapter, encoder, token
            pipe = pipeline(protocol, checkpoint, args.device)
            for item in (item for item in entries if item["step"] == step):
                result = pipe(item["prompt"],
                              num_inference_steps=protocol["sampling"]["num_inference_steps"],
                              guidance_scale=protocol["sampling"]["guidance_scale"],
                              generator=torch.Generator(device=args.device).manual_seed(item["seed"]),
                              cross_attention_kwargs=token_mask(
                                  pipe, item["prompt"], protocol["sampling"]["guidance_scale"]))
                target = args.output / item["image_path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                result.images[0].save(target)
                flags = getattr(result, "nsfw_content_detected", None)
                if flags is not None and len(flags) != 1:
                    raise ValueError("Expected one safety-checker result")
                ledger.write(json.dumps({**item, "image_sha256": sha(target),
                                         "safety_filtered": bool(flags[0]) if flags else False},
                                        sort_keys=True) + "\n")
                ledger.flush()
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    provenance["contact_sheet_sha256"] = contact_sheets(
        args.output, entries, final_transfer, final_rows)
    provenance["status"] = "succeeded"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Generated {len(entries)} step-transfer images: {args.output}")


if __name__ == "__main__":
    main()
