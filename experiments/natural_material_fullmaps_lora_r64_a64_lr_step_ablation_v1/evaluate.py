"""Evaluate one Material trajectory with transfer and Subject composition."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.lora_kv_material_v1.transfer_five_objects import token_mask
from experiments.lora_kv_subject_v1.evaluate import token_masks
from scripts.launch.colorpeel_run import read_config


WEIGHT_NAME = "pytorch_lora_kv_weights.bin"
MATERIALS = ("mailbox", "metal_spoon", "wood_spoon")
OBJECTS = ("vase", "chair", "mug", "sofa", "mailbox")
TRANSFER_ARMS = ("base", "literal", "token")
MATERIAL_STEPS = (600, 1000, 2000, 3000)
SUBJECT_STEPS = (1000, 2000, 3000)
LR_LABELS = {1.0e-5: "1em5", 5.0e-5: "5em5"}
LITERALS = {
    "mailbox": "red painted metal",
    "metal_spoon": "polished stainless steel",
    "wood_spoon": "light wood",
}
GROUPS = (
    {"id": "plain", "subject_prompt": "a photo of <S*> mailbox"},
    {"id": "red", "subject_prompt": "a photo of <S*> mailbox in red color"},
    {"id": "blue", "subject_prompt": "a photo of <S*> mailbox in blue color"},
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol(protocol: dict) -> None:
    expected = {
        "schema": "natural_material_fullmaps_lora_r64_a64_inference/v1",
        "base_model": "CompVis/stable-diffusion-v1-4",
        "material": {
            "training_study": "natural_material_fullmaps_lora_r64_a64_lr_step_ablation_v1",
            "mode": "token_local_kv", "rank": 64, "alpha": 64.0,
            "snapshot_steps": list(MATERIAL_STEPS),
            "kv_learning_rates": [1.0e-5, 5.0e-5],
            "literal_material": LITERALS,
        },
        "transfer": {
            "objects": list(OBJECTS), "arms": list(TRANSFER_ARMS),
            "token_prompt": "a photo of a {object} made of <M*>",
            "sampling": {"seeds": [42, 43, 44], "num_inference_steps": 100,
                         "guidance_scale": 3.5},
        },
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
        "composition": {
            "condition": "subject_material_token", "groups": list(GROUPS),
            "prompt_suffix": " made of <M*>",
            "sampling": {"seeds": [42, 43, 44, 45, 46],
                         "num_inference_steps": 100, "guidance_scale": 3.5},
        },
        "safety_checker": "enabled",
    }
    if protocol != expected:
        raise ValueError("inference protocol differs from the locked experiment")


def transfer_rows(protocol: dict) -> list[dict]:
    rows = []
    material = protocol["material_id"]
    literal = LITERALS[material]
    for material_step in MATERIAL_STEPS:
        for obj in OBJECTS:
            stem = f"a photo of a {obj}"
            prompts = {
                "base": stem,
                "literal": f"{stem} made of {literal}",
                "token": f"{stem} made of <M*>",
            }
            for seed in protocol["transfer"]["sampling"]["seeds"]:
                for arm in TRANSFER_ARMS:
                    sample_id = f"transfer__m{material_step}__{obj}__{arm}__seed{seed}"
                    rows.append({
                        "id": sample_id, "family": "transfer",
                        "material_id": material, "material_step": material_step,
                        "object": obj, "arm": arm, "seed": seed,
                        "prompt": prompts[arm],
                        "image_path": f"images/transfer/mstep{material_step}/{obj}/{arm}/seed{seed}.png",
                    })
    return rows


def composition_rows(protocol: dict) -> list[dict]:
    rows = []
    material = protocol["material_id"]
    for material_step in MATERIAL_STEPS:
        for subject_step in SUBJECT_STEPS:
            for group in GROUPS:
                prompt = group["subject_prompt"] + protocol["composition"]["prompt_suffix"]
                if prompt.count("<S*>") != 1 or prompt.count("<M*>") != 1:
                    raise ValueError(f"modifier token count differs: {prompt}")
                for seed in protocol["composition"]["sampling"]["seeds"]:
                    sample_id = (f"composition__m{material_step}__s{subject_step}__"
                                 f"{group['id']}__seed{seed}")
                    rows.append({
                        "id": sample_id, "family": "composition",
                        "condition": "subject_material_token", "material_id": material,
                        "material_step": material_step, "subject_step": subject_step,
                        "group": group["id"], "seed": seed, "prompt": prompt,
                        "image_path": (f"images/composition/mstep{material_step}/"
                                       f"sstep{subject_step}/{group['id']}/seed{seed}.png"),
                    })
    return rows


def verify_material_run(run: Path, protocol: dict, run_root: Path) -> tuple[str, float, Path, dict]:
    run = run.resolve()
    study = protocol["material"]["training_study"]
    if run.parent != run_root / study:
        raise ValueError("Material run belongs to a different study")
    manifest_path = run / "manifest.json"
    manifest = read_json(manifest_path)
    config = read_config(run / "config.yaml")
    material = config.get("source_lock", {}).get("material_id")
    learning_rate = config.get("args", {}).get("kv_learning_rate")
    if material not in MATERIALS or learning_rate not in LR_LABELS:
        raise ValueError("Material identity or learning rate differs")
    expected_run = {
        "study": study,
        "variant": (f"{material}_fullmaps_token_local_kv_r64_a64_"
                    f"kvlr{LR_LABELS[learning_rate]}_3000"),
        "seed": 42,
    }
    args = config.get("args", {})
    if (manifest.get("status") != "succeeded" or manifest.get("returncode") != 0
            or manifest.get("stage") != "train" or manifest.get("run") != expected_run
            or config.get("run") != expected_run
            or config.get("status") != "authorized_natural_material_fullmaps_lora_r64_a64_lr_step_ablation"
            or args.get("material_lora_mode") != "token_local_kv"
            or args.get("material_lora_rank") != 64 or args.get("material_lora_alpha") != 64
            or args.get("learning_rate") != 1.0e-5
            or args.get("max_train_steps") != 3000
            or args.get("checkpoint_steps") != list(MATERIAL_STEPS)):
        raise ValueError("Material training run differs from the locked experiment")
    final = run / "checkpoints"
    adaptation = read_json(final / "adaptation_config.json")
    if (adaptation.get("adaptation_mode") != "lora_material_kv"
            or adaptation.get("mode") != "token_local_kv"
            or adaptation.get("rank") != 64 or adaptation.get("alpha") != 64
            or adaptation.get("modifier_tokens") != ["<M*>"]
            or adaptation.get("weight_name") != WEIGHT_NAME):
        raise ValueError("Material adaptation config differs")
    return material, learning_rate, final, {
        "manifest_sha256": sha256(manifest_path),
        "weights_sha256": sha256(final / WEIGHT_NAME),
        "token_sha256": sha256(final / "<M*>.bin"),
        "adaptation_config_sha256": sha256(final / "adaptation_config.json"),
    }


def verify_subject_run(protocol: dict, run_root: Path) -> tuple[Path, Path, dict]:
    spec = protocol["subject"]
    run = (run_root / spec["run_relative_to_COLORPEEL_RUN_ROOT"]).resolve()
    if run.parent != run_root / spec["training_study"]:
        raise ValueError("Subject run belongs to a different study")
    manifest_path = run / "manifest.json"
    manifest = read_json(manifest_path)
    config = read_config(run / "config.yaml")
    expected_run = {"study": spec["training_study"], "variant": spec["variant"], "seed": 42}
    args = config.get("args", {})
    if (manifest.get("status") != "succeeded" or manifest.get("returncode") != 0
            or manifest.get("stage") != "train" or manifest.get("run") != expected_run
            or config.get("run") != expected_run
            or args.get("subject_lora_mode") != spec["mode"]
            or args.get("subject_lora_rank") != spec["rank"]
            or args.get("subject_lora_alpha") != spec["alpha"]
            or args.get("kv_learning_rate") != spec["kv_learning_rate"]
            or args.get("checkpoint_steps") != [600, 1000, 2000, 3000]):
        raise ValueError("Subject training run differs from the locked experiment")
    final = run / "checkpoints"
    adaptation = read_json(final / "adaptation_config.json")
    if (adaptation.get("adaptation_mode") != "lora_subject_kv"
            or adaptation.get("mode") != spec["mode"]
            or adaptation.get("rank") != spec["rank"]
            or adaptation.get("alpha") != spec["alpha"]
            or adaptation.get("modifier_tokens") != ["<S*>"]
            or adaptation.get("weight_name") != WEIGHT_NAME):
        raise ValueError("Subject adaptation config differs")
    return run, final, {
        "manifest_sha256": sha256(manifest_path),
        "weights_sha256": sha256(final / WEIGHT_NAME),
        "token_sha256": sha256(final / "<S*>.bin"),
        "adaptation_config_sha256": sha256(final / "adaptation_config.json"),
    }


def derive_snapshots(final: Path, token: str, steps: tuple[int, ...],
                     output: Path | None) -> dict[str, dict]:
    import torch

    audit = read_json(final / "embedding_update_audit.json")
    records = [item for item in audit["modifier_tokens"] if item["token"] == token]
    if len(records) != 1:
        raise ValueError(f"embedding audit differs for {token}")
    token_id = records[0]["token_id"]
    final_adapter = torch.load(final / WEIGHT_NAME, map_location="cpu")
    final_token = torch.load(final / f"{token}.bin", map_location="cpu")[token]
    derived = {}
    for step in steps:
        snapshot = final / f"checkpoint-{step}"
        adapter_path = snapshot / "pytorch_model.bin"
        encoder_path = snapshot / "pytorch_model_1.bin"
        adapter = torch.load(adapter_path, map_location="cpu")
        encoder = torch.load(encoder_path, map_location="cpu")
        embedding = encoder["text_model.embeddings.token_embedding.weight"][token_id].clone().contiguous()
        if (set(adapter) != set(final_adapter)
                or any(adapter[key].shape != final_adapter[key].shape for key in adapter)
                or embedding.shape != final_token.shape):
            raise ValueError(f"snapshot schema differs for {token} at step {step}")
        if step == steps[-1] and (not torch.equal(embedding, final_token)
                or any(not torch.equal(adapter[key], final_adapter[key]) for key in adapter)):
            raise ValueError(f"final snapshot differs from exported {token} artifacts")
        record = {"source_adapter_sha256": sha256(adapter_path),
                  "source_text_encoder_sha256": sha256(encoder_path)}
        if output is not None:
            checkpoint = output / f"step{step}"
            checkpoint.mkdir(parents=True)
            torch.save(adapter, checkpoint / WEIGHT_NAME)
            torch.save({token: embedding}, checkpoint / f"{token}.bin")
            shutil.copyfile(final / "adaptation_config.json", checkpoint / "adaptation_config.json")
            record.update({"path": str(checkpoint),
                           "derived_adapter_sha256": sha256(checkpoint / WEIGHT_NAME),
                           "derived_token_sha256": sha256(checkpoint / f"{token}.bin")})
        derived[str(step)] = record
    return derived


def load_material_pipeline(base_model: str, checkpoint: Path, device: str):
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
        raise ValueError("transfer requires the base safety checker and PNDM scheduler")
    pipe.load_textual_inversion(str(checkpoint), weight_name="<M*>.bin")
    return pipe


def load_composition_pipeline(base_model: str, subject: Path, material: Path, device: str):
    import torch
    from diffusers import DiffusionPipeline, PNDMScheduler
    from experiments.lora_kv_material_v1.attention import install_dual_token_local_lora_kv

    train_root = str(REPO_ROOT / "src" / "train")
    if train_root not in sys.path:
        sys.path.insert(0, train_root)
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(
        base_model, subfolder="unet", local_files_only=True, torch_dtype=torch.float16)
    subject_state = torch.load(subject / WEIGHT_NAME, map_location="cpu")
    material_state = torch.load(material / WEIGHT_NAME, map_location="cpu")
    install_dual_token_local_lora_kv(unet, subject_state, material_state, 64, 64)
    pipe = DiffusionPipeline.from_pretrained(
        base_model, unet=unet, low_cpu_mem_usage=False,
        torch_dtype=torch.float16, local_files_only=True).to(device)
    if pipe.safety_checker is None or not isinstance(pipe.scheduler, PNDMScheduler):
        raise ValueError("composition requires the base safety checker and PNDM scheduler")
    pipe.load_textual_inversion(str(subject), weight_name="<S*>.bin")
    pipe.load_textual_inversion(str(material), weight_name="<M*>.bin")
    return pipe


def generate_rows(pipe, rows: list[dict], sampling: dict, output: Path,
                  ledger, device: str, composition: bool) -> None:
    import torch

    for row in rows:
        masks = (token_masks(pipe, row["prompt"], sampling["guidance_scale"])
                 if composition else token_mask(
                     pipe, row["prompt"], sampling["guidance_scale"]))
        result = pipe(
            row["prompt"], num_inference_steps=sampling["num_inference_steps"],
            guidance_scale=sampling["guidance_scale"],
            generator=torch.Generator(device=device).manual_seed(row["seed"]),
            cross_attention_kwargs=masks,
        )
        target = output / row["image_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(target)
        flags = getattr(result, "nsfw_content_detected", None)
        if flags is not None and len(flags) != 1:
            raise ValueError("expected one safety-checker result")
        ledger.write(json.dumps({**row, "image_sha256": sha256(target),
                                 "safety_filtered": bool(flags[0]) if flags else False},
                                sort_keys=True) + "\n")
        ledger.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--material-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol = read_json(args.protocol)
    validate_protocol(protocol)
    run_root = Path(os.environ["COLORPEEL_RUN_ROOT"]).resolve()
    material, learning_rate, material_final, material_hashes = verify_material_run(
        args.material_run, protocol, run_root)
    subject_run, subject_final, subject_hashes = verify_subject_run(protocol, run_root)
    runtime_protocol = dict(protocol, material_id=material)
    transfers = transfer_rows(runtime_protocol)
    compositions = composition_rows(runtime_protocol)
    if (len(transfers) != 180 or len(compositions) != 180
            or len({row["id"] for row in transfers + compositions}) != 360):
        raise ValueError("inference row matrix differs")
    material_audit = derive_snapshots(material_final, "<M*>", MATERIAL_STEPS, None)
    subject_audit = derive_snapshots(subject_final, "<S*>", SUBJECT_STEPS, None)
    if args.dry_run:
        print(f"verified {material} kvLR={learning_rate}: 180 transfer + 180 composition rows")
        return

    output = args.output.resolve()
    for protected in (args.material_run.resolve(), subject_run):
        if output == protected or output.is_relative_to(protected):
            raise ValueError(f"output must not be inside training run: {protected}")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    material_derived = derive_snapshots(
        material_final, "<M*>", MATERIAL_STEPS, output / "derived_material")
    subject_derived = derive_snapshots(
        subject_final, "<S*>", SUBJECT_STEPS, output / "derived_subject")
    rows = transfers + compositions
    with (output / "generation_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    provenance = {
        "schema": protocol["schema"], "status": "running",
        "protocol_sha256": sha256(args.protocol), "material_id": material,
        "kv_learning_rate": learning_rate,
        "material_run": str(args.material_run.resolve()),
        "material_checkpoint_sha256": material_hashes,
        "material_snapshots": material_derived,
        "subject_run": str(subject_run), "subject_checkpoint_sha256": subject_hashes,
        "subject_snapshots": subject_derived, "row_count": len(rows),
        "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    import torch

    with (output / "generation_status.jsonl").open("w", encoding="utf-8") as ledger:
        for material_step in MATERIAL_STEPS:
            material_checkpoint = Path(material_derived[str(material_step)]["path"])
            pipe = load_material_pipeline(protocol["base_model"], material_checkpoint, args.device)
            generate_rows(
                pipe, [row for row in transfers if row["material_step"] == material_step],
                protocol["transfer"]["sampling"], output, ledger, args.device, False)
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            for subject_step in SUBJECT_STEPS:
                subject_checkpoint = Path(subject_derived[str(subject_step)]["path"])
                pipe = load_composition_pipeline(
                    protocol["base_model"], subject_checkpoint, material_checkpoint, args.device)
                generate_rows(
                    pipe, [row for row in compositions
                           if row["material_step"] == material_step
                           and row["subject_step"] == subject_step],
                    protocol["composition"]["sampling"], output, ledger, args.device, True)
                del pipe
                gc.collect()
                torch.cuda.empty_cache()
    if (verify_material_run(args.material_run, protocol, run_root)[3] != material_hashes
            or verify_subject_run(protocol, run_root)[2] != subject_hashes
            or derive_snapshots(material_final, "<M*>", MATERIAL_STEPS, None) != material_audit
            or derive_snapshots(subject_final, "<S*>", SUBJECT_STEPS, None) != subject_audit):
        raise ValueError("training checkpoints changed during inference")
    provenance["status"] = "succeeded"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"generated {len(rows)} images: {output}")


if __name__ == "__main__":
    main()
