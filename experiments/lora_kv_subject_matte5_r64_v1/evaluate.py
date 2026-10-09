"""Run the locked matte5 Subject comparison or historical transfer benchmark."""

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

from experiments.lora_kv_subject_v1.evaluate import token_masks
from scripts.launch.colorpeel_run import read_config


WEIGHT_NAME = "pytorch_lora_kv_weights.bin"
SUBJECT_STEPS = (1000, 2000, 3000)
MATERIAL_IDS = ("metal_spoon", "wood_spoon")
CONDITIONS = (
    "subject_only",
    "subject_literal_metal",
    "subject_material_token_metal_spoon",
    "subject_material_token_wood_spoon",
)
SOURCE_FIELDS = {
    "checkpoint_id", "checkpoint_steps", "color", "guidance_scale", "id",
    "image_path", "model_dir", "num_inference_steps", "prompt", "sampling_id", "seed",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def validate_protocol(protocol: dict) -> None:
    require(protocol.get("schema") == "lora_kv_subject_matte5_r64_inference/v1",
            "inference protocol schema differs")
    require(protocol.get("base_model") == "CompVis/stable-diffusion-v1-4",
            "base model differs")
    subject = protocol.get("subject", {})
    require(subject.get("training_study") == "lora_kv_subject_matte5_r64_v1"
            and subject.get("variant")
            == "mailbox_matte5_token_local_kv_r64_a64_kvlr5em5_3000"
            and subject.get("mode") == "token_local_kv"
            and subject.get("rank") == 64 and subject.get("alpha") == 64.0
            and subject.get("kv_learning_rate") == 5.0e-5
            and subject.get("snapshot_steps") == list(SUBJECT_STEPS)
            and set(subject.get("source_snapshot_sha256", {}))
            == {str(step) for step in SUBJECT_STEPS},
            "Subject protocol differs")
    materials = protocol.get("materials", {})
    require(tuple(materials) == MATERIAL_IDS, "Material selection differs")
    for material_id in MATERIAL_IDS:
        spec = materials[material_id]
        require(spec.get("variant")
                == f"{material_id}_fullmaps_token_local_kv_r64_a64_kvlr5em5_3000"
                and spec.get("snapshot_step") == 1000,
                f"{material_id} protocol differs")
    comparison = protocol.get("comparison", {})
    require(comparison.get("groups") == [
        {"id": "plain", "subject_prompt": "a photo of <S*> mailbox"},
        {"id": "red", "subject_prompt": "a photo of <S*> mailbox in red color"},
        {"id": "blue", "subject_prompt": "a photo of <S*> mailbox in blue color"},
    ] and comparison.get("conditions") == list(CONDITIONS)
        and comparison.get("literal_suffix") == " made of metal"
        and comparison.get("material_token_suffix") == " made of <M*>"
        and comparison.get("sampling") == {
            "seeds": [42, 43, 44, 45, 46],
            "num_inference_steps": 100,
            "guidance_scale": 3.5,
        }, "comparison matrix differs")
    transfer = protocol.get("transfer", {})
    require(transfer.get("row_count") == 140 and transfer.get("prompt_count") == 28
            and transfer.get("source_checkpoint_id") == "token-local-kv-5000"
            and transfer.get("source_checkpoint_steps") == 5000
            and transfer.get("sampling") == comparison["sampling"],
            "transfer protocol differs")
    require(protocol.get("safety_checker") == "enabled", "safety checker differs")


def verify_subject(protocol: dict, run_root: Path) -> tuple[Path, dict]:
    spec = protocol["subject"]
    run = (run_root / spec["run_relative_to_COLORPEEL_RUN_ROOT"]).resolve()
    require(run.parent == run_root / spec["training_study"], "Subject run study differs")
    manifest_path = run / "manifest.json"
    require(sha256(manifest_path) == spec["training_manifest_sha256"],
            "Subject training manifest changed")
    manifest = read_json(manifest_path)
    expected_run = {"study": spec["training_study"], "variant": spec["variant"], "seed": 42}
    require(manifest.get("status") == "succeeded" and manifest.get("returncode") == 0
            and manifest.get("stage") == "train" and manifest.get("run") == expected_run
            and manifest.get("lora_training_data", {}).get("row_count") == 5,
            "Subject training did not succeed as specified")
    config = read_config(run / "config.yaml")
    args = config.get("args", {})
    require(config.get("status") == "authorized_lora_subject_matte5_r64"
            and config.get("run") == expected_run
            and args.get("subject_lora_mode") == spec["mode"]
            and args.get("subject_lora_rank") == spec["rank"]
            and args.get("subject_lora_alpha") == spec["alpha"]
            and args.get("kv_learning_rate") == spec["kv_learning_rate"]
            and args.get("max_train_steps") == 3000
            and args.get("checkpointing_steps") == 1000,
            "Subject training config differs")
    final = run / "checkpoints"
    adaptation = read_json(final / "adaptation_config.json")
    require(adaptation.get("adaptation_mode") == "lora_subject_kv"
            and adaptation.get("mode") == "token_local_kv"
            and adaptation.get("rank") == 64 and adaptation.get("alpha") == 64
            and adaptation.get("modifier_tokens") == ["<S*>"]
            and adaptation.get("weight_name") == WEIGHT_NAME,
            "Subject adaptation config differs")
    return final, {"run": run, "manifest_sha256": sha256(manifest_path)}


def verify_material(protocol: dict, material_id: str, run_root: Path) -> tuple[Path, dict]:
    require(material_id in MATERIAL_IDS, f"unsupported Material: {material_id}")
    spec = protocol["materials"][material_id]
    run = (run_root / spec["run_relative_to_COLORPEEL_RUN_ROOT"]).resolve()
    require(run.parent == run_root / "natural_material_fullmaps_lora_r64_a64_lr_step_ablation_v1",
            "Material run study differs")
    manifest_path = run / "manifest.json"
    require(sha256(manifest_path) == spec["training_manifest_sha256"],
            f"{material_id} training manifest changed")
    manifest = read_json(manifest_path)
    expected_run = {
        "study": "natural_material_fullmaps_lora_r64_a64_lr_step_ablation_v1",
        "variant": spec["variant"], "seed": 42,
    }
    require(manifest.get("status") == "succeeded" and manifest.get("returncode") == 0
            and manifest.get("stage") == "train" and manifest.get("run") == expected_run,
            f"{material_id} training did not succeed as specified")
    config = read_config(run / "config.yaml")
    args = config.get("args", {})
    require(config.get("run") == expected_run
            and config.get("status")
            == "authorized_natural_material_fullmaps_lora_r64_a64_lr_step_ablation"
            and config.get("source_lock", {}).get("material_id") == material_id
            and args.get("material_lora_mode") == "token_local_kv"
            and args.get("material_lora_rank") == 64
            and args.get("material_lora_alpha") == 64
            and args.get("kv_learning_rate") == 5.0e-5
            and args.get("max_train_steps") == 3000
            and args.get("checkpoint_steps") == [600, 1000, 2000, 3000],
            f"{material_id} training config differs")
    final = run / "checkpoints"
    adaptation = read_json(final / "adaptation_config.json")
    require(adaptation.get("adaptation_mode") == "lora_material_kv"
            and adaptation.get("mode") == "token_local_kv"
            and adaptation.get("rank") == 64 and adaptation.get("alpha") == 64
            and adaptation.get("modifier_tokens") == ["<M*>"]
            and adaptation.get("weight_name") == WEIGHT_NAME,
            f"{material_id} adaptation config differs")
    return final, {"run": run, "manifest_sha256": sha256(manifest_path)}


def derive_snapshot(final: Path, token: str, step: int, expected: dict,
                    output: Path | None, verify_final: bool = False) -> dict:
    import torch

    audit = read_json(final / "embedding_update_audit.json")
    records = [item for item in audit["modifier_tokens"] if item["token"] == token]
    require(len(records) == 1, f"embedding audit differs for {token}")
    token_id = records[0]["token_id"]
    adapter_path = final / f"checkpoint-{step}" / "pytorch_model.bin"
    encoder_path = final / f"checkpoint-{step}" / "pytorch_model_1.bin"
    require(sha256(adapter_path) == expected["adapter"]
            and sha256(encoder_path) == expected["text_encoder"],
            f"source snapshot changed for {token} step {step}")
    adapter = torch.load(adapter_path, map_location="cpu")
    encoder = torch.load(encoder_path, map_location="cpu")
    embedding = encoder["text_model.embeddings.token_embedding.weight"][token_id].clone().contiguous()
    final_adapter = torch.load(final / WEIGHT_NAME, map_location="cpu")
    final_token = torch.load(final / f"{token}.bin", map_location="cpu")[token]
    require(set(adapter) == set(final_adapter)
            and all(adapter[key].shape == final_adapter[key].shape for key in adapter)
            and embedding.shape == final_token.shape,
            f"snapshot schema differs for {token} step {step}")
    if verify_final:
        require(torch.equal(embedding, final_token)
                and all(torch.equal(adapter[key], final_adapter[key]) for key in adapter),
                f"final snapshot differs for {token}")
    record = {"source_adapter_sha256": sha256(adapter_path),
              "source_text_encoder_sha256": sha256(encoder_path)}
    if output is not None:
        output.mkdir(parents=True)
        torch.save(adapter, output / WEIGHT_NAME)
        torch.save({token: embedding}, output / f"{token}.bin")
        shutil.copyfile(final / "adaptation_config.json", output / "adaptation_config.json")
        record.update({"path": str(output),
                       "derived_adapter_sha256": sha256(output / WEIGHT_NAME),
                       "derived_token_sha256": sha256(output / f"{token}.bin")})
    return record


def comparison_rows(protocol: dict, subject_step: int) -> list[dict]:
    require(subject_step in SUBJECT_STEPS, "unsupported Subject snapshot")
    rows = []
    for group in protocol["comparison"]["groups"]:
        subject = group["subject_prompt"]
        prompts = {
            "subject_only": subject,
            "subject_literal_metal": subject + protocol["comparison"]["literal_suffix"],
            "subject_material_token_metal_spoon": (
                subject + protocol["comparison"]["material_token_suffix"]),
            "subject_material_token_wood_spoon": (
                subject + protocol["comparison"]["material_token_suffix"]),
        }
        for condition in CONDITIONS:
            material_id = next((item for item in MATERIAL_IDS if condition.endswith(item)), None)
            prompt = prompts[condition]
            require(prompt.count("<S*>") == 1
                    and prompt.count("<M*>") == int(material_id is not None),
                    f"modifier token count differs: {condition}")
            for seed in protocol["comparison"]["sampling"]["seeds"]:
                sample_id = f"s{subject_step}__{group['id']}__{condition}__seed{seed}"
                rows.append({
                    "id": sample_id, "family": "comparison", "subject_step": subject_step,
                    "group": group["id"], "condition": condition, "material_id": material_id,
                    "material_step": 1000 if material_id else None,
                    "prompt": prompt, "seed": seed,
                    "image_path": f"images/{group['id']}/{condition}/seed{seed}.png",
                })
    require(len(rows) == 60 and len({row["id"] for row in rows}) == 60
            and len({row["image_path"] for row in rows}) == 60,
            "comparison matrix differs")
    return rows


def load_transfer_rows(protocol: dict, run_root: Path) -> tuple[Path, list[dict], dict]:
    spec = protocol["transfer"]
    path = (run_root / spec["prompt_manifest_relative_to_COLORPEEL_RUN_ROOT"]).resolve()
    require(sha256(path) == spec["prompt_manifest_sha256"], "transfer prompt manifest changed")
    source_run = path.parent.parent
    require(sha256(source_run / "manifest.json") == spec["source_run_manifest_sha256"]
            and sha256(path.parent / "provenance.json") == spec["source_provenance_sha256"],
            "transfer source provenance changed")
    source_manifest = read_json(source_run / "manifest.json")
    require(source_manifest.get("status") == "succeeded"
            and source_manifest.get("returncode") == 0,
            "transfer source run did not succeed")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    require(len(rows) == spec["row_count"]
            and all(set(row) == SOURCE_FIELDS for row in rows),
            "transfer source schema differs")
    require(len({row["id"] for row in rows}) == len(rows)
            and len({row["image_path"] for row in rows}) == len(rows),
            "transfer source ids or paths are not unique")
    pairs: dict[tuple[str, str], set[int]] = {}
    sampling = spec["sampling"]
    for row in rows:
        require(row["checkpoint_id"] == spec["source_checkpoint_id"]
                and row["checkpoint_steps"] == spec["source_checkpoint_steps"]
                and row["sampling_id"] is None
                and row["seed"] in sampling["seeds"]
                and row["num_inference_steps"] == sampling["num_inference_steps"]
                and row["guidance_scale"] == sampling["guidance_scale"]
                and row["prompt"].count("<S*>") == 1
                and "<M*>" not in row["prompt"] and "<C*>" not in row["prompt"],
                f"transfer source row differs: {row.get('id')}")
        pairs.setdefault((row["color"], row["prompt"]), set()).add(row["seed"])
    require(len(pairs) == spec["prompt_count"]
            and all(seeds == set(sampling["seeds"]) for seeds in pairs.values()),
            "transfer prompt/seed grid differs")
    return path, rows, {
        "prompt_manifest_sha256": sha256(path),
        "source_run_manifest_sha256": sha256(source_run / "manifest.json"),
        "source_provenance_sha256": sha256(path.parent / "provenance.json"),
    }


def transfer_rows(source_rows: list[dict], subject_step: int) -> list[dict]:
    rows = []
    for index, source in enumerate(source_rows):
        rows.append({
            "id": f"s{subject_step}__transfer__{index:03d}", "family": "transfer",
            "subject_step": subject_step, "source_id": source["id"],
            "source_color": source["color"], "prompt": source["prompt"],
            "seed": source["seed"], "num_inference_steps": source["num_inference_steps"],
            "guidance_scale": source["guidance_scale"],
            "image_path": f"images/{index:03d}__{source['color']}__seed{source['seed']}.png",
        })
    require(len(rows) == 140 and len({row["id"] for row in rows}) == 140
            and len({row["image_path"] for row in rows}) == 140,
            "transfer replay matrix differs")
    return rows


def subject_mask(pipe, prompt: str, guidance_scale: float) -> dict:
    import torch

    ids = pipe.tokenizer(
        prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(pipe.unet.device)
    mask = ids == pipe.tokenizer.convert_tokens_to_ids("<S*>")
    require(int(mask.sum()) == prompt.count("<S*>"), f"Subject tokenization changed: {prompt}")
    if guidance_scale > 1:
        mask = torch.cat([torch.zeros_like(mask), mask], dim=0)
    return {"modifier_token_mask": mask}


def load_pipeline(base_model: str, subject: Path, device: str, material: Path | None = None):
    import torch
    from diffusers import DiffusionPipeline, PNDMScheduler
    from experiments.lora_kv_subject_v1.attention import install_subject_lora_kv

    train_root = str(REPO_ROOT / "src" / "train")
    if train_root not in sys.path:
        sys.path.insert(0, train_root)
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(
        base_model, subfolder="unet", local_files_only=True, torch_dtype=torch.float16)
    subject_state = torch.load(subject / WEIGHT_NAME, map_location="cpu")
    if material is None:
        install_subject_lora_kv(unet, subject_state, "token_local_kv", 64, 64)
    else:
        from experiments.lora_kv_material_v1.attention import install_dual_token_local_lora_kv
        material_state = torch.load(material / WEIGHT_NAME, map_location="cpu")
        install_dual_token_local_lora_kv(unet, subject_state, material_state, 64, 64)
    pipe = DiffusionPipeline.from_pretrained(
        base_model, unet=unet, low_cpu_mem_usage=False,
        torch_dtype=torch.float16, local_files_only=True).to(device)
    require(pipe.safety_checker is not None and isinstance(pipe.scheduler, PNDMScheduler),
            "inference requires the base safety checker and PNDM scheduler")
    pipe.load_textual_inversion(str(subject), weight_name="<S*>.bin")
    if material is not None:
        pipe.load_textual_inversion(str(material), weight_name="<M*>.bin")
    return pipe


def generate(pipe, rows: list[dict], output: Path, ledger, device: str,
             sampling: dict, composition: bool) -> None:
    import torch

    for row in rows:
        steps = row.get("num_inference_steps", sampling["num_inference_steps"])
        guidance = row.get("guidance_scale", sampling["guidance_scale"])
        masks = token_masks(pipe, row["prompt"], guidance) if composition else subject_mask(
            pipe, row["prompt"], guidance)
        result = pipe(
            row["prompt"], num_inference_steps=steps, guidance_scale=guidance,
            generator=torch.Generator(device=device).manual_seed(row["seed"]),
            cross_attention_kwargs=masks,
        )
        target = output / row["image_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(target)
        flags = getattr(result, "nsfw_content_detected", None)
        require(flags is None or len(flags) == 1, "expected one safety-checker result")
        filtered = bool(flags[0]) if flags is not None else False
        ledger.write(json.dumps({
            **row, "status": "safety_filtered" if filtered else "ok",
            "image_sha256": sha256(target), "nsfw_content_detected": filtered,
        }, sort_keys=True) + "\n")
        ledger.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--task", choices=("comparison", "transfer"), required=True)
    parser.add_argument("--subject-step", type=int, choices=SUBJECT_STEPS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol = read_json(args.protocol)
    validate_protocol(protocol)
    run_root = Path(os.environ["COLORPEEL_RUN_ROOT"]).resolve()
    subject_final, subject_info = verify_subject(protocol, run_root)
    subject_expected = protocol["subject"]["source_snapshot_sha256"][str(args.subject_step)]
    subject_audit = derive_snapshot(
        subject_final, "<S*>", args.subject_step, subject_expected, None,
        verify_final=args.subject_step == SUBJECT_STEPS[-1])
    material_inputs = {}
    source_info = None
    if args.task == "comparison":
        rows = comparison_rows(protocol, args.subject_step)
        for material_id in MATERIAL_IDS:
            final, info = verify_material(protocol, material_id, run_root)
            spec = protocol["materials"][material_id]
            expected = {"adapter": spec["source_adapter_sha256"],
                        "text_encoder": spec["source_text_encoder_sha256"]}
            audit = derive_snapshot(final, "<M*>", spec["snapshot_step"], expected, None)
            material_inputs[material_id] = {"final": final, "info": info, "audit": audit}
        sampling = protocol["comparison"]["sampling"]
    else:
        source_path, source_rows, source_info = load_transfer_rows(protocol, run_root)
        rows = transfer_rows(source_rows, args.subject_step)
        sampling = protocol["transfer"]["sampling"]
    if args.dry_run:
        print(f"verified {args.task} step {args.subject_step}: {len(rows)} rows")
        return

    output = args.output.resolve()
    protected = [subject_info["run"]]
    protected.extend(item["info"]["run"] for item in material_inputs.values())
    if args.task == "transfer":
        protected.append(source_path.parent.parent)
    require(all(output != path and not output.is_relative_to(path) for path in protected),
            "output must not be inside an input run")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    derived_subject = derive_snapshot(
        subject_final, "<S*>", args.subject_step, subject_expected,
        output / "derived_subject", verify_final=args.subject_step == SUBJECT_STEPS[-1])
    derived_material = {}
    for material_id, item in material_inputs.items():
        spec = protocol["materials"][material_id]
        expected = {"adapter": spec["source_adapter_sha256"],
                    "text_encoder": spec["source_text_encoder_sha256"]}
        derived_material[material_id] = derive_snapshot(
            item["final"], "<M*>", spec["snapshot_step"], expected,
            output / "derived_material" / material_id)
    with (output / "generation_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    provenance = {
        "schema": protocol["schema"], "status": "running", "task": args.task,
        "subject_step": args.subject_step, "protocol_sha256": sha256(args.protocol),
        "subject_run": str(subject_info["run"]),
        "subject_manifest_sha256": subject_info["manifest_sha256"],
        "subject_snapshot": derived_subject,
        "material_inputs": {
            key: {"run": str(value["info"]["run"]),
                  "manifest_sha256": value["info"]["manifest_sha256"],
                  "snapshot": derived_material[key]}
            for key, value in material_inputs.items()
        },
        "transfer_source": source_info, "row_count": len(rows),
        "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    import torch

    subject_checkpoint = Path(derived_subject["path"])
    with (output / "generation_status.jsonl").open("w", encoding="utf-8") as ledger:
        if args.task == "comparison":
            plain_rows = [row for row in rows if row["material_id"] is None]
            pipe = load_pipeline(protocol["base_model"], subject_checkpoint, args.device)
            generate(pipe, plain_rows, output, ledger, args.device, sampling, False)
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            for material_id in MATERIAL_IDS:
                pipe = load_pipeline(
                    protocol["base_model"], subject_checkpoint,
                    args.device, Path(derived_material[material_id]["path"]))
                generate(pipe, [row for row in rows if row["material_id"] == material_id],
                         output, ledger, args.device, sampling, True)
                del pipe
                gc.collect()
                torch.cuda.empty_cache()
        else:
            pipe = load_pipeline(protocol["base_model"], subject_checkpoint, args.device)
            generate(pipe, rows, output, ledger, args.device, sampling, False)
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
    require(derive_snapshot(
        subject_final, "<S*>", args.subject_step, subject_expected, None,
        verify_final=args.subject_step == SUBJECT_STEPS[-1]) == subject_audit,
        "Subject snapshot changed during inference")
    if args.task == "transfer":
        require(load_transfer_rows(protocol, run_root)[2] == source_info,
                "transfer source changed during inference")
    else:
        for material_id, item in material_inputs.items():
            spec = protocol["materials"][material_id]
            expected = {"adapter": spec["source_adapter_sha256"],
                        "text_encoder": spec["source_text_encoder_sha256"]}
            require(derive_snapshot(item["final"], "<M*>", 1000, expected, None)
                    == item["audit"], f"{material_id} snapshot changed during inference")
    provenance["status"] = "succeeded"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"generated {len(rows)} {args.task} images: {output}")


if __name__ == "__main__":
    main()
