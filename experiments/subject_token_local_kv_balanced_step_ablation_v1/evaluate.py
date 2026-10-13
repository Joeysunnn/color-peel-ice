"""Evaluate five balanced Subject token-local K/V snapshots."""

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

from scripts.launch.colorpeel_run import read_config
from scripts.methods.colorpeel_ice.generate_color_material_diagnostic import verify_checkpoint
from scripts.methods.colorpeel_ice.generate_subject_material_diagnostic import attention_masks


WEIGHTS = "pytorch_token_local_kv_weights.bin"
SUBJECT_STEPS = (300, 500, 700, 1000, 2000)
CONDITIONS = ("subject_only", "material_only", "subject_material", "literal_metal")
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
    require(
        protocol.get("schema") == "subject_token_local_kv_balanced_step_ablation_inference/v1",
        "inference protocol schema differs",
    )
    require(protocol.get("base_model") == "CompVis/stable-diffusion-v1-4", "base model differs")
    subject = protocol.get("subject", {})
    require(
        subject.get("study") == "subject_token_local_kv_balanced_step_ablation_v1"
        and subject.get("variant") == "balanced_aligned_token_local_kv_steps300_500_700_1000_2000"
        and subject.get("source_config_path")
        == "experiments/subject_token_local_kv_balanced_step_ablation_v1/configs/train.yaml"
        and subject.get("source_config_sha256")
        == "b2007c4f46d98c5dbdb05c647a9e3df0be9d20a7df1929fb38c7239b72f61fa3"
        and subject.get("training_protocol_path")
        == "experiments/subject_token_local_kv_balanced_step_ablation_v1/protocols/training_v1.json"
        and subject.get("training_protocol_sha256")
        == "508f11235bbd0a1b6edd0b5ac33e46c618e907476c6d7bfee31a1e6aac726fdf"
        and subject.get("seed") == 42
        and subject.get("snapshot_steps") == list(SUBJECT_STEPS)
        and subject.get("mode") == "token_local_kv"
        and subject.get("learning_rate") == 1.0e-5
        and subject.get("kv_learning_rate") == 1.0e-5
        and subject.get("max_train_steps") == 2000,
        "Subject protocol differs",
    )
    require(
        sha256(REPO_ROOT / subject["source_config_path"]) == subject["source_config_sha256"]
        and sha256(REPO_ROOT / subject["training_protocol_path"])
        == subject["training_protocol_sha256"],
        "Subject source config or training protocol changed",
    )
    material = protocol.get("material", {})
    require(
        material.get("selection_path")
        == "experiments/material_token_local_pilot_v1/configs/selected_material_source.json"
        and material.get("selection_sha256")
        == "1aefb6c8d941e73b369cfee2301429046c970c8786158b94e9b9452289bd1d90"
        and material.get("snapshot_step") == 5000
        and material.get("adaptation_mode") == "token_local_kv",
        "Material selection differs",
    )
    comparison = protocol.get("comparison", {})
    require(
        [group.get("id") for group in comparison.get("groups", [])]
        == ["red", "blue", "city_street"]
        and comparison.get("conditions") == list(CONDITIONS)
        and comparison.get("sampling")
        == {"seeds": [42, 43, 44], "num_inference_steps": 100, "guidance_scale": 3.5},
        "Subject/Material comparison matrix differs",
    )
    source_comparison_path = REPO_ROOT / comparison.get("source_protocol", "")
    require(
        comparison.get("source_protocol")
        == "experiments/subject_material_composition_v1/protocols/mailbox_metal_diagnostic_v1.json"
        and comparison.get("source_protocol_sha256")
        == "2c660f047ececb9b602fdf94284ff4c11db62a68fb3e82f1d2ad0fabc7893373"
        and sha256(source_comparison_path) == comparison["source_protocol_sha256"],
        "source Subject/Material protocol changed",
    )
    source_comparison = read_json(source_comparison_path)
    require(
        comparison["groups"] == source_comparison.get("groups")
        and comparison["sampling"] == source_comparison.get("sampling"),
        "Subject/Material prompts differ from the source protocol",
    )
    transfer = protocol.get("transfer", {})
    require(
        transfer.get("row_count") == 140
        and transfer.get("prompt_count") == 28
        and transfer.get("source_checkpoint_id") == "token-local-kv-5000"
        and transfer.get("source_checkpoint_steps") == 5000
        and transfer.get("sampling")
        == {"seeds": [42, 43, 44, 45, 46], "num_inference_steps": 100, "guidance_scale": 3.5},
        "transfer protocol differs",
    )
    require(protocol.get("safety_checker") == "enabled", "safety checker differs")


def verify_subject(protocol: dict, run_root: Path, requested_run: Path) -> tuple[Path, dict]:
    spec = protocol["subject"]
    run = requested_run.resolve()
    require(run.parent == run_root / spec["study"], "Subject run is outside the locked study")
    manifest_path = run / "manifest.json"
    manifest = read_json(manifest_path)
    expected_run = {"study": spec["study"], "variant": spec["variant"], "seed": spec["seed"]}
    require(
        manifest.get("status") == "succeeded"
        and manifest.get("returncode") == 0
        and manifest.get("stage") == "train"
        and manifest.get("run") == expected_run,
        "Subject training did not succeed as specified",
    )
    require(
        manifest.get("subject_training_data") == {
            "cohort": "balanced_aligned",
            "concepts_sha256": "109e2b336bde5a465db4cf9a29cc578f83cda48899fe2f91b4956aafcc0010f3",
            "asset_manifest_sha256": "05bd9f15dbca82a33531dae0cd9620a14554200556c384e53fbc64bc2e782e90",
            "row_count": 10,
        },
        "Subject training data receipt differs",
    )
    config_path = run / "config.yaml"
    config = read_config(config_path)
    source_config = read_config(REPO_ROOT / spec["source_config_path"])
    args = config.get("args", {})
    require(
        config == source_config
        and config.get("status") == "authorized_subject_token_local_kv_balanced_step_ablation"
        and config.get("run") == expected_run
        and args.get("pretrained_model_name_or_path") == protocol["base_model"]
        and args.get("token_local_kv") is True
        and args.get("modifier_token") == "<S*>"
        and args.get("initializer_token") == "mailbox"
        and args.get("learning_rate") == spec["learning_rate"]
        and args.get("kv_learning_rate") == spec["kv_learning_rate"]
        and args.get("max_train_steps") == spec["max_train_steps"]
        and args.get("checkpoint_steps") == spec["snapshot_steps"],
        "Subject training config differs",
    )
    final = run / "checkpoints"
    adaptation = read_json(final / "adaptation_config.json")
    audit = read_json(final / "embedding_update_audit.json")
    require(
        adaptation.get("adaptation_mode") == "token_local_kv"
        and adaptation.get("weight_name") == WEIGHTS
        and adaptation.get("modifier_tokens") == ["<S*>"]
        and len(audit.get("modifier_tokens", [])) == 1
        and audit["modifier_tokens"][0].get("token") == "<S*>",
        "Subject final checkpoint differs",
    )
    for step in SUBJECT_STEPS:
        snapshot = final / f"checkpoint-{step}"
        for name in ("pytorch_model.bin", "pytorch_model_1.bin"):
            require((snapshot / name).is_file(), f"missing Subject snapshot file: {snapshot / name}")
    return final, {
        "run": run,
        "manifest_sha256": sha256(manifest_path),
        "config_sha256": sha256(config_path),
        "token_id": audit["modifier_tokens"][0]["token_id"],
    }


def verify_material(protocol: dict, run_root: Path) -> tuple[Path, dict]:
    spec = protocol["material"]
    selection_path = REPO_ROOT / spec["selection_path"]
    require(sha256(selection_path) == spec["selection_sha256"], "Material selection changed")
    selection = read_json(selection_path)
    checkpoint = selection.get("checkpoint", {})
    require(
        checkpoint.get("step") == spec["snapshot_step"]
        and checkpoint.get("adaptation_mode") == spec["adaptation_mode"]
        and checkpoint.get("modifier_token") == "<M*>",
        "selected Material checkpoint differs",
    )
    run = (run_root / selection["training_run_relative_to_COLORPEEL_RUN_ROOT"]).resolve()
    manifest_path = run / "manifest.json"
    require(sha256(manifest_path) == selection["training_manifest_sha256"], "Material manifest changed")
    manifest = read_json(manifest_path)
    require(
        manifest.get("status") == "succeeded"
        and manifest.get("returncode") == 0
        and manifest.get("stage") == "train",
        "Material training did not succeed",
    )
    directory = run / checkpoint["directory_relative_to_training_run"]
    verify_checkpoint(directory, {
        "weights": checkpoint["token_local_kv_weights_sha256"],
        "adaptation": checkpoint["adaptation_config_sha256"],
        "token": checkpoint["token_embedding_sha256"],
    }, "<M*>")
    return directory, {
        "run": run,
        "selection_sha256": sha256(selection_path),
        "manifest_sha256": sha256(manifest_path),
        "weights_sha256": sha256(directory / WEIGHTS),
        "token_sha256": sha256(directory / "<M*>.bin"),
    }


def derive_subject_snapshot(final: Path, step: int, token_id: int, output: Path | None) -> dict:
    import torch

    require(step in SUBJECT_STEPS, "unsupported Subject snapshot")
    final_adapter = torch.load(final / WEIGHTS, map_location="cpu")
    tensor_keys = {key for key, value in final_adapter.items() if isinstance(value, torch.Tensor)}
    empty = {key: value for key, value in final_adapter.items() if not isinstance(value, torch.Tensor)}
    require(
        len(tensor_keys) == 32
        and len(empty) == 16
        and all(value == {} and key.endswith("attn1.processor") for key, value in empty.items()),
        "Subject final adapter schema differs",
    )
    snapshot = final / f"checkpoint-{step}"
    adapter_path = snapshot / "pytorch_model.bin"
    encoder_path = snapshot / "pytorch_model_1.bin"
    adapter = torch.load(adapter_path, map_location="cpu")
    encoder = torch.load(encoder_path, map_location="cpu")
    require(
        set(adapter) == tensor_keys
        and all(adapter[key].shape == final_adapter[key].shape for key in adapter),
        f"Subject adapter snapshot schema differs at step {step}",
    )
    embedding = encoder["text_model.embeddings.token_embedding.weight"][token_id].clone().contiguous()
    final_embedding = torch.load(final / "<S*>.bin", map_location="cpu")["<S*>"]
    require(embedding.shape == final_embedding.shape, f"Subject embedding shape differs at step {step}")
    if step == SUBJECT_STEPS[-1]:
        require(
            torch.equal(embedding, final_embedding)
            and all(torch.equal(adapter[key], final_adapter[key]) for key in tensor_keys),
            "step-2000 snapshot differs from final Subject export",
        )
    record = {
        "source_adapter_sha256": sha256(adapter_path),
        "source_text_encoder_sha256": sha256(encoder_path),
    }
    if output is not None:
        output.mkdir(parents=True)
        adapter.update(empty)
        torch.save(adapter, output / WEIGHTS)
        torch.save({"<S*>": embedding}, output / "<S*>.bin")
        shutil.copyfile(final / "adaptation_config.json", output / "adaptation_config.json")
        record.update({
            "path": str(output),
            "derived_adapter_sha256": sha256(output / WEIGHTS),
            "derived_token_sha256": sha256(output / "<S*>.bin"),
        })
    return record


def comparison_rows(protocol: dict, subject_step: int) -> list[dict]:
    require(subject_step in SUBJECT_STEPS, "unsupported Subject snapshot")
    rows = []
    for group in protocol["comparison"]["groups"]:
        prompts = group.get("prompts", {})
        require(set(prompts) == set(CONDITIONS), f"comparison prompts differ: {group.get('id')}")
        for condition in CONDITIONS:
            prompt = prompts[condition]
            require(
                ("<S*>" in prompt) == (condition != "material_only")
                and ("<M*>" in prompt) == (condition in {"material_only", "subject_material"}),
                f"modifier tokens differ: {group['id']}/{condition}",
            )
            for seed in protocol["comparison"]["sampling"]["seeds"]:
                sample_id = f"s{subject_step}__{group['id']}__{condition}__seed{seed}"
                rows.append({
                    "id": sample_id,
                    "family": "subject_material_comparison",
                    "subject_step": subject_step,
                    "material_step": 5000 if "<M*>" in prompt else None,
                    "group": group["id"],
                    "condition": condition,
                    "prompt": prompt,
                    "seed": seed,
                    "image_path": f"images/{group['id']}/{condition}/seed{seed}.png",
                })
    require(
        len(rows) == 36
        and len({row["id"] for row in rows}) == 36
        and len({row["image_path"] for row in rows}) == 36,
        "comparison matrix differs",
    )
    return rows


def load_transfer_rows(protocol: dict, run_root: Path) -> tuple[Path, list[dict], dict]:
    spec = protocol["transfer"]
    path = (run_root / spec["prompt_manifest_relative_to_COLORPEEL_RUN_ROOT"]).resolve()
    require(sha256(path) == spec["prompt_manifest_sha256"], "transfer prompt manifest changed")
    source_run = path.parent.parent
    require(
        sha256(source_run / "manifest.json") == spec["source_run_manifest_sha256"]
        and sha256(path.parent / "provenance.json") == spec["source_provenance_sha256"],
        "transfer source provenance changed",
    )
    source_manifest = read_json(source_run / "manifest.json")
    require(
        source_manifest.get("status") == "succeeded" and source_manifest.get("returncode") == 0,
        "transfer source run did not succeed",
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    require(
        len(rows) == spec["row_count"]
        and all(set(row) == SOURCE_FIELDS for row in rows)
        and len({row["id"] for row in rows}) == len(rows)
        and len({row["image_path"] for row in rows}) == len(rows),
        "transfer source rows differ",
    )
    pairs: dict[tuple[str, str], set[int]] = {}
    sampling = spec["sampling"]
    for row in rows:
        require(
            row["checkpoint_id"] == spec["source_checkpoint_id"]
            and row["checkpoint_steps"] == spec["source_checkpoint_steps"]
            and row["sampling_id"] is None
            and row["seed"] in sampling["seeds"]
            and row["num_inference_steps"] == sampling["num_inference_steps"]
            and row["guidance_scale"] == sampling["guidance_scale"]
            and row["prompt"].count("<S*>") == 1
            and "<M*>" not in row["prompt"]
            and "<C*>" not in row["prompt"],
            f"transfer source row differs: {row.get('id')}",
        )
        pairs.setdefault((row["color"], row["prompt"]), set()).add(row["seed"])
    require(
        len(pairs) == spec["prompt_count"]
        and all(seeds == set(sampling["seeds"]) for seeds in pairs.values()),
        "transfer prompt/seed grid differs",
    )
    return path, rows, {
        "prompt_manifest_sha256": sha256(path),
        "source_run_manifest_sha256": sha256(source_run / "manifest.json"),
        "source_provenance_sha256": sha256(path.parent / "provenance.json"),
    }


def transfer_rows(source_rows: list[dict], subject_step: int) -> list[dict]:
    require(subject_step in SUBJECT_STEPS, "unsupported Subject snapshot")
    rows = [{
        "id": f"s{subject_step}__transfer__{index:03d}",
        "family": "transfer",
        "subject_step": subject_step,
        "source_id": source["id"],
        "source_color": source["color"],
        "prompt": source["prompt"],
        "seed": source["seed"],
        "num_inference_steps": source["num_inference_steps"],
        "guidance_scale": source["guidance_scale"],
        "image_path": f"images/{index:03d}__{source['color']}__seed{source['seed']}.png",
    } for index, source in enumerate(source_rows)]
    require(
        len(rows) == 140
        and len({row["id"] for row in rows}) == 140
        and len({row["image_path"] for row in rows}) == 140,
        "transfer replay matrix differs",
    )
    return rows


def subject_mask(pipe, prompt: str, guidance_scale: float) -> dict:
    import torch

    ids = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(pipe.unet.device)
    mask = ids == pipe.tokenizer.convert_tokens_to_ids("<S*>")
    require(int(mask.sum()) == prompt.count("<S*>"), f"Subject tokenization changed: {prompt}")
    if guidance_scale > 1:
        mask = torch.cat([torch.zeros_like(mask), mask], dim=0)
    return {"modifier_token_mask": mask}


def load_pipeline(base_model: str, subject: Path, device: str, material: Path | None = None):
    import torch
    from diffusers import DiffusionPipeline, PNDMScheduler

    train_root = str(REPO_ROOT / "src" / "train")
    if train_root not in sys.path:
        sys.path.insert(0, train_root)
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(
        base_model,
        subfolder="unet",
        local_files_only=True,
        torch_dtype=torch.float16,
    )
    unet.load_attn_procs(str(subject), weight_name=WEIGHTS, adaptation_mode="token_local_kv")
    if material is not None:
        from src.methods.colorpeel_ice.dual_token_local_kv import install_dual_token_local_kv

        material_state = torch.load(material / WEIGHTS, map_location="cpu")
        install_dual_token_local_kv(unet, material_state, primary_label="subject")
    pipe = DiffusionPipeline.from_pretrained(
        base_model,
        unet=unet,
        low_cpu_mem_usage=False,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to(device)
    require(
        pipe.safety_checker is not None and isinstance(pipe.scheduler, PNDMScheduler),
        "inference requires the base safety checker and PNDM scheduler",
    )
    pipe.load_textual_inversion(str(subject), weight_name="<S*>.bin")
    if material is not None:
        pipe.load_textual_inversion(str(material), weight_name="<M*>.bin")
    return pipe


def generate(pipe, rows: list[dict], output: Path, ledger, device: str, sampling: dict,
             composition: bool) -> None:
    import torch

    for row in rows:
        steps = row.get("num_inference_steps", sampling["num_inference_steps"])
        guidance = row.get("guidance_scale", sampling["guidance_scale"])
        masks = attention_masks(pipe, row["prompt"], guidance) if composition else subject_mask(
            pipe, row["prompt"], guidance)
        result = pipe(
            row["prompt"],
            num_inference_steps=steps,
            guidance_scale=guidance,
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
            **row,
            "status": "safety_filtered" if filtered else "ok",
            "image_sha256": sha256(target),
            "nsfw_content_detected": filtered,
        }, sort_keys=True) + "\n")
        ledger.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--subject-run", type=Path, required=True)
    parser.add_argument("--task", choices=("comparison", "transfer"), required=True)
    parser.add_argument("--subject-step", type=int, choices=SUBJECT_STEPS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol = read_json(args.protocol)
    validate_protocol(protocol)
    run_root = Path(os.environ["COLORPEEL_RUN_ROOT"]).resolve()
    subject_final, subject_info = verify_subject(protocol, run_root, args.subject_run)
    subject_audit = derive_subject_snapshot(
        subject_final, args.subject_step, subject_info["token_id"], None)
    material_dir = None
    material_info = None
    transfer_path = None
    transfer_info = None
    if args.task == "comparison":
        material_dir, material_info = verify_material(protocol, run_root)
        rows = comparison_rows(protocol, args.subject_step)
        sampling = protocol["comparison"]["sampling"]
    else:
        transfer_path, source_rows, transfer_info = load_transfer_rows(protocol, run_root)
        rows = transfer_rows(source_rows, args.subject_step)
        sampling = protocol["transfer"]["sampling"]
    if args.dry_run:
        print(f"verified {args.task} step {args.subject_step}: {len(rows)} rows")
        return

    output = args.output.resolve()
    protected = [subject_info["run"]]
    if material_info is not None:
        protected.append(material_info["run"])
    if transfer_path is not None:
        protected.append(transfer_path.parent.parent)
    require(
        all(output != path and not output.is_relative_to(path) for path in protected),
        "output must not be inside an input run",
    )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    derived_subject = derive_subject_snapshot(
        subject_final,
        args.subject_step,
        subject_info["token_id"],
        output / "derived_subject",
    )
    with (output / "generation_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    provenance = {
        "schema": protocol["schema"],
        "status": "running",
        "task": args.task,
        "subject_step": args.subject_step,
        "protocol_sha256": sha256(args.protocol),
        "subject_run": str(subject_info["run"]),
        "subject_manifest_sha256": subject_info["manifest_sha256"],
        "subject_config_sha256": subject_info["config_sha256"],
        "subject_snapshot": derived_subject,
        "material_input": (
            {**material_info, "run": str(material_info["run"])}
            if material_info is not None else None
        ),
        "transfer_source": transfer_info,
        "row_count": len(rows),
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    subject_checkpoint = Path(derived_subject["path"])
    with (output / "generation_status.jsonl").open("w", encoding="utf-8") as ledger:
        pipe = load_pipeline(protocol["base_model"], subject_checkpoint, args.device, material_dir)
        generate(pipe, rows, output, ledger, args.device, sampling, args.task == "comparison")
        del pipe
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    require(
        derive_subject_snapshot(subject_final, args.subject_step, subject_info["token_id"], None)
        == subject_audit,
        "Subject snapshot changed during inference",
    )
    if args.task == "comparison":
        require(verify_material(protocol, run_root)[1] == material_info,
                "Material source changed during inference")
    else:
        require(load_transfer_rows(protocol, run_root)[2] == transfer_info,
                "transfer source changed during inference")
    provenance["status"] = "succeeded"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"generated {len(rows)} {args.task} images: {output}")


if __name__ == "__main__":
    main()
