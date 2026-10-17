"""Load completed SD1.5 full-rank or rank64 LoRA runs without crossing bases."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.sd15_token_local_kv_v1.contract import (
    PROTOCOL, read_json, sha256, validate_config,
)
from experiments.sd_native_comparison_v1.generate import (
    git_info, output_directory, package_versions, write_json,
)


def verify_run(run, token):
    manifest = read_json(run / "manifest.json")
    from scripts.launch.colorpeel_run import read_config
    config = read_config(run / "config.yaml")
    validate_config(config, {}, verify_assets=False)
    if (manifest.get("status") != "succeeded" or manifest.get("returncode") != 0
            or manifest.get("stage") != "train" or manifest.get("run") != config["run"]
            or not manifest.get("sd15_migration_inputs", {}).get("assets_verified")
            or config["args"]["modifier_token"] != token):
        raise ValueError("expected a completed SD1.5 run for the requested token")
    checkpoint = run / "checkpoints"
    adaptation = read_json(checkpoint / "adaptation_config.json")
    method = "fullrank" if config["args"].get("token_local_kv") else "lora_r64"
    expected_mode = "token_local_kv" if method == "fullrank" else (
        "lora_subject_kv" if token == "<S*>" else "lora_material_kv")
    if (adaptation.get("adaptation_mode") != expected_mode
            or adaptation.get("modifier_tokens") != [token]
            or (method == "lora_r64" and any(adaptation.get(k) != v for k, v in
                {"rank": 64, "alpha": 64, "mode": "token_local_kv"}.items()))):
        raise ValueError("adapter method, token, rank or alpha differs")
    weight = ("pytorch_token_local_kv_weights.bin" if method == "fullrank"
              else "pytorch_lora_kv_weights.bin")
    files = [run / "manifest.json", run / "config.yaml", checkpoint / weight,
             checkpoint / "adaptation_config.json", checkpoint / f"{token}.bin"]
    return {"run": str(run), "checkpoint": str(checkpoint), "method": method,
            "token": token, "weight": weight,
            "training_steps": config["args"]["max_train_steps"],
            "smoke": config["sd15_migration"]["smoke"],
            "hashes": {str(p): sha256(p) for p in files}}


def load_pipeline(selections, device):
    import torch
    from diffusers import DiffusionPipeline, PNDMScheduler
    train_root = str(ROOT / "src/train")
    sys.path.insert(0, train_root)
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel

    methods = {s["method"] for s in selections}
    if selections and len(methods) != 1:
        raise ValueError("full-rank and LoRA adapters require separate comparisons")
    model = read_json(PROTOCOL)["model"]["root"]
    receipt = read_json(Path(model) / "model_source.json")
    if receipt["model"] != read_json(PROTOCOL)["model"] or any(
            sha256(Path(model) / name) != value for name, value in receipt["files"].items()):
        raise ValueError("SD1.5 model receipt changed")
    unet = UNet2DConditionModel.from_pretrained(
        model, subfolder="unet", local_files_only=True, torch_dtype=torch.float16)
    if not selections:
        pipe = DiffusionPipeline.from_pretrained(
            model, unet=unet, low_cpu_mem_usage=False, local_files_only=True,
            torch_dtype=torch.float16).to(device)
        if pipe.safety_checker is None or not isinstance(pipe.scheduler, PNDMScheduler):
            raise ValueError("expected shared safety checker and PNDM scheduler")
        return pipe
    first = selections[0]
    if first["method"] == "fullrank":
        unet.load_attn_procs(first["checkpoint"], weight_name=first["weight"],
                             adaptation_mode="token_local_kv")
        if len(selections) == 2:
            from src.methods.colorpeel_ice.dual_token_local_kv import install_dual_token_local_kv
            material = selections[1]
            install_dual_token_local_kv(unet, torch.load(
                Path(material["checkpoint"]) / material["weight"], map_location="cpu"),
                primary_label="subject")
    else:
        states = [torch.load(Path(s["checkpoint"]) / s["weight"], map_location="cpu")
                  for s in selections]
        if len(states) == 2:
            from experiments.lora_kv_material_v1.attention import install_dual_token_local_lora_kv
            install_dual_token_local_lora_kv(unet, *states, rank=64, alpha=64)
        else:
            from experiments.lora_kv_subject_v1.attention import install_subject_lora_kv
            install_subject_lora_kv(unet, states[0], "token_local_kv", 64, 64)
    pipe = DiffusionPipeline.from_pretrained(
        model, unet=unet, low_cpu_mem_usage=False, local_files_only=True,
        torch_dtype=torch.float16).to(device)
    if pipe.safety_checker is None or not isinstance(pipe.scheduler, PNDMScheduler):
        raise ValueError("expected shared safety checker and PNDM scheduler")
    for s in selections:
        pipe.load_textual_inversion(s["checkpoint"], weight_name=s["token"] + ".bin")
    return pipe


def attention_masks(pipe, prompt, tokens, guidance):
    import torch
    ids = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids.to(pipe.unet.device)
    masks = {}
    for token in tokens:
        mask = ids == pipe.tokenizer.convert_tokens_to_ids(token)
        if int(mask.sum()) != prompt.count(token) or prompt.count(token) > 1:
            raise ValueError("modifier was repeated or truncated")
        if guidance > 1:
            mask = torch.cat([torch.zeros_like(mask), mask])
        masks["subject" if token == "<S*>" else "material"] = mask
    return {"modifier_token_mask": masks if len(tokens) == 2 else next(iter(masks.values()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-run", type=Path)
    parser.add_argument("--material-run", type=Path)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.subject_run is None and args.material_run is None:
        parser.error("provide at least one completed training run")
    revision = git_info()
    selections = [verify_run(run.resolve(), token) for run, token in (
        (args.subject_run, "<S*>"), (args.material_run, "<M*>")) if run is not None]
    if len({s["method"] for s in selections}) != 1 or len({s["smoke"] for s in selections}) != 1:
        raise ValueError("do not mix methods or smoke/full training runs")
    source = [json.loads(line) for line in args.prompt_manifest.read_text(encoding="utf-8").splitlines()]
    tokens = [s["token"] for s in selections]
    rows = []
    for index, row in enumerate(source):
        if (not isinstance(row.get("prompt"), str) or type(row.get("seed")) is not int
                or any(t in row["prompt"] for t in {"<S*>", "<M*>"} - set(tokens))
                or row.get("num_inference_steps", 100) != 100
                or row.get("guidance_scale", 3.5) != 3.5):
            raise ValueError("prompt tokens, seed or sampling differ")
        rows.append({"id": f"sample_{index:04d}", "source_id": row.get("id"),
                     "prompt": row["prompt"], "seed": row["seed"],
                     "image_path": f"images/sample_{index:04d}.png"})
    if not rows:
        raise ValueError("prompt manifest must contain at least one row")
    output = output_directory(args.output_dir)
    output.mkdir(parents=True)
    provenance = {"status": "dry_run" if args.dry_run else "running", "git": revision,
                  "training": selections, "protocol_sha256": sha256(PROTOCOL),
                  "prompt_manifest": str(args.prompt_manifest.resolve()),
                  "prompt_manifest_sha256": sha256(args.prompt_manifest),
                  "sampling": {"steps": 100, "guidance_scale": 3.5, "resolution": 512},
                  "row_count": len(rows), "environment": package_versions()}
    write_json(output / "provenance.json", provenance)
    (output / "generation_manifest.jsonl").write_text("".join(
        json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    if args.dry_run:
        return
    try:
        import torch
        pipe = load_pipeline(selections, args.device)
        with (output / "generation_status.jsonl").open("w", encoding="utf-8") as ledger:
            for row in rows:
                result = pipe(row["prompt"], num_inference_steps=100, guidance_scale=3.5,
                              width=512, height=512,
                              generator=torch.Generator(device=args.device).manual_seed(row["seed"]),
                              cross_attention_kwargs=attention_masks(pipe, row["prompt"], tokens, 3.5))
                image = output / row["image_path"]
                image.parent.mkdir(exist_ok=True)
                result.images[0].save(image)
                if result.nsfw_content_detected is None or len(result.nsfw_content_detected) != 1:
                    raise ValueError("missing safety checker result")
                filtered = bool(result.nsfw_content_detected[0])
                ledger.write(json.dumps({**row, "status": "safety_filtered" if filtered else "ok",
                                         "image_sha256": sha256(image)}) + "\n")
                ledger.flush()
        if any(sha256(Path(p)) != digest for s in selections for p, digest in s["hashes"].items()):
            raise ValueError("training artifacts changed during inference")
        provenance["status"] = "succeeded"
    except Exception as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "provenance.json", provenance)


if __name__ == "__main__":
    main()
