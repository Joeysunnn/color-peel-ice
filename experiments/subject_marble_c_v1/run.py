"""Run six independent official MARBLE edits; never create a Subject training set."""

import argparse
import gc
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.subject_marble_c_v1 import assets, quality
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("protocol.json")


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def grid(protocol):
    if (protocol["metallic"] != {"low": 20, "high": -20}
            or protocol["roughness"] != {"low": -1, "medium": 0, "high": 1}):
        raise ValueError("phase1 uses only the declared signed six-case official control grid")
    rows = []
    for metallic, m in protocol["metallic"].items():
        for roughness, r in protocol["roughness"].items():
            material = "metallic" if metallic == "high" else "non-metallic painted"
            adjective = {"low": "smooth", "medium": "moderately rough", "high": "matte rough"}[roughness]
            rows.append({"id": f"metallic_{metallic}__roughness_{roughness}",
                "metallic_state": metallic, "metallic_control": m,
                "roughness_state": roughness, "roughness_control": r,
                "caption": f"a photo of <S*> mailbox in orange color with a {adjective} {material} surface",
                "caption_status": "proposed conditioning label; actual appearance requires visual validation",
                "training_eligible": False})
    return rows


def restore_background(original, edited, mask):
    result = original.copy()
    result[mask > 0] = edited[mask > 0]
    return result


def check_locks(locks):
    if any(sha256(path) != expected for path, expected in locks.items()):
        raise ValueError("frozen source/model content changed")


def run(asset_root, output, gpu):
    asset_root, output = Path(asset_root).resolve(), resolve_run_dir(Path(output))
    if (type(gpu) is not int or gpu < 0 or output.exists() or output.is_relative_to(ROOT)
            or output.is_relative_to(asset_root) or asset_root.is_relative_to(output)):
        raise ValueError("select a GPU and fresh external output outside all assets")
    revision, protocol = git_info(), read_json(PROTOCOL)
    receipt_path = asset_root / "assets.json"
    receipt = read_json(receipt_path)
    official = Path(receipt["official"])
    if receipt["official_commit"] != assets.OFFICIAL_COMMIT or receipt["models"] != assets.MODELS:
        raise ValueError("official MARBLE model/source revisions differ")
    assets.official_state(official)
    source, mask_path = ROOT / protocol["source"], Path(protocol["mask"])
    locks = {**receipt["sha256"], str(receipt_path): sha256(receipt_path), str(PROTOCOL): sha256(PROTOCOL),
             str(source): protocol["source_sha256"], str(mask_path): protocol["mask_sha256"]}
    check_locks(locks)
    with Image.open(source) as image:
        if image.mode != "RGB" or image.size != (512, 512):
            raise ValueError("require frozen original512 RGB")
        original = image.copy()
    with Image.open(mask_path) as image:
        if image.mode != "L" or image.size != original.size or set(np.unique(image)) != {0, 255}:
            raise ValueError("require binary existing512 subject mask")
        mask = image.copy()
    rows = grid(protocol)
    output.mkdir(parents=True)
    for name in ("source", "raw", "images", "repro_outputs"):
        (output / name).mkdir()
    shutil.copyfile(source, output / "source/original.jpg")
    shutil.copyfile(mask_path, output / "source/object_mask.png")
    write(output / "plan.json", {"git": revision, "protocol": protocol, "assets_receipt": str(receipt_path),
          "source_hashes": locks, "rows": rows,
          "command": [sys.executable, "-B", "-m", "experiments.subject_marble_c_v1.run",
                      "--assets", str(asset_root), "--output", str(output), "--gpu", str(gpu)]})
    state = {"status": "initializing", "git": revision, "expected_images": 6, "generated_images": 0,
             "training_started": False, "training_eligible_images": 0, "expanded_to30": False}
    write(output / "status.json", state)
    report = output / "repro_outputs"
    (report / "COMMANDS.md").write_text(
        f"Run adapted official parametric_control.ipynb using the pinned clean MARBLE checkout.\n\n"
        f"`{sys.executable} -B -m experiments.subject_marble_c_v1.run --assets {asset_root} --output {output} --gpu {gpu}`\n",
        encoding="utf8")
    (report / "SCIENTIFIC_CHANGELOG.md").write_text(
        "MARBLE code/model semantics are unchanged. Notebook9-case demo becomes six signed M/R endpoints. "
        "Original mailbox replaces toycar; target itself replaces optional metalbowl texture reference. "
        "Explicit existing RGBmask and once-computed official DPTdepth are reused. Local pinned model paths, "
        "Torch2.5.1 and CPUoffload are declared runtime adaptations. Separate final512background composition "
        "does not replace retained raw1024outputs. No hue editing or Subject training.\n", encoding="utf8")
    (report / "COMPARABILITY_REPORT.md").write_text(
        "Relative M/R edit strengths cannot be compared numerically with prior PBR M/R. "
        "No A/B/C learned-token evaluation is run in this phase. Color drift and structural proxies are "
        "confounded by material changes, and copied background does not prove raw model invariance.\n", encoding="utf8")
    generated = []
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        sys.path.insert(0, str(official))
        import torch
        import marble
        from transformers import DPTForDepthEstimation, DPTImageProcessor

        depth_model = DPTForDepthEstimation.from_pretrained(receipt["paths"]["depth"]).to("cuda", dtype=torch.float16)
        processor = DPTImageProcessor.from_pretrained(receipt["paths"]["depth"])
        depth_model.eval()
        with torch.inference_mode():
            depth = marble.run_dpt_depth(original, depth_model, processor, device="cuda")
        depth.save(output / "source/depth.png")
        del depth_model, processor
        gc.collect()
        torch.cuda.empty_cache()
        # Adapt only local model paths; official model code and inference math stay intact.
        marble.base_model_path = receipt["paths"]["base"]
        marble.controlnet_path = receipt["paths"]["controlnet"]
        adapter = marble.setup_pipeline(device="cuda", dtype=torch.float16)
        adapter.pipe.enable_model_cpu_offload(gpu_id=0)
        mlps = {name: marble.setup_control_mlp(name, device="cuda", dtype=torch.float16)
                for name in ("metallic", "roughness")}
        parameters = {"pipeline": type(adapter.pipe).__name__, "scheduler": type(adapter.pipe.scheduler).__name__,
            "scheduler_config": dict(adapter.pipe.scheduler.config), "device": torch.cuda.get_device_name(0),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "diffusers", "transformers", "huggingface_hub", "accelerate", "numpy", "rembg")},
            "prompt": "best quality, high quality",
            "negative_prompt": "monochrome, lowres, bad anatomy, worst quality, low quality",
            "official_defaults": "strength1, ControlNet1, IPAdapter1; CFG from unchanged pipeline signature",
            "guidance_scale": inspect.signature(adapter.pipe.__call__).parameters["guidance_scale"].default,
            "mask": "RGB input mask; official resize1024; raw1024 saved before any composition",
            "memory_adaptation": "CPU offload; torch2.5.1 CUDA runtime reused, other inference package pins follow official requirements"}
        write(output / "runtime.json", parameters)
        state["status"] = "generating"
        write(output / "status.json", state)
        original_pixels, mask_pixels = np.asarray(original), np.asarray(mask)
        for row in rows:
            with torch.inference_mode():
                result = marble.run_parametric_control(adapter, original.copy(),
                    {mlps["metallic"]: row["metallic_control"], mlps["roughness"]: row["roughness_control"]},
                    texture_image=None, num_inference_steps=protocol["num_inference_steps"], seed=protocol["seed"],
                    depth_map=depth.copy(), mask=mask.convert("RGB"))
            if result.mode != "RGB" or result.size != (1024, 1024):
                raise ValueError("official MARBLE output must be1024 RGB")
            raw_path, final_path = output / "raw" / f"{row['id']}.png", output / "images" / f"{row['id']}.png"
            result.save(raw_path)
            resized = np.asarray(result.resize(original.size, Image.Resampling.LANCZOS))
            final = restore_background(original_pixels, resized, mask_pixels)
            Image.fromarray(final).save(final_path)
            qc = quality.evaluate(original_pixels, resized, final, mask_pixels)
            generated.append({**row, "raw_image": str(raw_path.relative_to(output)),
                "raw_sha256": sha256(raw_path), "image": str(final_path.relative_to(output)),
                "image_sha256": sha256(final_path), "qc": qc})
            with (output / "generation_manifest.jsonl").open("a", encoding="utf8") as stream:
                stream.write(json.dumps(generated[-1]) + "\n")
            state["generated_images"] = len(generated)
            write(output / "status.json", state)
            print("Generated", row["id"], qc["flags"], flush=True)
        pairs = []
        for roughness in protocol["roughness"]:
            selected = {row["metallic_state"]: row for row in generated if row["roughness_state"] == roughness}
            a, b = [np.asarray(Image.open(output / selected[key]["image"]), dtype=float)[mask_pixels > 0]
                    for key in ("low", "high")]
            gain = float(np.sum(a * b) / max(np.sum(a * a), 1e-12))
            pairs.append({"roughness_state": roughness, "M_pair_MAE_255": float(np.abs(a - b).mean()),
                "best_brightness_gain": gain,
                "relative_non_gain_residual": float(np.linalg.norm(b - gain * a) / max(np.linalg.norm(b), 1e-12))})
        write(output / "pairwise_response.json", {"pairs": pairs,
            "interpretation": "Image-response diagnostics only; differences can include hue/shape/texture leakage, not proof of metallic accuracy."})
        quality.write_review(output, output / "source/original.jpg", output / "source/object_mask.png", generated)
        check_locks(locks)
        assets.official_state(official)
        if git_info() != revision:
            raise ValueError("checkout changed during official inference")
        state["status"] = "six_generated_pending_visual_review"
        return state
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(output / "status.json", state)
        write(output / "repro_outputs/status.json", state)
        (output / "repro_outputs/SUMMARY.md").write_text(
            f"Status:{state['status']}. Generated{len(generated)}/6. No training or30-image expansion. "
            "All RGBs are candidates requiring visual review; controls are relative, not PBR measurements.\n", encoding="utf8")
        (report / "LOG.md").write_text(
            f"Execution status:{state['status']}. Generated{len(generated)}/6. "
            "See external launch.log, runtime.json, plan.json, generation_manifest.jsonl and raw/final review files.\n",
            encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.assets, args.output, args.gpu)))
