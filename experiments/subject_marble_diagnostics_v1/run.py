"""Pinned MARBLE sanity checks; outputs are diagnostics, never training data."""

import argparse
import gc
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.subject_marble_c_v1 import assets, quality
from experiments.subject_marble_c_v1.run import check_locks, write
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]


def official_rows():
    return [{"id": f"M{m:+d}_R{r:+d}", "controls": {"metallic": m, "roughness": r},
             "label": f"M={m:+d} | R={r:+d}"}
            for m in (-20, 0, 20) for r in (-1, 0, 1)]


def sheet(output, rows, sources, filename="contact_sheet.png"):
    """Show actual source inputs and uncomposited outputs, with explicit labels."""
    width, tile, header = 1080, 360, 410
    canvas = Image.new("RGB", (width, header + ((len(rows) + 2) // 3) * 410), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    draw.text((12, 8), "MARBLE diagnostics | relative controls | seed42 / 30steps | NO training", fill="black", font=font)

    def panel(path, x, y, size):
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
            canvas.paste(image, (x, y))

    for index, (label, path) in enumerate(sources):
        x = index * (width // len(sources)) + 8
        draw.text((x, 42), label, font=font, fill="black")
        panel(output / path, x, 68, min(320, width // len(sources) - 16))
    for index, row in enumerate(rows):
        x, y = (index % 3) * tile + 8, header + (index // 3) * 410
        draw.text((x, y), row["label"], font=font, fill="black")
        panel(output / row["raw_image"], x, y + 30, 344)
        draw.text((x, y + 378), "Raw model output; manual QC required", font=font, fill="black")
    canvas.save(output / filename)


def report(output, state, command, deviations):
    target = output / "repro_outputs"
    target.mkdir(exist_ok=True)
    write(target / "status.json", state)
    (target / "SUMMARY.md").write_text(
        f"# MARBLE diagnostics\n\nStatus: {state['status']}. Generated {state['generated_images']}"
        f"/{state['expected_images']}. No training or expansion to30.\n"
        "Outputs require visual review; generation completion is not paper reproduction quality.\n", encoding="utf8")
    (target / "COMMANDS.md").write_text("# Adapted documented inference\n\n```bash\n" + command + "\n```\n"
        "\nSource: official parametric_control.ipynb at " + assets.OFFICIAL_COMMIT + "\n", encoding="utf8")
    (target / "LOG.md").write_text(
        f"Status: {state['status']}. See plan.json, runtime.json, launch.log, manifest.jsonl and source/.\n",
        encoding="utf8")
    (target / "SCIENTIFIC_CHANGELOG.md").write_text("\n".join(deviations) + "\n", encoding="utf8")
    (target / "COMPARABILITY_REPORT.md").write_text(
        "Official notebook has no saved reference PNGs; visual agreement cannot establish pixel identity.\n"
        "Relative M/R controls are not measured PBR parameters. Mask/depth are official-function outputs, "
        "cached once and reused, rather than recomputed for every identical target.\n"
        "No external background compositing is applied to raw outputs.\n", encoding="utf8")
    (target / "PATCHES.md").write_text(
        f"Branch: {state['git']['branch']}; execution commit: {state['git']['commit']}.\n"
        "New isolated diagnostic runner only. Existing official and experiment code is unchanged.\n"
        "Local paths, CPU offload and cached conditions are declared adaptations.\n", encoding="utf8")


def run(asset_root, output, gpu):
    asset_root, output = Path(asset_root).resolve(), resolve_run_dir(Path(output))
    if (gpu < 0 or output.exists() or output.is_relative_to(ROOT)
            or output.is_relative_to(asset_root) or asset_root.is_relative_to(output)):
        raise ValueError("use a fresh external output separate from code and assets")
    revision = git_info()
    receipt_path = asset_root / "assets.json"
    receipt = read_json(receipt_path)
    if receipt["official_commit"] != assets.OFFICIAL_COMMIT or receipt["models"] != assets.MODELS:
        raise ValueError("pinned official source/model revisions differ")
    official = Path(receipt["official"])
    assets.official_state(official)
    locks = {**receipt["sha256"], str(receipt_path): sha256(receipt_path)}
    check_locks(locks)
    rows = official_rows()
    output.mkdir(parents=True)
    for folder in ("source", "raw", "repro_outputs"):
        (output / folder).mkdir()
    target_path = official / "input_images/context_image/toy_car.png"
    reference_path = official / "input_images/texture/metal_bowl.png"
    shutil.copyfile(target_path, output / "source/target.png")
    shutil.copyfile(reference_path, output / "source/reference.png")
    with Image.open(target_path) as image:
        target = image.copy()
    with Image.open(reference_path) as image:
        reference = image.copy()  # preserve the official RGBA input mode
    if target.size != (1024, 1024) or target.mode != "RGB":
        raise ValueError("official toy-car target changed")
    command = f"{sys.executable} -B -m experiments.subject_marble_diagnostics_v1.run --assets {asset_root} --output {output} --gpu {gpu}"
    deviations = [
        "Official 9-case toy_car + metal_bowl notebook grid, unchanged edit math and default sampling.",
        "Pinned local model paths; only M/R MLPs loaded because other attributes are unused.",
        "Official rembg mask and DPT depth computed once, saved, and reused for nine independent calls.",
        "Torch2.5.1+cu124 and Python3.11 reuse the isolated existing environment; official Torch pin is2.3.",
        "Model CPU offload enabled; no attention processor or scheduler replacement.",
    ]
    write(output / "plan.json", {"git": revision, "official_commit": assets.OFFICIAL_COMMIT,
        "stage": "official_notebook_grid", "rows": rows, "command": command,
        "source_hashes": {str(target_path): sha256(target_path), str(reference_path): sha256(reference_path)},
        "seed": 42, "num_inference_steps": 30, "deviations": deviations})
    state = {"status": "initializing", "git": revision, "expected_images": len(rows), "generated_images": 0,
             "training_started": False, "expanded_to30": False, "training_eligible_images": 0}
    write(output / "status.json", state)
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["U2NET_HOME"] = str(asset_root / "diagnostics_rembg")
        sys.path.insert(0, str(official))
        import torch
        import marble
        from transformers import DPTForDepthEstimation, DPTImageProcessor

        # Same official extraction operations, with explicit local pinned checkpoint paths.
        depth_model = DPTForDepthEstimation.from_pretrained(receipt["paths"]["depth"]).to("cuda", dtype=torch.float16)
        processor = DPTImageProcessor.from_pretrained(receipt["paths"]["depth"])
        depth_model.eval()
        with torch.inference_mode():
            depth = marble.run_dpt_depth(target, depth_model, processor, device="cuda")
        del depth_model, processor
        gc.collect()
        torch.cuda.empty_cache()
        mask = marble.prepare_mask(target)
        init = marble.prepare_init_image(target, mask)
        depth.save(output / "source/depth.png")
        mask.save(output / "source/mask.png")
        init.save(output / "source/initialization.png")
        write(output / "source/conditions.json", {"target_size": target.size, "reference_size": reference.size,
            "reference_mode": reference.mode, "mask_size": mask.size, "depth_size": depth.size,
            "foreground_fraction": float((np.asarray(mask.convert('L')) >= 128).mean()),
            "mask_policy": "unmodified official prepare_mask/rembg; not alpha replacement",
            "condition_hashes": {p.name: sha256(p) for p in (output / "source").glob("*.png")}})

        marble.base_model_path = receipt["paths"]["base"]
        marble.controlnet_path = receipt["paths"]["controlnet"]
        adapter = marble.setup_pipeline(device="cuda", dtype=torch.float16)
        adapter.pipe.enable_model_cpu_offload(gpu_id=0)
        mlps = {name: marble.setup_control_mlp(name, device="cuda", dtype=torch.float16)
                for name in ("metallic", "roughness")}
        write(output / "runtime.json", {"pipeline": type(adapter.pipe).__name__,
            "scheduler": type(adapter.pipe.scheduler).__name__, "scheduler_config": dict(adapter.pipe.scheduler.config),
            "guidance_scale": inspect.signature(adapter.pipe.__call__).parameters["guidance_scale"].default,
            "unet_in_channels": adapter.pipe.unet.config.in_channels,
            "device": torch.cuda.get_device_name(0), "strength": 1, "controlnet_scale": 1, "ip_adapter_scale": 1,
            "seed": 42, "num_inference_steps": 30,
            "prompt": "best quality, high quality", "negative_prompt": "monochrome, lowres, bad anatomy, worst quality, low quality",
            "packages": {p: importlib.metadata.version(p) for p in
                         ("torch", "diffusers", "transformers", "huggingface_hub", "accelerate", "Pillow", "numpy", "rembg")}})
        state["status"] = "generating"
        generated = []
        binary_mask = (np.asarray(mask.convert("L")) >= 128).astype(np.uint8) * 255
        for row in rows:
            start = time.monotonic()
            with torch.inference_mode():
                result = marble.run_parametric_control(adapter, target.copy(),
                    {mlps[name]: value for name, value in row["controls"].items()},
                    texture_image=reference.copy(), depth_map=depth.copy(), mask=mask.copy())
            if result.size != (1024, 1024) or result.mode != "RGB":
                raise ValueError("unexpected official RGB output")
            path = output / "raw" / f"{row['id']}.png"
            result.save(path)
            generated.append({**row, "raw_image": str(path.relative_to(output)), "sha256": sha256(path),
                "seconds": time.monotonic() - start, "training_eligible": False,
                "qc": quality.evaluate(np.asarray(target), np.asarray(result), np.asarray(result), binary_mask)})
            with (output / "manifest.jsonl").open("a", encoding="utf8") as stream:
                stream.write(json.dumps(generated[-1]) + "\n")
            state["generated_images"] = len(generated)
            write(output / "status.json", state)
            print("Generated", row["id"], generated[-1]["seconds"], flush=True)
        sheet(output, generated, [("Official context", "source/target.png"), ("Metal reference", "source/reference.png"),
            ("Official rembg mask", "source/mask.png"), ("Shared official DPT", "source/depth.png")])
        check_locks(locks)
        assets.official_state(official)
        if git_info() != revision:
            raise ValueError("project checkout changed during generation")
        state["status"] = "generated_pending_visual_review"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(output / "status.json", state)
        report(output, state, command, deviations)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.assets, args.output, args.gpu)))
