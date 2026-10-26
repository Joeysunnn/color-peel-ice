"""Test stock SD1.5 mailbox material prompts without learned tokens or adapters."""

import argparse
import json
from pathlib import Path
import sys
import time
import textwrap

from PIL import Image, ImageDraw, ImageFont

from experiments.sd15_token_local_kv_v1.contract import PROTOCOL, read_json, sha256
from experiments.sd_native_comparison_v1.generate import package_versions, write_json
from experiments.subject_multicolor_multimaterial_b.inference import MATERIALS, SAMPLING, prompt_rows
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
TOKENS = ("<S*>", "<C*>", "<M*>", "<T*>")


def rows():
    return [{"id": row["id"].replace("fullrank__", "sd15_native__"), "material": row["material"],
        "seed": row["seed"], "prompt": row["prompt"].replace("<S*>", "a"),
        "image_path": f"images/{row['material'].replace(' ', '_')}__seed{row['seed']}.png"}
        for row in prompt_rows(500) if row["method"] == "fullrank"]


def contact_sheet(output, records):
    try:
        face = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        face = ImageFont.load_default()
    canvas = Image.new("RGB", (2560, 1290), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), "Native SD1.5 | stock Diffusers UNet | NO learned tokens / K/V / LoRA | NO training images", font=face, fill="black")
    draw.text((10, 37), "PNDM100 / CFG3.5 / 512px / fp16 / fresh CUDA generator per image / shared safety checker", font=face, fill="black")
    lookup = {(row["material"], row["seed"]): row for row in records}
    for column, material in enumerate(MATERIALS):
        draw.text((column * 512 + 8, 75), material, font=face, fill="black")
        for line, seed in enumerate(SAMPLING["seeds"]):
            row = lookup[material, seed]
            x, y = column * 512 + 6, 110 + line * 590
            with Image.open(output / row["image_path"]) as image:
                canvas.paste(image.convert("RGB").resize((500, 500)), (x, y))
            draw.text((x, y + 505), f"seed{seed} | {row['status']}", font=face,
                fill="red" if row["status"] == "safety_filtered" else "black")
            draw.multiline_text((x, y + 534), "\n".join(textwrap.wrap(row["prompt"], 46)), font=face, fill="black", spacing=3)
    canvas.save(output / "contact_sheet.png")


def report(output, state, command):
    folder = output / "repro_outputs"
    folder.mkdir(exist_ok=True)
    write_json(output / "provenance.json", state)
    write_json(folder / "status.json", state)
    documents = {
        "SUMMARY.md": f"{state['status']}; {len(state['records'])}/10 images. Native SD1.5 material-prompt sanity test, not identity reconstruction.\n",
        "COMMANDS.md": command + "\n",
        "LOG.md": "See generation_manifest.jsonl, generation_status.jsonl, provenance.json and contact_sheet.png.\n",
        "SCIENTIFIC_CHANGELOG.md": "Same frozen SD1.5 generation weights and sampling as Subject tests; stock Diffusers UNet. "
            "Replace learned <S*> in each prompt with the article a. No adapter, learned token, mask, ControlNet, reference image or training data.\n",
        "COMPARABILITY_REPORT.md": "Five existing material words, seeds42/43,PNDM100,CFG3.5,512px,fp16. "
            "Fresh device generator per image, matching Subject inference; do not substitute older CPU-latent comparison protocol. "
            "Safety checker remains the project's shared SD1.4 checker. Shape/composition may change with material words; no fixed-subject identity claim.\n",
        "PATCHES.md": f"Independent script at {state['git']['commit']}. No old code, model weights or result files are modified.\n"}
    for name, content in documents.items():
        (folder / name).write_text(content, encoding="utf8")


def run(output, device="cuda:0", execute=False):
    output = resolve_run_dir(Path(output))
    revision = git_info()
    spec = read_json(PROTOCOL)["model"]
    model = Path(spec["root"])
    if output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(model):
        raise ValueError("choose a fresh external output, preserving old models and results")
    receipt_path = model / "model_source.json"
    receipt = read_json(receipt_path)
    if receipt["model"] != spec:
        raise ValueError("frozen SD1.5 source differs")
    locks = {str(model / name): value for name, value in receipt["files"].items()}
    locks.update({str(receipt_path): sha256(receipt_path), str(PROTOCOL): sha256(PROTOCOL),
                  str(Path(__file__)): sha256(__file__)})
    if any(sha256(Path(path)) != digest for path, digest in locks.items()):
        raise ValueError("base model or source hashes differ")
    planned = rows()
    if len(planned) != 10 or any(token in row["prompt"] for row in planned for token in TOKENS):
        raise ValueError("native matrix must contain ten ordinary prompts")
    output.mkdir(parents=True)
    state = {"status": "running" if execute else "dry_run_verified", "git": revision,
        "model": spec, "source_hashes": locks, "sampling": SAMPLING, "training": [],
        "learned_tokens": [], "adapters": False, "expected_images": 10, "records": [],
        "environment": package_versions(), "safety_checker_source": receipt["safety_checker_source"]}
    command = f"{sys.executable} -B -m experiments.sd15_native_mailbox_material_v1.run --output {output} --device {device}" + (" --execute" if execute else "")
    (output / "generation_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in planned), encoding="utf8")
    report(output, state, command)
    if not execute:
        return state
    try:
        import torch
        from diffusers import PNDMScheduler, StableDiffusionPipeline
        scheduler = PNDMScheduler.from_pretrained(str(model), subfolder="scheduler", local_files_only=True)
        pipe = StableDiffusionPipeline.from_pretrained(str(model), scheduler=scheduler,
            torch_dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=False).to(device)
        processors = sorted({type(value).__module__ + "." + type(value).__name__ for value in pipe.unet.attn_processors.values()})
        if (not type(pipe.unet).__module__.startswith("diffusers.")
                or any(not name.startswith("diffusers.") for name in processors)
                or any(token in pipe.tokenizer.get_vocab() for token in TOKENS)
                or pipe.safety_checker is None or not isinstance(pipe.scheduler, PNDMScheduler)):
            raise ValueError("expected stock UNet/attention, unmodified vocabulary, PNDM and enabled checker")
        pipe.set_progress_bar_config(disable=True)
        state["runtime"] = {"unet_class": type(pipe.unet).__module__ + "." + type(pipe.unet).__name__,
            "attention_processors": processors, "scheduler_config": dict(pipe.scheduler.config),
            "gpu": torch.cuda.get_device_name(device), "model_dtype": str(pipe.unet.dtype), "safety_checker": True,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic}
        for row in planned:
            start = time.monotonic()
            with torch.inference_mode():
                result = pipe(row["prompt"], num_inference_steps=SAMPLING["num_inference_steps"],
                    guidance_scale=SAMPLING["guidance_scale"], width=512, height=512,
                    generator=torch.Generator(device=device).manual_seed(row["seed"]))
            if result.nsfw_content_detected is None or len(result.nsfw_content_detected) != 1:
                raise ValueError("missing explicit safety checker result")
            path = output / row["image_path"]
            path.parent.mkdir(exist_ok=True)
            result.images[0].save(path)
            record = {**row, "status": "safety_filtered" if result.nsfw_content_detected[0] else "ok",
                "image_sha256": sha256(path), "seconds": time.monotonic() - start}
            state["records"].append(record)
            with (output / "generation_status.jsonl").open("a", encoding="utf8") as handle:
                handle.write(json.dumps(record) + "\n")
            report(output, state, command)
            print("SD15_NATIVE_MAILBOX", row["id"], record["status"], flush=True)
        contact_sheet(output, state["records"])
        (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Native SD1.5 mailbox materials</title>'
            '<p>No training images, learned tokens or adapters. Rows: seeds42/43; columns: metallic, matte, glossy plastic, ceramic, wood.</p>'
            '<a href="contact_sheet.png"><img width="1500" src="contact_sheet.png"></a>', encoding="utf8")
        if git_info() != revision or any(sha256(Path(path)) != value for path, value in locks.items()):
            raise ValueError("code or frozen model changed during generation")
        state.update(status="succeeded", safety_filtered=sum(row["status"] == "safety_filtered" for row in state["records"]))
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report(output, state, command)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(run(args.output, args.device, args.execute)["status"])
