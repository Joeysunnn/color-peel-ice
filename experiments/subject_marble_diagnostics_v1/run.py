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
from experiments.subject_marble_diagnostics_v1 import identity, strength
from experiments.subject_marble_c_v1.run import check_locks, write
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("official", "mailbox", "mailbox_bowl", "mailbox_self", "mailbox_identity", "mailbox_strength", "mailbox_rgb_init")
OLD_RUN = Path("/var/tmp/colorpeel-r12user5/runs/subject_marble_c_v1/"
               "20261021-125714__phase1_six__86fb6a2__42")


def official_rows():
    return [{"id": f"M{m:+d}_R{r:+d}", "controls": {"metallic": m, "roughness": r},
             "label": f"M={m:+d} | R={r:+d}"}
            for m in (-20, 0, 20) for r in (-1, 0, 1)]


def mailbox_rows():
    controls = [("none", {}, "No attribute MLP"),
                ("joint_zero", {"metallic": 0, "roughness": 0}, "M=0 + R=0"),
                *[(f"M{m:+d}", {"metallic": m}, f"M={m:+d} only") for m in (-20, 0, 20)],
                *[(f"R{r:+d}", {"roughness": r}, f"R={r:+d} only") for r in (-1, 0, 1)]]
    rows = [{"id": f"{reference}__{name}", "controls": values, "reference": reference,
             "target_size": 1024, "label": f"{reference} | {label}"}
            for reference in ("full", "masked_gray") for name, values, label in controls]
    # Intentional old-size diagnostic controls, never used as the fixed default.
    rows += [{**row, "id": "legacy512__" + row["id"], "target_size": 512,
              "label": "Legacy512 | " + row["label"]}
             for row in rows if row["id"] in ("full__none", "full__M-20")]
    return rows


def joint_mailbox_rows(stage):
    if stage not in ("mailbox_bowl", "mailbox_self"):
        raise ValueError("select a mailbox joint-grid stage")
    reference = "bowl" if stage == "mailbox_bowl" else "self"
    return [{**row, "reference": reference, "target_size": 1024} for row in official_rows()]


def aligned_inputs(image, mask):
    if image.size != mask.size:
        raise ValueError("target and mask must start in the same coordinates")
    # Match the official RGB mask interpolation; align target BEFORE ImageChops.
    size = (1024, 1024)
    return (image.convert("RGB").resize(size, Image.Resampling.BICUBIC),
            mask.convert("RGB").resize(size, Image.Resampling.BICUBIC))


def masked_reference(image, mask):
    """Remove only background; preserve foreground pixels, canvas and object scale."""
    return Image.composite(image, Image.new("RGB", image.size, (127, 127, 127)), mask.convert("L"))


def sheet(output, rows, sources, filename="contact_sheet.png", columns=3, crop=None, title=None):
    """Show actual source inputs and uncomposited outputs, with explicit labels."""
    width, header = 1080, 410
    tile = width // columns
    row_height = tile + 50
    canvas = Image.new("RGB", (width, header + ((len(rows) + columns - 1) // columns) * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    draw.text((12, 8), title or "MARBLE diagnostics | relative controls | seed42 / 30steps | NO training", fill="black", font=font)

    def panel(path, x, y, size, box=None):
        with Image.open(path) as image:
            image = image.convert("RGB")
            if box is not None:
                image = image.crop(box)
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
            canvas.paste(image, (x, y))

    for index, (label, path) in enumerate(sources):
        x = index * (width // len(sources)) + 8
        draw.text((x, 42), label, font=font, fill="black")
        panel(output / path, x, 68, min(320, width // len(sources) - 16))
    for index, row in enumerate(rows):
        x, y = (index % columns) * tile + 8, header + (index // columns) * row_height
        draw.text((x, y), row["label"], font=font, fill="black")
        panel(output / row["raw_image"], x, y + 30, tile - 16, crop)
        draw.text((x, y + tile + 18), "Raw1024" + (" / crop" if crop else "") + " / review pending", font=font, fill="black")
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
        "Relative M/R controls are not measured PBR parameters. The plan identifies whether this is the "
        "official example or the authorized mailbox diagnostics. Conditions are frozen per stage.\n"
        "No external background compositing is applied to raw outputs.\n", encoding="utf8")
    (target / "PATCHES.md").write_text(
        f"Branch: {state['git']['branch']}; execution commit: {state['git']['commit']}.\n"
        "New isolated diagnostic runner only. Existing official and experiment code is unchanged.\n"
        "Mailbox target/mask alignment is fixed in the new caller; explicit old-size controls retain the prior error for comparison.\n"
        "Local paths, CPU offload and cached conditions are declared adaptations.\n", encoding="utf8")


def run(asset_root, output, gpu, stage="official"):
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
    if stage not in STAGES:
        raise ValueError("unknown diagnostic stage")
    rows = mailbox_rows() if stage == "mailbox" else official_rows()
    if stage in ("mailbox_bowl", "mailbox_self"):
        rows = joint_mailbox_rows(stage)
    elif stage == "mailbox_identity":
        rows = identity.rows()
    elif stage == "mailbox_strength":
        rows = strength.rows()
    elif stage == "mailbox_rgb_init":
        rows = strength.rgb_rows()
    if stage != "official":
        protocol = read_json(ROOT / "experiments/subject_marble_c_v1/protocol.json")
        source_path = ROOT / protocol["source"]
        mask_path = Path(protocol["mask"])
        locks.update({str(source_path): protocol["source_sha256"], str(mask_path): protocol["mask_sha256"]})
        if stage == "mailbox":
            depth_path = OLD_RUN / "source/depth.png"
            locks[str(depth_path)] = sha256(depth_path)
        check_locks(locks)
    output.mkdir(parents=True)
    for folder in ("source", "raw", "repro_outputs"):
        (output / folder).mkdir()
    if stage == "official":
        target_path = official / "input_images/context_image/toy_car.png"
        reference_path = official / "input_images/texture/metal_bowl.png"
        shutil.copyfile(target_path, output / "source/target.png")
        shutil.copyfile(reference_path, output / "source/reference.png")
        with Image.open(target_path) as image:
            target = image.copy()
        with Image.open(reference_path) as image:
            reference = image.copy()  # preserve the official RGBA input mode
        references = {"official": reference}
        source_hashes = {str(target_path): sha256(target_path), str(reference_path): sha256(reference_path)}
    else:
        shutil.copyfile(source_path, output / "source/original512.jpg")
        shutil.copyfile(mask_path, output / "source/object_mask512.png")
        with Image.open(source_path) as image:
            original = image.convert("RGB")
        with Image.open(mask_path) as image:
            original_mask = image.convert("L")
        target, mask = aligned_inputs(original, original_mask)
        if stage == "mailbox":
            references = {"full": original.copy(), "masked_gray": masked_reference(original, original_mask)}
            with Image.open(depth_path) as image:
                depth = image.copy()
        elif stage == "mailbox_bowl":
            reference_path = official / "input_images/texture/metal_bowl.png"
            with Image.open(reference_path) as image:
                references = {"bowl": image.copy()}
        else:
            references = {"self": target.copy()}
        if stage in ("mailbox_bowl", "mailbox_self", "mailbox_identity", "mailbox_strength", "mailbox_rgb_init"):
            # A common review ROI only; bowl generation later uses official rembg.
            mask.save(output / "source/review_mask.png")
        for name, image in references.items():
            image.save(output / "source" / f"reference_{name}.png")
        target.save(output / "source/target.png")
        source_hashes = {str(path): locks[str(path)] for path in (source_path, mask_path)}
        if stage == "mailbox":
            source_hashes[str(depth_path)] = locks[str(depth_path)]
        elif stage == "mailbox_bowl":
            source_hashes[str(reference_path)] = sha256(reference_path)
    if target.size != (1024, 1024) or target.mode != "RGB":
        raise ValueError("require a1024 RGB target before official initialization")
    command = f"{sys.executable} -B -m experiments.subject_marble_diagnostics_v1.run --assets {asset_root} --output {output} --gpu {gpu} --stage {stage}"
    deviations = [
        "Official 9-case toy_car + metal_bowl notebook grid, unchanged edit math and default sampling.",
        "Pinned local model paths; only M/R MLPs loaded because other attributes are unused.",
        "Official rembg mask and DPT depth computed once, saved, and reused for nine independent calls.",
        "Torch2.5.1+cu124 and Python3.11 reuse the isolated existing environment; official Torch pin is2.3.",
        "Model CPU offload enabled; no attention processor or scheduler replacement.",
    ]
    if stage == "mailbox":
        deviations[0] = "Authorized mailbox diagnostics, not an official notebook reproduction."
        deviations[2] = "Existing mailbox mask and old512-target DPT cache reused without altering the old run."
        deviations += [
            "Fix: target resized to1024 before official ImageChops; mask keeps official RGB BICUBIC resize.",
            "Full reference explicitly remains original512; masked-gray reference changes only background, not object scale.",
            "16 aligned cases plus2 intentional old512 controls (none and M-20 only); these18 RGBs are diagnostics, not a dataset.",
            "No-attribute MLP still regenerates; zero controls may have nonzero learned offsets.",
        ]
    elif stage in ("mailbox_bowl", "mailbox_self"):
        deviations[0] = "Authorized full official M/R joint grid on the original mailbox, nine independent edits."
        deviations[2] = "Official DPT freshly computed from the aligned1024 target, cached within this run only; no old depth cache."
        deviations += [
            "Target and Mask aligned to1024 before initialization; all raw outputs retained without background copying.",
            ("Official native RGBA metal_bowl reference and unmodified official rembg auto-mask. Existing mailbox mask is review ROI only."
             if stage == "mailbox_bowl" else
             "Target itself is the1024 material reference via official texture_image=None; existing mailbox Mask is retained."),
            "Two groups differ in BOTH Mask and reference, so between-group differences are not a reference-only ablation.",
        ]
    elif stage == "mailbox_identity":
        deviations[0] = "Identity reconstruction diagnostics, not counterfactual dataset expansion."
        deviations[2] = "Fresh official DPT from1024 target; frozen existing mailbox Mask;1024 target itself as reference."
        deviations += [
            "Seed42: empty MLP, joint0, M-only[-20,0,20], R-only[-1,0,1]. Seeds43/44: empty MLP and joint0 only; twelve RGBs total.",
            "All other official parameters unchanged. Empty MLP is still strength1 regeneration, not an identity passthrough.",
            "Evaluate visible pattern, text and structure preservation; RGB differences do not establish albedo leakage.",
            "No intrinsic ground truth: albedo preservation remains unverified. Three seeds provide only a limited stability check.",
        ]
    elif stage == "mailbox_strength":
        deviations[0] = "Authorized strength-only identity test, not an official default reproduction or dataset expansion."
        deviations[2] = "Fresh official1024 DPT, frozen mailbox Mask, same1024 target as reference; official grayscale-subject initialization unchanged."
        deviations += [
            "Four seed42 empty-MLP calls: strength1/.75/.5/.25; adapter proxy overrides only the official caller's hard-coded strength.",
            "All models, prompts, nominal30 steps, guidance5, ControlNet1 and IPAdapter1 unchanged. No RGB-init substitution.",
            "Lower strength retains more of the grayscale initialization, not original RGB; actual denoising counts are recorded separately.",
            "RGB differences are not albedo leakage evidence; manually review new patterns, lettering and visible part details.",
        ]
    elif stage == "mailbox_rgb_init":
        deviations[0] = "Authorized RGB-vs-official-gray initialization ablation, not official default reproduction or training data."
        deviations[2] = "Fresh1024 DPT, frozen mailbox Mask, identical1024 self reference; only initialization and strength vary."
        deviations += [
            "Two init groups each strength1/.75/.5/.25, seed42, empty MLP dict; eight raw1024 images total.",
            "Gray uses official prepare_init_image unchanged; RGB overrides only image=target.copy() at the adapter boundary.",
            "All remaining models, prompts, nominal30 steps, guidance5, ControlNet1, IPAdapter1 unchanged; no background compositing.",
            "Init and strength recorded per image; actual denoising counts read after generation. No RGB-based albedo verdict.",
        ]
    write(output / "plan.json", {"git": revision, "official_commit": assets.OFFICIAL_COMMIT,
        "stage": stage, "rows": rows, "command": command, "source_hashes": source_hashes,
        "seed": 42, "seeds": sorted({row.get("seed", 42) for row in rows}),
        "num_inference_steps": 30, "deviations": deviations})
    state = {"status": "initializing", "git": revision, "expected_images": len(rows), "generated_images": 0,
             "stage": stage,
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

        if stage != "mailbox":
            # Same official extraction operations, with explicit local pinned checkpoint paths.
            depth_model = DPTForDepthEstimation.from_pretrained(receipt["paths"]["depth"]).to("cuda", dtype=torch.float16)
            processor = DPTImageProcessor.from_pretrained(receipt["paths"]["depth"])
            depth_model.eval()
            with torch.inference_mode():
                depth = marble.run_dpt_depth(target, depth_model, processor, device="cuda")
            del depth_model, processor
            gc.collect()
            torch.cuda.empty_cache()
            if stage in ("official", "mailbox_bowl"):
                mask = marble.prepare_mask(target)
        init = marble.prepare_init_image(target, mask)
        depth.save(output / "source/depth.png")
        mask.save(output / "source/mask.png")
        init.save(output / "source/initialization.png")
        if stage == "mailbox_rgb_init":
            target.save(output / "source/initialization_rgb.png")
        if stage == "mailbox":
            marble.prepare_init_image(original, mask).save(output / "source/legacy_initialization.png")
        locks.update({str(p): sha256(p) for p in (asset_root / "diagnostics_rembg").glob("*.onnx")})
        write(output / "source/conditions.json", {"target_size": target.size,
            "references": {name: {"size": image.size, "mode": image.mode} for name, image in references.items()},
            "mask_size": mask.size, "depth_size": depth.size,
            "depth_input_size": [512, 512] if stage == "mailbox" else list(target.size),
            "depth_policy": "old512 DPT cache" if stage == "mailbox" else "fresh official DPT from this1024 target",
            "foreground_fraction": float((np.asarray(mask.convert('L')) >= 128).mean()),
            "mask_policy": "unmodified official prepare_mask/rembg" if stage in ("official", "mailbox_bowl") else "frozen existing mask; official RGB BICUBIC resize",
            "rembg_weights": {str(p): sha256(p) for p in (asset_root / "diagnostics_rembg").glob("*.onnx")},
            "condition_hashes": {p.name: sha256(p) for p in (output / "source").glob("*.png")}})

        marble.base_model_path = receipt["paths"]["base"]
        marble.controlnet_path = receipt["paths"]["controlnet"]
        adapter = marble.setup_pipeline(device="cuda", dtype=torch.float16)
        adapter.pipe.enable_model_cpu_offload(gpu_id=0)
        mlps = {name: marble.setup_control_mlp(name, device="cuda", dtype=torch.float16)
                for name in ("metallic", "roughness")}
        if stage == "mailbox_identity":
            write(output / "zero_control_offsets.json", identity.zero_offsets(adapter, mlps, target))
        write(output / "runtime.json", {"pipeline": type(adapter.pipe).__name__,
            "scheduler": type(adapter.pipe.scheduler).__name__, "scheduler_config": dict(adapter.pipe.scheduler.config),
            "guidance_scale": inspect.signature(adapter.pipe.__call__).parameters["guidance_scale"].default,
            "unet_in_channels": adapter.pipe.unet.config.in_channels,
            "device": torch.cuda.get_device_name(0), "strength": [row["strength"] for row in rows] if stage in ("mailbox_strength", "mailbox_rgb_init") else 1,
            "controlnet_scale": 1, "ip_adapter_scale": 1,
            "seed": 42, "seeds": sorted({row.get("seed", 42) for row in rows}), "num_inference_steps": 30,
            "prompt": "best quality, high quality", "negative_prompt": "monochrome, lowres, bad anatomy, worst quality, low quality",
            "packages": {p: importlib.metadata.version(p) for p in
                         ("torch", "diffusers", "transformers", "huggingface_hub", "accelerate", "Pillow", "numpy", "rembg")}})
        state["status"] = "generating"
        generated = []
        binary_mask = (np.asarray(mask.convert("L")) >= 128).astype(np.uint8) * 255
        for row in rows:
            start = time.monotonic()
            with torch.inference_mode():
                row_target = original if row.get("target_size") == 512 else target
                row_adapter = adapter
                if stage in ("mailbox_strength", "mailbox_rgb_init"):
                    row_adapter = strength.StrengthOverride(adapter, row["strength"],
                        init_image=target if row.get("initialization") == "rgb" else None)
                result = marble.run_parametric_control(row_adapter, row_target.copy(),
                    {mlps[name]: value for name, value in row["controls"].items()},
                    texture_image=None if stage in ("mailbox_self", "mailbox_identity", "mailbox_strength", "mailbox_rgb_init") else references[row.get("reference", "official")].copy(),
                    num_inference_steps=30, seed=row.get("seed", 42), depth_map=depth.copy(), mask=mask.copy())
            if result.size != (1024, 1024) or result.mode != "RGB":
                raise ValueError("unexpected official RGB output")
            path = output / "raw" / f"{row['id']}.png"
            result.save(path)
            generated.append({**row, "raw_image": str(path.relative_to(output)), "sha256": sha256(path),
                "seconds": time.monotonic() - start, "training_eligible": False,
                "qc": identity.evaluate(target, result, binary_mask) if stage in ("mailbox_identity", "mailbox_strength", "mailbox_rgb_init") else
                      quality.evaluate(np.asarray(target), np.asarray(result), np.asarray(result), binary_mask)})
            if stage in ("mailbox_strength", "mailbox_rgb_init"):
                generated[-1]["actual_denoising_steps"] = adapter.pipe.num_timesteps
            if stage == "mailbox_rgb_init":
                generated[-1]["initialization_sha256"] = sha256(output / row["initialization_image"])
            with (output / "manifest.jsonl").open("a", encoding="utf8") as stream:
                stream.write(json.dumps(generated[-1]) + "\n")
            state["generated_images"] = len(generated)
            write(output / "status.json", state)
            print("Generated", row["id"], generated[-1]["seconds"], flush=True)
        if stage == "official":
            sheet(output, generated, [("Official context", "source/target.png"), ("Metal reference", "source/reference.png"),
                ("Official rembg mask", "source/mask.png"), ("Shared official DPT", "source/depth.png")])
        elif stage == "mailbox_identity":
            identity.write_review(output, generated, sheet)
        elif stage == "mailbox_strength":
            strength.write_review(output, generated, sheet)
        elif stage == "mailbox_rgb_init":
            strength.write_rgb_review(output, generated, sheet)
        elif stage in ("mailbox_bowl", "mailbox_self"):
            reference = "bowl" if stage == "mailbox_bowl" else "self"
            sources = [("Mailbox1024 target", "source/target.png"), (reference + " material reference", f"source/reference_{reference}.png"),
                       ("Actual generation Mask", "source/mask.png"), ("Fresh1024-input DPT", "source/depth.png")]
            sheet(output, generated, sources)
            with Image.open(output / "source/review_mask.png") as image:
                bbox = image.convert("L").getbbox()
            crop = (max(0, bbox[0] - 24), max(0, bbox[1] - 24), min(1024, bbox[2] + 24), min(1024, bbox[3] + 24))
            write(output / "review_crop.json", {"bbox": crop, "policy": "same existing mailbox ROI for both groups; does not change generation Mask"})
            sheet(output, generated, sources, "subject_crop.png", crop=crop)
        else:
            bbox = mask.convert("L").getbbox()
            crop = (max(0, bbox[0] - 24), max(0, bbox[1] - 24), min(1024, bbox[2] + 24), min(1024, bbox[3] + 24))
            for name in references:
                selected = [row for row in generated if row["reference"] == name and row["target_size"] == 1024]
                sources = [("Original input", "source/original512.jpg"), ("Actual material reference", f"source/reference_{name}.png"),
                           ("Fixed aligned initialization", "source/initialization.png")]
                sheet(output, selected, sources, f"contact_sheet_{name}.png")
                sheet(output, selected, sources, f"subject_crop_{name}.png", crop=crop)
            by_id = {row["id"]: row for row in generated}
            paired = [by_id[prefix + name] for name in ("full__none", "full__M-20") for prefix in ("", "legacy512__")]
            sheet(output, paired, [("Aligned1024 init", "source/initialization.png"),
                  ("Legacy512 init", "source/legacy_initialization.png")], "size_comparison.png", columns=2, crop=crop)
            comparisons = []
            for first, second in zip(paired[::2], paired[1::2]):
                a, b = [np.asarray(Image.open(output / row["raw_image"]), dtype=float) for row in (first, second)]
                comparisons.append({"aligned": first["id"], "legacy": second["id"],
                    "foreground_MAE_255": float(np.abs(a - b)[binary_mask > 0].mean()),
                    "background_MAE_255": float(np.abs(a - b)[binary_mask == 0].mean()),
                    "identical_pixels": bool(np.array_equal(a, b))})
            write(output / "size_pair_metrics.json", {"pairs": comparisons, "fixed_review_crop": crop})
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
    parser.add_argument("--stage", choices=STAGES, default="official")
    args = parser.parse_args()
    print(json.dumps(run(args.assets, args.output, args.gpu, args.stage)))
