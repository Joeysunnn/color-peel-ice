"""Remove the original scene from all MARBLE image conditions; never train."""

import argparse
import gc
import importlib.metadata
import inspect
from io import BytesIO
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.subject_marble_c_v1 import assets
from experiments.subject_marble_c_v1.run import check_locks
from experiments.subject_marble_color30_v1 import data
from experiments.subject_marble_color30_v1.run import image, save_bytes, write
from experiments.subject_marble_diagnostics_v1.identity import evaluate
from experiments.subject_marble_foreground_v1.preprocess import NEUTRAL, mask_depth, prepare_inputs
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("protocol.json")


def prepare(output):
    policy = read_json(PROTOCOL)
    source_protocol = ROOT / policy["source_protocol"]
    config = {**read_json(source_protocol), **policy}
    frozen_directories = (Path(config["original_root"]), Path(config["comparison_run"]), Path(config["mask"]).parent)
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in frozen_directories):
        raise ValueError("new output must not modify any original asset or previous result directory")
    locks = {str(PROTOCOL): sha256(PROTOCOL), str(source_protocol): sha256(source_protocol),
             config["mask"]: config["mask_sha256"]}
    for color in config["colors"]:
        source = Path(config["original_root"]) / "category" / color["source_color"] / "images/image.png"
        locks[str(source)] = color["image_sha256"]
    comparison = Path(config["comparison_run"])
    for name in ("plan.json", "runtime.json", "generation_manifest.json"):
        locks[str(comparison / name)] = sha256(comparison / name)
    previous = read_json(comparison / "generation_manifest.json")
    if previous["status"] != "generated35_pending_user_review" or len(previous["records"]) != 30:
        raise ValueError("require the complete original scene-conditioned thirty-case comparison")
    for row in previous["records"]:
        locks[str(comparison / row["raw_image"])] = row["raw_sha256"]
    check_locks(locks)
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    shutil.copyfile(config["mask"], source / "object_mask512.png")
    mask = image(source / "object_mask512.png")
    sources, boxes = {}, {}
    for color in config["colors"]:
        name = color["name"]
        original_path = Path(config["original_root"]) / "category" / color["source_color"] / "images/image.png"
        prepared = prepare_inputs(image(original_path), mask)
        folder = source / name
        folder.mkdir()
        shutil.copyfile(original_path, folder / "original_source512.png")
        for key, filename in (("target", "target.png"), ("reference", "reference.png"),
                              ("cutout", "cutout512.png"), ("original_neutral", "original_neutral512.png")):
            prepared[key].save(folder / filename)
        prepared["mask"].convert("RGB").save(source / "mask.png")
        sources[name] = {"original": f"source/{name}/original_neutral512.png",
                         "target": f"source/{name}/target.png", "reference": f"source/{name}/reference.png",
                         "mask": "source/mask.png"}
        boxes[name] = prepared["reference_box"]
    rows = data.rows(config)
    plan = {"config": config, "rows": rows, "sources": sources, "reference_boxes": boxes,
            "source_hashes": locks, "diagnostics": [{"id": color + "__no_attribute", "color": color,
                "controls": {}, "training_eligible": False} for color in sources]}
    write(output / "config.json", config)
    write(output / "plan.json", plan)
    paths = [output / "config.json", output / "plan.json", *source.rglob("*.png")]
    write(output / "prepare_receipt.json", {"output_hashes": {
        path.relative_to(output).as_posix(): sha256(path) for path in paths}})
    data.verify_prepared(output)
    return plan


def generate_edit(marble, adapter, mlps, row, target, reference, depth, mask):
    if any(value.size != (1024, 1024) for value in (target, reference, depth, mask)):
        raise ValueError("all image conditions must be aligned1024")
    return marble.run_parametric_control(adapter, target.copy(),
        {mlps[name]: value for name, value in row["controls"].items()}, texture_image=reference.copy(),
        num_inference_steps=30, seed=42, depth_map=depth.copy(), mask=mask.convert("RGB"))


def package(output, plan, records):
    if len(records) != 30 or {row["id"] for row in records} != {row["id"] for row in plan["rows"]}:
        raise ValueError("require every M/R case before assembling the new candidates")
    mask512 = image(output / "source/object_mask512.png")
    rows = [{"id": color + "__original", "color": color, "kind": "original_recolor_cutout",
        "source_image": source["original"],
        "caption": f"a photo of <S*> mailbox in {color} color made of glossy painted metal",
        "caption_status": "requested conditioning words; physical appearance requires review"}
        for color, source in plan["sources"].items()]
    rows += [{**row, "kind": "marble_counterfactual", "source_image": row["raw_image"]} for row in records]
    concepts = []
    for row in rows:
        value = image(output / row["source_image"]).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        # Retain generated foreground pixels. Only replace the exterior with neutral padding.
        value = Image.composite(value, Image.new("RGB", value.size, NEUTRAL), mask512)
        location = output / "dataset/images" / row["id"] / "image.png"
        buffer = BytesIO()
        value.save(buffer, format="PNG")
        save_bytes(location, buffer.getvalue())
        mask_location = output / "dataset/masks" / row["id"] / "image.png"
        save_bytes(mask_location, (output / "source/object_mask512.png").read_bytes())
        cutout = Image.composite(value, Image.new("RGB", value.size), mask512).convert("RGBA")
        cutout.putalpha(mask512)
        # RGBA previews remain outside the one-image training directories.
        preview = output / "cutouts" / (row["id"] + ".png")
        preview.parent.mkdir(exist_ok=True)
        cutout.save(preview)
        row.update(image=location.relative_to(output / "dataset").as_posix(), image_sha256=sha256(location),
            mask=mask_location.relative_to(output / "dataset").as_posix(), mask_sha256=sha256(mask_location),
            foreground_rgba=preview.relative_to(output).as_posix(), training_eligible=False,
            review_status="pending_user_review")
        concepts.append({"instance_prompt": [row["caption"]], "instance_data_dir": str(location.parent),
                         "instance_mask_dir": str(mask_location.parent)})
    write(output / "dataset/manifest.json", {"records": rows, "counterfactuals": 30, "original_recolors": 5,
        "training_started": False, "training_eligible_images": 0,
        "interpretation": "Generated foreground retained; only exterior replaced with neutral. No original RGB foreground pasted back."})
    write(output / "dataset/concepts.json", concepts)


def report(output, state, command):
    folder = output / "repro_outputs"
    folder.mkdir(exist_ok=True)
    write(output / "status.json", state)
    write(folder / "status.json", state)
    documents = {
        "SUMMARY.md": f"Status: {state['status']}; {len(state['records'])}/30 controls; "
            f"{len(state['identity_records'])}/5 no-attribute diagnostics. No training.\n",
        "COMMANDS.md": command + "\n",
        "LOG.md": "See generation_manifest.json, plan.json, source/conditions.json and runtime.json.\n",
        "SCIENTIFIC_CHANGELOG.md": "Neutralize source background before resizing; explicit object-cropped CLIP reference; "
            "fresh foreground-only DPT with zero exterior; isolated gray initialization and VAE context; no old depth cache.\n",
        "COMPARABILITY_REPORT.md": "Same seed42,30steps,Euler,CFG5,strength1,IPAdapter1,ControlNet1 and M/R controls. "
            "Target coordinates unchanged. Reference crop increases CLIP subject occupancy; this is not a background-only single-factor test. "
            "Gray padding/model priors and reflections already inside the object remain. RGB changes are not albedo evidence.\n",
        "PATCHES.md": f"Independent module at {state['git']['commit']}; frozen official source/weights and old runs untouched.\n"}
    for name, text in documents.items():
        (folder / name).write_text(text, encoding="utf8")


def run(asset_root, output, gpu):
    output, asset_root = resolve_run_dir(Path(output)), Path(asset_root).resolve()
    if (gpu < 0 or output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(asset_root)
            or asset_root.is_relative_to(output)):
        raise ValueError("use a new external output separate from code, assets and old results")
    revision = git_info()
    plan = prepare(output)
    receipt = read_json(asset_root / "assets.json")
    if receipt["official_commit"] != assets.OFFICIAL_COMMIT or receipt["models"] != assets.MODELS:
        raise ValueError("official source or models differ")
    official = Path(receipt["official"])
    assets.official_state(official)
    locks = {**receipt["sha256"], **plan["source_hashes"], str(asset_root / "assets.json"): sha256(asset_root / "assets.json"),
             str(Path(__file__)): sha256(__file__), str(Path(inspect.getfile(prepare_inputs))): sha256(inspect.getfile(prepare_inputs))}
    state = {"status": "initializing", "git": revision, "records": [], "identity_records": [],
        "expected_images": 35, "training_started": False, "training_eligible_images": 0, "source_locks": locks}
    command = f"{sys.executable} -B -m experiments.subject_marble_foreground_v1.run --assets {asset_root} --output {output} --gpu {gpu}"
    write(output / "generation_manifest.json", state)
    report(output, state, command)
    try:
        check_locks(locks)
        from experiments.subject_marble_foreground_v1.review import inputs
        inputs(output)
        os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
        sys.path.insert(0, str(official))
        import torch
        import marble
        from transformers import DPTForDepthEstimation, DPTImageProcessor
        if not Path(marble.__file__).resolve().is_relative_to(official):
            raise ValueError("wrong MARBLE module")
        model = DPTForDepthEstimation.from_pretrained(receipt["paths"]["depth"], local_files_only=True).to("cuda", dtype=torch.float16).eval()
        processor = DPTImageProcessor.from_pretrained(receipt["paths"]["depth"], local_files_only=True)
        conditions = {}
        for color, source in plan["sources"].items():
            target, mask = image(output / source["target"]), image(output / source["mask"])
            folder = output / "source" / color
            with torch.inference_mode():
                raw_depth = marble.run_dpt_depth(target, model, processor, device="cuda")
            raw_depth.save(folder / "depth_before_mask.png")
            mask_depth(raw_depth, mask.convert("L")).save(folder / "depth.png")
            marble.prepare_init_image(target, mask).save(folder / "initialization.png")
            model_inputs = [folder / name for name in ("target.png", "reference.png", "depth.png", "initialization.png")]
            model_inputs.append(output / source["mask"])
            conditions[color] = {"depth_input": source["target"], "reference": source["reference"],
                "hashes": {str(p.relative_to(output)): sha256(p) for p in model_inputs}}
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
        write(output / "source/conditions.json", conditions)
        marble.base_model_path, marble.controlnet_path = receipt["paths"]["base"], receipt["paths"]["controlnet"]
        adapter = marble.setup_pipeline(device="cuda", dtype=torch.float16)
        adapter.pipe.enable_model_cpu_offload(gpu_id=0)
        mlps = {name: marble.setup_control_mlp(name, device="cuda", dtype=torch.float16) for name in ("metallic", "roughness")}
        baseline = read_json(Path(plan["config"]["comparison_run"]) / "runtime.json")
        runtime = {"scheduler": type(adapter.pipe.scheduler).__name__, "scheduler_config": dict(adapter.pipe.scheduler.config),
            "guidance_scale": inspect.signature(adapter.pipe.__call__).parameters["guidance_scale"].default,
            "unet_in_channels": adapter.pipe.unet.config.in_channels,
            "packages": {name: importlib.metadata.version(name) for name in baseline["packages"]},
            "inference": plan["config"]["inference"], "official_commit": assets.OFFICIAL_COMMIT}
        def scheduler_values(value):
            return {key: sorted(item) if key == "_use_default_values" else item for key, item in value.items()}
        if (runtime["scheduler"] != baseline["scheduler"] or runtime["guidance_scale"] != baseline["guidance_scale"]
                or runtime["packages"] != baseline["packages"]
                or scheduler_values(runtime["scheduler_config"]) != scheduler_values(baseline["scheduler_config"])):
            raise ValueError("sampling environment differs from the previous thirty-image experiment")
        write(output / "runtime.json", runtime)
        for row in plan["diagnostics"] + plan["rows"]:
            source = plan["sources"][row["color"]]
            target, reference = image(output / source["target"]), image(output / source["reference"])
            mask, depth = image(output / source["mask"]), image(output / f"source/{row['color']}/depth.png")
            start = time.monotonic()
            with torch.inference_mode():
                result = generate_edit(marble, adapter, mlps, row, target, reference, depth, mask)
            if result.mode != "RGB" or result.size != (1024, 1024):
                raise ValueError("expected native1024 RGB output")
            path = output / "raw" / (row["id"] + ".png")
            path.parent.mkdir(exist_ok=True)
            result.save(path)
            record = {**row, "raw_image": path.relative_to(output).as_posix(), "raw_sha256": sha256(path),
                "actual_denoising_steps": adapter.pipe.num_timesteps, "seconds": time.monotonic() - start,
                "condition_hashes": conditions[row["color"]]["hashes"],
                "qc": evaluate(target, result, np.asarray(mask.convert("L")) > 0)}
            state["records" if row["controls"] else "identity_records"].append(record)
            state["status"] = "generating"
            write(output / "generation_manifest.json", state)
            report(output, state, command)
            print("FOREGROUND_MARBLE", row["id"], flush=True)
        package(output, plan, state["records"])
        from experiments.subject_marble_foreground_v1.review import review
        review(output)
        check_locks(locks)
        for entry in conditions.values():
            check_locks({str(output / path): digest for path, digest in entry["hashes"].items()})
        data.verify_prepared(output)
        assets.official_state(official)
        if git_info() != revision:
            raise ValueError("checkout changed during generation")
        state.update(status="generated35_candidates_and5_diagnostics_pending_review")
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(output / "generation_manifest.json", state)
        report(output, state, command)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()
    print(run(args.assets, args.output, args.gpu)["status"])
