"""Generate the authorized five-color MARBLE dataset; never start Subject training."""

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
from io import BytesIO
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from experiments.subject_marble_c_v1 import assets
from experiments.subject_marble_c_v1.run import check_locks
from experiments.subject_marble_color30_v1 import data
from experiments.subject_marble_diagnostics_v1.identity import evaluate
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

PROTOCOL = Path(__file__).with_name("protocol.json")


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.write(temporary, value)
    temporary.replace(path)


def image(path):
    with Image.open(path) as value:
        return value.copy()


def validate_records(output, plan, records):
    expected = {row["id"]: row for row in plan["rows"]}
    seen = set()
    for row in records:
        if row["id"] in seen or row["id"] not in expected:
            raise ValueError("duplicate or unknown generated case")
        if any(row[key] != value for key, value in expected[row["id"]].items()):
            raise ValueError("saved controls or caption differ from the frozen plan")
        path = (output / row["raw_image"]).resolve()
        if not path.is_relative_to(output) or data.sha256(path) != row["raw_sha256"]:
            raise ValueError("completed raw output changed")
        for relative, digest in row["condition_hashes"].items():
            condition = (output / relative).resolve()
            if not condition.is_relative_to(output) or data.sha256(condition) != digest:
                raise ValueError("completed image condition changed")
        seen.add(row["id"])
    return seen


def save_bytes(path, payload):
    """Resume packaging only when an existing artifact is byte-identical."""
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"candidate package changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def package(output, plan, generated):
    if len(validate_records(output, plan, generated)) != 30:
        raise ValueError("require all30 counterfactuals before assembling35 candidates")
    dataset = output / "dataset"
    records = []
    for color in plan["sources"]:
        records.append({"id": color + "__original", "color": color, "kind": "original_recolor",
            "source_image": plan["sources"][color]["original"],
            "caption": f"a photo of <S*> mailbox in {color} color made of glossy painted metal",
            "caption_status": "legacy appearance-conditioning label; pending review for this new dataset"})
    records += [{**row, "kind": "marble_counterfactual", "source_image": row["raw_image"]} for row in generated]
    mask_payload = (output / "source/object_mask512.png").read_bytes()
    concepts = []
    for row in records:
        source = output / row["source_image"]
        if row["kind"] == "original_recolor":
            payload = source.read_bytes()
        else:
            buffer = BytesIO()
            image(source).resize((512, 512), Image.Resampling.LANCZOS).save(buffer, format="PNG")
            payload = buffer.getvalue()
        row.update(image=f"images/{row['id']}/image.png", mask=f"masks/{row['id']}/image.png",
            image_sha256=hashlib.sha256(payload).hexdigest(), mask_sha256=hashlib.sha256(mask_payload).hexdigest(),
            training_eligible=False, review_status="pending_user_review")
        save_bytes(dataset / row["image"], payload)
        save_bytes(dataset / row["mask"], mask_payload)
        concepts.append({"instance_prompt": [row["caption"]],
            "instance_data_dir": str((dataset / row["image"]).parent),
            "instance_mask_dir": str((dataset / row["mask"]).parent)})
    manifest = {"records": records, "original_recolors": 5, "counterfactuals": 30,
        "training_started": False, "training_eligible_images": 0,
        "interpretation": "Proposed captions describe requested controls, not verified physical material or preserved color."}
    # Image files are checked above; these derived summaries may be regenerated.
    write(dataset / "manifest.json", manifest)
    write(dataset / "concepts.json", concepts)
    return manifest


def report(output, state, command):
    folder = output / "repro_outputs"
    folder.mkdir(exist_ok=True)
    write(folder / "status.json", state)
    write_status(output, state)
    documents = {
        "SUMMARY.md": f"# MARBLE color30\n\n{state['status']}; {len(state['records'])}/30 raw images. "
            "Five original recolors are additional candidate images. No training; user review pending.\n",
        "COMMANDS.md": command + "\n",
        "LOG.md": "See generation_manifest.json, conditions_manifest.json, runtime.json, dataset/manifest.json and source/.\n",
        "SCIENTIFIC_CHANGELOG.md": "User selected historical mailbox_self: grayscale initialization, strength1, "
            "self reference, existing mask; each1024 color gets fresh official DPT. Five existing RGB-LCh assets are unchanged.\n",
        "COMPARABILITY_REPORT.md": "M=-20/+20 and R=-1/0/+1 are relative CLIP edit strengths, not PBR parameters. "
            "Raw1024 remains untouched;512 training candidates are LANCZOS resizes without background compositing. "
            "RGB diagnostics do not establish albedo leakage or physical accuracy.\n",
        "PATCHES.md": f"Generation commit: {state['git']['commit']}. Independent wrapper; official code and all old experiments remain unchanged.\n",
    }
    for name, content in documents.items():
        (folder / name).write_text(content, encoding="utf8")


def write_status(output, state):
    write(output / "status.json", {"status": state["status"], "generated_images": len(state["records"]),
        "expected_generated_images": 30, "original_recolors": 5, "training_started": False,
        "training_eligible_images": 0, "git": state["git"]})


def generate_edit(marble, adapter, mlps, row, target, depth, mask):
    if target.mode != "RGB" or any(value.size != (1024, 1024) for value in (target, depth, mask)):
        raise ValueError("align all inputs before the official grayscale initialization")
    return marble.run_parametric_control(adapter, target.copy(),
        {mlps[name]: value for name, value in row["controls"].items()}, texture_image=None,
        num_inference_steps=30, seed=42, depth_map=depth.copy(), mask=mask.copy())


def run(asset_root, output, gpu, config_path=PROTOCOL, resume=False):
    asset_root, output = Path(asset_root).resolve(), resolve_run_dir(Path(output))
    if gpu < 0 or output.is_relative_to(asset_root) or asset_root.is_relative_to(output):
        raise ValueError("use a fresh external run separate from model assets")
    revision = git_info()
    if output.exists() and not resume:
        raise FileExistsError("old results are preserved; choose a new run or explicit --resume")
    plan = data.verify_prepared(output) if output.exists() else data.prepare(config_path, output)
    if plan["config"] != data.read(config_path):
        raise ValueError("resume config differs")
    config = plan["config"]
    reference_runtime = data.read(Path(config["baseline_run"]) / "runtime.json")
    if any(reference_runtime[key] != value for key, value in config["inference"].items()):
        raise ValueError("sampling differs from the selected historical mailbox_self group")
    if data.sha256(output / "source/mask.png") != data.sha256(Path(config["baseline_run"]) / "source/mask.png"):
        raise ValueError("generation mask differs from the selected group")
    receipt = data.read(asset_root / "assets.json")
    if receipt["official_commit"] != assets.OFFICIAL_COMMIT or receipt["models"] != assets.MODELS:
        raise ValueError("official model/source revisions differ")
    official = Path(receipt["official"])
    assets.official_state(official)
    locks = {**receipt["sha256"], str(asset_root / "assets.json"): data.sha256(asset_root / "assets.json"),
        **{str(p): data.sha256(p) for p in (Path(__file__), output / "prepare_receipt.json",
            Path(config["baseline_run"]) / "runtime.json", Path(inspect.getfile(evaluate)))}}
    check_locks(locks)
    state_path = output / "generation_manifest.json"
    state = data.read(state_path) if state_path.exists() else {
        "status": "initializing", "git": revision, "source_locks": locks,
        "records": [], "training_started": False, "training_eligible_images": 0}
    if state["source_locks"] != locks:
        raise ValueError("resume requires unchanged generation code and frozen assets")
    state.update(last_execution_git=revision)
    state.pop("error", None)
    completed = validate_records(output, plan, state["records"])
    command = f"{sys.executable} -B -m experiments.subject_marble_color30_v1.run --assets {asset_root} --output {output} --gpu {gpu} --resume"
    write(state_path, state)
    report(output, state, command)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    try:
        if len(completed) < 30:
            sys.path.insert(0, str(official))
            import torch
            import marble
            from transformers import DPTForDepthEstimation, DPTImageProcessor

            if not Path(marble.__file__).resolve().is_relative_to(official):
                raise ValueError("wrong MARBLE module imported")
            conditions_path = output / "conditions_manifest.json"
            conditions = data.read(conditions_path) if conditions_path.exists() else {}
            for saved in conditions.values():
                check_locks({str(output / name): digest for name, digest in saved["hashes"].items()})
            missing = [color for color in plan["sources"] if color not in conditions]
            if missing:
                model = DPTForDepthEstimation.from_pretrained(receipt["paths"]["depth"], local_files_only=True).to("cuda", dtype=torch.float16).eval()
                processor = DPTImageProcessor.from_pretrained(receipt["paths"]["depth"], local_files_only=True)
                for color in missing:
                    target = image(output / plan["sources"][color]["target"])
                    depth_path, init_path = output / f"source/{color}/depth.png", output / f"source/{color}/initialization.png"
                    if depth_path.exists() or init_path.exists():
                        raise FileExistsError("incomplete condition attempt preserved; use a fresh output")
                    with torch.inference_mode():
                        marble.run_dpt_depth(target, model, processor, device="cuda").save(depth_path)
                    marble.prepare_init_image(target, image(output / "source/mask.png")).save(init_path)
                    conditions[color] = {"depth_input": plan["sources"][color]["target"],
                        "hashes": {p.relative_to(output).as_posix(): data.sha256(p) for p in (depth_path, init_path)}}
                    write(conditions_path, conditions)
                del model, processor
                gc.collect()
                torch.cuda.empty_cache()
            marble.base_model_path, marble.controlnet_path = receipt["paths"]["base"], receipt["paths"]["controlnet"]
            adapter = marble.setup_pipeline(device="cuda", dtype=torch.float16)
            adapter.pipe.enable_model_cpu_offload(gpu_id=0)
            mlps = {name: marble.setup_control_mlp(name, device="cuda", dtype=torch.float16) for name in ("metallic", "roughness")}
            runtime = {"scheduler": type(adapter.pipe.scheduler).__name__, "scheduler_config": dict(adapter.pipe.scheduler.config),
                "guidance_scale": inspect.signature(adapter.pipe.__call__).parameters["guidance_scale"].default,
                "unet_in_channels": adapter.pipe.unet.config.in_channels, "gpu": gpu,
                "packages": {name: importlib.metadata.version(name) for name in reference_runtime["packages"]},
                "inference": config["inference"], "official_commit": assets.OFFICIAL_COMMIT}
            if runtime["scheduler"] != reference_runtime["scheduler"] or runtime["guidance_scale"] != 5:
                raise ValueError("unexpected official scheduler or CFG")
            def scheduler_values(values):
                return {key: sorted(value) if key == "_use_default_values" else value for key, value in values.items()}
            if scheduler_values(runtime["scheduler_config"]) != scheduler_values(reference_runtime["scheduler_config"]):
                raise ValueError("scheduler parameters differ from the selected historical group")
            if runtime["packages"] != reference_runtime["packages"]:
                raise ValueError("runtime package versions differ from the selected historical group")
            write(output / "runtime.json", runtime)
            for row in plan["rows"]:
                if row["id"] in completed:
                    continue
                source = plan["sources"][row["color"]]
                target, mask = image(output / source["target"]), image(output / source["mask"])
                depth = image(output / f"source/{row['color']}/depth.png")
                attempt, folder = 1, output / "raw" / row["id"]
                folder.mkdir(parents=True, exist_ok=True)
                while (folder / f"attempt{attempt:03d}.png").exists():
                    attempt += 1
                start = time.monotonic()
                with torch.inference_mode():
                    result = generate_edit(marble, adapter, mlps, row, target, depth, mask)
                if result.mode != "RGB" or result.size != (1024, 1024):
                    raise ValueError("require native1024 RGB output")
                path = folder / f"attempt{attempt:03d}.png"
                result.save(path)
                record = {**row, "raw_image": path.relative_to(output).as_posix(), "raw_sha256": data.sha256(path),
                    "seconds": time.monotonic() - start, "actual_denoising_steps": adapter.pipe.num_timesteps,
                    "condition_hashes": {**conditions[row["color"]]["hashes"], **{
                        source[name]: data.sha256(output / source[name]) for name in ("target", "reference", "mask")}},
                    "qc": evaluate(target, result, np.asarray(mask.convert("L")) >= 128)}
                state["records"].append(record)
                state["status"] = "generating"
                write(state_path, state)
                write_status(output, state)
                print("MARBLE_COLOR30", len(state["records"]), row["id"], flush=True)
        package(output, plan, state["records"])
        from experiments.subject_marble_color30_v1.review import review
        review(output)
        assets.official_state(official)
        check_locks(locks)
        data.verify_prepared(output)
        if git_info() != revision:
            raise ValueError("checkout changed during generation")
        state.update(status="generated35_pending_user_review", generated_counterfactuals=30, original_recolors=5)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(state_path, state)
        report(output, state, command)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROTOCOL)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args.assets, args.output, args.gpu, args.config, args.resume)
    print(result["status"])
