"""Generate independent, resumable ZeST diagnostics; never auto-start token training."""

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import time

from PIL import Image

from experiments.sd15_token_local_kv_v1.contract import sha256
from experiments.zest_material_counterfactual_v1 import assets, official
from experiments.zest_material_counterfactual_v1.render_bank import write
from scripts.launch.colorpeel_run import git_info

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def load_plan(output):
    receipt = read(output / "prepare_receipt.json")
    for relative, digest in receipt["output_hashes"].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output) or sha256(path) != digest:
            raise ValueError(f"Prepared input changed: {relative}")
    return read(output / "plan.json")


def jobs(plan, bank, stage, variants):
    references = {r["id"]: r for r in bank["records"]}
    if bank["status"] != "complete":
        raise ValueError("complete the PBR reference bank first")
    result = []
    for subject in plan["subjects"]:
        for variant in variants:
            if variant not in plan["variants"]:
                raise ValueError("unknown variant")
            selected = ["self", "estimated_original"] if stage == "identity" else [name for name in references if name != "estimated_original"]
            for name in selected:
                reference = references.get(name)
                result.append({"id": f"{stage}__{subject['id']}__{variant}__{name}",
                    "stage": stage, "subject_id": subject["id"], "variant": variant, "reference_id": name,
                    "reference_image": subject["source_files"]["target"] if name == "self" else reference["image"],
                    "reference_sha256": None if name == "self" else reference["image_sha256"],
                    "reference_target": None if name == "self" else {key: reference[key] for key in ("metallic", "roughness")}})
    return result


def report(output, state, command):
    folder = output / "repro_outputs"
    folder.mkdir(exist_ok=True)
    write(folder / "status.json", state)
    (folder / "SUMMARY.md").write_text(
        f"# ZeST candidate generation\n\n{state['status']}; {len(state['records'])} completed diagnostics. "
        "No token training. Generation completion does not imply counterfactual quality.\n", encoding="utf8")
    (folder / "COMMANDS.md").write_text(command + "\n", encoding="utf8")
    (folder / "LOG.md").write_text("See generation_manifest.json, cases/*/attempt*/, source/ and runtime_*.json.\n", encoding="utf8")
    (folder / "SCIENTIFIC_CHANGELOG.md").write_text(
        "Frozen existing mailbox Mask/Depth are explicit adaptations of ZeST; full Tencent IPAdapter is used.\n"
        "Color is estimated-Albedo chromaticity initialization, a soft condition, not a hard constraint.\n"
        "Texture is an added Albedo-edge Canny ControlNet, not native ZeST. No RGB output compositing.\n", encoding="utf8")
    (folder / "COMPARABILITY_REPORT.md").write_text(
        "Reference M/R is renderer ground truth for the reference sphere only. Generated M/R and Albedo are estimated.\n"
        "Self reference and original-estimate reference are distinct controls. Identity must be reviewed before training eligibility.\n", encoding="utf8")
    (folder / "PATCHES.md").write_text(
        f"Project generation commit: {state['git']['commit']}. Independent experiment module; original token learning unchanged.\n"
        "Official ZeST/Tencent files and shared models remain unmodified. Added Canny and color initialization are declared ablations.\n", encoding="utf8")


def run(output, asset_root, stage, gpu, variants=None, resume=False):
    output = Path(output).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("use an external prepared run")
    plan = load_plan(output)
    if any("control_depth" not in subject["source_files"] for subject in plan["subjects"]):
        raise ValueError("provide a frozen1024 Depth condition for each prepared carrier")
    if stage in ("grid", "all"):
        latest = read(output / "quality/latest.json")
        reviewed = read(output / latest["report"])
        controls = {(row["subject_id"], row["variant"], row["reference_id"]) for row in reviewed["records"] if row["stage"] == "identity"}
        if any((subject["id"], "baseline", "self") not in controls for subject in plan["subjects"]):
            raise ValueError("assess the original ZeST self-reference Identity before running the grid")
    bank = read(output / "bank/manifest.json")
    for row in bank["records"]:
        if sha256(row["image"]) != row["image_sha256"]:
            raise ValueError("reference changed")
    selected = [row for part in (("identity", "grid") if stage == "all" else (stage,))
                for row in jobs(plan, bank, part, variants or plan["variants"])]
    receipt = assets.read(asset_root)
    locks = {str(p): sha256(p) for p in (Path(__file__), Path(official.__file__), output / "plan.json", output / "bank/manifest.json")}
    locks[str(Path(asset_root).resolve() / "assets.json")] = sha256(Path(asset_root) / "assets.json")
    manifest_path = output / "generation_manifest.json"
    state = {"status": "generating", "git": git_info(), "source_locks": locks, "records": [], "training_started": False}
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("generation exists; use --resume with unchanged inputs")
        state = read(manifest_path)
        if state["source_locks"] != locks:
            raise ValueError("generator, reference bank or model receipt changed")
        for row in state["records"]:
            if sha256(output / row["raw_image"]) != row["raw_sha256"]:
                raise ValueError("saved generation changed")
    completed = {row["id"] for row in state["records"]}
    pending = [row for row in selected if row["id"] not in completed]
    if not pending:
        assets.verify(receipt)
        print("All requested outputs verified", flush=True)
        return state
    state.update(status="generating")
    state.pop("error", None)
    write(manifest_path, state)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    subjects = {row["id"]: row for row in plan["subjects"]}
    adapter = None
    command = f"python -B -m experiments.zest_material_counterfactual_v1.run --run {output} --assets {asset_root} --stage {stage} --gpu {gpu} --variants {' '.join(variants or plan['variants'])} --resume"
    try:
        for with_edges in (False, True):
            group = [row for row in pending if (row["variant"] in ("texture", "combined")) == with_edges]
            if not group:
                continue
            adapter = official.setup(receipt, with_edges=with_edges)
            runtime = official.compatibility(adapter)
            runtime["scheduler_config"] = dict(adapter.pipe.scheduler.config)
            runtime["packages"] = {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "accelerate", "Pillow")}
            write(output / f"runtime_{'depth_canny' if with_edges else 'depth'}.json", runtime)
            for row in group:
                source = subjects[row["subject_id"]]["source_files"]
                def image(name, mode="RGB"):
                    with Image.open(output / source[name]) as value:
                        return value.convert(mode)
                reference_path = Path(row["reference_image"])
                if not reference_path.is_absolute():
                    reference_path = output / reference_path
                with Image.open(reference_path) as value:
                    reference = value.convert("RGB").resize((1024, 1024), Image.Resampling.BICUBIC)
                folder = output / "cases" / row["id"]
                attempt = 1
                while (folder / f"attempt{attempt:03d}").exists():
                    attempt += 1
                folder = folder / f"attempt{attempt:03d}"
                folder.mkdir(parents=True)
                start = time.monotonic()
                result = official.generate(adapter, image("target"), image("edit_mask"), image("control_depth"), reference,
                    init_image=image("initialization_color") if row["variant"] in ("color", "combined") else None,
                    edge_image=image("albedo_edges") if with_edges else None, **plan["inference"])
                path = folder / "raw.png"
                result.save(path)
                record = {**row, "reference_sha256": sha256(reference_path),
                    "raw_image": path.relative_to(output).as_posix(), "raw_sha256": sha256(path),
                    "actual_denoising_steps": adapter.pipe.num_timesteps, "inference": plan["inference"],
                    "initialization": "albedo_chroma" if row["variant"] in ("color", "combined") else "official_gray",
                    "controlnets": "depth+canny" if with_edges else "depth", "seconds": time.monotonic() - start,
                    "training_eligible": False}
                write(folder / "record.json", record)
                state["records"].append(record)
                write(manifest_path, state)
                print("ZEST_GENERATED", len(state["records"]), row["id"], record["actual_denoising_steps"], flush=True)
            del adapter
            adapter = None
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        assets.verify(receipt)
        load_plan(output)
        if any(sha256(path) != digest for path, digest in locks.items()):
            raise ValueError("generation inputs changed")
        state["status"] = "generated_pending_quality_review"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(manifest_path, state)
        report(output, state, command)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--stage", choices=("identity", "grid", "all"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.run, args.assets, args.stage, args.gpu, args.variants, args.resume)
