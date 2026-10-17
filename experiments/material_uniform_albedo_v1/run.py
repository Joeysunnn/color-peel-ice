"""Render uniform-albedo Material data and train the four existing K/V recipes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from experiments.material_uniform_albedo_v1 import data
from experiments.sd15_token_local_kv_v1.contract import sha256
from experiments.sd_native_comparison_v1.generate import git_info, write_json
from experiments.texture_material_disentangled_v1 import step_ablation, train
from experiments.texture_material_disentangled_v1.step_snapshots import export_step
from scripts.launch.colorpeel_run import read_config, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
STEPS = (500, 750, 1000)


def recipes():
    return [recipe for concept in ("polished_metal", "rough_dielectric")
            for recipe in train.recipes() if recipe["branch"] == "material" and recipe["concept_id"] == concept]


def build_config(recipe, staging):
    if recipe not in recipes():
        raise ValueError("only the four Material recipes may be trained")
    baseline = step_ablation.build_config(recipe, staging)
    return {**baseline,
        "run": {**baseline["run"], "study": "material_uniform_albedo_v1",
                "variant": f"{recipe['concept_id']}_{recipe['method']}_uniform_steps1000"},
        "texture_material": {**baseline["texture_material"], "protocol_sha256": data.protocol_sha256()},
        "args": {**baseline["args"], "checkpoint_steps": list(STEPS)}}


def run(output, render_gpu, train_gpus):
    revision = git_info()
    output = resolve_run_dir(Path(output))
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError("use a new output directory outside the checkout")
    if render_gpu < 0 or not train_gpus or len(set(train_gpus)) != len(train_gpus) or min(train_gpus) < 0:
        raise ValueError("select valid distinct training GPUs")
    protocol, model = data.read_protocol(), train.verify_model()
    output.mkdir(parents=True)
    if shutil.disk_usage(output).free < 20 * (1 << 30):
        raise RuntimeError("less than20GiB free for new training snapshots")
    for name in ("logs", "training/configs", "training/logs", "training/audits", "train_outputs"):
        (output / name).mkdir(parents=True)
    report, batch = output / "train_outputs", output / "training"
    status = {"status": "preparing_data", "git": revision, "protocol": protocol,
              "protocol_sha256": data.protocol_sha256(), "model": model, "steps": [],
              "planned_training_runs": 4, "max_steps_per_run": 1000, "checkpoint_steps": list(STEPS),
              "resume": False, "texture_branch": "unchanged; no Texture data generation or training",
              "best_metric": None, "best_checkpoint": None, "completed_steps": 0}
    write_json(output / "status.json", status)
    write_json(report / "status.json", status)
    (report / "SUMMARY.md").write_text("Data preparation started; training is not yet verified. See status.json.\n", encoding="utf-8")
    (report / "LOG.md").write_text("See ../logs and ../training/gpu_*.json for current execution evidence.\n", encoding="utf-8")
    python, module = [sys.executable, "-B", "-m"], "experiments.material_uniform_albedo_v1."
    commands = [
        ("plan", [*python, module + "data", "plan", "--output", str(output / "plan")], None),
        ("render", [protocol["renderer"]["blender"], "--background", "--python-exit-code", "1",
            "--python", str(Path(__file__).with_name("render.py")), "--", "--plan",
            str(output / "plan/plan.json"), "--output", str(output / "renders"), "--gpu", str(render_gpu)], render_gpu),
        ("stage", [*python, module + "data", "stage", "--renders", str(output / "renders"),
                   "--output", str(output / "staging")], None)]
    (report / "COMMANDS.md").write_text("\n\n".join(
        f"{stage}\n\n```bash\n" + (f"CUDA_VISIBLE_DEVICES={gpu} " if gpu is not None else "")
        + shlex.join(command) + "\n```" for stage, command, gpu in commands) + "\n", encoding="utf-8")
    (report / "SCIENTIFIC_CHANGELOG.md").write_text(
        "Material-only dataset revision: uniform0.5 albedo times the original controlled linear RGB; "
        "fixed target R/M, no texture image/procedural pattern/normal/bump/displacement. "
        "Texture and old results remain unchanged. Caption omits texture. "
        "Other optimizer, model, seed, initialization, mask loss and learning rates match the old1000-step runs. "
        "Save only500/750/1000. No new loss or architecture.\n", encoding="utf-8")
    (report / "COMPARABILITY_REPORT.md").write_text(
        "Two shapes x two colors x two lights gives8 images per material, previously16 with two nuisance textures. "
        "No duplicate padding: fixed1000 updates now imply twice the average exposure per image. "
        "Constant albedo0.5 approximates the old normalized pattern mean; not per-image brightness matching. "
        "Fullrank KV LR1e-5; rank64 alpha64 KV LR5e-5; embedding LR1e-5, initializer metal, seed42. "
        "Execution and clean maps do not establish semantic independence.\n", encoding="utf-8")
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "DIFFUSERS_OFFLINE": "1"}
    records = []
    try:
        for stage, command, gpu in commands:
            if git_info() != revision:
                raise ValueError("source revision changed before preparation stage")
            state = {"stage": stage, "status": "running", "command": command}
            status["steps"].append(state)
            write_json(output / "status.json", status)
            with (output / "logs" / (stage + ".log")).open("x") as log:
                result = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, env={**environment,
                    **({"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else {})})
            state.update(status="succeeded" if result.returncode == 0 else "failed", returncode=result.returncode)
            write_json(output / "status.json", status)
            if result.returncode:
                raise RuntimeError(f"{stage} failed; see new output logs; old results preserved")
        status.update(dataset=data.verify_staging(output / "staging"), status="training")
        jobs, comparisons = [], []
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        for recipe in recipes():
            config = build_config(recipe, output / "staging")
            baseline = step_ablation.build_config(recipe, output / "staging")
            differences = {key: {"old": baseline["args"].get(key), "new": config["args"].get(key)}
                           for key in set(baseline["args"]) | set(config["args"])
                           if baseline["args"].get(key) != config["args"].get(key)}
            if set(differences) != {"checkpoint_steps"}:
                raise ValueError("unexpected scientific training-argument change")
            comparisons.append({"recipe": recipe, "argument_differences_at_same_staging": differences})
            variant = config["run"]["variant"]
            config_path = batch / "configs" / (variant + ".json")
            payload = json.dumps(config, indent=2).replace("1e-08", "0.00000001").replace(
                "1e-05", "0.00001").replace("5e-05", "0.00005")
            config_path.write_text(payload + "\n", encoding="utf-8")
            if read_config(config_path) != config:
                raise ValueError("new training config does not round-trip through the launcher")
            directory = batch / f"{stamp}__material_uniform_albedo_v1__{variant}__{revision['commit'][:7]}__42"
            jobs.append({"recipe": recipe, "config": config, "run": str(directory),
                         "command": [sys.executable, "-B", str(ROOT / "scripts/launch/colorpeel_run.py"),
                                     "--config", str(config_path), "--run-dir", str(directory)]})
        write_json(report / "CONFIG_COMPARISON.json", {"comparisons": comparisons})
        with (report / "COMMANDS.md").open("a", encoding="utf-8") as handle:
            handle.write("\n\n".join("```bash\n" + shlex.join(job["command"]) + "\n```" for job in jobs) + "\n")
        write_json(output / "status.json", status)
        write_json(report / "status.json", status)
        with ThreadPoolExecutor(max_workers=len(train_gpus)) as pool:
            futures = [pool.submit(train.run_queue, gpu, jobs[index::len(train_gpus)], batch)
                       for index, gpu in enumerate(train_gpus)]
            records = [item for future in futures for item in future.result()]
        status.update(completed_steps=sum(item.get("audit", {}).get("steps", 0) for item in records),
                      completed_training_runs=sum(item["status"] == "verified" for item in records))
        if len(records) != 4 or any(item["status"] != "verified" for item in records):
            raise RuntimeError("Material training or final export audit failed; see training/gpu_*.json")
        selections = [item["audit"] for item in records]
        write_json(batch / "training_index.json", {"git": train.git_info(), "git_full": revision,
            "protocol_sha256": data.protocol_sha256(), "study": "material_uniform_albedo_v1",
            "runs": selections, "dataset": status["dataset"], "texture_branch": "unchanged"})
        status.update(status="exporting", completed_steps=4000, completed_training_runs=4)
        write_json(output / "status.json", status)
        for step in STEPS:
            directory = output / f"step{step}"
            derived = [export_step(item, step, directory / "checkpoints" /
                       f"{item['method']}__{item['concept_id']}") for item in selections]
            write_json(directory / "training_index.json", {"git": train.git_info(), "git_full": revision,
                "study": "material_uniform_albedo_v1", "protocol_sha256": data.protocol_sha256(),
                "training_steps": step, "source_training_steps": 1000, "runs": derived,
                "dataset_audit_sha256": sha256(output / "staging/audit.json")})
        data.verify_staging(output / "staging")
        if git_info() != revision or data.read_protocol() != protocol:
            raise ValueError("source checkout or protocol changed during training")
        status.update(status="execution_verified", stop_reason="four1000-step runs audited; twelve paired exports saved")
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}", stop_reason="execution_error")
        raise
    finally:
        write_json(output / "status.json", status)
        write_json(report / "status.json", status)
        (report / "SUMMARY.md").write_text(
            f"Uniform-albedo Material training: {status['status']}. "
            f"Verified updates: {status['completed_steps']}/4000. No Texture retraining. "
            "Planned500/750/1000 exports; no inference requested or automatically launched.\n", encoding="utf-8")
        (report / "LOG.md").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-gpu", type=int, required=True)
    parser.add_argument("--train-gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    run(args.output, args.render_gpu, args.train_gpus)
