"""Keep Phase 1 fixed and replay paired 500/750/1000-step checkpoints."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.texture_material_disentangled_v1 import data, infer, train
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd_native_comparison_v1.generate import git_info, write_json
from scripts.launch.colorpeel_run import read_config, resolve_run_dir

STEPS = (500, 750, 1000)
CHECKPOINTS = (300, *STEPS)
ROOT = Path(__file__).resolve().parents[2]


def build_config(recipe, staging):
    original = train.build_config(recipe, staging, "pilot")
    return {**original, "run": {**original["run"], "variant": original["run"]["variant"] + "_steps1000"},
            "args": {**original["args"], "max_train_steps": 1000, "checkpoint_steps": list(CHECKPOINTS)}}


def verify_source(source):
    source = Path(source).resolve()
    if read_json(source / "status.json").get("status") != "execution_verified":
        raise ValueError("expected the completed original Phase 1 source")
    index_path = source / "pilot/training_index.json"
    index = read_json(index_path)
    infer.verify_inputs(index, data.read_protocol())
    audit = data.verify_staging(source / "staging")
    expected = {(recipe["method"], recipe["branch"], recipe["concept_id"]): recipe for recipe in train.recipes()}
    for selection in index["runs"]:
        recipe = expected[(selection["method"], selection["branch"], selection["concept_id"])]
        run = Path(selection["run"])
        manifest = read_json(run / "manifest.json")
        if (read_config(run / "config.yaml") != train.build_config(recipe, source / "staging", "pilot")
                or manifest.get("status") != "succeeded" or manifest.get("returncode") != 0
                or manifest.get("git") != index["git"]):
            raise ValueError("original training recipe or provenance changed")
    return index, {"source": str(source), "source_git": index["git"], "dataset": audit,
                   "source_index_sha256": sha256(index_path),
                   "render_manifest_sha256": sha256(source / "renders/manifest.json")}


def run(source, output, gpus, inference_gpu):
    from experiments.texture_material_disentangled_v1.step_snapshots import export_step, compare_baseline
    revision = git_info()
    source, output = Path(source).resolve(), resolve_run_dir(Path(output))
    if output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(source):
        raise ValueError("step study must be a new external directory outside the original experiment")
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0 or inference_gpu < 0:
        raise ValueError("select distinct nonnegative training GPUs and a valid inference GPU")
    original_index, receipt = verify_source(source)
    model = train.verify_model()
    output.mkdir(parents=True)
    if shutil.disk_usage(output).free < 40 * (1 << 30):
        raise RuntimeError("less than 40 GiB free for the new training states")
    batch = output / "training"
    for name in ("configs", "logs", "audits", "train_outputs"):
        (batch / name).mkdir(parents=True)
    (output / "logs").mkdir()
    status = {"status": "training", "git": revision, "source": receipt, "model": model,
              "protocol_sha256": data.protocol_sha256(), "requested_steps": list(STEPS),
              "checkpoint_steps": list(CHECKPOINTS), "resume": False,
              "unchanged": "data, initialization, seed, optimizer, learning rates, architecture, prompts and sampling",
              "steps": []}
    write_json(output / "status.json", status)
    jobs = []
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    for recipe in train.recipes():
        config = build_config(recipe, source / "staging")
        variant = config["run"]["variant"]
        config_path = batch / "configs" / (variant + ".json")
        config_path.write_text(json.dumps(config, indent=2).replace("1e-08", "0.00000001").replace(
            "1e-05", "0.00001").replace("5e-05", "0.00005") + "\n", encoding="utf-8")
        if read_config(config_path) != config:
            raise ValueError("training recipe does not round-trip through the launcher")
        run_dir = batch / f"{stamp}__texture_material_disentangled_v1__{variant}__{revision['commit'][:7]}__42"
        jobs.append({"recipe": recipe, "config": config, "run": str(run_dir),
                     "command": [sys.executable, "-B", str(ROOT / "scripts/launch/colorpeel_run.py"),
                                 "--config", str(config_path), "--run-dir", str(run_dir)]})
    report = batch / "train_outputs"
    import shlex
    (report / "COMMANDS.md").write_text("\n\n".join(
        "```bash\n" + shlex.join(job["command"]) + "\n```" for job in jobs) + "\n", encoding="utf-8")
    (report / "SCIENTIFIC_CHANGELOG.md").write_text(
        "Only max_train_steps changes from300to1000; explicit save steps300/500/750/1000. "
        "Old data and300-step results are read-only. No resume support is assumed. "
        "Each new300-step snapshot must pass a recorded numerical comparison with its original final adapter and token.\n", encoding="utf-8")
    (report / "COMPARABILITY_REPORT.md").write_text(
        "Same eight concepts/methods and exact original staged files; same seed42 and constant LR. "
        "Same256-image prompts, seeds42/43, PNDM100/CFG3.5/512/fp16/safety checker per selected step. "
        "No claim of improved disentanglement follows from a longer training dose.\n", encoding="utf-8")
    records = []
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(train.run_queue, gpu, jobs[index::len(gpus)], batch)
                       for index, gpu in enumerate(gpus)]
            records = [item for future in futures for item in future.result()]
        if len(records) != 8 or any(item["status"] != "verified" for item in records):
            raise RuntimeError("1000-step training or final export audit failed; see training/gpu_*.json")
        final = [item["audit"] for item in records]
        originals = {(item["method"], item["branch"], item["concept_id"]): item for item in original_index["runs"]}
        prefixes = [compare_baseline(item, originals[(item["method"], item["branch"], item["concept_id"])],
                                     allow_roundoff=True)
                    for item in final]
        write_json(output / "baseline300_comparison.json", {"comparisons": prefixes})
        status.update(status="generating", completed_training_runs=8, completed_training_steps=8000,
                      baseline300_matches=all(item["adapter_equal"] and item["token_equal"] for item in prefixes),
                      baseline300_numerically_equivalent=True)
        write_json(output / "status.json", status)
        for step in STEPS:
            directory = output / f"step{step}"
            directory.mkdir()
            selections = [export_step(item, step, directory / "checkpoints" /
                          f"{item['method']}__{item['branch']}__{item['concept_id']}") for item in final]
            index_path = directory / "training_index.json"
            write_json(index_path, {"git": train.git_info(), "git_full": revision, "mode": "pilot",
                "protocol_sha256": data.protocol_sha256(), "runs": selections,
                "training_steps": step, "source_training_steps": 1000,
                "source_receipt": receipt, "baseline300_comparison_sha256": sha256(output / "baseline300_comparison.json")})
            state = {"step": step, "status": "running", "output": str(directory)}
            status["steps"].append(state)
            write_json(output / "status.json", status)
            commands = [
                ("inference", [sys.executable, "-B", "-m", "experiments.texture_material_disentangled_v1.infer",
                    "--training-index", str(index_path), "--output", str(directory / "inference"),
                    "--training-steps", str(step), "--device", "cuda:0"]),
                ("evaluation", [sys.executable, "-B", "-m", "experiments.texture_material_disentangled_v1.evaluate",
                    "--inference", str(directory / "inference"), "--renders", str(source / "renders"),
                    "--output", str(directory / "evaluation")]),
            ]
            for stage, command in commands:
                with (output / "logs" / f"step{step}_{stage}.log").open("x") as log:
                    state[stage + "_command"] = command
                    result = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(inference_gpu),
                        "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "DIFFUSERS_OFFLINE": "1"})
                if result.returncode:
                    raise RuntimeError(f"step{step} {stage} failed; old outputs preserved")
            state["status"] = "succeeded"
            write_json(output / "status.json", status)
        verify_source(source)
        if git_info() != revision:
            raise ValueError("source checkout changed during the experiment")
        status["status"] = "execution_verified"
        (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>T/M training steps</title>'
            + "".join(f'<p><a href="step{step}/evaluation/index.html">{step} training steps</a></p>'
                      for step in STEPS), encoding="utf-8")
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "status.json", status)
        write_json(report / "status.json", status)
        (report / "SUMMARY.md").write_text(
            f"Step study: {status['status']}. Requested500/750/1000; same original data and inference matrix. "
            "8000 planned training updates,768 planned inference images. Scientific quality is not established by execution.\n",
            encoding="utf-8")
        (report / "LOG.md").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--inference-gpu", type=int, required=True)
    args = parser.parse_args()
    run(args.source_run, args.output, args.gpus, args.inference_gpu)
