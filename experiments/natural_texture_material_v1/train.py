"""Train the twelve frozen natural T/M recipes and export paired step snapshots."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from experiments.natural_texture_material_v1 import dataset_protocol as schema
from experiments.natural_texture_material_v1.training_config import STEPS, build_config, compare_config, recipes
from experiments.natural_texture_material_v1.training_source import verify_dataset
from experiments.texture_material_disentangled_v1 import train as shared
from experiments.texture_material_disentangled_v1.step_snapshots import export_step
from scripts.launch.colorpeel_run import git_info, read_config, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def run(source, output, gpus):
    source, output = Path(source).resolve(), resolve_run_dir(Path(output))
    if (output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(source)
            or not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0):
        raise ValueError("Use a new external training directory and distinct nonnegative GPUs")
    git = git_info()
    protocol, dataset, locks = verify_dataset(source)
    model = shared.verify_model()
    if shutil.disk_usage(output.parent).free < 40 * (1 << 30):
        raise RuntimeError("Less than40GiB free for twelve new training runs and snapshots")
    output.mkdir(parents=True)
    batch, report = output / "training", output / "train_outputs"
    for folder in (batch / "configs", batch / "logs", batch / "audits", output / "preflight", report):
        folder.mkdir(parents=True)
    state = {"status": "preflight", "git": git, "source_dataset": str(source),
             "source_dataset_git": json.loads((source / "status.json").read_text())["git"],
             "protocol_sha256": schema.protocol_sha256(protocol), "dataset": dataset, "model": model,
             "planned_training_runs": 12, "max_steps": 12000, "max_steps_per_run": 1000,
             "checkpoint_steps": list(STEPS), "completed_steps": 0, "completed_training_runs": 0,
             "best_metric": None, "best_checkpoint": None, "stop_reason": None,
             "resume": False, "inference_launched": False, "gpus": list(gpus)}
    write(output / "source_locks.json", locks)
    write(output / "protocol.json", protocol)
    source_refs = source / "staging/training_references.json"
    references = json.loads(source_refs.read_text())
    write(output / "training_references.json", {
        "source": str(source_refs), "source_sha256": schema.sha256(source_refs),
        "groups": {concept: {key: group[key] for key in ("source_id", "branch", "source", "image_count",
                   "training_assets_manifest", "images")} for concept, group in references.items()},
        "recipes": recipes(), "steps": list(STEPS), "same_dataset_for_original_and_rank64": True})
    jobs, comparisons, completed = [], [], []
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        for recipe in recipes():
            config = build_config(recipe, source / "staging", protocol)
            comparisons.append(compare_config(recipe, config, source / "staging", protocol))
            variant = config["run"]["variant"]
            path = batch / "configs" / f"{variant}.json"
            text = json.dumps(config, indent=2).replace("1e-08", "0.00000001").replace(
                "1e-05", "0.00001").replace("5e-05", "0.00005")
            path.write_text(text + "\n", encoding="utf8")
            if read_config(path) != config:
                raise ValueError("Training config fails the shared launcher's YAML roundtrip")
            name = f"{stamp}__natural_texture_material_v1__{variant}__{git['commit'][:7]}__42"
            command = [sys.executable, "-B", str(ROOT / "scripts/launch/colorpeel_run.py"),
                       "--config", str(path), "--run-dir", str(batch / name)]
            jobs.append({"recipe": recipe, "config": config, "run": str(batch / name), "command": command})
        write(output / "jobs.json", jobs)
        write(report / "CONFIG_COMPARISON.json", {"comparisons": comparisons})
        write(output / "status.json", state)
        write(report / "status.json", state)
        (report / "COMMANDS.md").write_text("\n\n".join(
            f"GPU{gpus[index % len(gpus)]}: " + shlex.join(job["command"])
            for index, job in enumerate(jobs)) +
            "\n\nEach config is first dry-run in the separate ../preflight/ directory.\n", encoding="utf8")
        (report / "SCIENTIFIC_CHANGELOG.md").write_text(
            "Train the accepted natural72-image dataset only; no regeneration. Six independent concepts, "
            "two methods each. Embeddings plus token-local K/V, frozen base and ordinary embedding rows; "
            "no new loss or architecture, no Subject/Color modification. Save500/750/1000 paired snapshots.\n",
            encoding="utf8")
        (report / "COMPARABILITY_REPORT.md").write_text(
            "All training args match the existing Phase1 same-method/branch1000-step recipe except "
            "natural concept data path and explicit snapshot list. Fullrank KV LR1e-5, rank64/alpha64 KV "
            "LR5e-5, embedding LR1e-5, seed42, batch1, mixed_precision=no. T16/M8 images per source, "
            "same data across methods; different per-image exposures across branches at equal1000 updates. "
            "Natural extraction limitations remain. No inference or disentanglement score is produced.\n",
            encoding="utf8")
        (report / "SUMMARY.md").write_text(
            "Twelve1000-step runs requested; launcher preflight in progress. Training has not started.\n",
            encoding="utf8")
        (report / "LOG.md").write_text(
            "Preflight: ../preflight/*.log. Live queues: ../training/gpu_*.json. "
            "Trainer logs: ../training/*/logs/stdout.log.\n", encoding="utf8")
        for job in jobs:
            if git_info() != git:
                raise ValueError("Checkout changed before training preflight")
            directory = output / "preflight" / Path(job["run"]).name
            command = [*job["command"][:-1], str(directory), "--dry-run"]
            with (output / "preflight" / (job["config"]["run"]["variant"] + ".log")).open("x") as log:
                result = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT)
            receipt = json.loads((directory / "manifest.json").read_text()) if directory.exists() else {}
            if (result.returncode or receipt.get("status") != "dry-run" or receipt.get("git") != git
                    or read_config(directory / "config.yaml") != job["config"]):
                raise RuntimeError("Shared training launcher dry-run failed; see preflight logs")
        if any(schema.sha256(path) != digest for path, digest in locks.items()):
            raise ValueError("Frozen dataset changed before starting training")
        state["status"] = "training"
        write(output / "status.json", state)
        write(report / "status.json", state)
        (report / "SUMMARY.md").write_text(
            "Full kickoff: twelve1000-step natural T/M runs queued. Training in progress; completion unverified.\n",
            encoding="utf8")
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(shared.run_queue, gpu, jobs[index::len(gpus)], batch)
                       for index, gpu in enumerate(gpus)]
            completed = [record for future in futures for record in future.result()]
        state.update(completed_steps=sum(record.get("audit", {}).get("steps", 0) for record in completed),
                     completed_training_runs=sum(record["status"] == "verified" for record in completed))
        if len(completed) != 12 or any(record["status"] != "verified" for record in completed):
            raise RuntimeError("Training or final checkpoint audit failed; see training/gpu_*.json")
        selections = [record["audit"] for record in completed]
        write(batch / "training_index.json", {"git": git, "runs": selections,
              "source_dataset": str(source), "protocol_sha256": state["protocol_sha256"]})
        state["status"] = "exporting"
        write(output / "status.json", state)
        for step in STEPS:
            folder = output / f"step{step}"
            derived = [export_step(selection, step, folder / "checkpoints" /
                       f"{selection['method']}__{selection['concept_id']}") for selection in selections]
            write(folder / "training_index.json", {"git": git, "training_git": git,
                "dataset_git": state["source_dataset_git"], "protocol_sha256": state["protocol_sha256"],
                "training_steps": step, "source_training_steps": 1000, "runs": derived,
                "training_references": str(output / "training_references.json"),
                "training_references_sha256": schema.sha256(output / "training_references.json"),
                "source_dataset": str(source), "source_locks_sha256": schema.sha256(output / "source_locks.json")})
        verify_dataset(source)
        if git_info() != git or any(schema.sha256(path) != digest for path, digest in locks.items()):
            raise ValueError("Dataset or checkout changed during training")
        state.update(status="execution_verified", stop_reason="twelve1000-step runs audited;36 paired exports saved")
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", stop_reason="execution_error")
        raise
    finally:
        write(output / "status.json", state)
        write(report / "status.json", state)
        (report / "SUMMARY.md").write_text(
            f"Natural T/M training: {state['status']}. Audited updates{state['completed_steps']}/12000. "
            "Full kickoff from base, no resume or inference. See GPU queues for active-job progress.\n", encoding="utf8")
        (report / "LOG.md").write_text(
            "Live queues: ../training/gpu_*.json; actual trainer logs: ../training/*/logs/stdout.log.\n" +
            json.dumps(completed, indent=2) + "\n", encoding="utf8")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps({"status": run(args.dataset, args.output, args.gpus)["status"]}))
