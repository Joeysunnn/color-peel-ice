"""Run the mailbox and metal_spoon initializer experiments without GPU overlap."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.natural_material_initializer_v1.run import ROOT
from experiments.natural_texture_material_v1.probe import write
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

SOURCES = ("mailbox", "metal_spoon")


def run(source_training, source_transfer, output, gpus):
    source_training, source_transfer = Path(source_training).resolve(), Path(source_transfer).resolve()
    output = resolve_run_dir(Path(output))
    if (output.exists() or any(output.is_relative_to(path) for path in (ROOT, source_training, source_transfer))
            or len(gpus) != 3 or len(set(gpus)) != 3 or min(gpus) < 0):
        raise ValueError("Use a fresh external output and three distinct GPUs")
    git = git_info()
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    report = output / "train_outputs"
    report.mkdir()
    jobs = [{"source_id": source, "output": str(output / source), "status": "queued",
        "command": [sys.executable, "-B", "-m", "experiments.natural_material_initializer_v1.run",
            "--source-training", str(source_training), "--source-transfer", str(source_transfer),
            "--output", str(output / source), "--source-id", source, "--gpus", *map(str, gpus)]}
        for source in SOURCES]
    state = {"status": "running", "git": git, "jobs": jobs, "max_steps": 4000,
        "completed_steps": 0, "completed_training_runs": 0, "planned_training_runs": 4,
        "checkpoint_steps": [500, 750, 1000], "best_metric": None, "best_checkpoint": None,
        "resume": False, "stop_reason": None}
    (report / "COMMANDS.md").write_text("\n\n".join(shlex.join(job["command"]) for job in jobs) + "\n")
    (report / "SCIENTIFIC_CHANGELOG.md").write_text(
        "Mailbox and metal_spoon Material only:initializer metal->material. No R/M,RGB,mask,caption,"
        "loss,architecture or other training argument change. Texture unchanged.\n")
    (report / "COMPARABILITY_REPORT.md").write_text(
        "Each source retains its own eight training RGB and original sampler/prompts. "
        "Source-specific configs,hash locks,paired exports and actual training references are checked by the shared runner.\n")
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Material initializer comparisons</title>'
        '<p>Mailbox runs first;metal_spoon follows automatically. Each source has separate500/750/1000 comparisons.</p>'
        + "".join(f'<p><a href="{source}/index.html">{source}</a></p>' for source in SOURCES))

    def save():
        write(output / "status.json", state)
        write(report / "status.json", state)
        (report / "SUMMARY.md").write_text(
            f"Material initializer batch:{state['status']}. Audited updates{state['completed_steps']}/4000. "
            "Sources execute sequentially;see source-level status for live progress.\n")

    save()
    try:
        for job in jobs:
            if git_info() != git:
                raise ValueError("Checkout changed before the next source")
            job["status"] = "running"
            save()
            with (output / "logs" / (job["source_id"] + ".log")).open("x") as log:
                result = subprocess.run(job["command"], cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
            job["returncode"] = result.returncode
            status_path = Path(job["output"]) / "status.json"
            summary = json.loads(status_path.read_text()) if status_path.exists() else {}
            job["summary"] = summary
            if (result.returncode or summary.get("status") != "execution_verified"
                    or summary.get("source_id") != job["source_id"] or summary.get("git") != git
                    or summary.get("completed_training_runs") != 2 or summary.get("completed_steps") != 2000
                    or summary.get("new_images") != 48 or summary.get("reused_images") != 144):
                job["status"] = "failed"
                raise RuntimeError(f"{job['source_id']} training/transfer incomplete;next source not started")
            job["status"] = "verified"
            state["completed_steps"] += 2000
            state["completed_training_runs"] += 2
            save()
        if git_info() != git:
            raise ValueError("Checkout changed during the batch")
        state.update(status="execution_verified", stop_reason="four training runs and both source transfers verified",
                     new_images=96, reused_images=288)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", stop_reason="execution_error")
        raise
    finally:
        save()
        (report / "LOG.md").write_text("Source logs:../logs/mailbox.log and ../logs/metal_spoon.log.\n"
                                      + json.dumps(jobs, indent=2))
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-training", type=Path, required=True)
    parser.add_argument("--source-transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=3, required=True)
    args = parser.parse_args()
    print(json.dumps({"status": run(args.source_training, args.source_transfer, args.output, args.gpus)["status"]}))
