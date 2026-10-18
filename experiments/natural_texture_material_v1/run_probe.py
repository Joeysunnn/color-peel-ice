"""Audit twelve completed runs, then generate all three neutral checkpoint probes."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.natural_texture_material_v1.probe_source import audit_training
from experiments.natural_texture_material_v1.probe import PROMPTS, STEPS, write
from experiments.natural_texture_material_v1.dataset_protocol import sha256
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]


def run(source, output, gpus):
    source, output = Path(source).resolve(), resolve_run_dir(Path(output))
    if (output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(source)
            or len(gpus) != 3 or len(set(gpus)) != 3 or min(gpus) < 0):
        raise ValueError("Use a new external output and three distinct nonnegative GPUs")
    git, prompt_hash = git_info(), sha256(PROMPTS)
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    report = output / "repro_outputs"
    report.mkdir()
    state = {"status": "auditing_training", "git": git, "source_run": str(source),
             "expected_images": 864, "steps": [], "retraining": False, "human_ratings": "pending"}
    write(output / "status.json", state)
    processes = []
    try:
        audit = audit_training(source)
        write(output / "training_audit.json", audit)
        state.update(training_git=audit["training_git"], completed_training_runs=audit["completed_training_runs"],
                     verified_training_updates=audit["completed_steps"], verified_paired_exports=audit["paired_exports"])
        table = ["| Source concept | Method | Updates | First loss | Last loss | Ordinary rows changed |",
                 "|---|---|---:|---:|---:|---:|"]
        for row in audit["runs"]:
            losses = row["metrics"]["total_loss"]
            table.append(f"| {row['concept_id']} | {row['method']} | {row['steps']} | "
                         f"{losses['first']:.6f} | {losses['last']:.6f} | 0 |")
        (report / "TRAINING_AUDIT.md").write_text(
            "Twelve runs/12000 finite-loss updates;36 paired exports tensor-verified. "
            "Losses are stochastic training observations, not disentanglement scores.\n\n" + "\n".join(table) + "\n",
            encoding="utf8")
        (report / "SCIENTIFIC_CHANGELOG.md").write_text(
            "Inference only. User-approved natural literal controls recorded in probe_protocol.json; "
            "prior neutral templates, contexts,seeds42/43,PNDM100,CFG3.5,512px,fp16,safety retained. "
            "Texture prompts/adapters omit Material; Material prompts/adapters omit Texture. "
            "Base/literal use native SD1.5. Learned arm loads one selected adapter/embedding only.\n", encoding="utf8")
        (report / "COMPARABILITY_REPORT.md").write_text(
            "288 images and24 contact sheets per step,864 images/72 sheets total. "
            "Actual T16/M8 staged RGB training images appear on each corresponding learned-column sheet. "
            "Native/literal have no concept-specific training. No automatic physical BRDF or independence score.\n",
            encoding="utf8")
        (output / "probe_protocol.json").write_bytes(PROMPTS.read_bytes())
        for step, gpu in zip(STEPS, gpus):
            if git_info() != git or sha256(PROMPTS) != prompt_hash:
                raise ValueError("Checkout or prompt protocol changed before inference")
            command = [sys.executable, "-B", "-m", "experiments.natural_texture_material_v1.probe",
                       "--source-run", str(source), "--output", str(output / f"step{step}"),
                       "--step", str(step), "--device", "cuda:0",
                       "--training-audit", str(output / "training_audit.json")]
            with (output / "logs" / f"step{step}.log").open("x") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONDONTWRITEBYTECODE": "1"})
            record = {"step": step, "gpu": gpu, "pid": process.pid, "status": "running", "command": command}
            state["steps"].append(record)
            processes.append((process, record))
        state["status"] = "generating"
        write(output / "status.json", state)
        write(report / "status.json", state)
        (report / "SUMMARY.md").write_text(
            "Twelve training runs and36 paired exports verified. Neutral inference generating864 images; "
            "completion pending.\n", encoding="utf8")
        (report / "LOG.md").write_text("Per-step generation logs: ../logs/step{500,750,1000}.log\n", encoding="utf8")
        (report / "COMMANDS.md").write_text("\n\n".join(
            f"CUDA_VISIBLE_DEVICES={job['gpu']} " + shlex.join(job["command"]) for job in state["steps"]) + "\n",
            encoding="utf8")
        (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Natural T/M probes</title>'
            '<p>Native / literal / learned branch. Training RGB references on each contact sheet.</p>' +
            "".join(f'<p><a href="step{step}/index.html">step{step}</a></p>' for step in STEPS), encoding="utf8")
        for process, record in processes:
            record["returncode"] = process.wait()
            folder = output / f"step{record['step']}"
            summary = json.loads((folder / "summary.json").read_text()) if (folder / "summary.json").exists() else {}
            sheets = summary.get("contact_sheets", [])
            complete = (record["returncode"] == 0 and summary.get("generated_images") == 288
                        and summary.get("missing_images") == 0 and summary.get("status") == "succeeded"
                        and summary.get("training_steps") == record["step"]
                        and len(sheets) == len(set(sheets)) == 24
                        and all((folder / path).is_file() for path in sheets))
            record["status"] = "succeeded" if complete else "failed"
            record["summary"] = summary
            write(output / "status.json", state)
        if any(record["status"] != "succeeded" for _, record in processes):
            raise RuntimeError("One or more step probes failed; see new output logs")
        if (git_info() != git or sha256(PROMPTS) != prompt_hash
                or any(sha256(path) != digest for path, digest in audit["source_locks"].items())):
            raise ValueError("Training source, actual training RGB, prompts or checkout changed")
        state.update(status="execution_verified", generated_images=864)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(output / "status.json", state)
        write(report / "status.json", state)
        (report / "SUMMARY.md").write_text(
            f"Natural neutral T/M inference: {state['status']}. Expected864 new images; "
            "see per-step summaries and training_audit.json. No retraining.\n", encoding="utf8")
        (report / "LOG.md").write_text("Per-step generation logs: ../logs/step{500,750,1000}.log\n", encoding="utf8")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=3, required=True)
    args = parser.parse_args()
    print(json.dumps({"status": run(args.source_run, args.output, args.gpus)["status"]}))
