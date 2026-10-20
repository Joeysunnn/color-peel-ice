"""Run the approved Subject B training and then all three material-prompt grids."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.subject_multicolor_multimaterial_b import data, inference, training
from scripts.launch.colorpeel_run import git_info, read_config, resolve_run_dir


def commands(dataset, output, gpu):
    prefix = [sys.executable, "-B", "-m"]
    jobs = []
    for method in training.METHODS:
        jobs.append({"name": method, "phase": "training", "environment": {}, "command": [
            *prefix, "experiments.subject_multicolor_multimaterial_b.training",
            "--dataset", str(dataset), "--output", str(output / "training" / method),
            "--method", method, "--gpu", str(gpu), "--execute-training"]})
    for step in inference.STEPS:
        jobs.append({"name": f"step{step}", "phase": "inference",
                     "environment": {"CUDA_VISIBLE_DEVICES": str(gpu)}, "command": [
            *prefix, "experiments.subject_multicolor_multimaterial_b.inference",
            "--dataset", str(dataset), "--output", str(output / f"step{step}"),
            "--step", str(step), "--training-index", str(output / "training_index.json"),
            "--device", "cuda:0", "--execute-inference"]})
    return jobs


def _report(output, state):
    training._write(output / "status.json", state)
    training._write(output / "train_outputs/status.json", state)
    (output / "train_outputs/SUMMARY.md").write_text(
        f"Status: {state['status']}. Mode: full kickoff, no resume.\n"
        f"Completed audited training steps: {state['completed_steps']} (1000 per method).\n"
        f"Completed inference steps: {state['inference_steps']}; 20 images per step.\n"
        f"Stop reason: {state['stop_reason']}. Best metric/checkpoint: not selected.\n",
        encoding="utf-8")


def run(dataset, output, gpu, execute=False):
    if type(gpu) is not int or gpu < 0:
        raise ValueError("select a nonnegative GPU index")
    dataset, output = Path(dataset).resolve(), resolve_run_dir(Path(output))
    if (output.exists() or output.is_relative_to(data.ROOT)
            or output.is_relative_to(dataset.parent) or dataset.is_relative_to(output)):
        raise ValueError("use a fresh external output outside the approved data run")
    git = git_info()
    data.verify_dataset(dataset)
    # Reuse exactly the recipes shown alongside the accepted data preview.
    for method in training.METHODS:
        approved = dataset.parent / "training_plan/configs" / f"{method}.json"
        if read_config(approved) != training.build_config(method, dataset):
            raise ValueError("training recipe differs from the approved data preview")
    jobs = commands(dataset, output, gpu)
    report = output / "train_outputs"
    report.mkdir(parents=True)
    training._write(report / "COMMANDS.json", jobs)
    (report / "COMMANDS.md").write_text("\n\n".join(
        f"{job['name']} (cwd {data.ROOT}):\n```sh\n"
        + " ".join(f"{key}={shlex.quote(value)}" for key, value in job["environment"].items())
        + " " + shlex.join(job["command"]) + "\n```" for job in jobs) + "\n", encoding="utf-8")
    (report / "LOG.md").write_text(
        "Serial GPU execution; subprocess logs: fullrank.log, lora_r64.log, "
        "step500.log, step750.log, step1000.log. Detailed trainer logs and audits "
        "are in ../training/{method}/train_outputs/.\n", encoding="utf-8")
    (report / "SCIENTIFIC_CHANGELOG.md").write_text(
        "No trainer, loss, optimizer, dataset or sampling changes. The accepted "
        "Experiment B recipes train <S*> with original full-rank token-local K/V "
        "and rank64/alpha64 token-local K/V; seed42, 1000 updates, saves500/750/1000. "
        "The new entrypoint only schedules training, existing audits and inference.\n", encoding="utf-8")
    (report / "COMPARABILITY_REPORT.md").write_text(
        "The exact accepted ten RGB/mask pairs and material-conditioned captions "
        "are reused. Each recipe must equal its approved training_plan config. "
        "The existing balanced Subject baseline already used these images with "
        "different material captions; this is a caption ablation. Inference uses "
        "metallic, matte, glossy plastic, ceramic, wood; seeds42/43; PNDM100, "
        "CFG3.5, 512px, fp16, safety checker. No matched old750-step baseline "
        "or disentanglement score is claimed. All result sheets include the ten "
        "actual training images and full captions.\n", encoding="utf-8")
    state = {"status": "planned", "git": git, "dataset": str(dataset), "gpu": gpu,
             "execute": execute, "resume": False, "max_steps_per_method": 1000,
             "completed_steps": {method: 0 for method in training.METHODS},
             "checkpoint_steps": list(inference.STEPS), "inference_steps": [],
             "best_metric": None, "best_checkpoint": None, "stop_reason": None}
    _report(output, state)
    try:
        if not execute:
            state.update(status="dry_run_verified", stop_reason="execution not requested")
            return state
        selections = []
        for job in jobs:
            if git_info() != git:
                raise ValueError("checkout changed during Experiment B")
            state.update(status=job["phase"], active_job=job["name"])
            _report(output, state)
            with (report / f"{job['name']}.log").open("x", encoding="utf-8") as log:
                result = subprocess.run(job["command"], cwd=data.ROOT,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **job["environment"]},
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"{job['name']} failed; see train_outputs/{job['name']}.log")
            if job["phase"] == "training":
                method = job["name"]
                directory = output / "training" / method
                status = training.read_json(directory / "status.json")
                index = training.read_json(directory / "training_index.json")
                runs = index.get("runs", [])
                if (status.get("status") != "training_verified" or status.get("completed_steps") != 1000
                        or index.get("git") != git or len(runs) != 1
                        or any(runs[0].get(key) != value for key, value in {
                            "status": "verified", "method": method, "branch": "subject",
                            "token": "<S*>", "steps": 1000}.items())):
                    raise ValueError(f"{method} did not produce a completed Subject audit")
                selections.extend(runs)
                state["completed_steps"][method] = 1000
                if len(selections) == 2:
                    training._write(output / "training_index.json", {"git": git, "runs": selections})
            else:
                step = int(job["name"][4:])
                provenance = training.read_json(output / job["name"] / "provenance.json")
                if (provenance.get("status") != "succeeded" or provenance.get("generated_images") != 20
                        or provenance.get("step") != step or provenance.get("sampling") != inference.SAMPLING
                        or len(provenance.get("contact_sheets", [])) != 3):
                    raise ValueError(f"{job['name']} inference audit is incomplete")
                state["inference_steps"].append(step)
            _report(output, state)
        data.verify_dataset(dataset)
        if git_info() != git:
            raise ValueError("checkout changed during Experiment B")
        state.update(status="execution_verified", active_job=None, stop_reason="all requested jobs completed")
        return state
    except BaseException as error:
        state.update(status="failed", stop_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        _report(output, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    run(args.dataset, args.output, args.gpu, args.execute)


if __name__ == "__main__":
    main()
