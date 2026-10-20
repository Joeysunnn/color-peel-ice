"""Train approved Subject C original/rank64, then infer all three paired steps."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess

from experiments.subject_albedo_pbr_c import data, inference, training
from experiments.subject_multicolor_multimaterial_b import run as baseline
from scripts.launch.colorpeel_run import git_info, resolve_run_dir


def commands(dataset, output, gpu):
    jobs = baseline.commands(dataset, output, gpu)
    for job in jobs:
        job["command"][3] = job["command"][3].replace(
            "subject_multicolor_multimaterial_b", "subject_albedo_pbr_c")
    return jobs


def run(dataset, output, gpu, execute=False):
    dataset, output = Path(dataset).resolve(), resolve_run_dir(Path(output))
    if (type(gpu) is not int or gpu < 0 or output.exists()
            or output.is_relative_to(data.ROOT) or output.is_relative_to(dataset)
            or dataset.is_relative_to(output)):
        raise ValueError("select a GPU and fresh external output outside the dataset and its parents")
    git = git_info()
    data.verify_dataset(dataset)
    training.build_configuration_bundle(dataset, output / "training_plan")
    jobs = commands(dataset, output, gpu)
    report = output / "train_outputs"
    report.mkdir()
    training._write(report / "COMMANDS.json", jobs)
    (report / "COMMANDS.md").write_text("\n\n".join(
        f"{job['name']} (cwd {data.ROOT}):\n```sh\n"
        + " ".join(f"{key}={shlex.quote(value)}" for key, value in job["environment"].items())
        + " " + shlex.join(job["command"]) + "\n```" for job in jobs) + "\n", encoding="utf8")
    (report / "LOG.md").write_text(
        "Serial isolated-GPU execution: fullrank.log, lora_r64.log, step500.log, "
        "step750.log, step1000.log. Each trainer first performs launcher dry-run "
        "validation. Detailed trainer audits are in ../training/{method}/train_outputs/.\n", encoding="utf8")
    (report / "SCIENTIFIC_CHANGELOG.md").write_text(
        "Replace B's ten RGB-edited metallic/matte pairs with the accepted C thirty "
        "fixed-Albedo Cycles A/M/R RGBs and their color/material captions. Same S token, "
        "mailbox initializer, original full-rank or rank64/alpha64 token-local K/V; "
        "unchanged trainer, losses, optimizers, seed42 and1000 updates with500/750/1000 saves. "
        "More unique training images changes exposure per image at fixed updates.\n", encoding="utf8")
    (report / "COMPARABILITY_REPORT.md").write_text(
        "C build_config reuses B build_config verbatim except run/status metadata and "
        "dataset paths. CONFIG_COMPARISON.json records frozen baseline arguments. "
        "Inference exactly reuses B's metallic, matte, glossy plastic, ceramic, wood "
        "prompts; seeds42/43, PNDM100, CFG3.5,512px,fp16 and safety checker. "
        "Every result sheet includes all30 actual training RGBs and captions. "
        "Manual2.5D scene and Albedo-extraction limits are retained in the data audit. "
        "No old baseline is regenerated or overwritten; no disentanglement score is implied.\n", encoding="utf8")
    state = {"status": "planned", "git": git, "dataset": str(dataset), "gpu": gpu,
             "execute": execute, "resume": False, "max_steps_per_method": 1000,
             "completed_steps": {method: 0 for method in training.METHODS},
             "checkpoint_steps": list(inference.STEPS), "inference_steps": [],
             "best_metric": None, "best_checkpoint": None, "stop_reason": None}
    baseline._report(output, state)
    try:
        if not execute:
            state.update(status="dry_run_verified", stop_reason="execution not requested")
            return state
        selections = []
        for job in jobs:
            if git_info() != git:
                raise ValueError("checkout changed during Experiment C")
            state.update(status=job["phase"], active_job=job["name"])
            baseline._report(output, state)
            with (report / f"{job['name']}.log").open("x", encoding="utf8") as log:
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
                    raise ValueError(f"{method} has no completed Subject audit")
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
                    raise ValueError(f"{job['name']} inference is incomplete")
                state["inference_steps"].append(step)
            baseline._report(output, state)
        data.verify_dataset(dataset)
        if git_info() != git:
            raise ValueError("checkout changed during Experiment C")
        state.update(status="execution_verified", active_job=None, stop_reason="all requested jobs completed")
        return state
    except BaseException as error:
        state.update(status="failed", stop_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        baseline._report(output, state)


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
