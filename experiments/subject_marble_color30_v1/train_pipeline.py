"""Train original/rank64 on approved MARBLE data, then infer the three saved steps."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from experiments.subject_marble_color30_v1 import inference, stage_training, training
from experiments.subject_multicolor_multimaterial_b import run as baseline
from scripts.launch.colorpeel_run import git_info, resolve_run_dir


def commands(dataset, output, gpu):
    jobs = baseline.commands(dataset, output, gpu)
    for job in jobs:
        job["command"][3] = job["command"][3].replace("subject_multicolor_multimaterial_b", "subject_marble_color30_v1")
    return jobs


def run(candidates, output, gpu, execute=False, pink_cyan_only=False):
    candidates, output = Path(candidates).resolve(), resolve_run_dir(Path(output))
    if (type(gpu) is not int or gpu < 0 or output.exists() or output.is_relative_to(training.ROOT)
            or output.is_relative_to(candidates.parent) or candidates.is_relative_to(output)):
        raise ValueError("select a GPU and fresh external output outside the original candidate run")
    git = git_info()
    dataset = output / "training_dataset"
    dataset_audit = stage_training.stage(candidates, dataset, pink_cyan_only)
    count = dataset_audit["rows"]
    colors = dataset_audit["selected_colors"]
    originals = len(colors)
    training.build_configuration_bundle(dataset, output / "training_plan")
    jobs = commands(dataset, output, gpu)
    report = output / "train_outputs"
    report.mkdir()
    training._write(report / "COMMANDS.json", jobs)
    (report / "COMMANDS.md").write_text("\n\n".join(
        f"{job['name']} (cwd {training.ROOT}):\n```sh\n"
        + " ".join(f"{key}={shlex.quote(value)}" for key, value in job["environment"].items())
        + " " + shlex.join(job["command"]) + "\n```" for job in jobs) + "\n", encoding="utf8")
    (report / "LOG.md").write_text(
        "Serial isolated-GPU jobs: fullrank.log, lora_r64.log, step500.log, step750.log, step1000.log. "
        "Each method performs the existing shared-launcher preflight and writes its training audit.\n", encoding="utf8")
    (report / "SCIENTIFIC_CHANGELOG.md").write_text(
        f"Use {count - originals} approved MARBLE outputs plus {originals} byte-identical original recolors from colors {colors}. "
        "Keep SD1.5, mailbox initializer, S embedding, original full-rank or rank64/alpha64 token-local K/V, "
        "optimizer, seed42,1000 updates,500/750/1000 saves and original losses. No MARBLE inference is rerun.\n", encoding="utf8")
    if dataset_audit["source_mode"] == "foreground_only":
        with (report / "SCIENTIFIC_CHANGELOG.md").open("a", encoding="utf8") as document:
            document.write(f"This run uses {count} foreground-only candidates. "
                "All selected candidate PNGs/masks are copied byte-for-byte; no scene-background data are mixed in. "
                "Five no-attribute diagnostic outputs are excluded from training. No new extraction or generation is performed.\n")
    references = [json.loads(line) for line in (dataset / "training_assets_manifest.jsonl").read_text(encoding="utf8").splitlines()]
    exposures = {color: sum(references[i % count]["color"] == color for i in range(1000)) for color in colors}
    (report / "COMPARABILITY_REPORT.md").write_text(
        "build_config reuses B verbatim except study/status and data paths. CONFIG_COMPARISON.json records exact arguments. "
        "Same five prompts (metallic/matte/glossy plastic/ceramic/wood),seeds42/43,PNDM100,CFG3.5,512px,fp16,safety checker. "
        f"Sheets show all {count} actual training images and captions. Sequential DataLoader remains unchanged: "
        f"1000 updates={1000 // count} full {count}-image cycles+first{1000 % count}. Color exposures: {exposures}. "
        "Fewer unique images at fixed updates increases exposure per image. Extra originals are glossy metal, so M/R are not exactly balanced. "
        "Control labels are relative MARBLE supervision, not verified physical BRDF. Old artifacts stay untouched.\n", encoding="utf8")
    state = {"status": "planned", "git": git, "dataset": str(dataset), "candidates": str(candidates), "gpu": gpu,
        "source_mode": dataset_audit["source_mode"],
        "selected_colors": colors, "training_images": count,
        "execute": execute, "resume": False, "max_steps_per_method": 1000,
        "completed_steps": {method: 0 for method in training.METHODS}, "checkpoint_steps": list(inference.STEPS),
        "inference_steps": [], "best_metric": None, "best_checkpoint": None, "stop_reason": None}
    baseline._report(output, state)
    try:
        if not execute:
            state.update(status="dry_run_verified", stop_reason="execution not requested")
            return state
        selections = []
        for job in jobs:
            if git_info() != git:
                raise ValueError("checkout changed during MARBLE35 training/inference")
            state.update(status=job["phase"], active_job=job["name"])
            baseline._report(output, state)
            with (report / f"{job['name']}.log").open("x", encoding="utf8") as log:
                result = subprocess.run(job["command"], cwd=training.ROOT,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **job["environment"]},
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"{job['name']} failed; see train_outputs/{job['name']}.log")
            if job["phase"] == "training":
                directory = output / "training" / job["name"]
                status = training.read_json(directory / "status.json")
                index = training.read_json(directory / "training_index.json")
                runs = index.get("runs", [])
                if (status.get("status") != "training_verified" or status.get("completed_steps") != 1000
                        or index.get("git") != git or len(runs) != 1
                        or any(runs[0].get(key) != value for key, value in {
                            "status": "verified", "method": job["name"], "branch": "subject",
                            "token": "<S*>", "steps": 1000}.items())):
                    raise ValueError("training did not produce a completed Subject audit")
                selections.extend(runs)
                state["completed_steps"][job["name"]] = 1000
                if len(selections) == 2:
                    training._write(output / "training_index.json", {"git": git, "runs": selections})
            else:
                step = int(job["name"][4:])
                provenance = training.read_json(output / job["name"] / "provenance.json")
                if (provenance.get("status") != "succeeded" or provenance.get("generated_images") != 20
                        or provenance.get("step") != step or provenance.get("sampling") != inference.SAMPLING
                        or len(provenance.get("contact_sheets", [])) != 3):
                    raise ValueError("incomplete Subject material-prompt inference")
                state["inference_steps"].append(step)
            baseline._report(output, state)
        training.verify_dataset(dataset)
        if git_info() != git:
            raise ValueError("checkout changed during MARBLE35 training/inference")
        state.update(status="execution_verified", active_job=None, stop_reason="all requested jobs completed")
        return state
    except BaseException as error:
        state.update(status="failed", stop_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        baseline._report(output, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--pink-cyan-only", action="store_true")
    args = parser.parse_args()
    run(args.candidates, args.output, args.gpu, args.execute, args.pink_cyan_only)
