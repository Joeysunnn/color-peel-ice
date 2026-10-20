"""Prepare and validate Experiment B; this entrypoint never starts training."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

from experiments.subject_multicolor_multimaterial_b.data import read_json, stage, verify_dataset, write_json
from experiments.subject_multicolor_multimaterial_b.review import write_review
from experiments.subject_multicolor_multimaterial_b.inference import run as prepare_inference
from experiments.subject_multicolor_multimaterial_b.training import ROOT, STUDY, build_configuration_bundle
from scripts.launch.colorpeel_run import git_info, read_config, resolve_run_dir


def run(original_root, reviewed_root, output):
    original_root, reviewed_root = Path(original_root).resolve(), Path(reviewed_root).resolve()
    output = resolve_run_dir(Path(output))
    if output.exists() or any(output.is_relative_to(path) for path in (ROOT, original_root, reviewed_root)):
        raise ValueError("Use a fresh external Experiment B directory")
    git = git_info()
    output.mkdir(parents=True)
    state = {"status": "preparing_data", "git": git, "training_started": False,
             "inference_executed": False, "training_rows": 10, "preflights": []}
    write_json(output / "status.json", state)
    try:
        dataset = output / "dataset"
        stage(original_root, reviewed_root, dataset)
        state["review"] = write_review(dataset, output / "review")
        bundle = build_configuration_bundle(dataset, output / "training_plan")
        state["configs"] = bundle["paths"]
        preflight = output / "preflight"
        preflight.mkdir()
        for method, config in bundle["configs"].items():
            name = f"{datetime.now():%Y%m%d-%H%M%S}__{STUDY}__{config['run']['variant']}__{git['commit'][:7]}__42"
            directory = preflight / name
            command = [sys.executable, "-B", str(ROOT / "scripts/launch/colorpeel_run.py"),
                       "--config", bundle["paths"][method], "--run-dir", str(directory), "--dry-run"]
            record = {"method": method, "command": command, "run": str(directory)}
            with (preflight / (method + ".log")).open("x") as log:
                result = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            if (result.returncode or read_json(directory / "manifest.json").get("status") != "dry-run"
                    or read_config(directory / "config.yaml") != config):
                raise RuntimeError(f"{method} configuration preflight failed")
            record["status"] = "dry-run"
            state["preflights"].append(record)
            write_json(output / "status.json", state)
        state["inference_plans"] = []
        for step in (500, 750, 1000):
            folder = output / "inference_plan" / f"step{step}"
            result = prepare_inference(dataset, folder, step)
            if result["status"] != "dry_run" or result["generated_images"] != 0:
                raise RuntimeError("Inference planning unexpectedly executed a model")
            state["inference_plans"].append({"step": step, "output": str(folder),
                "status": "dry_run", "expected_images": result["expected_images"], "generated_images": 0})
        state["dataset_audit"] = verify_dataset(dataset)
        if git_info() != git:
            raise ValueError("Checkout changed during data preparation")
        state["status"] = "data_and_configuration_verified"
        (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Subject Experiment B</title>'
            '<h1>Experiment B prepared; training has not started</h1><p><a href="review/index.html">Actual training images and captions</a></p>'
            '<p><a href="review/training_pairs.jpg">Full contact sheet</a></p>'
            '<p><a href="training_plan/CONFIG_COMPARISON.json">Exact baseline configuration differences</a></p>'
            + ''.join(f'<p><a href="inference_plan/step{step}/index.html">step{step} inference plan; no generated images</a></p>'
                      for step in (500, 750, 1000)), encoding="utf8")
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "status.json", state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--reviewed-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps({"status": run(args.original_root, args.reviewed_root, args.output)["status"]}))
