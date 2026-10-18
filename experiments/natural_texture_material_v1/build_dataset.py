"""Build, render, verify and display natural counterfactual datasets without training."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.natural_texture_material_v1 import dataset_protocol as schema
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def run(extraction, output, gpu):
    output, extraction = resolve_run_dir(Path(output)), Path(extraction).resolve()
    if (output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(extraction)
            or gpu < 0):
        raise ValueError("Select a nonnegative GPU and a new external dataset directory")
    git, protocol = git_info(), schema.build_protocol(extraction)
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    report = output / "repro_outputs"
    report.mkdir()
    steps = [
        ("plan", [sys.executable, "-B", "-m", "experiments.natural_texture_material_v1.dataset_protocol",
                  "--extraction", str(extraction), "--output", str(output / "plan")]),
        ("render", [protocol["renderer"]["blender"], "--background", "--python-exit-code", "1",
                    "--python", str(Path(__file__).with_name("render_dataset.py")), "--",
                    "--plan", str(output / "plan/plan.json"), "--output", str(output / "renders"),
                    "--gpu", str(gpu)]),
        ("stage", [sys.executable, "-B", "-m", "experiments.natural_texture_material_v1.dataset_stage",
                   "--renders", str(output / "renders"), "--output", str(output / "staging")])]
    state = {"status": "preparing", "phase": 2, "stage": "counterfactual_dataset_only", "git": git,
             "extraction": str(extraction), "expected_images": 72, "training_launched": False,
             "gpu": gpu, "steps": []}
    write(output / "commands.json", {"cwd": str(ROOT), "gpu": gpu, "steps": steps})
    (report / "SCIENTIFIC_CHANGELOG.md").write_text(
        "Dataset only. Fixed accepted natural grayscale Texture with varied geometry/color/light/material; "
        "fixed SuperMat scalar R/M with uniform0.5 albedo and varied geometry/color/light. "
        "Reuse unchanged Phase1 renderers,UV,64 Cycles samples,512px,seed42+condition. "
        "No training, checkpoint export, model inference, new loss or architecture.\n", encoding="utf8")
    (report / "COMPARABILITY_REPORT.md").write_text(
        "Three sources;16Texture and8Material actual RGB training images per source. "
        "Both future adaptation methods use the same staged images. "
        "Accepted mailbox patch52x13 has limited detail; metal Albedo may retain illumination; "
        "wood roughness/gloss discrepancy remains. Dataset integrity is not evidence of learned disentanglement.\n",
        encoding="utf8")
    try:
        for name, command in steps:
            if git_info() != git:
                raise ValueError("Checkout changed between dataset stages")
            record = {"stage": name, "status": "running", "command": command}
            state["steps"].append(record)
            state["status"] = name
            write(output / "status.json", state)
            environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            if name == "render":
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            with (output / "logs" / f"{name}.log").open("x") as log:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT)
            record.update(returncode=result.returncode, status="succeeded" if result.returncode == 0 else "failed")
            if result.returncode:
                raise RuntimeError(f"Dataset {name} failed; see logs/{name}.log")
        from experiments.natural_texture_material_v1.dataset_stage import verify_staging
        audit = verify_staging(output / "staging")
        if (git_info() != git or any(schema.sha256(path) != digest
                for path, digest in protocol["source_locks"].items())):
            raise ValueError("Checkout or accepted extraction changed during dataset construction")
        state.update(status="dataset_verified", dataset=audit)
        (output / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Natural counterfactual datasets</title>'
            '<p>Dataset prepared and verified. Training has not started.</p>'
            '<p><a href="staging/index.html">Six datasets and all actual training RGB images</a></p>'
            '<p><a href="status.json">Execution audit</a></p>', encoding="utf8")
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(output / "status.json", state)
        write(report / "status.json", state)
        (report / "SUMMARY.md").write_text(
            f"Phase2 dataset status: {state['status']}. Training not started. See ../status.json.\n", encoding="utf8")
        (report / "COMMANDS.md").write_text("Exact invocation arrays, cwd and GPU: ../commands.json\n", encoding="utf8")
        (report / "LOG.md").write_text("Execution logs: ../logs/{plan,render,stage}.log\n", encoding="utf8")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps({"status": run(args.extraction, args.output, args.gpu)["status"]}))
