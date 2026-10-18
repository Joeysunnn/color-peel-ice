"""Audit completed Material training and replay the unchanged three-arm probe."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.material_uniform_albedo_v1 import audit, compare
from experiments.sd15_token_local_kv_v1.contract import sha256
from experiments.sd_native_comparison_v1.generate import git_info, write_json
from experiments.texture_material_disentangled_v1.material_probe import STEPS
from scripts.launch.colorpeel_run import resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]


def run_step(source, old, directory, step, gpu):
    (directory / "logs").mkdir(parents=True)
    state = {"step": step, "gpu": gpu, "status": "running", "stages": []}
    commands = [
        ("probe", [sys.executable, "-B", "-m", "experiments.texture_material_disentangled_v1.material_probe",
                   "--source-run", str(source), "--output", str(directory / "probe"),
                   "--step", str(step), "--device", "cuda:0", "--uniform-material"]),
        ("comparison", [sys.executable, "-B", "-m", "experiments.material_uniform_albedo_v1.compare",
                        "--old", str(old / f"step{step}"), "--new", str(directory / "probe"),
                        "--output", str(directory / "comparison")])]
    try:
        for name, command in commands:
            stage = {"stage": name, "command": command, "status": "running"}
            state["stages"].append(stage)
            write_json(directory / "status.json", state)
            with (directory / "logs" / (name + ".log")).open("x") as handle:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle,
                    stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1", "DIFFUSERS_OFFLINE": "1"})
                stage["pid"] = child.pid
                write_json(directory / "status.json", state)
                stage["returncode"] = child.wait()
            stage["status"] = "succeeded" if stage["returncode"] == 0 else "failed"
            if stage["returncode"]:
                raise RuntimeError(f"step{step} {name} failed; see its new logs")
        state["status"] = "succeeded"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(directory / "status.json", state)
    return state


def run(source, old_results, output, gpus):
    source, old = Path(source).resolve(), Path(old_results).resolve()
    output, revision = resolve_run_dir(Path(output)), git_info()
    if output.exists() or any(output.is_relative_to(path) for path in (source, old, ROOT)):
        raise ValueError("evaluation requires a new external directory outside both source experiments")
    if len(gpus) != 3 or len(set(gpus)) != 3 or min(gpus) < 0:
        raise ValueError("select three distinct nonnegative GPUs, one per step")
    output.mkdir(parents=True)
    report = output / "repro_outputs"
    report.mkdir()
    status = {"status": "auditing", "git": revision, "training_source": str(source),
              "old_results": str(old), "requested_steps": list(STEPS), "gpus": gpus,
              "planned_images": 288, "retraining": False, "steps": []}
    write_json(output / "status.json", status)
    (report / "COMMANDS.md").write_text("```bash\n" + shlex.join([sys.executable, "-B", "-m",
        "experiments.material_uniform_albedo_v1.evaluate", "--source-run", str(source),
        "--old-results", str(old), "--output", str(output), "--gpus", *map(str, gpus)]) + "\n```\n",
        encoding="utf-8")
    (report / "SCIENTIFIC_CHANGELOG.md").write_text(
        "Inference only: unchanged96-row neutral Material probe at each step. Native/literal controls use "
        "unadapted SD1.5; learned arm loads only the selected Material adapter/token. "
        "No new training or extraction. Old data, checkpoints and images are read-only.\n", encoding="utf-8")
    (report / "COMPARABILITY_REPORT.md").write_text(
        "Exact old prompts, seeds42/43, four contexts, PNDM100, CFG3.5,512px, fp16 and safety checker. "
        "New uniform8-image datasets replace old patterned16-image datasets; fixed1000 updates give more "
        "exposures per new image. Matched visual sheets and blind pairs do not alone establish physical "
        "BRDF recovery or a calibrated texture leakage score. Control pixel equality is diagnostic only.\n",
        encoding="utf-8")
    try:
        training = audit.audit_training(source)
        write_json(output / "training_audit.json", training)
        old_locks = {}
        for step in STEPS:
            _, _, _, locks = compare.validate_probe(old / f"step{step}")
            old_locks.update(locks)
        status.update(status="generating", training_audit="training_audit.json", old_source_locks=old_locks)
        write_json(output / "status.json", status)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(run_step, source, old, output / f"step{step}", step, gpu)
                       for step, gpu in zip(STEPS, gpus)]
            status["steps"] = [future.result() for future in futures]
        source_locks = {**training.get("source_locks", {}), **old_locks}
        if git_info() != revision or any(sha256(Path(path)) != digest for path, digest in source_locks.items()):
            raise ValueError("checkout, training artifacts or old results changed during comparison")
        status["status"] = "execution_verified"
        status["generated_images"] = 288
        status["scientific_review"] = "pending; inspect matched sheets and blind pairs"
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "status.json", status)
        write_json(report / "status.json", status)
        (report / "SUMMARY.md").write_text(f"Material comparison: {status['status']}. "
            "Three steps,288 planned new images; original and rank64. "
            "No disentanglement score inferred from execution.\n", encoding="utf-8")
        (report / "LOG.md").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Material comparison</title>'
        + "".join(f'<p><a href="step{step}/comparison/index.html">{step}: old/new comparison</a> | '
                  f'<a href="step{step}/probe/index.html">new three-arm probe</a></p>' for step in STEPS), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--old-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs=3, type=int, required=True)
    args = parser.parse_args()
    run(args.source_run, args.old_results, args.output, args.gpus)
