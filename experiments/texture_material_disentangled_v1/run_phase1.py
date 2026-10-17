"""Execute only the bounded Phase 1 oracle experiment, preserving every old run."""

import argparse
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd_native_comparison_v1.generate import git_info, write_json
from scripts.launch.colorpeel_run import resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("protocol.json")


def run(output, render_gpu, train_gpus):
    revision = git_info()
    output = resolve_run_dir(Path(output))
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError("Phase 1 requires a new external output directory")
    if render_gpu < 0 or not train_gpus or len(set(train_gpus)) != len(train_gpus) or min(train_gpus) < 0:
        raise ValueError("select valid GPU indices")
    protocol = read_json(PROTOCOL)
    output.mkdir(parents=True)
    if shutil.disk_usage(output).free < 30 * (1 << 30):
        raise RuntimeError("less than 30 GiB free for new oracle renders and training states")
    (output / "logs").mkdir()
    report = output / "repro_outputs"
    report.mkdir()
    status = {"status": "running", "phase": 1, "git": revision, "steps": [],
              "protocol": str(PROTOCOL), "protocol_file_sha256": sha256(PROTOCOL),
              "scientific_result": "pending; execution success is not disentanglement success",
              "phase2": "not_started", "phase3": "not_started"}
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "DIFFUSERS_OFFLINE": "1"}
    module = "experiments.texture_material_disentangled_v1."
    python = [sys.executable, "-B", "-m"]
    commands = [
        ("plan", [*python, module + "data", "plan", "--output", str(output / "plan")], None),
        ("render", [protocol["renderer"]["blender"], "--background", "--python-exit-code", "1",
                    "--python", str(Path(__file__).with_name("render.py")), "--", "--plan",
                    str(output / "plan/plan.json"), "--output", str(output / "renders"),
                    "--gpu", str(render_gpu)], render_gpu),
        ("stage", [*python, module + "data", "stage", "--renders", str(output / "renders"),
                   "--output", str(output / "staging")], None),
        ("smoke", [*python, module + "train", "--staging", str(output / "staging"),
                   "--output", str(output / "smoke"), "--mode", "smoke", "--gpus", *map(str, train_gpus)], None),
        ("pilot", [*python, module + "train", "--staging", str(output / "staging"),
                   "--output", str(output / "pilot"), "--mode", "pilot", "--gpus", *map(str, train_gpus)], None),
        ("inference", [*python, module + "infer", "--training-index", str(output / "pilot/training_index.json"),
                       "--output", str(output / "inference"), "--device", "cuda:0"], render_gpu),
        ("evaluation", [*python, module + "evaluate", "--inference", str(output / "inference"),
                        "--renders", str(output / "renders"), "--output", str(output / "evaluation")], None),
    ]
    write_json(report / "commands.json", {"commands": [{"stage": name, "command": command, "gpu": gpu}
                                                         for name, command, gpu in commands]})
    import shlex
    (report / "COMMANDS.md").write_text("\n\n".join(
        f"{name}\n\n```bash\n" + (f"CUDA_VISIBLE_DEVICES={gpu} " if gpu is not None else "")
        + shlex.join(command) + "\n```" for name, command, gpu in commands) + "\n", encoding="utf-8")
    (report / "SCIENTIFIC_CHANGELOG.md").write_text(
        "New experiment: SD1.5 oracle counterfactual T/M learning, independent token embeddings and token-local K/V. "
        "No existing S/C training, dataset, checkpoint, losses or results are changed. T is linear grayscale albedo only; "
        "M is scalar roughness/metallic only. No IID, attention loss, normal/bump/displacement or new attention architecture. "
        "The trainer allowlist admits T/texture only for token_local_kv LoRA; existing paths are preserved.\n", encoding="utf-8")
    (report / "COMPARABILITY_REPORT.md").write_text(
        "64 balanced train renders, 16 per concept. Target T uses two auxiliary M values; target M uses two auxiliary T maps. "
        "All four target T×M pairs are absent from train. Evaluation separates unseen pair, shape, color and light contexts. "
        "Both methods use identical images/embeddings LR; original KV LR1e-5 versus established rank64 KV LR5e-5. "
        "300 steps is an initial learning pilot, not established convergence. 256 images include matched literal and single-token controls. "
        "BRDF recovery and independence are not measured by pixel statistics or uncalibrated VLM labels. "
        "Filtered outputs are missing observations. Blind appearance review remains a separate scientific evaluation.\n", encoding="utf-8")
    (report / "PATCHES.md").write_text(
        "New experiment-local protocol, data/UV-render/train/inference/orchestration scripts and tests. "
        "Shared trainer changes only its LoRA supported token allowlist to admit T/texture. "
        "Existing scene, masks, geometry, shared trainer, SD1.5 loader, fullrank/LoRA processors and dual gates are reused.\n", encoding="utf-8")
    try:
        for name, command, gpu in commands:
            if git_info() != revision:
                raise ValueError("clean source revision changed between stages")
            step = {"stage": name, "command": command, "status": "running", "started_at": time.time()}
            status["steps"].append(step)
            write_json(output / "status.json", status)
            with (output / "logs" / (name + ".log")).open("x") as log:
                child = subprocess.Popen(command, cwd=ROOT,
                    env={**environment, **({"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else {})},
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                step["pid"] = child.pid
                write_json(output / "status.json", status)
                step["returncode"] = child.wait()
            step.update(status="succeeded" if step["returncode"] == 0 else "failed", finished_at=time.time())
            write_json(output / "status.json", status)
            if step["returncode"]:
                raise RuntimeError(f"Phase 1 stopped at {name}; see logs/{name}.log; no automatic retry")
        status.update(status="execution_verified", scientific_result="pending_blinded_appearance_review")
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "status.json", status)
        write_json(report / "status.json", status)
        (report / "SUMMARY.md").write_text(
            f"Phase 1 state: {status['status']}. Scientific result: {status['scientific_result']}.\n\n"
            "Stages: " + ", ".join(f"{step['stage']}={step['status']}" for step in status["steps"]) + "\n\n"
            f"Commit: {revision['commit']}. Output: {output}. Phase 2 and 3 have not started.\n", encoding="utf-8")
        (report / "LOG.md").write_text(json.dumps(status["steps"], indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-gpu", type=int, required=True)
    parser.add_argument("--train-gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    run(args.output, args.render_gpu, args.train_gpus)
