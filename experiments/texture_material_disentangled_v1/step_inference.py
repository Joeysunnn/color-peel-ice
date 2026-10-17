"""Generate the locked step evaluation from completed training, without retraining."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd_native_comparison_v1.generate import git_info, write_json
from experiments.texture_material_disentangled_v1 import data, step_ablation, train
from experiments.texture_material_disentangled_v1.step_snapshots import _verify, compare_baseline, export_step
from scripts.launch.colorpeel_run import read_config, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]


def completed_selections(batch, source, receipt):
    """Recheck the preserved audits against their original training revision."""
    status_path = batch / "status.json"
    status = read_json(status_path)
    training_git = {key: status["git"][key] for key in ("branch", "commit")}
    if (status.get("source") != receipt or status.get("protocol_sha256") != data.protocol_sha256()):
        raise ValueError("completed training source or protocol differs")
    expected = {(item["method"], item["branch"], item["concept_id"]): item for item in train.recipes()}
    audit_paths = sorted((batch / "training/audits").glob("*.json"))
    selections, locks, seen = [], {str(status_path): sha256(status_path)}, set()
    if len(audit_paths) != len(expected):
        raise ValueError("expected exactly eight completed training audits")
    for path in audit_paths:
        selection = read_json(path)
        key = tuple(selection.get(field) for field in ("method", "branch", "concept_id"))
        if key not in expected or key in seen:
            raise ValueError("completed training concepts or methods differ")
        run = Path(selection["run"]).resolve()
        manifest = read_json(run / "manifest.json")
        if (run.parent != batch / "training" or Path(selection["checkpoint"]).resolve() != run / "checkpoints"
                or selection.get("steps") != 1000 or selection.get("token") != expected[key]["token"]
                or read_config(run / "config.yaml") != step_ablation.build_config(expected[key], source / "staging")
                or manifest.get("status") != "succeeded" or manifest.get("returncode") != 0
                or manifest.get("git") != training_git):
            raise ValueError("completed training config, path, steps or provenance differs")
        _verify(selection)
        seen.add(key)
        selections.append(selection)
        locks[str(path)] = sha256(path)
    return selections, training_git, locks


def run(source, completed_training_run, output, inference_gpu):
    revision = git_info()
    source, batch = Path(source).resolve(), Path(completed_training_run).resolve()
    output = resolve_run_dir(Path(output))
    if (output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(source)
            or output.is_relative_to(batch) or inference_gpu < 0):
        raise ValueError("inference requires a new external directory and a nonnegative GPU")
    original, receipt = step_ablation.verify_source(source)
    selections, training_git, locks = completed_selections(batch, source, receipt)
    model = train.verify_model()
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    status = {"status": "checking", "git": revision, "training_git": training_git,
              "source": receipt, "completed_training_run": str(batch), "source_locks": locks,
              "model": model, "protocol_sha256": data.protocol_sha256(), "retraining": False,
              "requested_steps": list(step_ablation.STEPS), "steps": []}
    write_json(output / "status.json", status)
    try:
        originals = {(item["method"], item["branch"], item["concept_id"]): item for item in original["runs"]}
        comparisons = [compare_baseline(item, originals[(item["method"], item["branch"], item["concept_id"])],
                                        allow_roundoff=True) for item in selections]
        comparison_path = output / "baseline300_comparison.json"
        write_json(comparison_path, {"comparisons": comparisons})
        status.update(status="generating", completed_training_runs=8, source_training_steps=1000)
        snapshot_locks = {}
        for step in step_ablation.STEPS:
            directory = output / f"step{step}"
            directory.mkdir()
            derived = [export_step(item, step, directory / "checkpoints" /
                       f"{item['method']}__{item['branch']}__{item['concept_id']}") for item in selections]
            for item in derived:
                snapshot_locks.update(item["hashes"])
            index_path = directory / "training_index.json"
            write_json(index_path, {"git": train.git_info(), "git_full": revision, "training_git": training_git,
                "mode": "pilot", "protocol_sha256": data.protocol_sha256(), "runs": derived,
                "training_steps": step, "source_training_steps": 1000, "source_receipt": receipt,
                "completed_training_run": str(batch), "source_locks": locks,
                "baseline300_comparison_sha256": sha256(comparison_path)})
            state = {"step": step, "status": "running", "output": str(directory)}
            status["steps"].append(state)
            write_json(output / "status.json", status)
            commands = [
                ("inference", [sys.executable, "-B", "-m", "experiments.texture_material_disentangled_v1.infer",
                    "--training-index", str(index_path), "--output", str(directory / "inference"),
                    "--training-steps", str(step), "--device", "cuda:0"]),
                ("evaluation", [sys.executable, "-B", "-m", "experiments.texture_material_disentangled_v1.evaluate",
                    "--inference", str(directory / "inference"), "--renders", str(source / "renders"),
                    "--output", str(directory / "evaluation")]),
            ]
            for stage, command in commands:
                state[stage + "_command"] = command
                write_json(output / "status.json", status)
                with (output / "logs" / f"step{step}_{stage}.log").open("x") as log:
                    result = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(inference_gpu),
                        "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "DIFFUSERS_OFFLINE": "1"})
                if result.returncode:
                    raise RuntimeError(f"step{step} {stage} failed; see the new output logs")
            state["status"] = "succeeded"
            write_json(output / "status.json", status)
        if step_ablation.verify_source(source)[1] != receipt or git_info() != revision:
            raise ValueError("original experiment or inference checkout changed during generation")
        for item in selections:
            _verify(item)
        for path, digest in snapshot_locks.items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"snapshot or derived artifact changed: {path}")
        if any(sha256(Path(path)) != digest for path, digest in locks.items()):
            raise ValueError("completed training batch or audits changed during generation")
        status["status"] = "execution_verified"
        (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>T/M steps</title>'
            + "".join(f'<p><a href="step{step}/evaluation/index.html">{step} training steps</a></p>'
                      for step in step_ablation.STEPS), encoding="utf-8")
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        (output / "logs/failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        write_json(output / "status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--completed-training-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inference-gpu", type=int, required=True)
    args = parser.parse_args()
    run(args.source_run, args.completed_training_run, args.output, args.inference_gpu)
