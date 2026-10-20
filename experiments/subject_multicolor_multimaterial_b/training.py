"""Frozen Subject recipes; dry-run unless training is explicitly requested."""

import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.texture_material_disentangled_v1 import train as shared
from scripts.launch.colorpeel_run import git_info, read_config, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
STUDY = "subject_multicolor_multimaterial_b"
METHODS = ("fullrank", "lora_r64")
BASELINES = {method: ROOT / "experiments/sd15_token_local_kv_v1/configs" /
             ("subject_fullrank.json" if method == "fullrank" else "subject_lora_r64.json")
             for method in METHODS}
ARG_CHANGES = {"concepts_list", "max_train_steps", "checkpoint_steps"}


def _write(path, value):
    # The shared launcher reads JSON through YAML; scientific notation needs a decimal.
    text = json.dumps(value, indent=2, sort_keys=True).replace("1e-08", "0.00000001").replace(
        "1e-05", "0.00001").replace("5e-05", "0.00005")
    path.write_text(text + "\n", encoding="utf-8")


def build_config(method, dataset):
    if method not in METHODS:
        raise ValueError("expected fullrank or lora_r64")
    dataset = Path(dataset).resolve()
    config = copy.deepcopy(read_json(BASELINES[method]))
    config.pop("sd15_migration")
    config["status"] = "subject_experiment_b_prepared"
    config["run"] = {"study": STUDY, "variant": f"mailbox_{method}_steps1000", "seed": 42}
    config["environment"].pop("COLORPEEL_MAILBOX_MATTE_ROOT", None)
    config["data_manifest"] = str(dataset / "training_assets_manifest.jsonl")
    config["args"].update(concepts_list=str(dataset / "concepts.json"),
                          max_train_steps=1000, checkpoint_steps=[500, 750, 1000])
    return config


def build_configuration_bundle(dataset, output):
    from experiments.subject_multicolor_multimaterial_b.data import verify_dataset

    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    audit = verify_dataset(dataset)
    if output.exists() or output.is_relative_to(dataset):
        raise ValueError("configuration output must be fresh and outside the dataset")
    configs, comparison = {}, {}
    for method in METHODS:
        baseline = read_json(BASELINES[method])
        config = build_config(method, dataset)
        differences = {key: {"old": baseline["args"].get(key), "new": config["args"].get(key)}
                       for key in set(baseline["args"]) | set(config["args"])
                       if baseline["args"].get(key) != config["args"].get(key)}
        if set(differences) != ARG_CHANGES:
            raise ValueError("Subject baseline arguments changed outside data and checkpoint dose")
        configs[method] = config
        comparison[method] = {"baseline": str(BASELINES[method]),
                              "baseline_sha256": sha256(BASELINES[method]), "args_diff": differences}
    source_hashes = {str(dataset / name): sha256(dataset / name) for name in
                     ("concepts.json", "training_assets_manifest.jsonl", "dataset_audit.json", "source_locks.json")}
    (output / "configs").mkdir(parents=True)
    paths = {}
    for method, config in configs.items():
        path = output / "configs" / f"{method}.json"
        _write(path, config)
        if read_config(path) != config:
            raise ValueError("training configuration fails shared-launcher YAML roundtrip")
        paths[method] = str(path)
    record = {"study": STUDY, "comparison": comparison, "source_hashes": source_hashes,
              "dataset_audit": audit, "checkpoint_steps": [500, 750, 1000]}
    _write(output / "CONFIG_COMPARISON.json", record)
    return {"configs": configs, "paths": paths, "comparison": record}


def run(dataset, output, method, gpu, execute_training=False):
    from experiments.subject_multicolor_multimaterial_b.data import verify_dataset

    if method not in METHODS or type(gpu) is not int or gpu < 0:
        raise ValueError("select one supported method and a nonnegative GPU index")
    dataset, output = Path(dataset).resolve(), resolve_run_dir(Path(output))
    if output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(dataset):
        raise ValueError("use a fresh external output outside the dataset")
    git = git_info()
    bundle = build_configuration_bundle(dataset, output)
    state = {"status": "preflight", "git": git, "method": method, "gpu": gpu,
             "execute_training": execute_training, "max_steps": 1000, "completed_steps": 0,
             "checkpoint_steps": [500, 750, 1000], "resume": False,
             "best_metric": None, "best_checkpoint": None, "stop_reason": None}
    _write(output / "status.json", state)
    report = output / "train_outputs"
    report.mkdir()
    try:
        state["model"] = shared.verify_model()
        config = bundle["configs"][method]
        name = f"{datetime.now():%Y%m%d-%H%M%S}__{STUDY}__{config['run']['variant']}__{git['commit'][:7]}__42"
        command = [sys.executable, "-B", str(ROOT / "scripts/launch/colorpeel_run.py"),
                   "--config", bundle["paths"][method], "--run-dir", str(output / "runs" / name)]
        preflight = [*command[:-1], str(output / "preflight" / name), "--dry-run"]
        _write(report / "COMMANDS.json", {"environment": {"CUDA_VISIBLE_DEVICES": str(gpu)},
                                         "preflight": preflight, "training": command})
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONDONTWRITEBYTECODE": "1"}
        with (report / "preflight.log").open("x") as log:
            result = subprocess.run(preflight, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
        preflight_run = Path(preflight[preflight.index("--run-dir") + 1])
        if (result.returncode or read_json(preflight_run / "manifest.json").get("status") != "dry-run"
                or read_config(preflight_run / "config.yaml") != config):
            raise RuntimeError("shared launcher preflight failed")
        verify_dataset(dataset)
        if git_info() != git:
            raise ValueError("checkout changed after preparation")
        if not execute_training:
            state.update(status="dry_run_verified", stop_reason="training not requested")
            return state
        state["status"] = "training"
        _write(output / "status.json", state)
        with (report / "training.log").open("x") as log:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("shared trainer failed; see training.log")
        recipe = {"method": method, "branch": "subject", "concept_id": "mailbox", "token": "<S*>"}
        audit = shared.audit_run(recipe, Path(command[-1]), config)
        _write(report / "training_audit.json", audit)
        _write(output / "training_index.json", {"git": git, "runs": [audit]})
        verify_dataset(dataset)
        if git_info() != git:
            raise ValueError("checkout changed during training")
        state.update(status="training_verified", completed_steps=audit["steps"],
                     run=command[-1], stop_reason="max_steps reached")
        return state
    except Exception as error:
        state.update(status="failed", stop_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        _write(output / "status.json", state)
        _write(report / "status.json", state)
        (report / "SUMMARY.md").write_text(
            f"Status: {state['status']}. Completed steps: {state['completed_steps']}/1000. "
            f"Stop reason: {state['stop_reason']}. No inference was started.\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--execute-training", action="store_true")
    args = parser.parse_args()
    run(args.dataset, args.output, args.method, args.gpu, args.execute_training)


if __name__ == "__main__":
    main()
