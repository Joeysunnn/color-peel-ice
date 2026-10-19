"""Train wood Material with one changed initializer, then run matched transfer."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from experiments.natural_material_initializer_v1.source import build_config, prepare
from experiments.natural_texture_material_v1.probe import STEPS, write
from experiments.natural_texture_material_v1.dataset_protocol import sha256
from experiments.texture_material_disentangled_v1 import train as shared
from experiments.texture_material_disentangled_v1.step_snapshots import export_step
from scripts.launch.colorpeel_run import git_info, read_config, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]


def verify_locks(locks):
    if any(sha256(path) != value for path, value in locks.items()):
        raise ValueError("Frozen source artifacts changed")


def run(source_training, source_transfer, output, gpus):
    source_training, source_transfer = Path(source_training).resolve(), Path(source_transfer).resolve()
    output = resolve_run_dir(Path(output))
    if (output.exists() or any(output.is_relative_to(path) for path in (ROOT, source_training, source_transfer))
            or len(gpus) != 3 or len(set(gpus)) != 3 or min(gpus) < 0):
        raise ValueError("Use a fresh external output and three distinct GPUs")
    git = git_info()
    output.mkdir(parents=True)
    batch, report = output / "training", output / "train_outputs"
    for folder in (batch / "configs", batch / "logs", batch / "audits", output / "preflight", report):
        folder.mkdir(parents=True)
    state = {"status": "auditing_sources", "git": git, "source_training": str(source_training),
        "source_transfer": str(source_transfer), "initializer_change": {"old": "metal", "new": "material"},
        "planned_training_runs": 2, "max_steps": 2000, "max_steps_per_run": 1000, "completed_steps": 0,
        "completed_training_runs": 0, "best_metric": None, "best_checkpoint": None,
        "checkpoint_steps": list(STEPS), "resume": False, "stop_reason": None,
        "expected_new_images": 48, "expected_reused_images": 144, "transfer": []}
    write(output / "status.json", state)
    completed = []
    try:
        plan = prepare(source_training, source_transfer)
        model = shared.verify_model()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model["root"], subfolder="tokenizer", local_files_only=True)
        ids = tokenizer.encode("material", add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError("material must encode to one CLIP token")
        if shutil.disk_usage(output).free < 15 * (1 << 30):
            raise RuntimeError("Less than15GiB free for new training snapshots")
        locks = plan.pop("source_locks")
        plan.update(git=git, source_training=str(source_training), source_transfer=str(source_transfer),
                    initializer={"word": "material", "token_ids": ids}, model=model)
        write(output / "plan.json", plan)
        write(output / "source_locks.json", locks)
        jobs, differences = [], {}
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for recipe in plan["recipes"]:
            method = recipe["method"]
            baseline = plan["configs"][method]
            config = build_config(baseline, method)
            differences[method] = {key: {"old": baseline["args"].get(key), "new": config["args"].get(key)}
                for key in set(baseline["args"]) | set(config["args"])
                if baseline["args"].get(key) != config["args"].get(key)}
            if differences[method] != {"initializer_token": {"old": "metal", "new": "material"}}:
                raise ValueError("Only the initializer may differ from the original training arguments")
            variant = config["run"]["variant"]
            path = batch / "configs" / (variant + ".json")
            text = json.dumps(config, indent=2).replace("1e-08", "0.00000001").replace(
                "1e-05", "0.00001").replace("5e-05", "0.00005")
            path.write_text(text + "\n", encoding="utf8")
            if read_config(path) != config:
                raise ValueError("Training config fails YAML roundtrip")
            destination = batch / f"{stamp}__natural_material_initializer_v1__{variant}__{git['commit'][:7]}__42"
            jobs.append({"recipe": recipe, "config": config, "run": str(destination),
                "command": [sys.executable, "-B", str(ROOT / "scripts/launch/colorpeel_run.py"),
                            "--config", str(path), "--run-dir", str(destination)]})
        write(output / "jobs.json", jobs)
        write(report / "CONFIG_COMPARISON.json", differences)
        (report / "COMMANDS.md").write_text("\n\n".join(
            f"CUDA_VISIBLE_DEVICES={gpu} " + shlex.join(job["command"])
            for gpu, job in zip(gpus, jobs)) + "\n", encoding="utf8")
        (report / "SCIENTIFIC_CHANGELOG.md").write_text(
            "Wood Material only;initializer metal->material. Same eight RGB/masks/captions and raw R/M. "
            "No resume,new loss or architecture. Original/fullrank and rank64 both1000 updates,save500/750/1000.\n",
            encoding="utf8")
        (report / "COMPARABILITY_REPORT.md").write_text(
            "Exact argument diff is initializer_token only;run metadata changes for isolation. "
            "Seed42,embedding LR1e-5,KV LR1e-5 original or5e-5 rank64/alpha64 retained. "
            "Transfer retains chair/mug/vase/sofa,seeds42/43,PNDM100,CFG3.5,512px,fp16,safety. "
            "Native/literal/old-token PNGs are reused with source hashes;only the new-token arm is generated.\n",
            encoding="utf8")
        (report / "PATCHES.md").write_text(
            "Independent initializer-ablation runner reuses source audits,trainer,queues,snapshot exporter and sampler. "
            "LoRA token-local allowlist additionally admits (<M*>,material);all other guards retained.\n", encoding="utf8")
        (report / "SUMMARY.md").write_text("Two1000-step initializer runs selected;preflight in progress.\n", encoding="utf8")
        (report / "LOG.md").write_text("Training queues:../training/gpu_*.json;transfer logs:../logs/step*.log.\n", encoding="utf8")
        state["status"] = "preflight"
        write(output / "status.json", state)
        for job in jobs:
            directory = output / "preflight" / Path(job["run"]).name
            command = [*job["command"][:-1], str(directory), "--dry-run"]
            with (output / "preflight" / (job["config"]["run"]["variant"] + ".log")).open("x") as log:
                result = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            if (result.returncode or json.loads((directory / "manifest.json").read_text()).get("status") != "dry-run"
                    or read_config(directory / "config.yaml") != job["config"]):
                raise RuntimeError("Shared launcher preflight failed")
        verify_locks(locks)
        if git_info() != git:
            raise ValueError("Checkout changed before training")
        state["status"] = "training"
        write(output / "status.json", state)
        write(report / "status.json", state)
        (report / "SUMMARY.md").write_text(
            "Full kickoff:two1000-step runs;completion unverified. Transfer automatically follows successful training/export audits.\n",
            encoding="utf8")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(shared.run_queue, gpu, [job], batch) for gpu, job in zip(gpus, jobs)]
            completed = [record for future in futures for record in future.result()]
        state.update(completed_training_runs=sum(record["status"] == "verified" for record in completed),
                     completed_steps=sum(record.get("audit", {}).get("steps", 0) for record in completed))
        if len(completed) != 2 or state["completed_training_runs"] != 2:
            raise RuntimeError("Wood initializer training or checkpoint audit failed")
        selections = [record["audit"] for record in completed]
        write(batch / "training_index.json", {"git": git, "runs": selections})
        state["status"] = "exporting"
        write(output / "status.json", state)
        for step in STEPS:
            folder = output / "exports" / f"step{step}"
            exported = [export_step(item, step, folder / "checkpoints" / item["method"]) for item in selections]
            references = {}
            for item in exported:
                old = plan["references"][str(step)][item["method"]]
                references[item["method"]] = {**old, "baseline_run": old["run"],
                    "baseline_checkpoint": old["checkpoint"], "run": item["run"], "checkpoint": item["checkpoint"]}
            write(folder / "index.json", {"git": git, "step": step, "runs": exported, "references": references})
        verify_locks(locks)
        state["status"] = "generating"
        processes = []
        (output / "logs").mkdir()
        for step, gpu in zip(STEPS, gpus):
            command = [sys.executable, "-B", "-m", "experiments.natural_material_initializer_v1.transfer",
                       "--run-root", str(output), "--step", str(step)]
            with (output / "logs" / f"step{step}.log").open("x") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONDONTWRITEBYTECODE": "1"})
            record = {"step": step, "gpu": gpu, "pid": process.pid, "command": command, "status": "running"}
            state["transfer"].append(record)
            processes.append((process, record))
        write(output / "status.json", state)
        (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Wood Material initializer</title>'
            '<p>Native / literal wood / metal initialization / material initialization. '
            'Both learned columns share the same eight training RGB images shown.</p>' +
            "".join(f'<p><a href="step{step}/index.html">step{step}</a></p>' for step in STEPS), encoding="utf8")
        with (report / "COMMANDS.md").open("a", encoding="utf8") as handle:
            handle.write("\n\n" + "\n\n".join(f"CUDA_VISIBLE_DEVICES={job['gpu']} " + shlex.join(job["command"])
                                                   for job in state["transfer"]) + "\n")
        for process, record in processes:
            record["returncode"] = process.wait()
            summary_path = output / f"step{record['step']}" / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            sheets = summary.get("contact_sheets", [])
            complete = (record["returncode"] == 0 and summary.get("status") == "succeeded"
                        and summary.get("generated_images") == 16 and summary.get("reused_images") == 48
                        and summary.get("total_images") == 64 and summary.get("missing_images") == 0
                        and summary.get("training_steps") == record["step"]
                        and len(sheets) == len(set(sheets)) == 4
                        and all((output / f"step{record['step']}" / path).is_file() for path in sheets))
            record.update(status="succeeded" if complete else "failed", summary=summary)
            write(output / "status.json", state)
        if any(record["status"] != "succeeded" for _, record in processes):
            raise RuntimeError("One or more initializer transfer workers failed")
        verify_locks(locks)
        if git_info() != git:
            raise ValueError("Checkout changed during the experiment")
        state.update(status="execution_verified", new_images=48, reused_images=144,
                     stop_reason="two training runs,six paired exports and three transfer comparisons verified")
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", stop_reason="execution_error")
        raise
    finally:
        write(output / "status.json", state)
        write(report / "status.json", state)
        (report / "SUMMARY.md").write_text(
            f"Wood Material initializer ablation:{state['status']}. Audited updates{state['completed_steps']}/2000. "
            "No R/M calibration or Texture retraining. See status and GPU queues for partial progress.\n", encoding="utf8")
        (report / "LOG.md").write_text("Training:../training/gpu_*.json and trainer logs;transfer:../logs/step*.log.\n" +
                                       json.dumps(completed, indent=2), encoding="utf8")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-training", type=Path, required=True)
    parser.add_argument("--source-transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=3, required=True)
    args = parser.parse_args()
    print(json.dumps({"status": run(args.source_training, args.source_transfer, args.output, args.gpus)["status"]}))
