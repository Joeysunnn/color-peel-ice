"""Run the six frozen full recipes, sequentially on each selected GPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.sd_native_comparison_v1.generate import git_info, write_json
from experiments.sd15_token_local_kv_v1.contract import PROTOCOL, read_json, validate_config
from scripts.launch.colorpeel_run import read_config, resolve_run_dir


def make_queues(batch, gpus, commit):
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("select distinct nonnegative GPU indices")
    queues = {gpu: [] for gpu in gpus}
    stamp = batch.name.split("__")[0]
    for index, recipe in enumerate(read_json(PROTOCOL)["recipes"]):
        config_path = Path(__file__).with_name("configs") / (recipe + ".json")
        config = read_config(config_path)
        validate_config(config, {}, verify_assets=False)
        if config["sd15_migration"]["smoke"]:
            raise ValueError("full training must not use a smoke configuration")
        run = batch.parent / f"{stamp}__sd15_token_local_kv_v1__{recipe}__{commit[:7]}__42"
        if run.exists():
            raise FileExistsError(run)
        gpu = gpus[index % len(gpus)]
        queues[gpu].append({
            "recipe": recipe, "gpu": gpu, "status": "queued", "run_dir": str(run),
            "config": str(config_path), "max_steps": config["args"]["max_train_steps"],
            "command": [sys.executable, "-B", str(ROOT / "scripts/launch/colorpeel_run.py"),
                        "--config", str(config_path), "--run-dir", str(run)],
        })
    return queues


def run_queue(gpu, jobs, batch, environment):
    status_path = batch / f"gpu_{gpu}.json"
    write_json(status_path, {"gpu": gpu, "jobs": jobs})
    for index, job in enumerate(jobs):
        job["status"] = "running"
        job["started_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(status_path, {"gpu": gpu, "jobs": jobs})
        try:
            # A new root with ample space is required; never prune old snapshots.
            if shutil.disk_usage(batch).free < 40 * (1 << 30):
                raise RuntimeError("less than 40 GiB free on the new output filesystem")
            with (batch / (job["recipe"] + ".launcher.log")).open("x", encoding="utf-8") as log:
                child = subprocess.Popen(
                    job["command"], cwd=ROOT, env={**environment, "CUDA_VISIBLE_DEVICES": str(gpu)},
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                job["launcher_pid"] = child.pid
                write_json(status_path, {"gpu": gpu, "jobs": jobs})
                job["returncode"] = child.wait()
            job["status"] = "succeeded" if job["returncode"] == 0 else "failed"
        except Exception as error:
            job.update(status="failed", error=f"{type(error).__name__}: {error}")
        job["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        if job["status"] == "failed":
            for pending in jobs[index + 1:]:
                pending.update(status="not_run", stop_reason="previous job in this GPU queue failed")
            write_json(status_path, {"gpu": gpu, "jobs": jobs})
            return False
        write_json(status_path, {"gpu": gpu, "jobs": jobs})
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    revision = git_info()
    batch = resolve_run_dir(args.batch_dir)
    if batch.is_relative_to(ROOT) or batch.exists():
        raise ValueError("batch must be a new directory outside the repository")
    queues = make_queues(batch, args.gpus, revision["commit"])
    batch.mkdir(parents=True)
    receipt = {"status": "preflight", "mode": "full_kickoff", "git": revision,
               "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "source_run_root": os.environ["COLORPEEL_RUN_ROOT"],
               "output_root": os.environ.get("COLORPEEL_OUTPUT_ROOT", os.environ["COLORPEEL_RUN_ROOT"]),
               "gpus": args.gpus, "queue_files": [f"gpu_{gpu}.json" for gpu in queues]}
    write_json(batch / "manifest.json", receipt)
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "DIFFUSERS_OFFLINE": "1"}
    try:
        if shutil.disk_usage(batch).free < 40 * (1 << 30):
            raise RuntimeError("less than 40 GiB free on the new output filesystem")
        for gpu, jobs in queues.items():
            for job in jobs:
                config = read_config(Path(job["config"]))
                job_environment = {**environment, **config.get("environment", {})}
                job["input_verification"] = validate_config(config, job_environment)
            write_json(batch / f"gpu_{gpu}.json", {"gpu": gpu, "jobs": jobs})
        receipt["status"] = "running"
        write_json(batch / "manifest.json", receipt)
        with ThreadPoolExecutor(max_workers=len(queues)) as pool:
            futures = [pool.submit(run_queue, gpu, jobs, batch, environment)
                       for gpu, jobs in queues.items()]
            codes = [future.result() for future in futures]
        receipt["status"] = "succeeded" if all(codes) else "failed"
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(batch / "manifest.json", receipt)
    if receipt["status"] != "succeeded":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
