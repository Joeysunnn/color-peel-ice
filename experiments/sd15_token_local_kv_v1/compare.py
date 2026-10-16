"""Audit six completed runs, then reproduce their frozen SD1.4 image matrices."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.sd15_token_local_kv_v1.contract import PROTOCOL, read_json, sha256
from experiments.sd_native_comparison_v1.generate import git_info, write_json
from scripts.launch.colorpeel_run import resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_PROTOCOL = Path(__file__).with_name("inference_protocol.json")


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def selection_key(source, row):
    """Select only the same method and training step as the historical image."""
    if source.startswith("fullrank_"):
        if source == "fullrank_material":
            return (None, None, "material_ground_reflection_fullrank", 5000)
        step = int(source.rsplit("_", 1)[1])
        material = "material_ground_reflection_fullrank" if "comparison" in source else None
        return ("subject_fullrank", step, material, 5000 if material else None)
    if source == "lora_subject":
        if row["condition"] == "subject_material_token":
            return None
        if row["condition"] not in ("subject_only", "subject_literal_material"):
            raise ValueError("unknown historical subject condition")
        return ("subject_lora_r64", row["step"], None, None)
    if source.startswith("lora_transfer_"):
        return ("subject_lora_r64", row["subject_step"], None, None)
    material = "material_" + source.removeprefix("lora_material_") + "_lora_r64"
    if row["family"] == "transfer":
        return (None, None, material, row["material_step"])
    if row["family"] != "composition":
        raise ValueError("unknown historical material family")
    return ("subject_lora_r64", row["subject_step"], material, row["material_step"])


def reference_rows(reference, root):
    directory = (root / reference["directory"]).resolve()
    if not directory.is_relative_to(root.resolve()):
        raise ValueError("reference escapes the immutable result root")
    for name, digest in reference["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"historical source changed: {directory / name}")
    provenance = read_json(directory / "provenance.json")
    if provenance.get("status", "succeeded") != "succeeded":
        raise ValueError("historical generation did not succeed")
    rows, ledger = jsonl(directory / "generation_manifest.jsonl"), jsonl(directory / "generation_status.jsonl")
    indexed = {row["id"]: row for row in ledger}
    if (len(indexed) != len(ledger) or len({row["id"] for row in rows}) != len(rows)
            or {row["id"] for row in rows} != set(indexed)):
        raise ValueError("historical manifest and ledger are incomplete or duplicated")
    result = []
    for row in rows:
        status = indexed[row["id"]]
        if any(status.get(key) != value for key, value in row.items()):
            raise ValueError("historical manifest and ledger row differ")
        if (row.get("num_inference_steps", 100) != 100 or row.get("guidance_scale", 3.5) != 3.5
                or type(row.get("seed")) is not int or not isinstance(row.get("prompt"), str)):
            raise ValueError("historical sampling differs")
        image = (directory / row["image_path"]).resolve()
        if not image.is_relative_to(directory) or sha256(image) != status["image_sha256"]:
            raise ValueError("historical image path or hash differs")
        if "status" in status and status["status"] not in ("ok", "safety_filtered"):
            raise ValueError("historical image status failed")
        filtered = status.get("safety_filtered", status.get("nsfw_content_detected"))
        if type(filtered) is not bool:
            raise ValueError("historical safety observation missing")
        result.append({**row, "reference_image": str(image),
                       "reference_sha256": status["image_sha256"], "reference_filtered": filtered})
    return result


def plan_jobs(protocol, audits, output):
    jobs = []
    for reference in protocol["references"]:
        groups = {}
        for row in reference_rows(reference, Path(protocol["reference_root"])):
            key = selection_key(reference["id"], row)
            if key is None:
                continue
            groups.setdefault(key, []).append(row)
        for key, rows in groups.items():
            selections = [audits[recipe]["checkpoints"][str(step)]
                          for recipe, step in (key[:2], key[2:]) if recipe]
            tokens = {selection["token"] for selection in selections}
            if any(token in row["prompt"] for row in rows for token in {"<S*>", "<M*>"} - tokens):
                raise ValueError("prompt requires an unselected adapter")
            identifier = reference["id"] + "__" + "__".join(
                f"{recipe}_s{step}" for recipe, step in (key[:2], key[2:]) if recipe)
            rows = [{**row, "source_id": row["id"], "image_path": f"images/{index:04d}.png"}
                    for index, row in enumerate(rows)]
            jobs.append({"id": identifier, "output": str(output / "jobs" / identifier),
                         "reference": reference, "selections": selections, "rows": rows})
    counts = {method: sum(len(job["rows"]) for job in jobs
                         if job["selections"][0]["method"] == method)
              for method in ("fullrank", "lora_r64")}
    if counts != {k: protocol["expected_images"][k] for k in counts} or len(jobs) != 66:
        raise ValueError(f"historical matrix differs: {counts}, {len(jobs)} jobs")
    return jobs


def prepare(output):
    from experiments.sd15_token_local_kv_v1.snapshots import audit_training
    revision = git_info()
    output = resolve_run_dir(output)
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError("comparison requires a new output directory outside the repository")
    output.mkdir(parents=True)
    if shutil.disk_usage(output).free < 15 * (1 << 30):
        raise RuntimeError("less than 15 GiB free for snapshot exports and generated images")
    protocol, audits = read_json(INFERENCE_PROTOCOL), {}
    (output / "training_audits").mkdir()
    status = {"status": "auditing", "git": revision}
    write_json(output / "status.json", status)
    try:
        for source in protocol["historical_protocols"]:
            import hashlib
            blob = subprocess.check_output(["git", "show", source["commit"] + ":" + source["path"]], cwd=ROOT)
            if hashlib.sha256(blob).hexdigest() != source["sha256"] or json.loads(blob) != source["payload"]:
                raise ValueError("frozen historical inference protocol differs")
        for recipe in read_json(PROTOCOL)["recipes"]:
            run = Path(protocol["training_root"]) / (protocol["training_prefix"] + recipe + protocol["training_suffix"])
            print(f"Auditing {recipe}", flush=True)
            audits[recipe] = audit_training(recipe, run, output / "snapshots" / recipe)
            write_json(output / "training_audits" / (recipe + ".json"), audits[recipe])
        jobs = plan_jobs(protocol, audits, output)
        plan = {"git": revision, "sampling": protocol["sampling"], "jobs": jobs,
                "inference_protocol_sha256": sha256(INFERENCE_PROTOCOL),
                "excluded": protocol["excluded"], "expected_images": protocol["expected_images"]}
        write_json(output / "comparison_plan.json", plan)
        status.update(status="prepared", job_count=len(jobs), image_count=sum(len(job["rows"]) for job in jobs),
                      plan_sha256=sha256(output / "comparison_plan.json"))
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "status.json", status)


def run_queue(gpu, jobs, output, environment):
    records = []
    for job in jobs:
        command = [sys.executable, "-B", "-m", "experiments.sd15_token_local_kv_v1.compare_worker",
                   "--plan", str(output / "comparison_plan.json"), "--job", job["id"], "--device", "cuda:0"]
        record = {"id": job["id"], "status": "running", "command": command}
        records.append(record)
        write_json(output / f"gpu_{gpu}.json", {"gpu": gpu, "jobs": records})
        with (output / "logs" / (job["id"] + ".log")).open("x", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, env={**environment, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            record["pid"] = child.pid
            write_json(output / f"gpu_{gpu}.json", {"gpu": gpu, "jobs": records})
            record["returncode"] = child.wait()
        record["status"] = "succeeded" if record["returncode"] == 0 else "failed"
        write_json(output / f"gpu_{gpu}.json", {"gpu": gpu, "jobs": records})
        if record["returncode"]:
            return False
    return True


def run(output, gpus):
    from experiments.sd15_token_local_kv_v1.comparison_bundle import bundle_all
    output = resolve_run_dir(output)
    plan, status = read_json(output / "comparison_plan.json"), read_json(output / "status.json")
    if (git_info() != plan["git"] or status["status"] != "prepared"
            or sha256(output / "comparison_plan.json") != status["plan_sha256"]
            or sha256(INFERENCE_PROTOCOL) != plan["inference_protocol_sha256"]):
        raise ValueError("prepared comparison or clean Git revision changed")
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("select distinct nonnegative GPUs")
    queues = {gpu: plan["jobs"][index::len(gpus)] for index, gpu in enumerate(gpus)}
    (output / "logs").mkdir()
    status.update(status="running", gpus=gpus)
    write_json(output / "status.json", status)
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "DIFFUSERS_OFFLINE": "1"}
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(run_queue, gpu, jobs, output, environment) for gpu, jobs in queues.items()]
            passed = [future.result() for future in futures]
        if not all(passed):
            raise RuntimeError("inference queue failed; see gpu status and job logs")
        bundle_all(output)
        status["status"] = "succeeded"
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    prepare(args.output) if args.action == "prepare" else run(args.output, args.gpus)


if __name__ == "__main__":
    main()
