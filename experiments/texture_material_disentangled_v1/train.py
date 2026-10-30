"""Run frozen two-step checks or 300-step T/M pilots through the shared trainer."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd15_token_local_kv_v1.snapshots import adapter_schema, _check_state, _metrics, _embedding_audit
from experiments.sd_native_comparison_v1.generate import write_json, git_info as full_git_info
from scripts.launch.colorpeel_run import git_info, read_config, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("protocol.json")


def protocol_hash():
    import hashlib
    return hashlib.sha256(json.dumps(read_json(PROTOCOL), sort_keys=True,
                                    separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def recipes():
    protocol = read_json(PROTOCOL)
    return [{"method": method, "branch": branch, "concept_id": concept,
             "token": "<T*>" if branch == "texture" else "<M*>"}
            for method in ("fullrank", "lora_r64")
            for branch, key in (("texture", "textures"), ("material", "materials"))
            for concept, spec in protocol[key].items() if spec["role"] == "target"]


def build_config(recipe, staging, mode):
    if recipe not in recipes() or mode not in ("smoke", "pilot"):
        raise ValueError("unknown target, method or training mode")
    protocol = read_json(PROTOCOL)
    training = protocol["training"]
    steps = training["smoke_steps"] if mode == "smoke" else training["steps"]
    # Reuse the established safe optimizer/precision defaults, changing only
    # this study's data, concept, base, explicit pilot dose and adapter method.
    baseline = ROOT / "experiments/sd15_token_local_kv_v1/configs/subject_fullrank.json"
    args = dict(read_json(baseline)["args"])
    args.update(pretrained_model_name_or_path=protocol["base_model"],
                concepts_list=str(Path(staging).resolve() / recipe["concept_id"] / "concepts.json"),
                max_train_steps=steps, checkpointing_steps=steps, checkpoint_steps=[steps],
                modifier_token=recipe["token"],
                initializer_token="texture" if recipe["branch"] == "texture" else "metal",
                learning_rate=training["embedding_lr"],
                kv_learning_rate=training["fullrank_kv_lr"],
                adam_weight_decay=training["weight_decay"], cos_weight=training["cos_weight"],
                freeze_model="crossattn_kv")
    if recipe["method"] == "lora_r64":
        args.pop("token_local_kv")
        args.update(subject_lora_mode="token_local_kv", subject_lora_rank=training["lora_rank"],
                    subject_lora_alpha=training["lora_alpha"], kv_learning_rate=training["lora_kv_lr"])
    variant = f"{recipe['branch']}_{recipe['concept_id']}_{recipe['method']}_{mode}"
    return {"schema_version": 1, "stage": "train",
            "run": {"study": protocol["study"], "variant": variant, "seed": protocol["seed"]},
            "environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "DIFFUSERS_OFFLINE": "1"},
            "data_manifest": str(Path(staging).resolve() / recipe["concept_id"] / "training_assets_manifest.jsonl"),
            "texture_material": {**recipe, "mode": mode, "protocol_sha256": protocol_hash()}, "args": args}


def verify_model():
    protocol = read_json(PROTOCOL)
    model = Path(protocol["base_model"])
    receipt = read_json(model / "model_source.json")
    frozen = read_json(ROOT / "experiments/sd15_token_local_kv_v1/protocol.json")["model"]
    if (receipt["model"] != frozen or frozen["root"] != str(model)
            or frozen["revision"] != protocol["model_revision"]):
        raise ValueError("frozen SD1.5 model receipt differs")
    for name, digest in receipt["files"].items():
        if sha256(model / name) != digest:
            raise ValueError(f"base model component changed: {name}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), subfolder="tokenizer", local_files_only=True)
    for initializer in ("texture", "metal"):
        if len(tokenizer.encode(initializer, add_special_tokens=False)) != 1:
            raise ValueError("initializer must be one token")
    return {"root": str(model), "receipt_sha256": sha256(model / "model_source.json")}


def audit_run(recipe, run, expected_config, *, expected_git=None):
    import torch
    run = Path(run)
    config = read_config(run / "config.yaml")
    manifest = read_json(run / "manifest.json")
    if (config != expected_config or manifest.get("status") != "succeeded"
            or manifest.get("returncode") != 0
            or manifest.get("git") != (git_info() if expected_git is None else expected_git)):
        raise ValueError("training run failed, changed recipe or differs from this revision")
    checkpoint = run / "checkpoints"
    token, method = recipe["token"], recipe["method"]
    weight = "pytorch_token_local_kv_weights.bin" if method == "fullrank" else "pytorch_lora_kv_weights.bin"
    adaptation = read_json(checkpoint / "adaptation_config.json")
    expected_mode = "token_local_kv" if method == "fullrank" else (
        "lora_material_kv" if token == "<M*>" else
        "lora_color_kv" if token == "<C*>" else "lora_subject_kv")
    if (adaptation.get("modifier_tokens") != [token] or adaptation.get("adaptation_mode") != expected_mode
            or (method == "lora_r64" and any(adaptation.get(key) != value for key, value in
                {"rank": 64, "alpha": 64, "mode": "token_local_kv"}.items()))):
        raise ValueError("exported token, adapter method or rank differs")
    steps = config["args"]["max_train_steps"]
    metrics = _metrics(checkpoint / "training_metrics.jsonl", steps, token)
    token_id, embedding_audit = _embedding_audit(checkpoint / "embedding_update_audit.json", steps, token)
    if embedding_audit["non_modifier_embedding_drift"].get("changed_rows") != 0:
        raise ValueError("ordinary token embeddings changed")
    shapes, empty = adapter_schema(method)
    adapter = torch.load(checkpoint / weight, map_location="cpu")
    _check_state(torch, adapter, shapes, empty, "final")
    embeddings = torch.load(checkpoint / (token + ".bin"), map_location="cpu")
    if (set(embeddings) != {token} or tuple(embeddings[token].shape) != (768,)
            or not bool(torch.isfinite(embeddings[token]).all())):
        raise ValueError("exported token embedding invalid")
    snapshot = checkpoint / f"checkpoint-{steps}"
    saved_adapter = torch.load(snapshot / "pytorch_model.bin", map_location="cpu")
    _check_state(torch, saved_adapter, shapes, {}, "snapshot")
    saved_encoder = torch.load(snapshot / "pytorch_model_1.bin", map_location="cpu")
    if (any(not torch.equal(adapter[key], saved_adapter[key]) for key in shapes)
            or not torch.equal(embeddings[token], saved_encoder[
                "text_model.embeddings.token_embedding.weight"][token_id])):
        raise ValueError("final paired snapshot differs from final export")
    files = [run / "config.yaml", run / "manifest.json", checkpoint / weight,
             checkpoint / (token + ".bin"), checkpoint / "adaptation_config.json",
             checkpoint / "training_metrics.jsonl", checkpoint / "embedding_update_audit.json"]
    return {**recipe, "status": "verified", "run": str(run.resolve()), "checkpoint": str(checkpoint.resolve()),
            "weight": weight, "steps": steps, "metrics": metrics, "embedding_audit": embedding_audit,
            "final_snapshot_matches_export": True, "hashes": {str(path.resolve()): sha256(path) for path in files}}


def run_queue(gpu, jobs, batch):
    records = []
    for job in jobs:
        record = {"recipe": job["recipe"], "run": job["run"], "status": "running", "command": job["command"]}
        records.append(record)
        write_json(batch / f"gpu_{gpu}.json", {"gpu": gpu, "jobs": records})
        try:
            with (batch / "logs" / (job["config"]["run"]["variant"] + ".log")).open("x") as log:
                child = subprocess.Popen(job["command"], cwd=ROOT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONDONTWRITEBYTECODE": "1"},
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                record["pid"] = child.pid
                write_json(batch / f"gpu_{gpu}.json", {"gpu": gpu, "jobs": records})
                record["returncode"] = child.wait()
            if record["returncode"]:
                raise RuntimeError("shared trainer failed; see new launcher log")
            record["audit"] = audit_run(job["recipe"], job["run"], job["config"])
            write_json(batch / "audits" / (job["config"]["run"]["variant"] + ".json"), record["audit"])
            record["status"] = "verified"
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(batch / f"gpu_{gpu}.json", {"gpu": gpu, "jobs": records})
        if record["status"] == "failed":
            return records
    return records


def train(staging, batch, mode, gpus):
    from experiments.texture_material_disentangled_v1.data import verify_staging
    revision = git_info()
    batch = resolve_run_dir(Path(batch))
    if batch.exists() or batch.is_relative_to(ROOT):
        raise ValueError("training batch output must be new and outside the repository")
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("select distinct nonnegative GPUs")
    data_audit = verify_staging(Path(staging))
    model_audit = verify_model()
    batch.mkdir(parents=True)
    for name in ("configs", "logs", "audits", "train_outputs"):
        (batch / name).mkdir()
    status = {"status": "running", "mode": mode, "git": revision, "git_full": full_git_info(),
              "protocol_sha256": protocol_hash(), "dataset": data_audit, "model": model_audit}
    write_json(batch / "status.json", status)
    jobs = []
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    for recipe in recipes():
        config = build_config(recipe, staging, mode)
        variant = config["run"]["variant"]
        source = batch / "configs" / (variant + ".json")
        # Decimal scientific notation must remain a YAML float for PyYAML too.
        text = json.dumps(config, indent=2).replace("1e-08", "0.00000001").replace("1e-05", "0.00001").replace("5e-05", "0.00005")
        source.write_text(text + "\n", encoding="utf-8")
        if read_config(source) != config:
            raise ValueError("generated frozen recipe does not round-trip through the launcher")
        run = batch / f"{stamp}__texture_material_disentangled_v1__{variant}__{revision['commit'][:7]}__42"
        jobs.append({"recipe": recipe, "config": config, "run": str(run),
                     "command": [sys.executable, "-B", str(ROOT / "scripts/launch/colorpeel_run.py"),
                                 "--config", str(source), "--run-dir", str(run)]})
    report = batch / "train_outputs"
    import shlex
    (report / "COMMANDS.md").write_text("\n\n".join(
        "```bash\n" + shlex.join(job["command"]) + "\n```" for job in jobs) + "\n", encoding="utf-8")
    (report / "SCIENTIFIC_CHANGELOG.md").write_text(
        "Independent oracle T/M branches, embedding plus token-local K/V only. "
        "No new loss; frozen base, ordinary embeddings, Subject/Color branches. "
        "Texture LoRA uses the same processor and explicit new token allowlist.\n", encoding="utf-8")
    (report / "COMPARABILITY_REPORT.md").write_text(
        "Both methods use identical sixteen-image per-concept data, seed42 and embedding LR1e-5. "
        "Fullrank KV LR1e-5; rank64 alpha64 KV LR5e-5. Two-step smoke or300-step initial pilot "
        "is not evidence of convergence or semantic independence. No best-checkpoint selection.\n", encoding="utf-8")
    records = []
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(run_queue, gpu, jobs[index::len(gpus)], batch)
                       for index, gpu in enumerate(gpus)]
            records = [record for future in futures for record in future.result()]
        if len(records) != len(jobs) or any(record["status"] != "verified" for record in records):
            raise RuntimeError("training queue failed or did not complete all eight targets")
        verify_staging(Path(staging))
        index = {"git": revision, "git_full": full_git_info(), "protocol_sha256": protocol_hash(),
                 "mode": mode, "runs": [record["audit"] for record in records]}
        write_json(batch / "training_index.json", index)
        status.update(status="verified", completed_runs=len(records),
                      completed_steps=sum(record["audit"]["steps"] for record in records),
                      stop_reason="all declared short runs and export audits completed")
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status.update(max_steps_per_run=read_json(PROTOCOL)["training"][
            "smoke_steps" if mode == "smoke" else "steps"], best_metric=None, best_checkpoint=None)
        write_json(batch / "status.json", status)
        write_json(report / "status.json", status)
        (report / "SUMMARY.md").write_text(
            f"{mode}: {status['status']}. Verified runs: {status.get('completed_runs', 0)}/8. "
            f"Verified completed steps: {status.get('completed_steps', 0)}. "
            "See per-GPU status files for partial/failed workers. No semantic independence claim.\n", encoding="utf-8")
        (report / "LOG.md").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    train(args.staging, args.output, args.mode, args.gpus)
