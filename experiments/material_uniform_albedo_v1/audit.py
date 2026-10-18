"""Read-only checks of uniform-albedo Material training and paired exports."""

from pathlib import Path

from experiments.material_uniform_albedo_v1 import data, run
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd15_token_local_kv_v1.snapshots import (
    STATE_FILES, _check_state, _embedding_audit, _metrics, adapter_schema,
)
from experiments.texture_material_disentangled_v1 import step_snapshots
from scripts.launch.colorpeel_run import read_config


def _lock(locks, path, digest=None):
    path = Path(path).resolve()
    expected = sha256(path) if digest is None else digest
    if str(path) in locks:
        if locks[str(path)] != expected:
            raise ValueError(f"conflicting artifact locks: {path}")
    elif sha256(path) != expected:
        raise ValueError(f"training artifact changed: {path}")
    locks[str(path)] = expected


def _members(index):
    recipes = {(r["method"], r["concept_id"]): r for r in run.recipes()}
    selections = index.get("runs", [])
    if (len(selections) != 4 or {(s.get("method"), s.get("concept_id")) for s in selections} != set(recipes)
            or any(any(s.get(k) != v for k, v in recipes[s["method"], s["concept_id"]].items())
                   or s.get("status") != "verified" for s in selections)):
        raise ValueError("expected four verified Material concept/method selections")
    return selections


def _source(source):
    source = Path(source).resolve()
    status_path, index_path = source / "status.json", source / "training/training_index.json"
    status, index = read_json(status_path), read_json(index_path)
    protocol_hash = data.protocol_sha256()
    revision = index.get("git", {})
    if (status.get("status") != "execution_verified" or status.get("completed_training_runs") != 4
            or status.get("completed_steps") != 4000 or not revision.get("commit")
            or {key: status.get("git", {}).get(key) for key in ("branch", "commit")} != revision
            or any(item.get("protocol_sha256") != protocol_hash for item in (status, index))
            or index.get("study") != "material_uniform_albedo_v1"):
        raise ValueError("expected completed uniform Material training with original provenance")
    dataset = data.verify_staging(source / "staging")
    if index.get("dataset") != dataset or status.get("dataset") != dataset:
        raise ValueError("training dataset audit differs")
    locks = {}
    paths = [status_path, index_path, source / "staging/audit.json"]
    paths += [p for p in (source / "staging").rglob("*") if p.is_file()]
    for path in paths:
        _lock(locks, path)
    selections = _members(index)
    for selection in selections:
        if selection.get("steps") != 1000:
            raise ValueError("expected complete1000-step Material runs")
        directory = Path(selection["run"]).resolve()
        recipe = {key: selection[key] for key in ("method", "branch", "concept_id", "token")}
        manifest = read_json(directory / "manifest.json")
        if (not directory.is_relative_to(source / "training")
                or Path(selection["checkpoint"]).resolve() != directory / "checkpoints"
                or read_config(directory / "config.yaml") != run.build_config(recipe, source / "staging")
                or manifest.get("status") != "succeeded" or manifest.get("returncode") != 0
                or manifest.get("git") != revision):
            raise ValueError("training config, success or original revision differs")
        step_snapshots._verify(selection)
        for path, digest in selection["hashes"].items():
            _lock(locks, path, digest)
    return source, index, selections, locks


def verify_source(source, step):
    """Verify four selected Material exports without requiring a T/M inference run."""
    if type(step) is not int or step not in run.STEPS:
        raise ValueError("select an existing500/750/1000-step snapshot")
    source, original, selections, locks = _source(source)
    index_path = source / f"step{step}/training_index.json"
    index = read_json(index_path)
    if (index.get("git") != original["git"] or index.get("study") != "material_uniform_albedo_v1"
            or index.get("protocol_sha256") != data.protocol_sha256()
            or index.get("training_steps") != step or index.get("source_training_steps") != 1000
            or index.get("dataset_audit_sha256") != locks[str(source / "staging/audit.json")]):
        raise ValueError("selected step, dataset or original training provenance differs")
    _lock(locks, index_path)
    originals = {(s["method"], s["concept_id"]): s for s in selections}
    derived = _members(index)
    for item in derived:
        previous = originals[item["method"], item["concept_id"]]
        checkpoint = Path(item["checkpoint"]).resolve()
        if (item.get("steps") != step or item.get("step") != step or item.get("training_steps") != 1000
                or item.get("run") != previous["run"] or item.get("weight") != previous["weight"]
                or item.get("source_checkpoint") != previous["checkpoint"]
                or item.get("source_hashes") != previous["hashes"]
                or checkpoint != source / f"step{step}/checkpoints/{item['method']}__{item['concept_id']}"):
            raise ValueError("derived Material snapshot pairing differs")
        raw = Path(previous["checkpoint"]) / f"checkpoint-{step}"
        snapshots = item.get("source_snapshot_hashes", {})
        expected = {str(raw / name) for name in ("pytorch_model.bin", "pytorch_model_1.bin")}
        required = set(previous["hashes"]) | expected | {str(checkpoint / name) for name in
                   (item["weight"], "<M*>.bin", "adaptation_config.json")}
        if set(snapshots) != expected or not required <= set(item.get("hashes", {})):
            raise ValueError("derived snapshot is missing source or export locks")
        for path, digest in {**item["hashes"], **snapshots}.items():
            if (path in previous["hashes"] and digest != previous["hashes"][path]
                    or path in snapshots and item["hashes"][path] != digest):
                raise ValueError("derived snapshot source locks differ")
            _lock(locks, path, digest)
        adaptation = read_json(checkpoint / "adaptation_config.json")
        mode = "token_local_kv" if item["method"] == "fullrank" else "lora_material_kv"
        if (adaptation.get("modifier_tokens") != ["<M*>"] or adaptation.get("adaptation_mode") != mode
                or (item["method"] == "lora_r64" and any(adaptation.get(key) != value for key, value in
                    {"rank": 64, "alpha": 64, "mode": "token_local_kv"}.items()))):
            raise ValueError("Material adapter token, mode or rank differs")
        _lock(locks, checkpoint / "receipt.json")
        if read_json(checkpoint / "receipt.json") != item:
            raise ValueError("derived export receipt differs from training index")
    return {**index, "training_git": original["git"]}, derived, locks


def audit_training(source):
    """Check all four complete trajectories and twelve paired inference exports."""
    import torch
    source, index, selections, locks = _source(source)
    by_step = {}
    for step in run.STEPS:
        _, derived, step_locks = verify_source(source, step)
        locks.update(step_locks)
        by_step[step] = {(s["method"], s["concept_id"]): s for s in derived}
    results = []
    for selection in selections:
        checkpoint, token = Path(selection["checkpoint"]), selection["token"]
        metrics_path, audit_path = checkpoint / "training_metrics.jsonl", checkpoint / "embedding_update_audit.json"
        for path in (metrics_path, audit_path):
            if str(path) not in selection["hashes"]:
                raise ValueError("training metrics or embedding audit lacks a source lock")
        metrics = _metrics(metrics_path, 1000, token)
        _, embedding = _embedding_audit(audit_path, 1000, token)
        if (embedding.get("non_modifier_embedding_drift") or {}).get("changed_rows") != 0:
            raise ValueError("ordinary token embeddings changed")
        snapshots = []
        if {p.name for p in checkpoint.glob("checkpoint-*") if p.is_dir()} != {
                f"checkpoint-{step}" for step in run.STEPS}:
            raise ValueError("saved checkpoint directories differ from training plan")
        for step in run.STEPS:
            raw = checkpoint / f"checkpoint-{step}"
            for name in STATE_FILES:
                if not (raw / name).is_file() or not (raw / name).stat().st_size:
                    raise ValueError(f"missing training state: {raw / name}")
            adapter, token_row, empty, record = step_snapshots._paired(torch, selection, step)
            derived = by_step[step][selection["method"], selection["concept_id"]]
            exported = Path(derived["checkpoint"])
            actual = torch.load(exported / derived["weight"], map_location="cpu")
            shapes, _ = adapter_schema(selection["method"])
            _check_state(torch, actual, shapes, empty, "derived")
            token_state = torch.load(exported / (token + ".bin"), map_location="cpu")
            if (not isinstance(token_state, dict) or set(token_state) != {token}
                    or not isinstance(token_state[token], torch.Tensor)
                    or not torch.equal(token_row, token_state[token])
                    or any(not torch.equal(adapter[key], actual[key]) for key in shapes)):
                raise ValueError("derived adapter or token differs from its paired raw snapshot")
            snapshots.append({"step": step, "paired_export_matches": True, **record})
        results.append({"method": selection["method"], "concept_id": selection["concept_id"],
                        "run": selection["run"], "steps": 1000, "metrics": metrics,
                        "embedding_audit": embedding, "snapshots": snapshots})
    for path, digest in locks.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"source changed during audit: {path}")
    return {"status": "verified", "source": str(source), "training_git": index["git"],
            "protocol_sha256": data.protocol_sha256(), "completed_training_runs": 4,
            "completed_steps": 4000, "paired_snapshots": 12, "runs": results, "source_locks": locks}
