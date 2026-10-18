"""Read-only audits of completed natural T/M training and its paired exports."""

import json
from pathlib import Path

from experiments.natural_texture_material_v1 import dataset_protocol as schema
from experiments.natural_texture_material_v1.training_config import STEPS, build_config, recipes
from experiments.natural_texture_material_v1.training_source import verify_dataset
from experiments.sd15_token_local_kv_v1.contract import read_json
from experiments.sd15_token_local_kv_v1.snapshots import _embedding_audit, _metrics
from experiments.texture_material_disentangled_v1 import step_snapshots
from scripts.launch.colorpeel_run import read_config


def _key(selection):
    return selection.get("method"), selection.get("concept_id")


def _members(selections):
    expected = {_key(recipe) for recipe in recipes()}
    if len(selections) != 12 or {_key(item) for item in selections} != expected:
        raise ValueError("Expected exactly twelve distinct natural T/M concept/method runs")
    return {_key(item): item for item in selections}


def _lock(locks, path, expected=None):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("Training and dataset locks must use absolute paths")
    digest = schema.sha256(path)
    if ((expected is not None and digest != expected)
            or (str(path) in locks and locks[str(path)] != digest)):
        raise ValueError(f"Frozen training or dataset artifact changed: {path}")
    locks[str(path)] = digest
    return digest


def _load(source):
    source = Path(source).resolve()
    state = read_json(source / "status.json")
    if (state.get("status") != "execution_verified" or state.get("planned_training_runs") != 12
            or state.get("completed_training_runs") != 12 or state.get("completed_steps") != 12000
            or state.get("checkpoint_steps") != list(STEPS) or state.get("max_steps_per_run") != 1000
            or state.get("resume") is not False or state.get("inference_launched") is not False):
        raise ValueError("Expected twelve completed 1000-step runs and all three exported snapshots")
    dataset = Path(state["source_dataset"])
    protocol, dataset_audit, locks = verify_dataset(dataset)
    digest, training_git = schema.protocol_sha256(protocol), state["git"]
    if (read_json(source / "protocol.json") != protocol or state.get("protocol_sha256") != digest
            or state.get("dataset") != dataset_audit or read_json(source / "source_locks.json") != locks
            or not training_git.get("commit")
            or state.get("source_dataset_git") != read_json(dataset / "status.json").get("git")):
        raise ValueError("Training source differs from the accepted dataset or its recorded provenance")
    index = read_json(source / "training/training_index.json")
    if (index.get("git") != training_git or index.get("source_dataset") != str(dataset)
            or index.get("protocol_sha256") != digest):
        raise ValueError("Training index provenance differs from its completed batch")
    selections = _members(index.get("runs", []))
    refs_path = source / "training_references.json"
    refs = read_json(refs_path)
    original_refs = read_json(dataset / "staging/training_references.json")
    expected_groups = {concept: {key: group[key] for key in ("source_id", "branch", "source",
                       "image_count", "training_assets_manifest", "images")}
                       for concept, group in original_refs.items()}
    if (refs.get("source") != str(dataset / "staging/training_references.json")
            or refs.get("source_sha256") != schema.sha256(refs["source"])
            or refs.get("groups") != expected_groups or refs.get("recipes") != recipes()
            or refs.get("steps") != list(STEPS)
            or refs.get("same_dataset_for_original_and_rank64") is not True):
        raise ValueError("Recorded training references differ from the actual shared RGB datasets")
    for name in ("status.json", "protocol.json", "source_locks.json", "training/training_index.json",
                 "training_references.json"):
        _lock(locks, source / name)
    runs = []
    for recipe in recipes():
        selection = selections[_key(recipe)]
        if any(selection.get(key) != value for key, value in recipe.items()):
            raise ValueError("Training selection recipe identity differs")
        run = Path(selection["run"])
        expected = build_config(recipe, dataset / "staging", protocol)
        manifest = read_json(run / "manifest.json")
        if (run.parent != source / "training" or read_config(run / "config.yaml") != expected
                or manifest.get("status") != "succeeded" or manifest.get("returncode") != 0
                or manifest.get("git") != training_git
                or manifest.get("data_manifest") != expected["data_manifest"]
                or selection.get("steps") != 1000
                or selection.get("checkpoint") != str(run / "checkpoints")
                or selection.get("final_snapshot_matches_export") is not True):
            raise ValueError("Training config, manifest, final step or recorded training revision differs")
        checkpoint = step_snapshots._verify(selection)
        metrics_path, embedding_path = checkpoint / "training_metrics.jsonl", checkpoint / "embedding_update_audit.json"
        if not {str(metrics_path), str(embedding_path)} <= set(selection["hashes"]):
            raise ValueError("Training metrics and embedding audit must be checkpoint-locked")
        metrics = _metrics(metrics_path, 1000, recipe["token"])
        _, embedding = _embedding_audit(embedding_path, 1000, recipe["token"])
        if (metrics != selection.get("metrics") or embedding != selection.get("embedding_audit")
                or (embedding.get("non_modifier_embedding_drift") or {}).get("changed_rows") != 0):
            raise ValueError("Finite-loss, token exposure or frozen ordinary embedding audit differs")
        if {path.name for path in checkpoint.glob("checkpoint-*") if path.is_dir()} != {
                f"checkpoint-{step}" for step in STEPS}:
            raise ValueError("Saved checkpoint directories differ from the three requested steps")
        for path, value in selection["hashes"].items():
            _lock(locks, path, value)
        runs.append({**recipe, "run": str(run), "steps": 1000, "metrics": metrics,
                     "embedding_audit": embedding, "final_snapshot_matches_export": True})
    exports = {}
    for step in STEPS:
        index_path = source / f"step{step}/training_index.json"
        derived_index = read_json(index_path)
        if (derived_index.get("git") != training_git or derived_index.get("training_git") != training_git
                or derived_index.get("dataset_git") != state["source_dataset_git"]
                or derived_index.get("protocol_sha256") != digest or derived_index.get("training_steps") != step
                or derived_index.get("source_training_steps") != 1000
                or derived_index.get("training_references") != str(refs_path)
                or derived_index.get("training_references_sha256") != schema.sha256(refs_path)
                or derived_index.get("source_dataset") != str(dataset)
                or derived_index.get("source_locks_sha256") != schema.sha256(source / "source_locks.json")):
            raise ValueError("Derived step index differs from its frozen training provenance")
        _lock(locks, index_path)
        derived = _members(derived_index.get("runs", []))
        for key, item in derived.items():
            parent = selections[key]
            checkpoint = source / f"step{step}/checkpoints/{item['method']}__{item['concept_id']}"
            snapshot = Path(parent["checkpoint"]) / f"checkpoint-{step}"
            snapshot_hashes = {str(snapshot / name): schema.sha256(snapshot / name)
                               for name in ("pytorch_model.bin", "pytorch_model_1.bin")}
            if (any(item.get(field) != parent.get(field) for field in
                    ("method", "branch", "concept_id", "source_id", "token", "weight", "run"))
                    or item.get("steps") != step or item.get("step") != step or item.get("training_steps") != 1000
                    or item.get("source_checkpoint") != parent["checkpoint"]
                    or item.get("checkpoint") != str(checkpoint) or item.get("path") != str(checkpoint)
                    or item.get("source_hashes") != parent["hashes"]
                    or item.get("source_snapshot_hashes") != snapshot_hashes
                    or item.get("source_adapter_sha256") != snapshot_hashes[str(snapshot / "pytorch_model.bin")]
                    or item.get("source_text_encoder_sha256") != snapshot_hashes[str(snapshot / "pytorch_model_1.bin")]
                    or item.get("paired_embedding_finite") is not True
                    or item.get("final_snapshot_matches_export") is not (step == 1000)
                    or read_json(checkpoint / "receipt.json") != item):
                raise ValueError("Derived checkpoint, paired snapshot or export receipt differs")
            required = {**parent["hashes"], **snapshot_hashes}
            required.update({str(checkpoint / name): schema.sha256(checkpoint / name)
                             for name in (parent["weight"], parent["token"] + ".bin", "adaptation_config.json")})
            if (item.get("hashes") != required or item.get("derived_adapter_sha256") !=
                    required[str(checkpoint / parent["weight"])] or item.get("derived_token_sha256") !=
                    required[str(checkpoint / (parent["token"] + ".bin"))]):
                raise ValueError("Derived adapter or token digests differ")
            if (read_json(checkpoint / "adaptation_config.json") !=
                    read_json(Path(parent["checkpoint"]) / "adaptation_config.json")):
                raise ValueError("Derived adapter configuration changed")
            for path, value in required.items():
                _lock(locks, path, value)
            _lock(locks, checkpoint / "receipt.json")
        exports[step] = derived_index["runs"]
    return source, state, protocol, selections, exports, refs, locks, runs


def audit_training(source):
    """Verify every finite training update and all 36 actual paired tensor exports."""
    import torch
    source, state, protocol, selections, exports, _, locks, runs = _load(source)
    snapshots = []
    for step, items in exports.items():
        for item in items:
            parent = selections[_key(item)]
            adapter, embedding, empty, record = step_snapshots._paired(torch, parent, step)
            checkpoint = Path(item["checkpoint"])
            exported = torch.load(checkpoint / item["weight"], map_location="cpu")
            shapes, _ = step_snapshots.adapter_schema(item["method"])
            step_snapshots._check_state(torch, exported, shapes, empty, "derived")
            token_state = torch.load(checkpoint / (item["token"] + ".bin"), map_location="cpu")
            if (not isinstance(token_state, dict) or set(token_state) != {item["token"]}
                    or not torch.equal(embedding, token_state[item["token"]])
                    or any(not torch.equal(adapter[key], exported[key]) for key in shapes)
                    or record["source_snapshot_hashes"] != item["source_snapshot_hashes"]):
                raise ValueError("Derived export tensors differ from the paired training snapshot")
            snapshots.append({"method": item["method"], "concept_id": item["concept_id"], "step": step,
                              "checkpoint": str(checkpoint), "paired_tensors_verified": True})
    if any(schema.sha256(path) != value for path, value in locks.items()):
        raise ValueError("Frozen source changed during the training audit")
    return {"status": "execution_verified", "source": str(source), "training_git": state["git"],
            "protocol_sha256": schema.protocol_sha256(protocol), "completed_training_runs": len(runs),
            "completed_steps": sum(run["steps"] for run in runs), "paired_exports": len(snapshots),
            "runs": runs, "snapshots": snapshots, "source_locks": locks}


def verify_source(source, step, audit=None):
    """Verify all immutable receipts and identify the selected step's actual training RGB."""
    if type(step) is not int or step not in STEPS:
        raise ValueError("Expected 500, 750 or 1000 steps")
    if audit is None:
        audit = audit_training(source)
    root, state, protocol, _, exports, refs, locks, runs = _load(source)
    snapshots = [{"method": item["method"], "concept_id": item["concept_id"], "step": number,
                  "checkpoint": item["checkpoint"], "paired_tensors_verified": True}
                 for number, items in exports.items() for item in items]
    if (audit.get("status") != "execution_verified" or audit.get("source") != str(root)
            or audit.get("training_git") != state["git"]
            or audit.get("protocol_sha256") != schema.protocol_sha256(protocol)
            or audit.get("completed_training_runs") != 12 or audit.get("completed_steps") != 12000
            or audit.get("paired_exports") != 36 or audit.get("runs") != runs
            or audit.get("snapshots") != snapshots or audit.get("source_locks") != locks):
        raise ValueError("Training audit is not bound to this unchanged twelve-run, 36-export source")
    references = {}
    for item in exports[step]:
        group = refs["groups"][item["concept_id"]]
        ledger = Path(group["training_assets_manifest"])
        rows = [json.loads(line) for line in ledger.read_text(encoding="utf8").splitlines()]
        references[_key(item)] = {"method": item["method"], "branch": item["branch"],
            "source_id": item["source_id"], "concept_id": item["concept_id"], "step": step,
            "run": item["run"], "checkpoint": item["checkpoint"], "source": group["source"],
            "dataset_manifest": str(ledger), "dataset_manifest_sha256": schema.sha256(ledger),
            "images": [{key: row[key] for key in ("id", "image", "image_sha256", "prompt", "shape",
                        "color", "light", "material_id", "roughness", "metallic", "seed") if key in row}
                       for row in rows]}
    return protocol, exports[step], locks, references
