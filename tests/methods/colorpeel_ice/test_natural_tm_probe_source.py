"""Historical training provenance and immutable paired-export acceptance checks."""

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.natural_texture_material_v1 import probe_source as module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value)
    if path.name == "config.yaml":
        text = text.replace("1e-08", "0.00000001").replace("1e-05", "0.00001").replace("5e-05", "0.00005")
    path.write_text(text, encoding="utf8")


@pytest.fixture
def trained(tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("Linux checkpoint filenames <T*>.bin and <M*>.bin are invalid on Windows")
    source, dataset = tmp_path / "training_batch", tmp_path / "dataset"
    git = {"commit": "a" * 40, "branch": "codex/texture-material-v1"}
    protocol = module.read_json(module.schema.PARENT)
    protocol.update(study="natural_texture_material_v1", phase=2)
    write(dataset / "status.json", {"git": git})
    groups = {}
    for recipe in module.recipes():
        concept, branch = recipe["concept_id"], recipe["branch"]
        folder = dataset / "staging" / concept
        rows = [{"id": f"{concept}_{i}", "image": str(folder / f"image_{i}.png"),
                 "image_sha256": "actual-rgb", "prompt": f"actual RGB caption {recipe['token']}",
                 "shape": "sphere", "color": "red", "light": "soft_front", "seed": 42,
                 "roughness": 0.3, "metallic": 0.8, "material_id": "actual_material"}
                for i in range(16 if branch == "texture" else 8)]
        folder.mkdir(parents=True, exist_ok=True)
        ledger = folder / "training_assets_manifest.jsonl"
        ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
        groups[concept] = {"source_id": recipe["source_id"], "branch": branch,
            "source": {"input": "natural.png", "texture": "extracted.png", "roughness": 0.3, "metallic": 0.8},
            "image_count": len(rows), "training_assets_manifest": str(ledger), "images": rows}
    source_refs = dataset / "staging/training_references.json"
    write(source_refs, groups)
    dataset_locks = {str(path): module.schema.sha256(path) for path in dataset.rglob("*") if path.is_file()}
    monkeypatch.setattr(module, "verify_dataset", lambda path: (copy.deepcopy(protocol), {"train_images": 72}, dict(dataset_locks)))
    state = {"status": "execution_verified", "planned_training_runs": 12, "completed_training_runs": 12,
        "completed_steps": 12000, "checkpoint_steps": [500, 750, 1000], "max_steps_per_run": 1000,
        "resume": False, "inference_launched": False, "source_dataset": str(dataset), "git": git,
        "source_dataset_git": git, "protocol_sha256": module.schema.protocol_sha256(protocol),
        "dataset": {"train_images": 72}}
    for name, value in (("status.json", state), ("protocol.json", protocol), ("source_locks.json", dataset_locks),
        ("training_references.json", {"source": str(source_refs), "source_sha256": module.schema.sha256(source_refs),
            "groups": groups, "recipes": module.recipes(), "steps": [500, 750, 1000],
            "same_dataset_for_original_and_rank64": True})):
        write(source / name, value)
    selections, exports = [], {step: [] for step in module.STEPS}
    for recipe in module.recipes():
        run = source / "training" / (recipe["method"] + "__" + recipe["concept_id"])
        checkpoint, token = run / "checkpoints", recipe["token"]
        config = module.build_config(recipe, dataset / "staging", protocol)
        write(run / "config.yaml", config)
        write(run / "manifest.json", {"status": "succeeded", "returncode": 0, "git": git,
                                      "data_manifest": config["data_manifest"]})
        mode = "token_local_kv" if recipe["method"] == "fullrank" else (
            "lora_subject_kv" if recipe["branch"] == "texture" else "lora_material_kv")
        adaptation = {"modifier_tokens": [token], "adaptation_mode": mode,
                      "rank": 64, "alpha": 64, "mode": "token_local_kv"}
        write(checkpoint / "adaptation_config.json", adaptation)
        weight = "pytorch_token_local_kv_weights.bin" if recipe["method"] == "fullrank" else "pytorch_lora_kv_weights.bin"
        write(checkpoint / weight, {"w": 1})
        write(checkpoint / (token + ".bin"), {token: 7})
        write(checkpoint / "embedding_update_audit.json", {"observed_optimization_steps": 1000,
            "modifier_tokens": [{"token": token, "token_id": 49408, "exposure_steps": 1000,
                "nonzero_gradient_steps": 1000, "initial_final_l2_delta": 0.1}],
            "non_modifier_embedding_drift": {"changed_rows": 0}})
        metrics_path = checkpoint / "training_metrics.jsonl"
        metrics_path.write_text("".join(json.dumps({"step": i, "losses_finite": {"all": True},
            "present_modifier_tokens": [token], **{field: 0.2 for field in
                ("reconstruction_loss", "caa_loss", "caa_weighted_loss", "total_loss", "learning_rate")}}) + "\n"
            for i in range(1, 1001)), encoding="utf8")
        files = [run / "config.yaml", run / "manifest.json", checkpoint / weight, checkpoint / (token + ".bin"),
                 checkpoint / "adaptation_config.json", checkpoint / "embedding_update_audit.json", metrics_path]
        selection = {**recipe, "status": "verified", "steps": 1000, "run": str(run),
            "checkpoint": str(checkpoint), "weight": weight, "final_snapshot_matches_export": True,
            "hashes": {str(path): module.schema.sha256(path) for path in files},
            "metrics": module._metrics(metrics_path, 1000, token),
            "embedding_audit": module._embedding_audit(checkpoint / "embedding_update_audit.json", 1000, token)[1]}
        selections.append(selection)
        for step in module.STEPS:
            snapshot = checkpoint / f"checkpoint-{step}"
            for name in ("pytorch_model.bin", "pytorch_model_1.bin"):
                write(snapshot / name, {"step": step})
            snapshot_hashes = {str(path): module.schema.sha256(path) for path in snapshot.iterdir()}
            exported = source / f"step{step}/checkpoints/{recipe['method']}__{recipe['concept_id']}"
            for name, value in ((weight, {"w": 1}), (token + ".bin", {token: 7}), ("adaptation_config.json", adaptation)):
                write(exported / name, value)
            payload = {str(path): module.schema.sha256(path) for path in exported.iterdir()}
            item = {**selection, "checkpoint": str(exported), "path": str(exported), "steps": step,
                "step": step, "training_steps": 1000, "source_checkpoint": str(checkpoint),
                "source_hashes": selection["hashes"], "source_snapshot_hashes": snapshot_hashes,
                "hashes": {**selection["hashes"], **snapshot_hashes, **payload}, "paired_embedding_finite": True,
                "final_snapshot_matches_export": step == 1000,
                "source_adapter_sha256": snapshot_hashes[str(snapshot / "pytorch_model.bin")],
                "source_text_encoder_sha256": snapshot_hashes[str(snapshot / "pytorch_model_1.bin")],
                "derived_adapter_sha256": payload[str(exported / weight)],
                "derived_token_sha256": payload[str(exported / (token + ".bin"))]}
            write(exported / "receipt.json", item)
            exports[step].append(item)
    write(source / "training/training_index.json", {"git": git, "runs": selections,
        "source_dataset": str(dataset), "protocol_sha256": state["protocol_sha256"]})
    for step in module.STEPS:
        write(source / f"step{step}/training_index.json", {"git": git, "training_git": git, "dataset_git": git,
            "protocol_sha256": state["protocol_sha256"], "training_steps": step, "source_training_steps": 1000,
            "runs": exports[step], "training_references": str(source / "training_references.json"),
            "training_references_sha256": module.schema.sha256(source / "training_references.json"),
            "source_dataset": str(dataset), "source_locks_sha256": module.schema.sha256(source / "source_locks.json")})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda path, **kw: module.read_json(path), equal=lambda a, b: a == b))
    monkeypatch.setattr(module.step_snapshots, "_check_state", lambda *args: None)
    monkeypatch.setattr(module.step_snapshots, "adapter_schema", lambda method: ({"w": (1,)}, {}))
    def paired(torch, parent, step):
        paths = Path(parent["checkpoint"]) / f"checkpoint-{step}"
        return {"w": 1}, 7, {}, {"source_snapshot_hashes": {str(path): module.schema.sha256(path) for path in paths.iterdir()}}
    monkeypatch.setattr(module.step_snapshots, "_paired", paired)
    return source, selections, exports


def test_all_twelve_runs_and_thirty_six_paired_exports_use_recorded_training_revision(trained):
    source, _, _ = trained
    before = {str(path): module.schema.sha256(path) for path in source.rglob("*") if path.is_file()}
    audit = module.audit_training(source)
    assert (audit["completed_training_runs"], audit["completed_steps"], audit["paired_exports"]) == (12, 12000, 36)
    assert audit["training_git"]["commit"] == "a" * 40
    json.dumps(audit)
    for step in module.STEPS:
        _, selections, locks, refs = module.verify_source(source, step, audit)
        assert len(selections) == len(refs) == 12
        assert {item["steps"] for item in selections} == {step}
        for selection in selections:
            ref = refs[module._key(selection)]
            assert len(ref["images"]) == (16 if selection["branch"] == "texture" else 8)
            assert ref["images"][0]["prompt"].startswith("actual RGB caption")
            assert ref["images"][0]["image"] != ref["source"]["texture"]
            assert str(Path(selection["checkpoint"]) / "receipt.json") in locks
    assert {str(path): module.schema.sha256(path) for path in source.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("failure", ["incomplete", "wrong_git", "duplicate", "config", "unrequested_step", "export", "snapshot", "reference"])
def test_reject_incomplete_or_changed_training_before_accepting_audit(trained, failure):
    source, selections, exports = trained
    if failure == "incomplete":
        path = source / "status.json"
        state = module.read_json(path)
        state["completed_training_runs"] = 11
        write(path, state)
    elif failure in ("wrong_git", "config"):
        path = Path(selections[0]["run"]) / ("manifest.json" if failure == "wrong_git" else "config.yaml")
        value = module.read_json(path)
        if failure == "wrong_git":
            value["git"]["commit"] = "b" * 40
        else:
            value["args"]["seed"] = 99
        write(path, value)
    elif failure == "duplicate":
        path = source / "step750/training_index.json"
        value = module.read_json(path)
        value["runs"][1] = value["runs"][0]
        write(path, value)
    elif failure == "unrequested_step":
        (Path(selections[0]["checkpoint"]) / "checkpoint-300").mkdir()
    elif failure == "export":
        write(Path(exports[500][0]["checkpoint"]) / selections[0]["weight"], {"wrong": 1})
    elif failure == "snapshot":
        write(Path(selections[0]["checkpoint"]) / "checkpoint-750/pytorch_model_1.bin", {"changed": True})
    else:
        path = source / "training_references.json"
        value = module.read_json(path)
        value["groups"]["mailbox_texture"]["images"][0]["image"] = "extracted_grayscale.png"
        write(path, value)
    with pytest.raises(ValueError):
        module.audit_training(source)


def test_supplied_audit_skips_tensor_reload_but_is_bound_to_frozen_locks(trained, monkeypatch):
    source, _, _ = trained
    audit = module.audit_training(source)
    monkeypatch.setattr(module.step_snapshots, "_paired", lambda *args: pytest.fail("must reuse actual paired audit"))
    module.verify_source(source, 500, audit)
    audit["source_locks"].pop(next(iter(audit["source_locks"])))
    with pytest.raises(ValueError, match="audit is not bound"):
        module.verify_source(source, 500, audit)


def test_audit_compares_actual_export_embedding_with_snapshot(trained, monkeypatch):
    source, _, _ = trained
    original = module.step_snapshots._paired
    def wrong_embedding(*args):
        adapter, embedding, empty, record = original(*args)
        return adapter, embedding + 1, empty, record
    monkeypatch.setattr(module.step_snapshots, "_paired", wrong_embedding)
    with pytest.raises(ValueError, match="tensors differ"):
        module.audit_training(source)


@pytest.mark.parametrize("failure", ["duplicate", "missing", "unknown"])
def test_concept_method_membership_rejects_wrong_twelve_run_sets(failure):
    selections = module.recipes()
    assert len(module._members(selections)) == 12
    if failure == "duplicate":
        selections[1] = selections[0]
    elif failure == "missing":
        selections.pop()
    else:
        selections[0]["concept_id"] = "unrelated_concept"
    with pytest.raises(ValueError, match="twelve distinct"):
        module._members(selections)


@pytest.mark.parametrize("step", [True, 300, 751, 1001])
def test_reject_unrequested_step_before_source_reads(step):
    with pytest.raises(ValueError, match="500, 750 or 1000"):
        module.verify_source("does_not_exist", step)


def test_training_config_fixture_roundtrips_shared_yaml_reader(tmp_path):
    protocol = module.read_json(module.schema.PARENT)
    for recipe in module.recipes():
        config = module.build_config(recipe, tmp_path / "staging", protocol)
        path = tmp_path / recipe["method"] / recipe["concept_id"] / "config.yaml"
        write(path, config)
        assert module.read_config(path) == config
