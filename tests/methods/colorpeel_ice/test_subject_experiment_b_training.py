"""Check Subject baseline preservation and the explicit training boundary."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from experiments.subject_multicolor_multimaterial_b import training


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for name in ("concepts.json", "training_assets_manifest.jsonl", "dataset_audit.json", "source_locks.json"):
        (dataset / name).write_text("{}\n", encoding="utf-8")
    checks = []

    def verify(path):
        assert path == dataset
        checks.append(path)
        return {"status": "verified", "rows": 10}

    monkeypatch.setitem(sys.modules, "experiments.subject_multicolor_multimaterial_b.data",
                        SimpleNamespace(verify_dataset=verify))
    return dataset, tmp_path / "new_output", checks


@pytest.mark.parametrize("method", training.METHODS)
def test_configs_preserve_each_method_except_requested_data_and_dose(tmp_path, method):
    original_bytes = training.BASELINES[method].read_bytes()
    baseline = training.read_json(training.BASELINES[method])
    config = training.build_config(method, tmp_path / "dataset")
    assert {key for key in baseline["args"] if baseline["args"][key] != config["args"][key]} == {
        "concepts_list", "max_train_steps", "checkpoint_steps"}
    assert config["args"]["max_train_steps"] == 1000
    assert config["args"]["checkpoint_steps"] == [500, 750, 1000]
    assert config["args"]["modifier_token"] == "<S*>"
    assert config["args"]["initializer_token"] == "mailbox"
    assert config["args"]["learning_rate"] == 1e-5
    assert config["args"]["kv_learning_rate"] == (1e-5 if method == "fullrank" else 5e-5)
    assert config["args"]["mixed_precision"] == "no"
    assert config["run"] == {"study": training.STUDY, "variant": f"mailbox_{method}_steps1000", "seed": 42}
    assert "sd15_migration" not in config
    assert "COLORPEEL_MAILBOX_MATTE_ROOT" not in config["environment"]
    assert training.BASELINES[method].read_bytes() == original_bytes


def test_bundle_records_yaml_readable_configs_and_source_locks(prepared):
    dataset, output, checks = prepared
    bundle = training.build_configuration_bundle(dataset, output)
    assert checks == [dataset]
    for method in training.METHODS:
        path = Path(bundle["paths"][method])
        assert training.read_config(path) == bundle["configs"][method]
        record = bundle["comparison"]["comparison"][method]
        assert record["baseline_sha256"] == training.sha256(training.BASELINES[method])
        assert set(record["args_diff"]) == training.ARG_CHANGES
    assert len(bundle["comparison"]["source_hashes"]) == 4
    assert all(training.sha256(Path(path)) == digest for path, digest in
               bundle["comparison"]["source_hashes"].items())
    assert training.read_json(output / "CONFIG_COMPARISON.json") == bundle["comparison"]


def test_bundle_refuses_existing_output_and_dataset_descendant(prepared):
    dataset, output, _ = prepared
    output.mkdir()
    (output / "keep.txt").write_text("existing result")
    with pytest.raises(ValueError, match="fresh"):
        training.build_configuration_bundle(dataset, output)
    assert (output / "keep.txt").read_text() == "existing result"
    with pytest.raises(ValueError, match="outside"):
        training.build_configuration_bundle(dataset, dataset / "new_configs")
    assert not (dataset / "new_configs").exists()


def test_unknown_method_is_rejected_before_reading_data(tmp_path):
    with pytest.raises(ValueError, match="fullrank"):
        training.build_config("rank4", tmp_path)


@pytest.fixture
def execution(prepared, monkeypatch):
    dataset, output, checks = prepared
    calls, audits = [], []
    mode = {"preflight_failure": False}
    git = {"commit": "a" * 40, "branch": "codex/subject-b", "dirty": False}
    monkeypatch.setattr(training, "git_info", lambda: git)
    monkeypatch.setattr(training, "resolve_run_dir", lambda path: path.resolve())
    monkeypatch.setattr(training.shared, "verify_model", lambda: {"root": "/frozen/sd15"})

    def launch(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2"
        if command[-1] == "--dry-run":
            directory = Path(command[command.index("--run-dir") + 1])
            directory.mkdir(parents=True)
            directory.joinpath("config.yaml").write_bytes(Path(command[command.index("--config") + 1]).read_bytes())
            training._write(directory / "manifest.json", {"status": "dry-run"})
            return SimpleNamespace(returncode=int(mode["preflight_failure"]))
        assert len(calls) == 2
        assert calls[0][-1] == "--dry-run"
        return SimpleNamespace(returncode=0)

    def audit(recipe, run, config):
        assert recipe == {"method": "lora_r64", "branch": "subject", "concept_id": "mailbox", "token": "<S*>"}
        assert config["args"]["subject_lora_rank"] == config["args"]["subject_lora_alpha"] == 64
        audits.append(str(run))
        return {"status": "verified", "steps": 1000}

    monkeypatch.setattr(training.subprocess, "run", launch)
    monkeypatch.setattr(training.shared, "audit_run", audit)
    return dataset, output, checks, calls, audits, mode


def test_default_stops_after_preflight_and_never_trains(execution):
    dataset, output, checks, calls, audits, _ = execution
    result = training.run(dataset, output, "lora_r64", 2)
    assert result["status"] == "dry_run_verified"
    assert result["completed_steps"] == 0
    assert len(calls) == 1 and calls[0][-1] == "--dry-run"
    assert not audits
    assert len(checks) == 2
    assert training.read_json(output / "status.json") == result
    assert "No inference was started" in (output / "train_outputs/SUMMARY.md").read_text()


def test_explicit_execute_requires_successful_preflight_and_audits_training(execution):
    dataset, output, checks, calls, audits, _ = execution
    result = training.run(dataset, output, "lora_r64", 2, execute_training=True)
    assert result["status"] == "training_verified"
    assert result["completed_steps"] == 1000
    assert len(calls) == 2 and "--dry-run" not in calls[1]
    assert audits == [calls[1][-1]]
    assert len(checks) == 3
    assert (output / "train_outputs/training_audit.json").is_file()
    assert training.read_json(output / "training_index.json") == {
        "git": result["git"], "runs": [{"status": "verified", "steps": 1000}]}


def test_failed_preflight_blocks_training_even_when_explicitly_requested(execution):
    dataset, output, _, calls, audits, mode = execution
    mode["preflight_failure"] = True
    with pytest.raises(RuntimeError, match="preflight"):
        training.run(dataset, output, "lora_r64", 2, execute_training=True)
    assert len(calls) == 1 and not audits
    state = training.read_json(output / "status.json")
    assert state["status"] == "failed" and state["completed_steps"] == 0


def test_cli_requires_explicit_gpu_and_defaults_to_no_execute(monkeypatch):
    calls = []
    monkeypatch.setattr(training, "run", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["training", "--dataset", "data", "--output", "out",
                                     "--method", "fullrank", "--gpu", "3"])
    training.main()
    assert calls == [(Path("data"), Path("out"), "fullrank", 3, False)]
    monkeypatch.setattr(sys, "argv", ["training", "--dataset", "data", "--output", "out", "--method", "fullrank"])
    with pytest.raises(SystemExit):
        training.main()
