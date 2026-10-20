"""Verify C preserves B recipes and fail-closed automated execution."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.subject_albedo_pbr_c import run as runner, training
from experiments.subject_multicolor_multimaterial_b import training as baseline


@pytest.mark.parametrize("method", training.METHODS)
def test_c_reuses_all_b_parameters_except_dataset_and_run_metadata(tmp_path, method):
    b = baseline.build_config(method, tmp_path / "b")
    c = training.build_config(method, tmp_path / "c")
    assert {k for k in b["args"] if b["args"][k] != c["args"][k]} == {"concepts_list"}
    assert c["run"] == {**b["run"], "study": training.STUDY}
    assert c["environment"] == b["environment"]
    assert c["args"]["checkpoint_steps"] == [500, 750, 1000]
    if method == "lora_r64":
        assert c["args"]["subject_lora_rank"] == c["args"]["subject_lora_alpha"] == 64


@pytest.fixture
def execution(tmp_path, monkeypatch):
    dataset = tmp_path / "immutable_data"
    dataset.mkdir()
    for name in ("concepts.json", "training_assets_manifest.jsonl", "dataset_audit.json", "source_locks.json"):
        (dataset / name).write_text("{}")
    original = {p: p.read_bytes() for p in dataset.iterdir()}
    git = {"commit": "c" * 40, "branch": "codex/c"}
    monkeypatch.setattr(runner, "git_info", lambda: git)
    monkeypatch.setattr(runner, "resolve_run_dir", lambda p: p)
    monkeypatch.setattr(runner.data, "verify_dataset", lambda p: {"status": "verified", "rows": 30})
    calls, failure = [], {"job": None, "incomplete": False}
    output = tmp_path / "results"

    def launch(command, **kwargs):
        assert command[3].startswith("experiments.subject_albedo_pbr_c.")
        is_training = "--method" in command
        name = command[command.index("--method") + 1] if is_training else "step" + command[command.index("--step") + 1]
        calls.append(name)
        if failure["job"] == name:
            return SimpleNamespace(returncode=1)
        directory = Path(command[command.index("--output") + 1])
        directory.mkdir(parents=True)
        if is_training:
            assert not (output / "training_index.json").exists()
            record = {"status": "verified", "method": name, "branch": "subject", "token": "<S*>", "steps": 1000}
            training._write(directory / "status.json", {"status": "training_verified", "completed_steps": 750 if failure["incomplete"] else 1000})
            training._write(directory / "training_index.json", {"git": git, "runs": [record]})
        else:
            index = json.loads((output / "training_index.json").read_text())
            assert {r["method"] for r in index["runs"]} == set(training.METHODS)
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2"
            training._write(directory / "provenance.json", {"status": "succeeded", "generated_images": 20,
                "step": int(name[4:]), "sampling": runner.inference.SAMPLING, "contact_sheets": ["train", "original", "rank64"]})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", launch)
    return dataset, output, calls, failure, original


def test_auto_inference_only_follows_both_successful_training_audits(execution):
    dataset, output, calls, _, original = execution
    state = runner.run(dataset, output, 2, execute=True)
    assert calls == ["fullrank", "lora_r64", "step500", "step750", "step1000"]
    assert state["status"] == "execution_verified"
    assert state["inference_steps"] == [500, 750, 1000]
    assert all(p.read_bytes() == value for p, value in original.items())


@pytest.mark.parametrize("failure", ("fullrank", "lora_r64", "incomplete"))
def test_failure_never_starts_inference_or_merges_unverified_index(execution, failure):
    dataset, output, calls, mode, original = execution
    mode["incomplete"] = failure == "incomplete"
    mode["job"] = None if mode["incomplete"] else failure
    with pytest.raises((ValueError, RuntimeError)):
        runner.run(dataset, output, 2, execute=True)
    assert not any(name.startswith("step") for name in calls)
    assert not (output / "training_index.json").exists()
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    assert all(p.read_bytes() == value for p, value in original.items())


def test_dry_run_does_not_execute_and_existing_output_is_protected(execution):
    dataset, output, calls, _, _ = execution
    result = runner.run(dataset, output, 2)
    assert result["status"] == "dry_run_verified" and not calls
    with pytest.raises(ValueError, match="fresh"):
        runner.run(dataset, output, 2, execute=True)
