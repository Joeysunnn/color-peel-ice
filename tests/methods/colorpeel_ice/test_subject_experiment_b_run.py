"""Check the explicit training boundary and fail-closed Subject B orchestration."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.subject_multicolor_multimaterial_b import run as module


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    accepted = tmp_path / "accepted_data"
    dataset = accepted / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "old_source.txt").write_text("immutable source bytes", encoding="utf8")
    configs = accepted / "training_plan/configs"
    configs.mkdir(parents=True)
    for method in module.training.METHODS:
        module.training._write(configs / f"{method}.json", module.training.build_config(method, dataset))
    original = {str(path): path.read_bytes() for path in accepted.rglob("*") if path.is_file()}
    checks = []
    revision = {"commit": "a" * 40, "branch": "codex/subject-b", "dirty": False}
    monkeypatch.setattr(module.data, "verify_dataset", lambda path: checks.append(path))
    monkeypatch.setattr(module, "git_info", lambda: revision)
    monkeypatch.setattr(module, "resolve_run_dir", lambda path: path.resolve())
    return SimpleNamespace(dataset=dataset, output=tmp_path / "new_results", checks=checks,
                           revision=revision, original=original, accepted=accepted)


@pytest.fixture
def execution(prepared, monkeypatch):
    calls, mode = [], {"returncode_failure": None, "invalid_training": None, "invalid_inference": None}

    def launch(command, **kwargs):
        training = command[3] == "experiments.subject_multicolor_multimaterial_b.training"
        name = command[command.index("--method") + 1] if training else "step" + command[command.index("--step") + 1]
        directory = Path(command[command.index("--output") + 1])
        assert command[command.index("--dataset") + 1] == str(prepared.dataset)
        assert kwargs["cwd"] == module.data.ROOT
        assert kwargs["stdin"] == module.subprocess.DEVNULL
        assert kwargs["stderr"] == module.subprocess.STDOUT
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        if training:
            assert command[command.index("--gpu") + 1] == "2"
            assert "--execute-training" in command
            # A merged index must not exist until BOTH child audits are accepted.
            assert not (prepared.output / "training_index.json").exists()
        else:
            assert "--execute-inference" in command
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2"
            assert command[command.index("--device") + 1] == "cuda:0"
            index_path = Path(command[command.index("--training-index") + 1])
            index = json.loads(index_path.read_text())
            assert index["git"] == prepared.revision
            assert [item["method"] for item in index["runs"]] == list(module.training.METHODS)
            assert all(item["status"] == "verified" for item in index["runs"])
        calls.append({"name": name, "command": command, "environment": kwargs["env"]})
        if mode["returncode_failure"] == name:
            return SimpleNamespace(returncode=1)
        directory.mkdir(parents=True)
        if training:
            selection = {"status": "verified", "method": name, "branch": "subject", "token": "<S*>",
                         "steps": 1000, "run": str(directory / "actual_training_run")}
            status = {"status": "training_verified", "completed_steps": 1000}
            index = {"git": prepared.revision, "runs": [selection]}
            if mode["invalid_training"] == name:
                status["completed_steps"] = 750
            module.training._write(directory / "status.json", status)
            module.training._write(directory / "training_index.json", index)
        else:
            provenance = {"status": "succeeded", "generated_images": 20, "step": int(name[4:]),
                          "sampling": module.inference.SAMPLING,
                          "contact_sheets": ["training.jpg", "fullrank.jpg", "rank64.jpg"]}
            if mode["invalid_inference"] == name:
                provenance["generated_images"] = 19
            module.training._write(directory / "provenance.json", provenance)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", launch)
    return prepared, calls, mode


def assert_sources_unchanged(prepared):
    assert {str(path): path.read_bytes() for path in prepared.accepted.rglob("*") if path.is_file()} == prepared.original


def test_default_only_writes_plan_without_child_execution(execution):
    prepared, calls, _ = execution
    result = module.run(prepared.dataset, prepared.output, 2)
    assert result["status"] == "dry_run_verified"
    assert result["stop_reason"] == "execution not requested"
    assert result["completed_steps"] == {"fullrank": 0, "lora_r64": 0}
    assert not calls and not (prepared.output / "training_index.json").exists()
    assert not (prepared.output / "training").exists()
    planned = json.loads((prepared.output / "train_outputs/COMMANDS.json").read_text())
    assert [job["name"] for job in planned] == ["fullrank", "lora_r64", "step500", "step750", "step1000"]
    assert_sources_unchanged(prepared)


def test_explicit_execution_audits_both_trainings_before_three_inference_steps(execution):
    prepared, calls, _ = execution
    result = module.run(prepared.dataset, prepared.output, 2, execute=True)
    assert [call["name"] for call in calls] == ["fullrank", "lora_r64", "step500", "step750", "step1000"]
    assert result["status"] == "execution_verified" and result["active_job"] is None
    assert result["completed_steps"] == {"fullrank": 1000, "lora_r64": 1000}
    assert result["inference_steps"] == [500, 750, 1000]
    assert result["checkpoint_steps"] == [500, 750, 1000]
    assert result["best_checkpoint"] is None and result["best_metric"] is None
    assert result["resume"] is False and result["gpu"] == 2
    assert len(prepared.checks) == 2
    assert json.loads((prepared.output / "train_outputs/status.json").read_text()) == result
    assert_sources_unchanged(prepared)


@pytest.mark.parametrize("failed_method", module.training.METHODS)
@pytest.mark.parametrize("failure", ("returncode_failure", "invalid_training"))
def test_failed_or_incomplete_training_never_merges_index_or_starts_inference(execution, failed_method, failure):
    prepared, calls, mode = execution
    mode[failure] = failed_method
    with pytest.raises((ValueError, RuntimeError)):
        module.run(prepared.dataset, prepared.output, 2, execute=True)
    assert [call["name"] for call in calls] == (["fullrank"] if failed_method == "fullrank" else ["fullrank", "lora_r64"])
    assert not (prepared.output / "training_index.json").exists()
    status = json.loads((prepared.output / "status.json").read_text())
    assert status["status"] == "failed" and status["inference_steps"] == []
    assert status["completed_steps"] == {"fullrank": 0 if failed_method == "fullrank" else 1000, "lora_r64": 0}
    assert_sources_unchanged(prepared)


@pytest.mark.parametrize("failure", ("returncode_failure", "invalid_inference"))
def test_inference_failure_preserves_completed_training_and_partial_step_status(execution, failure):
    prepared, calls, mode = execution
    mode[failure] = "step750"
    with pytest.raises((ValueError, RuntimeError)):
        module.run(prepared.dataset, prepared.output, 2, execute=True)
    assert [call["name"] for call in calls] == ["fullrank", "lora_r64", "step500", "step750"]
    status = json.loads((prepared.output / "status.json").read_text())
    assert status["status"] == "failed" and status["active_job"] == "step750"
    assert status["inference_steps"] == [500]
    assert status["completed_steps"] == {"fullrank": 1000, "lora_r64": 1000}
    assert (prepared.output / "training_index.json").is_file()
    assert "step750" in status["stop_reason"]
    assert not (prepared.output / "step1000").exists()
    assert_sources_unchanged(prepared)


def test_existing_output_is_refused_without_modification(execution):
    prepared, calls, _ = execution
    prepared.output.mkdir()
    marker = prepared.output / "old_result.txt"
    marker.write_text("old image results", encoding="utf8")
    with pytest.raises(ValueError, match="fresh external"):
        module.run(prepared.dataset, prepared.output, 2, execute=True)
    assert marker.read_text() == "old image results" and len(list(prepared.output.iterdir())) == 1
    assert not calls
    assert_sources_unchanged(prepared)


def test_changed_approved_recipe_is_refused_before_creating_results(execution):
    prepared, calls, _ = execution
    path = prepared.accepted / "training_plan/configs/lora_r64.json"
    changed = module.training.read_config(path)
    changed["args"]["kv_learning_rate"] = 1e-3
    module.training._write(path, changed)
    modified_bytes = path.read_bytes()
    with pytest.raises(ValueError, match="approved data preview"):
        module.run(prepared.dataset, prepared.output, 2, execute=True)
    assert not calls and not prepared.output.exists()
    assert path.read_bytes() == modified_bytes
    for source, original_bytes in prepared.original.items():
        if source != str(path):
            assert Path(source).read_bytes() == original_bytes


@pytest.mark.parametrize("gpu", (-1, True, "2"))
def test_invalid_gpu_is_rejected_before_source_reads(prepared, gpu):
    with pytest.raises(ValueError, match="nonnegative GPU"):
        module.run(prepared.dataset, prepared.output, gpu, execute=True)
    assert not prepared.checks and not prepared.output.exists()
