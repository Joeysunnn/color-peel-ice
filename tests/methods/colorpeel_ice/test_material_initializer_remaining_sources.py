"""Verify sequential source execution and aggregate completion evidence."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.natural_material_initializer_v1 import remaining_sources as batch


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setenv("COLORPEEL_OUTPUT_ROOT", str(tmp_path))
    training, transfer = tmp_path / "old_training", tmp_path / "old_transfer"
    for directory in (training, transfer):
        directory.mkdir()
        (directory / "original.txt").write_bytes(b"immutable old result")
    original = {str(path): path.read_bytes() for path in tmp_path.glob("old_*/*")}
    output = tmp_path / "new_batch"
    git = {"commit": "a" * 40, "branch": "codex/texture-material-v1", "dirty": False}
    calls = []
    failures = {}
    monkeypatch.setattr(batch, "git_info", lambda: git)

    def execute(command, **kwargs):
        source_id = command[command.index("--source-id") + 1]
        destination = Path(command[command.index("--output") + 1])
        assert command[:4] == [batch.sys.executable, "-B", "-m",
                               "experiments.natural_material_initializer_v1.run"]
        assert source_id == ("mailbox" if not calls else "metal_spoon")
        assert destination == output / source_id
        assert command[command.index("--source-training") + 1] == str(training)
        assert command[command.index("--source-transfer") + 1] == str(transfer)
        assert command[command.index("--gpus") + 1:] == ["1", "2", "3"]
        assert kwargs["cwd"] == batch.ROOT
        state = json.loads((output / "status.json").read_text())
        assert state["jobs"][len(calls)]["status"] == "running"
        if calls:
            assert calls == ["mailbox"]
            assert state["jobs"][0]["status"] == "verified"
            assert state["completed_training_runs"] == 2 and state["completed_steps"] == 2000
        else:
            assert not (output / "metal_spoon").exists()
        calls.append(source_id)
        destination.mkdir()
        summary = {"status": "execution_verified", "source_id": source_id, "git": git,
                   "completed_training_runs": 2, "completed_steps": 2000,
                   "new_images": 48, "reused_images": 144}
        failure = failures.get(source_id)
        if failure == "wrong_source":
            summary["source_id"] = "wood_spoon"
        elif failure == "incomplete_training":
            summary["completed_steps"] = 1000
        elif failure == "incomplete_transfer":
            summary["new_images"] = 47
        elif failure == "unverified":
            summary["status"] = "generating"
        if failure != "missing_summary":
            (destination / "status.json").write_text(json.dumps(summary))
        return SimpleNamespace(returncode=int(failure == "exit_failure"))

    monkeypatch.setattr(batch.subprocess, "run", execute)
    return SimpleNamespace(training=training, transfer=transfer, output=output, calls=calls,
                           failures=failures, original=original)


def execute(harness):
    return batch.run(harness.training, harness.transfer, harness.output, [1, 2, 3])


def assert_old_results_unchanged(harness):
    assert {str(path): path.read_bytes() for path in harness.training.parent.glob("old_*/*")} == harness.original


def test_both_sources_complete_sequentially_with_four_training_runs(harness):
    result = execute(harness)
    assert harness.calls == ["mailbox", "metal_spoon"]
    assert result["status"] == "execution_verified"
    assert result["completed_training_runs"] == result["planned_training_runs"] == 4
    assert result["completed_steps"] == result["max_steps"] == 4000
    assert result["checkpoint_steps"] == [500, 750, 1000]
    assert result["new_images"] == 96 and result["reused_images"] == 288
    assert [(job["source_id"], job["status"], job["returncode"]) for job in result["jobs"]] == [
        ("mailbox", "verified", 0), ("metal_spoon", "verified", 0)]
    assert json.loads((harness.output / "status.json").read_text()) == result
    assert json.loads((harness.output / "train_outputs/status.json").read_text()) == result
    assert_old_results_unchanged(harness)


@pytest.mark.parametrize("failure", ["exit_failure", "missing_summary", "wrong_source",
                                    "incomplete_training", "incomplete_transfer", "unverified"])
def test_first_source_failure_or_incomplete_evidence_prevents_second_source(harness, failure):
    harness.failures["mailbox"] = failure
    with pytest.raises(RuntimeError, match="next source not started"):
        execute(harness)
    assert harness.calls == ["mailbox"]
    assert not (harness.output / "metal_spoon").exists()
    state = json.loads((harness.output / "status.json").read_text())
    assert state["status"] == "failed" and state["stop_reason"] == "execution_error"
    assert state["completed_steps"] == state["completed_training_runs"] == 0
    assert [job["status"] for job in state["jobs"]] == ["failed", "queued"]
    assert_old_results_unchanged(harness)


def test_second_source_failure_preserves_verified_first_source_count(harness):
    harness.failures["metal_spoon"] = "incomplete_transfer"
    with pytest.raises(RuntimeError):
        execute(harness)
    assert harness.calls == ["mailbox", "metal_spoon"]
    state = json.loads((harness.output / "status.json").read_text())
    assert state["status"] == "failed"
    assert state["completed_steps"] == 2000 and state["completed_training_runs"] == 2
    assert [job["status"] for job in state["jobs"]] == ["verified", "failed"]
    assert_old_results_unchanged(harness)


def test_existing_output_is_rejected_before_any_process_or_file_change(harness):
    harness.output.mkdir()
    marker = harness.output / "existing_result.png"
    marker.write_bytes(b"preserve this existing experiment")
    with pytest.raises(ValueError, match="fresh external output"):
        execute(harness)
    assert harness.calls == []
    assert list(harness.output.iterdir()) == [marker]
    assert marker.read_bytes() == b"preserve this existing experiment"
    assert_old_results_unchanged(harness)
