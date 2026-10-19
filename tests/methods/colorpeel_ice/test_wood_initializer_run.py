"""Exercise initializer orchestration without training or GPU inference."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from experiments.natural_material_initializer_v1 import run as experiment
from experiments.natural_material_initializer_v1 import source, transfer
from experiments.natural_texture_material_v1.training_config import build_config, recipes


def baseline(method, protocol, source_id="wood_spoon"):
    recipe = next(row for row in recipes() if row["concept_id"] == f"{source_id}_material"
                  and row["method"] == method)
    return build_config(recipe, Path("staging"), protocol)


@pytest.fixture(params=[("wood_spoon", "material"), ("mailbox", "material"),
                        ("metal_spoon", "material"), ("wood_spoon", "wood")])
def harness(tmp_path, monkeypatch, request):
    source_id, initializer = request.param
    monkeypatch.setenv("COLORPEEL_OUTPUT_ROOT", str(tmp_path))
    protocol = json.loads(source.schema.PARENT.read_text())
    configs = {method: baseline(method, protocol, source_id) for method in ("fullrank", "lora_r64")}
    source_root, old_transfer = tmp_path / "old_training", tmp_path / "old_transfer"
    source_root.mkdir()
    old_transfer.mkdir()
    marker = source_root / "old_checkpoint.bin"
    marker.write_bytes(b"immutable original training checkpoint")
    old_digest = source.schema.sha256(marker)
    plan = {"protocol": protocol, "configs": configs,
        "recipes": [row for row in recipes() if row["concept_id"] == f"{source_id}_material"],
        "references": {str(step): {method: {"run": str(source_root / method),
            "checkpoint": str(source_root / method / f"step{step}"),
            "images": [{"id": f"rgb_{index}"} for index in range(8)]}
            for method in configs} for step in experiment.STEPS},
        "baseline_rows": {}, "source_locks": {str(marker): old_digest}}
    if source_id != "wood_spoon":
        plan["source_id"] = source_id
    events, exports, children, queues = [], [], [], []
    settings = {"gpus": [1, 2, 3]}
    mode = {"failure": None}
    output = tmp_path / "new_experiment"
    git = {"commit": "a" * 40, "branch": "codex/wood-init", "dirty": False}
    monkeypatch.setattr(experiment, "prepare", lambda *_: copy.deepcopy(plan))
    monkeypatch.setattr(experiment, "git_info", lambda: git)
    monkeypatch.setattr(experiment.shared, "verify_model", lambda: {"root": str(tmp_path / "model")})
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs:
        {"material": [7118], "wood": [1704]}.get(text, []))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)))
    monkeypatch.setattr(experiment.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * (1 << 30)))

    def dry_run(command, **kwargs):
        assert command[-1] == "--dry-run"
        directory = Path(command[command.index("--run-dir") + 1])
        config_path = Path(command[command.index("--config") + 1])
        directory.mkdir()
        (directory / "config.yaml").write_bytes(config_path.read_bytes())
        experiment.write(directory / "manifest.json", {"status": "dry-run"})
        events.append(("preflight", str(directory)))
        return SimpleNamespace(returncode=int(mode["failure"] == "preflight"))

    def queue(gpu, jobs, batch):
        assert len([event for event in events if event[0] == "preflight"]) == 2
        assert len(jobs) == (2 if len(settings["gpus"]) == 1 else 1)
        queues.append((gpu, [job["recipe"]["method"] for job in jobs]))
        results = []
        for job in jobs:
            method = job["recipe"]["method"]
            assert job["recipe"]["concept_id"] == f"{source_id}_material"
            assert job["recipe"]["source_id"] == source_id
            assert job["recipe"]["branch"] == "material"
            assert {key for key in configs[method]["args"]
                    if configs[method]["args"][key] != job["config"]["args"][key]} == {"initializer_token"}
            assert job["config"]["args"]["initializer_token"] == initializer
            events.append(("training", method))
            results.append({"status": "verified", "audit": {"method": method, "steps": 1000,
                "concept_id": f"{source_id}_material", "source_id": source_id, "branch": "material",
                "run": job["run"], "checkpoint": str(Path(job["run"]) / "checkpoints")}})
        return results

    def export(item, step, destination):
        assert len([event for event in events if event[0] == "training"]) == 2
        destination.mkdir(parents=True)
        exports.append((step, item["method"]))
        events.append(("export", step, item["method"]))
        return {**item, "steps": step, "checkpoint": str(destination)}

    class Child:
        def __init__(self, command, **kwargs):
            assert len(exports) == 6
            if len(settings["gpus"]) == 1:
                assert len([event for event in events if event[0] == "wait"]) == len(children)
            self.step = int(command[command.index("--step") + 1])
            self.pid = 1000 + self.step
            self.gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
            assert "experiments.natural_material_initializer_v1.transfer" in command
            children.append(self)
            events.append(("launch", self.step, self.gpu))
            folder = output / f"step{self.step}"
            folder.mkdir()
            sheets = [f"sheet_{number}.jpg" for number in range(4)]
            for name in sheets:
                (folder / name).write_bytes(b"new contact sheet")
            if mode["failure"] == "missing_sheet" and self.step == 750:
                (folder / sheets[-1]).unlink()
            experiment.write(folder / "summary.json", {"status": "succeeded", "generated_images": 16,
                "reused_images": 48, "total_images": 64, "missing_images": 0,
                "training_steps": self.step, "contact_sheets": sheets})

        def wait(self):
            if len(settings["gpus"]) == 3:
                assert len(children) == 3, "all GPU workers must start before waiting"
            else:
                assert len(children) == 1 + len([event for event in events if event[0] == "wait"])
            events.append(("wait", self.step))
            return int(mode["failure"] == "child" and self.step == 750)

    monkeypatch.setattr(experiment.subprocess, "run", dry_run)
    monkeypatch.setattr(experiment.subprocess, "Popen", Child)
    monkeypatch.setattr(experiment.shared, "run_queue", queue)
    monkeypatch.setattr(experiment, "export_step", export)
    return SimpleNamespace(output=output, source=source_root, transfer=old_transfer, plan=plan,
        source_id=source_id, initializer=initializer, events=events, exports=exports, children=children, mode=mode,
        marker=marker, digest=old_digest, queues=queues, settings=settings)


def run_experiment(harness):
    arguments = {} if harness.source_id == "wood_spoon" else {"source_id": harness.source_id}
    if harness.initializer != "material":
        arguments["initializer"] = harness.initializer
    return experiment.run(harness.source, harness.transfer, harness.output, harness.settings["gpus"], **arguments)


def test_two_preflights_two_initializer_only_runs_six_exports_then_parallel_transfer(harness):
    before = copy.deepcopy(harness.plan)
    result = run_experiment(harness)
    assert result["status"] == "execution_verified"
    assert result["completed_training_runs"] == 2
    assert result["completed_steps"] == 2000
    assert result["new_images"] == 48 and result["reused_images"] == 144
    assert len(harness.exports) == len(set(harness.exports)) == 6
    assert [(child.step, child.gpu) for child in harness.children] == [(500, "1"), (750, "2"), (1000, "3")]
    assert [event[0] for event in harness.events[:2]] == ["preflight", "preflight"]
    assert harness.plan == before
    assert source.schema.sha256(harness.marker) == harness.digest
    assert list(harness.source.iterdir()) == [harness.marker]
    comparison = json.loads((harness.output / "train_outputs/CONFIG_COMPARISON.json").read_text())
    assert set(comparison) == {"fullrank", "lora_r64"}
    assert all(value == {"initializer_token": {"old": "metal", "new": harness.initializer}}
               for value in comparison.values())
    saved_plan = json.loads((harness.output / "plan.json").read_text())
    assert saved_plan["initializer"] == {"word": harness.initializer,
        "token_ids": [7118] if harness.initializer == "material" else [1704]}
    assert saved_plan.get("source_id", "wood_spoon") == harness.source_id
    assert {row["concept_id"] for row in saved_plan["recipes"]} == {f"{harness.source_id}_material"}
    for step in experiment.STEPS:
        index = json.loads((harness.output / f"exports/step{step}/index.json").read_text())
        assert len(index["runs"]) == 2
        for method, reference in index["references"].items():
            assert reference["initializer"] == harness.initializer
            assert reference["baseline_run"] == before["references"][str(step)][method]["run"]
            assert reference["baseline_checkpoint"] == before["references"][str(step)][method]["checkpoint"]
            assert reference["images"] == before["references"][str(step)][method]["images"]


def test_single_gpu_serializes_both_training_jobs_and_all_transfer_steps(harness):
    harness.settings["gpus"] = [2]
    result = run_experiment(harness)
    assert result["status"] == "execution_verified"
    assert result["completed_training_runs"] == 2 and result["completed_steps"] == 2000
    assert result["new_images"] == 48 and result["reused_images"] == 144
    assert len(harness.queues) == 1
    assert harness.queues[0][0] == 2
    assert set(harness.queues[0][1]) == {"fullrank", "lora_r64"}
    assert [(child.step, child.gpu) for child in harness.children] == [(500, "2"), (750, "2"), (1000, "2")]
    assert [event[0] for event in harness.events if event[0] in ("launch", "wait")] == [
        "launch", "wait", "launch", "wait", "launch", "wait"]
    assert source.schema.sha256(harness.marker) == harness.digest


@pytest.mark.parametrize("failure", ["child", "missing_sheet"])
def test_single_gpu_failed_transfer_does_not_start_later_steps(harness, failure):
    harness.settings["gpus"] = [2]
    harness.mode["failure"] = failure
    with pytest.raises(RuntimeError):
        run_experiment(harness)
    assert [(child.step, child.gpu) for child in harness.children] == [(500, "2"), (750, "2")]
    state = json.loads((harness.output / "status.json").read_text())
    assert state["status"] == "failed"
    assert [record["status"] for record in state["transfer"]] == ["succeeded", "failed"]
    assert source.schema.sha256(harness.marker) == harness.digest


@pytest.mark.parametrize("failure", ["preflight", "child", "missing_sheet"])
def test_incomplete_preflight_or_transfer_cannot_claim_completion(harness, failure):
    harness.mode["failure"] = failure
    with pytest.raises(RuntimeError):
        run_experiment(harness)
    state = json.loads((harness.output / "status.json").read_text())
    assert state["status"] == "failed"
    assert state["stop_reason"] == "execution_error"
    assert source.schema.sha256(harness.marker) == harness.digest
    if failure == "preflight":
        assert not harness.exports and not harness.children
        assert not any(event[0] == "training" for event in harness.events)
        assert state["completed_steps"] == 0
    else:
        assert len(harness.children) == 3
        assert [record["status"] for record in state["transfer"]] == ["succeeded", "failed", "succeeded"]


@pytest.mark.parametrize("source_id", ["wood_spoon", "mailbox", "metal_spoon"])
def test_comparison_rows_reuse_exact_controls_and_add_only_sixteen_material_token_prompts(source_id):
    protocol = json.loads(source.schema.PARENT.read_text())
    for step in experiment.STEPS:
        old = []
        for row in transfer.probe.build_rows(protocol, step, transfer=True):
            if row["concept_id"] == f"{source_id}_material":
                old.append({**row, "source_image": "/original/" + row["image_path"], "status": "ok",
                    "texture_adapter_loaded": False,
                    "material_adapter_loaded": row["condition"] == "learned_material",
                    "image_sha256": "b" * 64})
        before = copy.deepcopy(old)
        arguments = {} if source_id == "wood_spoon" else {"source_id": source_id}
        rows = transfer.comparison_rows(protocol, old, step, **arguments)
        assert len(rows) == len({row["id"] for row in rows}) == 64
        assert {row["concept_id"] for row in rows} == {f"{source_id}_material"}
        assert {row["source_id"] for row in rows} == {source_id}
        assert sum(row["reused"] for row in rows) == 48
        for condition in ("base", "literal", "learned_metal_init", "learned_material_init"):
            selected = [row for row in rows if row["condition"] == condition]
            assert len(selected) == 16
            assert {(row["method"], row["context"], row["seed"]) for row in selected} == {
                (method, context, seed) for method in ("fullrank", "lora_r64")
                for context in ("chair", "mug", "vase", "sofa") for seed in (42, 43)}
            assert all(row["training_steps"] == step for row in selected)
        new = [row for row in rows if not row["reused"]]
        assert len(new) == 16
        assert all(row["condition"] == "learned_material_init" and "__learned_material_init__" in row["id"]
                   and row["branch"] == "material" and "<M*>" in row["prompt"]
                   and "<T*>" not in row["prompt"] and "texture" not in row["prompt"] for row in new)
        assert old == before
        assert all(row["source_image"].startswith("/original/") for row in rows if row["reused"])
