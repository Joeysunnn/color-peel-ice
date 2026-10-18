"""Dataset-only lifecycle, immutable inputs and actual shader evidence."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.natural_texture_material_v1 import build_dataset as runner
from experiments.natural_texture_material_v1 import render_dataset as render


@pytest.fixture
def setup_runner(tmp_path, monkeypatch):
    extraction = tmp_path / "accepted_extraction"
    extraction.mkdir()
    source = extraction / "texture.png"
    source.write_bytes(b"accepted extraction remains immutable")
    protocol = {"renderer": {"blender": "blender"},
                "source_locks": {str(source): runner.schema.sha256(source)}}
    revision = {"commit": "abc123", "branch": "codex/texture-material-v1"}
    monkeypatch.setattr(runner, "resolve_run_dir", lambda path: path.resolve())
    monkeypatch.setattr(runner, "git_info", lambda: revision.copy())
    monkeypatch.setattr(runner.schema, "build_protocol", lambda path: copy.deepcopy(protocol))
    return extraction, tmp_path / "new_dataset", source, revision


def test_success_runs_only_plan_render_stage_and_records_no_training(setup_runner, monkeypatch):
    extraction, output, source, _ = setup_runner
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        assert source.read_bytes() == b"accepted extraction remains immutable"
        return SimpleNamespace(returncode=0)

    from experiments.natural_texture_material_v1 import dataset_stage
    audit = {"status": "complete", "train_images": 72, "training_started": False}
    verified = []
    monkeypatch.setattr(runner.subprocess, "run", execute)
    monkeypatch.setattr(dataset_stage, "verify_staging", lambda path: verified.append(path) or audit)
    state = runner.run(extraction, output, 2)
    assert state["status"] == "dataset_verified"
    assert state["training_launched"] is False
    assert state["expected_images"] == state["dataset"]["train_images"] == 72
    assert [record["stage"] for record in state["steps"]] == ["plan", "render", "stage"]
    assert len(calls) == 3
    assert calls[0][0][3] == "experiments.natural_texture_material_v1.dataset_protocol"
    assert calls[1][0][0] == "blender"
    assert "render_dataset.py" == Path(calls[1][0][calls[1][0].index("--python") + 1]).name
    assert calls[2][0][3] == "experiments.natural_texture_material_v1.dataset_stage"
    assert calls[1][1]["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert all(kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1" for _, kwargs in calls)
    assert all(kwargs["stdin"] == runner.subprocess.DEVNULL for _, kwargs in calls)
    assert verified == [output / "staging"]
    assert json.loads((output / "status.json").read_text()) == state
    assert json.loads((output / "repro_outputs/status.json").read_text()) == state
    assert "Training has not started" in (output / "index.html").read_text()
    assert list(extraction.iterdir()) == [source]


@pytest.mark.parametrize("failed_stage", [0, 1, 2])
def test_failed_stage_stops_and_preserves_accepted_extraction(setup_runner, monkeypatch, failed_stage):
    extraction, output, source, _ = setup_runner
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=7 if len(calls) - 1 == failed_stage else 0)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    with pytest.raises(RuntimeError, match="Dataset .* failed"):
        runner.run(extraction, output, 1)
    assert len(calls) == failed_stage + 1
    state = json.loads((output / "status.json").read_text())
    assert state["status"] == "failed"
    assert state["training_launched"] is False
    assert state["steps"][-1]["returncode"] == 7
    assert not (output / "index.html").exists()
    assert source.read_bytes() == b"accepted extraction remains immutable"


def test_dirty_checkout_between_stages_stops_before_render(setup_runner, monkeypatch):
    extraction, output, _, revision = setup_runner
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs:
                        calls.append(command) or SimpleNamespace(returncode=0))
    checks = iter([revision, revision, {**revision, "commit": "changed"}])
    monkeypatch.setattr(runner, "git_info", lambda: next(checks))
    with pytest.raises(ValueError, match="Checkout changed"):
        runner.run(extraction, output, 1)
    assert len(calls) == 1
    assert json.loads((output / "status.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("target", ["existing", "extraction_child", "repo"])
def test_rejects_old_or_protected_output_before_subprocess(setup_runner, monkeypatch, target):
    extraction, output, source, _ = setup_runner
    if target == "existing":
        output.mkdir()
        sentinel = output / "old_result.png"
        sentinel.write_bytes(b"do not overwrite")
    elif target == "extraction_child":
        output = extraction / "new_dataset"
    else:
        output = runner.ROOT / "forbidden_dataset"
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("launched subprocess"))
    with pytest.raises(ValueError, match="new external dataset"):
        runner.run(extraction, output, 1)
    assert source.read_bytes() == b"accepted extraction remains immutable"
    if target == "existing":
        assert sentinel.read_bytes() == b"do not overwrite"
    else:
        assert not output.exists()


def test_render_rejects_modified_protocol_or_source_lock(tmp_path, monkeypatch):
    source = tmp_path / "texture.png"
    source.write_bytes(b"accepted")
    protocol = {"source_locks": {str(source): render.schema.sha256(source)}}
    rows = [{"id": str(index)} for index in range(72)]
    monkeypatch.setattr(render.schema, "records", lambda protocol: rows)
    plan = {"status": "complete", "protocol_sha256": render.schema.protocol_sha256(protocol), "records": rows}
    assert render.validate_plan(plan, protocol) is rows
    with pytest.raises(ValueError, match="Plan differs"):
        render.validate_plan({**plan, "protocol_sha256": "changed"}, protocol)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source inputs changed"):
        render.validate_plan(plan, protocol)


def shader_scene(texture=False, roughness_linked=False):
    inputs = {name: SimpleNamespace(is_linked=False)
              for name in ("Roughness", "Metallic", "Normal", "Base Color")}
    inputs["Base Color"].is_linked = texture
    inputs["Roughness"].is_linked = roughness_linked
    nodes = [SimpleNamespace(type="BSDF_PRINCIPLED", inputs=inputs)]
    if texture:
        nodes.append(SimpleNamespace(type="TEX_IMAGE"))
    material = SimpleNamespace(node_tree=SimpleNamespace(nodes=nodes))
    obj = SimpleNamespace(type="MESH", name="Subject", data=SimpleNamespace(materials=[material]))
    return SimpleNamespace(context=SimpleNamespace(scene=SimpleNamespace(objects=[obj])))


def test_actual_material_shader_rejects_texture_or_spatial_roughness():
    assert render.shader_audit(shader_scene(), "material")["texture_image_nodes"] == 0
    with pytest.raises(ValueError, match="Actual shader"):
        render.shader_audit(shader_scene(texture=True), "material")
    with pytest.raises(ValueError, match="Actual shader"):
        render.shader_audit(shader_scene(roughness_linked=True), "material")
    assert render.shader_audit(shader_scene(texture=True), "texture")["base_color_input_linked"] is True
