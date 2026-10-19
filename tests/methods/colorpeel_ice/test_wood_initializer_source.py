"""The initializer ablation changes one argument and reads immutable old controls."""

import copy
import json
from pathlib import Path

import pytest

from experiments.natural_material_initializer_v1 import source
from experiments.natural_texture_material_v1.training_config import build_config, recipes


def baseline(method, source_id="wood_spoon"):
    protocol = json.loads(source.schema.PARENT.read_text())
    recipe = next(row for row in recipes() if row["concept_id"] == source_id + "_material"
                  and row["method"] == method)
    return build_config(recipe, Path("staging"), protocol)


@pytest.mark.parametrize("method", ["fullrank", "lora_r64"])
@pytest.mark.parametrize("source_id", source.SOURCE_IDS)
def test_initializer_is_the_only_training_argument_changed_and_original_is_untouched(method, source_id):
    old = baseline(method, source_id)
    before = copy.deepcopy(old)
    new = source.build_config(old, method, source_id)
    assert old == before
    assert {key for key in old["args"] if old["args"][key] != new["args"][key]} == {"initializer_token"}
    assert new["args"]["initializer_token"] == "material"
    assert new["run"]["study"] == "natural_material_initializer_v1"
    assert new["run"]["variant"] == f"{source_id}_material_{method}_init_material_steps1000"
    assert new["run"]["seed"] == old["run"]["seed"] == 42
    assert new["data_manifest"] == old["data_manifest"]
    assert new["texture_material"] == old["texture_material"]


@pytest.mark.parametrize("method", ["fullrank", "lora_r64"])
def test_wood_initializer_changes_only_initializer_and_preserves_source(method):
    old = baseline(method)
    before = copy.deepcopy(old)
    new = source.build_config(old, method, "wood_spoon", "wood")
    assert old == before
    assert {key for key in old["args"] if old["args"][key] != new["args"][key]} == {"initializer_token"}
    assert new["args"]["initializer_token"] == "wood"
    assert new["run"]["variant"] == f"wood_spoon_material_{method}_init_wood_steps1000"
    assert new["run"]["seed"] == old["run"]["seed"] == 42
    assert new["data_manifest"] == old["data_manifest"]
    assert new["texture_material"] == old["texture_material"]


@pytest.mark.parametrize("source_id,initializer", [("mailbox", "wood"), ("metal_spoon", "wood"),
    ("wood_spoon", "metal"), ("wood_spoon", "unknown"), ("wood_spoon", "")])
def test_invalid_initializer_is_rejected_without_mutating_baseline(source_id, initializer):
    old = baseline("fullrank", source_id)
    before = copy.deepcopy(old)
    with pytest.raises(ValueError, match="initializer"):
        source.validate_initializer(source_id, initializer)
    with pytest.raises(ValueError, match="initializer"):
        source.build_config(old, "fullrank", source_id, initializer)
    assert old == before


@pytest.mark.parametrize("method", ["fullrank", "lora_r64"])
def test_wood_initializer_rejects_mixed_source_and_nonmetal_baseline(method):
    with pytest.raises(ValueError):
        source.build_config(baseline(method, "mailbox"), method, "wood_spoon", "wood")
    old = baseline(method)
    old["args"]["initializer_token"] = "wood"
    with pytest.raises(ValueError):
        source.build_config(old, method, "wood_spoon", "wood")


@pytest.mark.parametrize("key,value", [("initializer_token", "wood"), ("modifier_token", "<T*>"),
    ("max_train_steps", 750), ("checkpoint_steps", [500, 1000]), ("token_local_kv", False)])
def test_reject_changed_original_fullrank_recipe(key, value):
    old = baseline("fullrank")
    old["args"][key] = value
    with pytest.raises(ValueError, match="original wood_spoon Material"):
        source.build_config(old, "fullrank")


def test_reject_wrong_concept_method_and_lora_mode():
    old = baseline("lora_r64")
    for changed, method in [(old, "fullrank"), (old, "unknown")]:
        with pytest.raises(ValueError):
            source.build_config(changed, method)
    old["texture_material"]["concept_id"] = "metal_spoon_material"
    with pytest.raises(ValueError):
        source.build_config(old, "lora_r64")
    old = baseline("lora_r64")
    old["args"]["subject_lora_mode"] = "full_kv"
    with pytest.raises(ValueError):
        source.build_config(old, "lora_r64")


@pytest.mark.parametrize("source_id", source.SOURCE_IDS)
@pytest.mark.parametrize("key", ["concept_id", "source_id", "branch"])
def test_source_metadata_cannot_mix_concepts(source_id, key):
    old = baseline("fullrank", source_id)
    old["texture_material"][key] = "another_source"
    with pytest.raises(ValueError):
        source.build_config(old, "fullrank", source_id)


def test_unknown_source_is_rejected_before_reading_any_sources():
    with pytest.raises(ValueError, match="source_id"):
        source.build_config({}, "fullrank", "unknown")
    with pytest.raises(ValueError, match="source_id"):
        source._baseline_rows(Path("missing"), [], {}, "unknown")
    with pytest.raises(ValueError, match="source_id"):
        source.prepare("missing", "missing", "unknown")


def test_omitted_source_keeps_wood_spoon_compatibility():
    old = baseline("fullrank")
    assert source.build_config(old, "fullrank") == source.build_config(old, "fullrank", "wood_spoon")
    with pytest.raises(ValueError):
        source.build_config(baseline("fullrank", "mailbox"), "fullrank")


@pytest.fixture
def old_rows(tmp_path):
    protocol = json.loads(source.schema.PARENT.read_text())
    expected = source.build_rows(protocol, 500, True)
    rows = []
    for item in expected:
        path = tmp_path / item["image_path"]
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(item["id"].encode())
        learned = item["condition"].startswith("learned_")
        rows.append({**item, "status": "ok", "image_sha256": source.schema.sha256(path),
            "texture_adapter_loaded": learned and item["branch"] == "texture",
            "material_adapter_loaded": learned and item["branch"] == "material"})
    def write():
        (tmp_path / "generation_status.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
    write()
    return tmp_path, expected, rows, write


@pytest.mark.parametrize("source_id", source.SOURCE_IDS)
def test_baseline_controls_match_prompts_and_all_48_selected_images_are_locked(old_rows, source_id):
    folder, expected, _, _ = old_rows
    before = {str(path): source.schema.sha256(path) for path in folder.rglob("*") if path.is_file()}
    locks = {}
    rows = source._baseline_rows(folder, expected, locks, source_id)
    assert len(rows) == 48
    assert {row["concept_id"] for row in rows} == {source_id + "_material"}
    assert len(locks) == 49
    assert all(Path(row["source_image"]).is_absolute() for row in rows)
    assert {str(path): source.schema.sha256(path) for path in folder.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("failure", ["image", "adapter", "texture", "prompt", "duplicate", "status"])
@pytest.mark.parametrize("source_id", source.SOURCE_IDS)
def test_reject_changed_baseline_images_flags_and_prompt(old_rows, failure, source_id):
    folder, expected, rows, write = old_rows
    row = next(row for row in rows if row["concept_id"] == source_id + "_material"
               and row["condition"] == "learned_material")
    if failure == "image":
        (folder / row["image_path"]).write_bytes(b"changed")
    elif failure == "adapter":
        row["material_adapter_loaded"] = False
    elif failure == "texture":
        row["texture_adapter_loaded"] = True
    elif failure == "prompt":
        row["prompt"] = "changed"
    elif failure == "duplicate":
        rows[-1] = rows[0]
    else:
        row["status"] = "running"
    write()
    with pytest.raises(ValueError):
        source._baseline_rows(folder, expected, {}, source_id)


@pytest.fixture
def completed_sources(tmp_path, monkeypatch):
    training, transfer = tmp_path / "training", tmp_path / "transfer"
    training.mkdir()
    transfer.mkdir()
    protocol = json.loads(source.schema.PARENT.read_text())
    prompts = json.loads(source.TRANSFER_PROMPTS.read_text())
    training_git, transfer_git = {"commit": "training"}, {"commit": "transfer"}

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf8")

    groups, configs, selections = {}, {}, []
    for source_id in source.SOURCE_IDS:
        concept_id = source_id + "_material"
        manifest = tmp_path / (concept_id + ".jsonl")
        images = [{"id": f"{concept_id}_{index}", "image": str(tmp_path / f"{concept_id}_{index}.png"),
                   "image_sha256": f"image-{index}", "prompt": f"{source_id} image {index}",
                   "roughness": 0.3, "metallic": 0.1} for index in range(8)]
        manifest.write_text("".join(json.dumps(row) + "\n" for row in images), encoding="utf8")
        groups[concept_id] = {"training_assets_manifest": str(manifest),
                              "source": {"id": source_id}, "images": images}
        for method in ("fullrank", "lora_r64"):
            run = training / f"{concept_id}_{method}"
            configs[run] = baseline(method, source_id)
            selections.append({"method": method, "source_id": source_id, "branch": "material",
                               "concept_id": concept_id, "run": str(run), "checkpoint": str(run / "checkpoints")})
    monkeypatch.setattr(source, "read_config", lambda path: configs[path.parent])
    monkeypatch.setattr(source, "audit_training", lambda path: {"training_git": training_git, "source_locks": {}})
    write(training / "protocol.json", protocol)
    write(training / "training/training_index.json", {"runs": selections})
    write(training / "training_references.json", {"groups": groups})
    write(transfer / "status.json", {"status": "execution_verified", "generated_images": 864,
        "expected_images": 864, "source_run": str(training), "probe_suite": "ordinary_object_transfer",
        "training_git": training_git, "git": transfer_git,
        "prompt_protocol_sha256": source.schema.sha256(source.TRANSFER_PROMPTS)})
    write(transfer / "transfer_protocol.json", prompts)
    sampling = {**protocol["inference"], "conditions": {
        "texture": ["base", "literal", "learned_texture"],
        "material": ["base", "literal", "learned_material"]}}
    for step in source.STEPS:
        folder = transfer / f"step{step}"
        write(training / f"step{step}/training_index.json", {"runs": selections})
        write(folder / "summary.json", {"status": "succeeded", "generated_images": 288,
              "missing_images": 0, "training_steps": step})
        write(folder / "provenance.json", {"status": "succeeded", "source_run": str(training),
            "training_steps": step, "row_count": 288, "probe_suite": "ordinary_object_transfer",
            "git": transfer_git, "training_git": training_git, "sampling": sampling,
            "training": selections, "prompt_protocol": prompts,
            "prompt_protocol_sha256": source.schema.sha256(source.TRANSFER_PROMPTS),
            "protocol_sha256": source.schema.protocol_sha256(protocol)})
        refs = []
        for item in selections:
            group = groups[item["concept_id"]]
            manifest = Path(group["training_assets_manifest"])
            refs.append({**item, "step": step, "source": group["source"],
                "dataset_manifest": str(manifest), "dataset_manifest_sha256": source.schema.sha256(manifest),
                "images": group["images"], "conditions": {"base": None, "literal": None, "learned_material": "images"}})
        write(folder / "training_references.json", {"groups": refs})
        rows = []
        for item in source.build_rows(protocol, step, True):
            image = folder / item["image_path"]
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(item["id"].encode())
            learned = item["condition"].startswith("learned_")
            rows.append({**item, "status": "ok", "image_sha256": source.schema.sha256(image),
                "texture_adapter_loaded": learned and item["branch"] == "texture",
                "material_adapter_loaded": learned and item["branch"] == "material"})
        (folder / "generation_status.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
    return training, transfer


@pytest.mark.parametrize("source_id", source.SOURCE_IDS)
def test_prepare_selects_only_requested_source_configs_images_and_old_controls(completed_sources, source_id):
    training, transfer = completed_sources
    result = source.prepare(training, transfer, source_id)
    assert result["source_id"] == source_id
    assert len(result["recipes"]) == 2
    assert {recipe["concept_id"] for recipe in result["recipes"]} == {source_id + "_material"}
    assert set(result["configs"]) == {"fullrank", "lora_r64"}
    for config in result["configs"].values():
        assert config["texture_material"]["source_id"] == source_id
        assert config["args"]["initializer_token"] == "metal"
    for step in source.STEPS:
        refs = result["references"][str(step)]
        rows = result["baseline_rows"][str(step)]
        assert set(refs) == {"fullrank", "lora_r64"}
        assert all(ref["source_id"] == source_id and len(ref["images"]) == 8 for ref in refs.values())
        assert len(rows) == 48
        assert {row["concept_id"] for row in rows} == {source_id + "_material"}
