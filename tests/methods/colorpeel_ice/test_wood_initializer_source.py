"""The initializer ablation changes one argument and reads immutable old controls."""

import copy
import json
from pathlib import Path

import pytest

from experiments.natural_material_initializer_v1 import source
from experiments.natural_texture_material_v1.training_config import build_config, recipes


def baseline(method):
    protocol = json.loads(source.schema.PARENT.read_text())
    recipe = next(row for row in recipes() if row["concept_id"] == "wood_spoon_material"
                  and row["method"] == method)
    return build_config(recipe, Path("staging"), protocol)


@pytest.mark.parametrize("method", ["fullrank", "lora_r64"])
def test_initializer_is_the_only_training_argument_changed_and_original_is_untouched(method):
    old = baseline(method)
    before = copy.deepcopy(old)
    new = source.build_config(old, method)
    assert old == before
    assert {key for key in old["args"] if old["args"][key] != new["args"][key]} == {"initializer_token"}
    assert new["args"]["initializer_token"] == "material"
    assert new["run"]["study"] == "natural_material_initializer_v1"
    assert new["run"]["seed"] == old["run"]["seed"] == 42
    assert new["data_manifest"] == old["data_manifest"]
    assert new["texture_material"] == old["texture_material"]


@pytest.mark.parametrize("key,value", [("initializer_token", "wood"), ("modifier_token", "<T*>"),
    ("max_train_steps", 750), ("checkpoint_steps", [500, 1000]), ("token_local_kv", False)])
def test_reject_changed_original_fullrank_recipe(key, value):
    old = baseline("fullrank")
    old["args"][key] = value
    with pytest.raises(ValueError, match="original wood Material"):
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


def test_baseline_controls_match_prompts_and_all_48_selected_images_are_locked(old_rows):
    folder, expected, _, _ = old_rows
    before = {str(path): source.schema.sha256(path) for path in folder.rglob("*") if path.is_file()}
    locks = {}
    rows = source._baseline_rows(folder, expected, locks)
    assert len(rows) == 48
    assert {row["concept_id"] for row in rows} == {"wood_spoon_material"}
    assert len(locks) == 49
    assert all(Path(row["source_image"]).is_absolute() for row in rows)
    assert {str(path): source.schema.sha256(path) for path in folder.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("failure", ["image", "adapter", "texture", "prompt", "duplicate", "status"])
def test_reject_changed_baseline_images_flags_and_prompt(old_rows, failure):
    folder, expected, rows, write = old_rows
    row = next(row for row in rows if row["concept_id"] == "wood_spoon_material"
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
        source._baseline_rows(folder, expected, {})
