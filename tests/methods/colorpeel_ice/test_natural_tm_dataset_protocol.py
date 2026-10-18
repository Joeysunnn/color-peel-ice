"""Natural-source integrity and the exact separated counterfactual factorial."""

import collections
import copy
import itertools
import json

import pytest

from experiments.natural_texture_material_v1 import dataset_protocol as module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


@pytest.fixture
def extraction(tmp_path):
    root = tmp_path / "extraction"
    root.mkdir()
    upstream = tmp_path / "original_supermat.json"
    write(upstream, {"status": "complete"})
    write(root / "source_locks.json", {str(upstream): module.sha256(upstream)})
    write(root / "protocol.json", {"phase": 2, "stage": "extraction_only",
        "phase1_protocol_sha256": module.PARENT_SHA256,
        "texture_extraction": {"patch_sampling": {"no_inpainting": True}}})
    (root / "overview.png").write_bytes(b"extraction overview")
    manifest = {"schema": "natural_texture_material/extraction_v2", "phase": 2,
        "status": "complete", "phase1_protocol_sha256": module.PARENT_SHA256, "materials": {},
        "source_locks_sha256": module.sha256(root / "source_locks.json"),
        "protocol_sha256": module.sha256(root / "protocol.json"),
        "overview_sha256": module.sha256(root / "overview.png")}
    for index, name in enumerate(module.SOURCES):
        folder = root / name
        folder.mkdir()
        material = {"roughness": (90 - index * 10) / 255, "metallic": (2 + index * 50) / 255,
                    "semantic_status": "accepted", "representation": "scalar roughness/metallic only"}
        for filename in ("input.png", "mask.png", "albedo.png", "roughness.png", "metallic.png",
                         "gray_full.png", "luminance.png", "texture_inpaint_audit.png", "texture.png"):
            (folder / filename).write_bytes((name + ":" + filename).encode())
        write(folder / "material.json", material)
        sampling = {"method": "maximum_area_all_valid_axis_aligned_rectangle", "no_inpainting": True}
        record = {**material, "texture_sampling": sampling, "texture_statistics": {"patch_sampling": sampling},
                  "files": {path.name: {"path": str(path), "sha256": module.sha256(path)}
                            for path in folder.iterdir()}}
        write(folder / "extraction.json", record)
        manifest["materials"][name] = record
    write(root / "manifest.json", manifest)
    return root


def test_exact_separate_factorials_and_shared_context_seeds(extraction):
    protocol = module.build_protocol(extraction)
    rows = module.records(protocol)
    assert module.records(json.loads(json.dumps(protocol, sort_keys=True))) == rows
    assert len(rows) == len({row["id"] for row in rows}) == 72
    assert collections.Counter(row["branch"] for row in rows) == {"texture": 48, "material": 24}
    for source in module.SOURCES:
        for branch, count in (("texture", 16), ("material", 8)):
            chosen = [row for row in rows if row["source_id"] == source and row["branch"] == branch]
            assert len(chosen) == count
            assert {row["concept_id"] for row in chosen} == {source + "_" + branch}
            assert {(row["shape"], row["color"], row["light"]) for row in chosen} == set(
                itertools.product(("sphere", "cube"), ("red", "blue"), ("soft_front", "side_directional")))
            for row in chosen:
                prompt = module.caption(row, protocol)
                if branch == "texture":
                    assert row["texture_id"] == source + "_texture"
                    assert row["texture_path"] == str(extraction / source / "texture.png")
                    assert row["material_id"] in {"smooth_dielectric", "rough_metal"}
                    assert "with <T*> texture made of" in prompt and "<M*>" not in prompt
                else:
                    assert not any("texture" in key for key in row)
                    assert row["material_id"] == source + "_material"
                    assert (row["albedo_mode"], row["albedo_level"]) == ("uniform", .5)
                    assert "made of <M*> under" in prompt and "<T*>" not in prompt
    contexts = collections.defaultdict(list)
    for row in rows:
        contexts[(row["branch"], row["shape"], row["color"], row["light"], row["material_id"]
                  if row["branch"] == "texture" else None)].append(row["seed"])
    assert all(len(seeds) == 3 and len(set(seeds)) == 1 for seeds in contexts.values())
    parent = json.loads(module.PARENT.read_text(encoding="utf8"))
    assert protocol["renderer"] == parent["renderer"]
    assert protocol["inference"] == parent["inference"]
    assert protocol["dataset"]["heldout_contexts"] == parent["dataset"]["heldout_contexts"]
    assert protocol["training"]["checkpoint_steps"] == [500, 750, 1000]
    assert protocol["sources"]["wood_spoon"]["input"] == str(extraction / "wood_spoon/input.png")


@pytest.mark.parametrize("relative", ["mailbox/texture.png", "mailbox/texture_inpaint_audit.png",
    "mailbox/roughness.png", "metal_spoon/material.json", "wood_spoon/extraction.json", "overview.png", "protocol.json"])
def test_reject_changed_extraction_outputs(extraction, relative):
    (extraction / relative).write_bytes(b"changed")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        module.build_protocol(extraction)


def test_reject_upstream_change(extraction):
    upstream = next(iter(json.loads((extraction / "source_locks.json").read_text()).keys()))
    from pathlib import Path
    Path(upstream).write_bytes(b"changed upstream")
    with pytest.raises(ValueError, match="Upstream extraction source changed"):
        module.build_protocol(extraction)


@pytest.mark.parametrize("change", ["inpainted", "scalar", "missing_source", "legacy"])
def test_reject_unaccepted_semantics(extraction, change):
    manifest = json.loads((extraction / "manifest.json").read_text())
    if change == "inpainted":
        manifest["materials"]["mailbox"]["texture_sampling"]["no_inpainting"] = False
    elif change == "scalar":
        manifest["materials"]["mailbox"]["roughness"] = .99
    elif change == "missing_source":
        del manifest["materials"]["wood_spoon"]
    else:
        manifest["schema"] = "natural_texture_material/extraction_v1"
    write(extraction / "manifest.json", manifest)
    with pytest.raises(ValueError):
        module.build_protocol(extraction)


def test_plan_preserves_sources_and_refuses_overwrites(extraction, tmp_path, monkeypatch):
    monkeypatch.setenv("COLORPEEL_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(module, "git_info", lambda: {"commit": "a" * 40, "branch": "codex/test"})
    protocol = module.build_protocol(extraction)
    before = copy.deepcopy(protocol["source_locks"])
    output = tmp_path / "new_plan"
    plan = module.make_plan(extraction, output)
    assert plan["training_launched"] is False and plan["eval_images"] == 0
    assert plan["protocol_path"] == str(output / "protocol.json")
    assert plan["protocol_sha256"] == module.protocol_sha256(protocol)
    assert plan["source_locks"] == before
    assert all(module.sha256(path) == digest for path, digest in before.items())
    with pytest.raises(FileExistsError):
        module.make_plan(extraction, output)
    with pytest.raises(ValueError, match="external output"):
        module.make_plan(extraction, extraction / "new_plan")


def test_plan_rejects_checkout_change_before_writing(extraction, tmp_path, monkeypatch):
    monkeypatch.setenv("COLORPEEL_OUTPUT_ROOT", str(tmp_path))
    states = iter(({"commit": "a"}, {"commit": "b"}))
    monkeypatch.setattr(module, "git_info", lambda: next(states))
    output = tmp_path / "new_plan"
    with pytest.raises(ValueError, match="Checkout or extraction inputs changed"):
        module.make_plan(extraction, output)
    assert not output.exists()
