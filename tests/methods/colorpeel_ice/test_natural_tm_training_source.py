"""Bind accepted natural training RGB/masks to their original renderer evidence."""

import copy
import json

from PIL import Image, ImageOps
import pytest

from experiments.natural_texture_material_v1 import dataset_stage, training_source as module
from experiments.natural_texture_material_v1 import dataset_protocol as schema
from experiments.texture_material_disentangled_v1.data import write_json
from tests.methods.colorpeel_ice.test_natural_tm_dataset_stage import rendered


@pytest.fixture
def accepted(rendered, tmp_path, monkeypatch):
    renderroot, manifest = rendered
    protocol = dataset_stage.read(tmp_path / "plan/protocol.json")
    protocol["source_extraction"] = str(tmp_path / "accepted_extraction")
    write_json(tmp_path / "plan/protocol.json", protocol)
    plan = dataset_stage.read(tmp_path / "plan/plan.json")
    plan.update(status="complete", training_launched=False, source_locks=protocol["source_locks"],
                protocol_path=str(tmp_path / "plan/protocol.json"), protocol_sha256=schema.protocol_sha256(protocol))
    write_json(tmp_path / "plan/plan.json", plan)
    manifest.update(protocol_sha256=schema.protocol_sha256(protocol),
                    plan_sha256=schema.sha256(tmp_path / "plan/plan.json"))
    write_json(renderroot / "manifest.json", manifest)
    audit = dataset_stage.stage(renderroot, tmp_path / "staging")
    state = {"status": "dataset_verified", "phase": 2, "stage": "counterfactual_dataset_only",
             "training_launched": False, "extraction": protocol["source_extraction"], "dataset": audit}
    write_json(tmp_path / "status.json", state)
    (tmp_path / "index.html").write_text("dataset index", encoding="utf8")
    monkeypatch.setattr(schema, "build_protocol", lambda extraction: copy.deepcopy(protocol))
    return tmp_path, protocol, audit


def refresh_ledger(root, rows):
    """Keep the staging receipt chain valid, so tests reach cross-stage binding."""
    folder = root / "staging/mailbox_texture"
    ledger = folder / "training_assets_manifest.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
    receipt_path = folder / "dataset_receipt.json"
    receipt = dataset_stage.read(receipt_path)
    receipt["training_assets_manifest_sha256"] = schema.sha256(ledger)
    write_json(receipt_path, receipt)
    audit = dataset_stage.read(root / "staging/audit.json")
    audit["datasets"]["mailbox_texture"].update(receipt)
    audit["datasets"]["mailbox_texture"]["dataset_receipt_sha256"] = schema.sha256(receipt_path)
    write_json(root / "staging/audit.json", audit)
    state = dataset_stage.read(root / "status.json")
    state["dataset"] = audit
    write_json(root / "status.json", state)


def test_verified_source_locks_every_actual_training_and_raw_asset_read_only(accepted):
    root, expected, expected_audit = accepted
    before = {str(path): schema.sha256(path) for path in root.rglob("*") if path.is_file()}
    protocol, audit, locks = module.verify_dataset(root)
    assert protocol == expected and audit == expected_audit
    assert tuple(audit[key] for key in ("train_images", "texture_images", "material_images")) == (72, 48, 24)
    for path in root.rglob("*"):
        if path.is_file() and path.name != "extraction.json":
            assert str(path) in locks
    assert {str(path): schema.sha256(path) for path in root.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("failure", ["already_started", "count", "protocol", "source_rgb", "mask", "shader"])
def test_reject_changed_source_and_cross_stage_substitution(accepted, failure):
    root, _, _ = accepted
    if failure in ("already_started", "count"):
        state = dataset_stage.read(root / "status.json")
        if failure == "already_started":
            state["training_launched"] = True
        else:
            state["dataset"]["train_images"] = 71
        write_json(root / "status.json", state)
    elif failure == "protocol":
        protocol = dataset_stage.read(root / "plan/protocol.json")
        protocol["training"]["learning_rate"] = 999
        write_json(root / "plan/protocol.json", protocol)
    elif failure == "shader":
        path = root / "renders/manifest.json"
        manifest = dataset_stage.read(path)
        next(row for row in manifest["records"] if row["branch"] == "material")["texture_image_nodes"] = 1
        write_json(path, manifest)
    else:
        ledger = root / "staging/mailbox_texture/training_assets_manifest.jsonl"
        rows = [json.loads(line) for line in ledger.read_text(encoding="utf8").splitlines()]
        if failure == "source_rgb":
            rows[0]["source_image"] = str(root / "wrong_image.png")
        else:
            with Image.open(rows[0]["mask"]) as image:
                ImageOps.invert(image.convert("L")).save(rows[0]["mask"])
            rows[0]["mask_sha256"] = schema.sha256(rows[0]["mask"])
        refresh_ledger(root, rows)
        # Staging itself remains valid: the training source must catch the substitution.
        dataset_stage.verify_staging(root / "staging")
    with pytest.raises(ValueError):
        module.verify_dataset(root)
