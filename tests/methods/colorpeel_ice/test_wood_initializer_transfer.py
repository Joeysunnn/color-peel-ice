"""CPU tensor integration for paired exports and immutable transfer controls."""

import json
import os
from pathlib import Path

from PIL import Image
import pytest

from experiments.natural_material_initializer_v1 import transfer


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("The actual Linux <M*>.bin checkpoint filename is invalid on Windows")
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    root = tmp_path / "new_experiment"
    root.mkdir()
    git = {"commit": "a" * 40, "branch": "codex/texture-material-v1"}
    protocol = json.loads(transfer.probe.schema.PARENT.read_text())
    old_image = tmp_path / "old_control.png"
    Image.new("RGB", (8, 8), "red").save(old_image)
    old_hash = transfer.probe.schema.sha256(old_image)
    baseline = [{**row, "source_image": str(old_image), "image_sha256": old_hash, "status": "ok",
                 "texture_adapter_loaded": False,
                 "material_adapter_loaded": row["condition"] == "learned_material"}
                for row in transfer.probe.build_rows(protocol, 500, True)
                if row["concept_id"] == "wood_spoon_material"]
    write(root / "plan.json", {"git": git, "protocol": protocol,
        "baseline_rows": {"500": baseline}, "source_transfer": str(tmp_path / "old_transfer")})
    write(root / "source_locks.json", {str(old_image): old_hash})
    write(root / "status.json", {"status": "generating", "git": git})
    parents, selections, paired = [], [], {}
    for method, value in (("fullrank", 0.2), ("lora_r64", 0.4)):
        run = root / "training" / method
        source_marker = run / "source.txt"
        source_marker.parent.mkdir(parents=True)
        source_marker.write_text(method)
        source_hashes = {str(source_marker): transfer.probe.schema.sha256(source_marker)}
        parent = {"method": method, "run": str(run), "checkpoint": str(run / "checkpoints"),
                  "hashes": source_hashes}
        parents.append(parent)
        shapes, empty = transfer.step_snapshots.adapter_schema(method)
        adapter = {key: torch.full(shape, value) for key, shape in shapes.items()}
        embedding = torch.arange(768, dtype=torch.float32) + value
        snapshot = run / "snapshot.txt"
        snapshot.write_text("paired " + method)
        snapshot_hashes = {str(snapshot): transfer.probe.schema.sha256(snapshot)}
        paired[method] = (adapter, embedding, empty, {"source_snapshot_hashes": snapshot_hashes})
        checkpoint = root / "exports/step500/checkpoints" / method
        checkpoint.mkdir(parents=True)
        weight = ("pytorch_token_local_kv_weights.bin" if method == "fullrank"
                  else "pytorch_lora_kv_weights.bin")
        torch.save({**adapter, **empty}, checkpoint / weight)
        torch.save({"<M*>": embedding}, checkpoint / "<M*>.bin")
        hashes = {**source_hashes, **snapshot_hashes,
            **{str(path): transfer.probe.schema.sha256(path) for path in checkpoint.iterdir()}}
        selection = {"method": method, "run": str(run), "checkpoint": str(checkpoint),
            "weight": weight, "steps": 500, "training_steps": 1000,
            "source_checkpoint": parent["checkpoint"], "source_hashes": source_hashes,
            "source_snapshot_hashes": snapshot_hashes, "hashes": hashes}
        selections.append(selection)
        write(checkpoint / "receipt.json", selection)
    write(root / "training/training_index.json", {"git": git, "runs": parents})
    index = {"git": git, "step": 500, "runs": selections, "references": {}}
    write(root / "exports/step500/index.json", index)
    calls = {"paired": [], "loads": [], "generation": []}

    def paired_snapshot(torch_module, parent, step):
        assert torch_module is torch and step == 500
        calls["paired"].append(parent["method"])
        return paired[parent["method"]]

    def load_pipeline(items, device):
        assert len(items) == 1 and device == "cpu"
        calls["loads"].append(items[0]["method"])
        return object()

    def generate(pipe, rows, output, sampling, device, learned):
        assert len(rows) == 8 and learned is True and device == "cpu"
        assert sampling == protocol["inference"]
        calls["generation"].append(rows[0]["method"])
        with (output / "generation_status.jsonl").open("a", encoding="utf8") as ledger:
            for row in rows:
                path = output / row["image_path"]
                Image.new("RGB", (8, 8), "blue").save(path)
                ledger.write(json.dumps({**row, "status": "ok", "texture_adapter_loaded": False,
                    "material_adapter_loaded": True,
                    "image_sha256": transfer.probe.schema.sha256(path)}) + "\n")

    def review(output, rows, references):
        assert len(rows) == 64 and references == {}
        (output / "contact_sheets").mkdir()
        paths = [f"contact_sheets/sheet_{i}.jpg" for i in range(4)]
        for path in paths:
            Image.new("RGB", (8, 8), "white").save(output / path)
        return paths

    monkeypatch.setattr(transfer, "git_info", lambda: git)
    monkeypatch.setattr(transfer.step_snapshots, "_paired", paired_snapshot)
    monkeypatch.setattr(transfer.probe, "load_pipeline", load_pipeline)
    monkeypatch.setattr(transfer.probe, "generate", generate)
    monkeypatch.setattr(transfer, "review_bundle", review)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return root, index, paired, calls, torch


def test_paired_cpu_exports_accept_fullrank_empty_self_attention_and_reuse_controls(experiment):
    root, _, paired, calls, _ = experiment
    assert paired["fullrank"][2]
    transfer.run(root, 500, device="cpu")
    summary = json.loads((root / "step500/summary.json").read_text())
    assert (summary["total_images"], summary["generated_images"], summary["reused_images"]) == (64, 16, 48)
    assert len(summary["contact_sheets"]) == 4
    assert calls == {"paired": ["fullrank", "lora_r64"], "loads": ["fullrank", "lora_r64"],
                     "generation": ["fullrank", "lora_r64"]}
    assert json.loads((root / "step500/provenance.json").read_text())["status"] == "succeeded"


def test_mismatched_token_tensor_is_rejected_before_pipeline_generation(experiment):
    root, index, paired, calls, torch = experiment
    selection = index["runs"][0]
    checkpoint = Path(selection["checkpoint"])
    token_path = checkpoint / "<M*>.bin"
    torch.save({"<M*>": paired["fullrank"][1] + 1}, token_path)
    selection["hashes"][str(token_path)] = transfer.probe.schema.sha256(token_path)
    write(checkpoint / "receipt.json", selection)
    write(root / "exports/step500/index.json", index)
    with pytest.raises(ValueError, match="paired adapter/token export"):
        transfer.run(root, 500, device="cpu")
    assert calls["loads"] == calls["generation"] == []
    assert not (root / "step500").exists()
