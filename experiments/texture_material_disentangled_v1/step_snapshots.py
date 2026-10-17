"""Export paired T/M step snapshots without changing their training runs."""

import json
from pathlib import Path

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd15_token_local_kv_v1.snapshots import (
    EMBEDDING_KEY, _check_state, _new_output, _write_export, adapter_schema,
    checkpoint_plan,
)
from scripts.launch.colorpeel_run import read_config


def _verify(selection):
    method, branch = selection["method"], selection["branch"]
    token = {"texture": "<T*>", "material": "<M*>"}[branch]
    weight = {"fullrank": "pytorch_token_local_kv_weights.bin",
              "lora_r64": "pytorch_lora_kv_weights.bin"}[method]
    checkpoint = Path(selection["checkpoint"]).resolve()
    hashes = selection.get("hashes", {})
    required = [Path(selection["run"]) / name for name in ("config.yaml", "manifest.json")]
    required += [checkpoint / name for name in
                 (weight, token + ".bin", "adaptation_config.json", "embedding_update_audit.json")]
    if (selection.get("status") != "verified" or selection.get("token") != token
            or selection.get("weight") != weight or not hashes
            or any(str(path.resolve()) not in hashes for path in required)):
        raise ValueError("unverified or incomplete T/M training selection")
    for path, digest in hashes.items():
        if not Path(path).is_absolute() or sha256(Path(path)) != digest:
            raise ValueError(f"training artifact changed: {path}")
    adaptation = read_json(checkpoint / "adaptation_config.json")
    mode = "token_local_kv" if method == "fullrank" else (
        "lora_material_kv" if branch == "material" else "lora_subject_kv")
    if (adaptation.get("modifier_tokens") != [token]
            or adaptation.get("adaptation_mode") != mode
            or (method == "lora_r64" and any(adaptation.get(key) != value for key, value in
                {"rank": 64, "alpha": 64, "mode": "token_local_kv"}.items()))):
        raise ValueError("snapshot adapter token, mode or rank differs")
    return checkpoint


def _paired(torch, selection, step):
    checkpoint = _verify(selection)
    config = read_config(Path(selection["run"]) / "config.yaml")
    maximum = config["args"]["max_train_steps"]
    if (type(step) is not int or step not in checkpoint_plan(config["args"])
            or selection["steps"] != maximum):
        raise ValueError("requested snapshot is outside the audited training plan")
    shapes, empty = adapter_schema(selection["method"])
    directory = checkpoint / f"checkpoint-{step}"
    adapter_path, encoder_path = directory / "pytorch_model.bin", directory / "pytorch_model_1.bin"
    adapter = torch.load(adapter_path, map_location="cpu")
    _check_state(torch, adapter, shapes, {}, f"step {step}")
    token = selection["token"]
    audit = read_json(checkpoint / "embedding_update_audit.json")
    records = audit.get("modifier_tokens", [])
    if (len(records) != 1 or records[0].get("token") != token
            or audit.get("observed_optimization_steps") != maximum):
        raise ValueError("snapshot embedding audit token or steps differ")
    token_id = records[0].get("token_id")
    encoder = torch.load(encoder_path, map_location="cpu")
    table = encoder.get(EMBEDDING_KEY) if isinstance(encoder, dict) else None
    if (not isinstance(table, torch.Tensor) or len(table.shape) != 2 or table.shape[1] != 768
            or type(token_id) is not int or not 0 <= token_id < table.shape[0]):
        raise ValueError("snapshot embedding table or token ID differs")
    embedding = table[token_id].clone().contiguous()
    del encoder, table
    if not bool(torch.isfinite(embedding).all()):
        raise ValueError("snapshot token embedding is nonfinite")
    if step == maximum:
        final = torch.load(checkpoint / selection["weight"], map_location="cpu")
        _check_state(torch, final, shapes, empty, "final")
        token_state = torch.load(checkpoint / (token + ".bin"), map_location="cpu")
        if (not isinstance(token_state, dict) or set(token_state) != {token}
                or not torch.equal(embedding, token_state[token])
                or any(not torch.equal(adapter[key], final[key]) for key in shapes)):
            raise ValueError("final snapshot differs from the paired final export")
    record = {"source_adapter_sha256": sha256(adapter_path),
              "source_text_encoder_sha256": sha256(encoder_path),
              "source_snapshot_hashes": {str(path.resolve()): sha256(path)
                  for path in (adapter_path, encoder_path)}}
    return adapter, embedding, empty, record


def export_step(selection, step, output):
    """Derive an inference adapter and its matching token row in a new folder."""
    import torch
    output = _new_output(output, Path(selection["run"]).resolve())
    adapter, embedding, empty, record = _paired(torch, selection, step)
    _verify(selection)
    derived = _write_export(torch, adapter, embedding, empty, selection, step, record, output)
    derived.update(steps=step, training_steps=selection["steps"],
                   source_snapshot_hashes=record["source_snapshot_hashes"],
                   paired_embedding_finite=True, adapter_tensor_count=len(adapter),
                   final_snapshot_matches_export=step == selection["steps"])
    # Inference verifies both the derived payload and its original training locks.
    derived["hashes"] = {**selection["hashes"], **record["source_snapshot_hashes"], **derived["hashes"]}
    (output / "receipt.json").write_text(json.dumps(derived, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")
    return derived


def compare_baseline(selection, source_selection, step=300):
    """Require exact prefix equality against the preserved original pilot."""
    import torch
    keys = ("method", "branch", "concept_id", "token", "weight")
    if any(selection[key] != source_selection[key] for key in keys):
        raise ValueError("baseline comparison concept or adapter method differs")
    if source_selection["steps"] != step:
        raise ValueError("baseline must be the original pilot final step")
    adapter, embedding, _, record = _paired(torch, selection, step)
    original, original_embedding, _, original_record = _paired(torch, source_selection, step)
    equal = all(torch.equal(adapter[key], original[key]) for key in adapter)
    token_equal = torch.equal(embedding, original_embedding)
    _verify(selection)
    _verify(source_selection)
    result = {"status": "verified" if equal and token_equal else "mismatch", "step": step,
              "adapter_equal": equal, "token_equal": token_equal,
              "run": selection["run"], "baseline_run": source_selection["run"],
              "source_snapshot_hashes": record["source_snapshot_hashes"],
              "baseline_snapshot_hashes": original_record["source_snapshot_hashes"]}
    if not equal or not token_equal:
        raise ValueError(f"300-step training prefix differs from the original pilot: {result}")
    return result
