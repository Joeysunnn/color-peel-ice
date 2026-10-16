"""Audit completed SD1.5 training and derive paired, read-only-source snapshots."""

import json
import math
from pathlib import Path
import shutil

from experiments.sd15_token_local_kv_v1.contract import build_config, read_json, sha256
from experiments.sd15_token_local_kv_v1.evaluate import verify_run
from scripts.launch.colorpeel_run import read_config


EMBEDDING_KEY = "text_model.embeddings.token_embedding.weight"
LOSS_FIELDS = ("reconstruction_loss", "caa_loss", "caa_weighted_loss", "total_loss",
               "learning_rate")
STATE_FILES = ("pytorch_model.bin", "pytorch_model_1.bin", "optimizer.bin",
               "scheduler.bin", "random_states_0.pkl")


def checkpoint_plan(args):
    steps = args.get("checkpoint_steps")
    if steps is None:
        steps = list(range(args["checkpointing_steps"], args["max_train_steps"] + 1,
                           args["checkpointing_steps"]))
    if (not steps or any(type(step) is not int or step <= 0 for step in steps)
            or list(steps) != sorted(set(steps)) or steps[-1] != args["max_train_steps"]):
        raise ValueError("checkpoint plan must end at the final training step")
    return list(steps)


def adapter_schema(method):
    """The 16 SD1 cross-attention processors, with their frozen K/V dimensions."""
    processors = {}
    for block, hidden in ((0, 320), (1, 640), (2, 1280)):
        for attention in range(2):
            processors[f"down_blocks.{block}.attentions.{attention}.transformer_blocks.0"] = hidden
    processors["mid_block.attentions.0.transformer_blocks.0"] = 1280
    for block, hidden in ((1, 1280), (2, 640), (3, 320)):
        for attention in range(3):
            processors[f"up_blocks.{block}.attentions.{attention}.transformer_blocks.0"] = hidden
    shapes = {}
    for name, hidden in processors.items():
        for projection in ("k", "v"):
            prefix = name + ".attn2.processor."
            if method == "fullrank":
                shapes[prefix + f"delta_{projection}.weight"] = (hidden, 768)
            elif method == "lora_r64":
                shapes[prefix + f"to_{projection}_lora.down.weight"] = (64, 768)
                shapes[prefix + f"to_{projection}_lora.up.weight"] = (hidden, 64)
            else:
                raise ValueError("unknown adapter method")
    empty = {name + ".attn1.processor": {} for name in processors} if method == "fullrank" else {}
    return shapes, empty


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _metrics(path, maximum, token):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if ([row.get("step") for row in rows] != list(range(1, maximum + 1))
            or any(type(row.get("step")) is not int for row in rows)):
        raise ValueError("training metrics must contain every step exactly once in order")
    for row in rows:
        flags = row.get("losses_finite", {})
        if (not flags or any(value is not True for value in flags.values())
                or not all(_finite_number(row.get(field)) for field in LOSS_FIELDS)
                or any(not _finite_number(value) for field, value in row.items()
                       if "loss" in field and field != "losses_finite")
                or row.get("present_modifier_tokens") != [token]):
            raise ValueError(f"nonfinite loss or token exposure differs at step {row['step']}")
    return {"count": len(rows), "first_step": 1, "last_step": maximum,
            "losses_finite": True, "token_present_every_step": True,
            "total_loss": {"first": rows[0]["total_loss"], "last": rows[-1]["total_loss"],
                           "minimum": min(row["total_loss"] for row in rows),
                           "maximum": max(row["total_loss"] for row in rows)},
            "sha256": sha256(path)}


def _embedding_audit(path, maximum, token):
    audit = read_json(path)
    records = audit.get("modifier_tokens", [])
    if len(records) != 1 or records[0].get("token") != token:
        raise ValueError("embedding audit modifier token differs")
    record = records[0]
    token_id = record.get("token_id")
    gradients, delta = record.get("nonzero_gradient_steps"), record.get("initial_final_l2_delta")
    if (type(token_id) is not int or token_id < 0
            or audit.get("observed_optimization_steps") != maximum
            or record.get("exposure_steps") != maximum
            or type(gradients) is not int or not 0 < gradients <= maximum
            or not _finite_number(delta) or delta <= 0):
        raise ValueError("embedding audit has incomplete steps, gradients or token updates")
    return token_id, {"observed_optimization_steps": maximum, "modifier": record,
                      "nonzero_gradient_observed": True, "modifier_updated": True,
                      "non_modifier_embedding_drift": audit.get("non_modifier_embedding_drift"),
                      "optimizer": audit.get("optimizer"), "sha256": sha256(path)}


def _check_state(torch, state, shapes, empty, label):
    if not isinstance(state, dict) or set(state) != set(shapes) | set(empty):
        raise ValueError(f"{label} adapter keys differ")
    for key, shape in shapes.items():
        tensor = state[key]
        if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                or not bool(torch.isfinite(tensor).all())):
            raise ValueError(f"{label} adapter shape or finiteness differs: {key}")
    if any(not isinstance(state[key], dict) or state[key] for key in empty):
        raise ValueError(f"{label} self-attention entries must be empty dictionaries")


def _new_output(output, run):
    output = Path(output).resolve()
    if output == run or output.is_relative_to(run):
        raise ValueError("snapshot output must be outside the training run")
    if output.exists():
        raise FileExistsError(output)
    return output


def _write_export(torch, adapter, embedding, empty, selection, step, record, output):
    output.mkdir(parents=True)
    torch.save({**adapter, **empty}, output / selection["weight"])
    torch.save({selection["token"]: embedding}, output / (selection["token"] + ".bin"))
    shutil.copyfile(Path(selection["checkpoint"]) / "adaptation_config.json",
                    output / "adaptation_config.json")
    hashes = {str(path): sha256(path) for path in (
        output / selection["weight"], output / (selection["token"] + ".bin"),
        output / "adaptation_config.json")}
    derived = {**selection, "checkpoint": str(output), "path": str(output),
               "step": step, "source_checkpoint": selection["checkpoint"],
               "source_hashes": selection["hashes"], "hashes": hashes,
               "source_adapter_sha256": record["source_adapter_sha256"],
               "source_text_encoder_sha256": record["source_text_encoder_sha256"],
               "derived_adapter_sha256": hashes[str(output / selection["weight"])],
               "derived_token_sha256": hashes[str(output / (selection["token"] + ".bin"))]}
    (output / "receipt.json").write_text(json.dumps(derived, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
    return derived


def _audit_training(recipe, run, export_root=None, requested_step=None, output=None):
    import torch

    run = Path(run).resolve()
    frozen = build_config(recipe)
    token = frozen["args"]["modifier_token"]
    selection = verify_run(run, token)
    config = read_config(run / "config.yaml")
    if config != frozen or selection["smoke"]:
        raise ValueError("expected the requested complete, frozen training recipe")
    final = Path(selection["checkpoint"])
    steps = checkpoint_plan(config["args"])
    if requested_step is not None and (type(requested_step) is not int or requested_step not in steps):
        raise ValueError("requested snapshot step is outside the checkpoint plan")
    if export_root is not None:
        export_root = _new_output(export_root, run)
    if output is not None:
        output = _new_output(output, run)
    metrics = _metrics(final / "training_metrics.jsonl", steps[-1], token)
    token_id, embedding_audit = _embedding_audit(final / "embedding_update_audit.json", steps[-1], token)
    shapes, empty = adapter_schema(selection["method"])
    final_adapter = torch.load(final / selection["weight"], map_location="cpu")
    _check_state(torch, final_adapter, shapes, empty, "final")
    token_state = torch.load(final / (token + ".bin"), map_location="cpu")
    if not isinstance(token_state, dict) or set(token_state) != {token}:
        raise ValueError("final embedding export token differs")
    final_embedding = token_state[token]
    if (not isinstance(final_embedding, torch.Tensor) or tuple(final_embedding.shape) != (768,)
            or not bool(torch.isfinite(final_embedding).all())):
        raise ValueError("final token embedding shape or finiteness differs")
    actual_steps = {path.name for path in final.glob("checkpoint-*") if path.is_dir()}
    if actual_steps != {f"checkpoint-{step}" for step in steps}:
        raise ValueError("saved checkpoint directories differ from the frozen plan")
    result = {**selection, "recipe": recipe, "token_id": token_id,
              "metrics": metrics, "embedding_audit": embedding_audit,
              "checkpoint_steps": steps, "checkpoints": {}, "final_snapshot_matches_export": False}
    for step in steps:
        snapshot = final / f"checkpoint-{step}"
        for name in STATE_FILES:
            path = snapshot / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing or empty training state: {path}")
        adapter_path, encoder_path = snapshot / STATE_FILES[0], snapshot / STATE_FILES[1]
        adapter = torch.load(adapter_path, map_location="cpu")
        _check_state(torch, adapter, shapes, {}, f"step {step}")
        encoder = torch.load(encoder_path, map_location="cpu")
        table = encoder.get(EMBEDDING_KEY) if isinstance(encoder, dict) else None
        if (not isinstance(table, torch.Tensor) or len(table.shape) != 2
                or table.shape[1] != 768 or token_id >= table.shape[0]):
            raise ValueError(f"snapshot token ID or embedding table differs at step {step}")
        embedding = table[token_id].clone().contiguous()
        del table, encoder
        if not bool(torch.isfinite(embedding).all()):
            raise ValueError(f"nonfinite snapshot token embedding at step {step}")
        if step == steps[-1]:
            if (not torch.equal(embedding, final_embedding)
                    or any(not torch.equal(adapter[key], final_adapter[key]) for key in shapes)):
                raise ValueError("final snapshot tensors differ from the final exports")
            result["final_snapshot_matches_export"] = True
        record = {"step": step, "tensor_count": len(shapes), "paired_embedding_finite": True,
                  "source_adapter_sha256": sha256(adapter_path),
                  "source_text_encoder_sha256": sha256(encoder_path),
                  "state_files": list(STATE_FILES)}
        target = export_root / f"step{step}" if export_root is not None else (
            output if step == requested_step else None)
        if target is not None:
            record.update(_write_export(torch, adapter, embedding, empty, selection, step, record, target))
        result["checkpoints"][str(step)] = record
        del adapter, embedding
    # Keep the source immutable, including final exports and receipt/config locks.
    if any(sha256(Path(path)) != digest for path, digest in selection["hashes"].items()):
        raise ValueError("final training artifacts changed during audit")
    result["status"] = "verified"
    return result


def audit_training(recipe, run, export_root=None):
    """Audit every planned checkpoint once; optionally export each into a new root."""
    return _audit_training(recipe, run, export_root=export_root)


def export_snapshot(recipe, run, step, output):
    """Audit the complete run and export one paired snapshot to a new directory."""
    result = _audit_training(recipe, run, requested_step=step, output=output)
    return result["checkpoints"][str(step)]
