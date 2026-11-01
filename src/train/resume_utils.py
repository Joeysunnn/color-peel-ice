"""Resume evidence for completed, fixed-order single-process training runs."""

import hashlib
import json
import math
from pathlib import Path


STATE_FILES = (
    "pytorch_model.bin", "pytorch_model_1.bin", "optimizer.bin",
    "scheduler.bin", "random_states_0.pkl", "custom_checkpoint_0.pkl",
)
METRIC_FIELDS = (
    "reconstruction_loss", "caa_loss", "caa_weighted_loss", "total_loss", "learning_rate",
)


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_resume(checkpoint, output_dir, token_ids, batches_per_epoch, maximum, *,
                   num_processes, batch_size, accumulation, scheduler, warmup):
    """Validate the final parent evidence before loading or writing any state."""
    if (num_processes != 1 or batch_size != 1 or accumulation != 1
            or scheduler != "constant" or warmup != 0 or len(token_ids) != 1):
        raise ValueError("resume requires single-process batch1 accumulation1 constant LR, no warmup, one token")
    checkpoint, output = Path(checkpoint).resolve(), Path(output_dir).resolve()
    try:
        step = int(checkpoint.name.removeprefix("checkpoint-"))
    except ValueError as exc:
        raise ValueError("resume requires an explicit checkpoint-N directory") from exc
    if checkpoint.name != f"checkpoint-{step}" or not 0 < step < maximum:
        raise ValueError("resume step must be positive and below the new training limit")
    if batches_per_epoch <= 0:
        raise ValueError("resume requires a nonempty fixed-order dataloader")
    parent = checkpoint.parent
    if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError("resume output must be separate from the immutable parent training directory")
    if ((output / "training_metrics.jsonl").exists()
            or (output / "resume_receipt.json").exists()
            or list(output.glob("checkpoint-*"))):
        raise ValueError("resume output already contains training evidence")
    files = [checkpoint / name for name in STATE_FILES]
    metrics_path = parent / "training_metrics.jsonl"
    audit_path = parent / "embedding_update_audit.json"
    files += [metrics_path, audit_path]
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing resume state/evidence: {path}")
    metrics_text = metrics_path.read_text(encoding="utf-8")
    metrics = [json.loads(line) for line in metrics_text.splitlines() if line.strip()]
    if ([row.get("step") for row in metrics] != list(range(1, step + 1))
            or any(type(row.get("step")) is not int for row in metrics)):
        raise ValueError("resume checkpoint must be the parent's completed final step, with continuous metrics")
    tokens = list(token_ids)
    for row in metrics:
        flags = row.get("losses_finite", {})
        if (row.get("present_modifier_tokens") != tokens or not flags
                or any(value is not True for value in flags.values())
                or any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key])
                       for key in METRIC_FIELDS)):
            raise ValueError("parent metrics contain invalid losses or token exposure")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    records = audit.get("modifier_tokens", [])
    if audit.get("observed_optimization_steps") != step or len(records) != 1:
        raise ValueError("parent embedding audit does not match final checkpoint")
    record = records[0]
    token = tokens[0]
    if (record.get("token") != token or record.get("token_id") != token_ids[token]
            or record.get("exposure_steps") != step
            or type(record.get("nonzero_gradient_steps")) is not int
            or not 0 < record["nonzero_gradient_steps"] <= step
            or type(record.get("initial_final_l2_delta")) not in (int, float)
            or not math.isfinite(record["initial_final_l2_delta"])
            or record["initial_final_l2_delta"] <= 0):
        raise ValueError("parent embedding audit token/update counters are inconsistent")
    if audit.get("non_modifier_embedding_drift", {}).get("changed_rows") != 0:
        raise ValueError("parent non-modifier embeddings drifted")
    return {
        "checkpoint": str(checkpoint), "global_step": step,
        "first_epoch": step // batches_per_epoch, "skip_batches": step % batches_per_epoch,
        "batches_per_epoch": batches_per_epoch, "maximum": maximum,
        "metrics_text": metrics_text, "audit": audit,
        "source_sha256": {str(path): _hash(path) for path in files},
    }


def restore_resume_audit(plan, audit, restored_embeddings, output_dir):
    """Check loaded embedding identity against the parent, then extend its audit."""
    restored = audit.finalize(restored_embeddings)
    expected = plan["audit"]["modifier_tokens"][0]
    actual = restored["modifier_tokens"][0]
    if (not math.isclose(actual["initial_final_l2_delta"], expected["initial_final_l2_delta"],
                         rel_tol=1e-6, abs_tol=1e-8)
            or restored["non_modifier_embedding_drift"]["changed_rows"] != 0):
        raise ValueError("restored embeddings or original initialization differ from the parent audit")
    token = expected["token"]
    audit.exposure_steps[token] = expected["exposure_steps"]
    audit.nonzero_gradient_steps[token] = expected["nonzero_gradient_steps"]
    audit.observed_optimization_steps = plan["global_step"]
    output = Path(output_dir)
    prefix = plan["metrics_text"]
    (output / "training_metrics.jsonl").write_text(prefix.rstrip("\n") + "\n", encoding="utf-8")
    receipt = {key: value for key, value in plan.items() if key not in ("metrics_text", "audit")}
    receipt.update({
        "first_new_step": plan["global_step"] + 1,
        "state_restore": "accelerator.load_state: models, optimizer, scheduler, RNG, custom state",
        "audit_baseline": "original initialization before state restore",
        "data_order": "fixed order; skip_batches applies only to the first resumed epoch",
        "rng_scope": "accelerator RNG restored; dataloader worker RNG is not checkpointed; no bitwise guarantee",
    })
    (output / "resume_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def validate_loaded_state(plan, optimizer, scheduler):
    """Reject a weights-only restore or a mismatched AdamW/scheduler state."""
    step = plan["global_step"]
    counters = []
    for state in optimizer.state.values():
        value = state.get("step")
        value = value.item() if hasattr(value, "item") else value
        if value != step or not {"exp_avg", "exp_avg_sq"}.issubset(state):
            raise ValueError("restored AdamW state does not match the parent final step")
        counters.append(value)
    if not counters:
        raise ValueError("restored AdamW optimizer state is empty")
    state = scheduler.state_dict()
    if state.get("last_epoch") != step or state.get("_step_count") != step + 1:
        raise ValueError("restored constant scheduler counters do not match the parent final step")
    last_row = json.loads([line for line in plan["metrics_text"].splitlines() if line.strip()][-1])
    if not math.isclose(scheduler.get_last_lr()[0], last_row["learning_rate"], rel_tol=1e-12):
        raise ValueError("restored scheduler LR differs from the parent final metric")
    plan["loaded_state"] = {
        "optimizer_state_count": len(counters), "optimizer_step": step,
        "scheduler_last_epoch": state["last_epoch"], "scheduler_step_count": state["_step_count"],
    }
