"""Observation-only metrics for ColorPeel training.

The helpers in this module never participate in loss construction, gradient
masking, optimizer configuration, or parameter updates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


LOSS_METRIC_FIELDS = (
    "reconstruction_loss",
    "caa_loss",
    "caa_weighted_loss",
    "total_loss",
    "learning_rate",
)
MODIFIER_TOKENS = ("<s1*>", "<s2*>", "<s3*>", "<c1*>", "<c2*>", "<c3*>")
CHECKPOINT_ARTIFACT_NAMES = (
    "pytorch_custom_diffusion_weights.bin",
    *(f"{token}.bin" for token in MODIFIER_TOKENS),
)


def build_training_metric(
    *,
    step,
    reconstruction_loss,
    caa_loss,
    caa_weight,
    total_loss,
    learning_rate,
    present_modifier_tokens=(),
):
    """Build one JSON-serializable observation of the existing training loss."""
    reconstruction_loss = float(reconstruction_loss)
    caa_loss = float(caa_loss)
    total_loss = float(total_loss)
    return {
        "step": int(step),
        "reconstruction_loss": reconstruction_loss,
        "caa_loss": caa_loss,
        "caa_weight": float(caa_weight),
        "caa_weighted_loss": caa_loss * float(caa_weight),
        "total_loss": total_loss,
        "learning_rate": float(learning_rate),
        "present_modifier_tokens": list(present_modifier_tokens),
        "losses_finite": {
            "reconstruction": math.isfinite(reconstruction_loss),
            "caa": math.isfinite(caa_loss),
            "total": math.isfinite(total_loss),
            "all": all(
                math.isfinite(value)
                for value in (reconstruction_loss, caa_loss, total_loss)
            ),
        },
    }


def append_jsonl(path, record):
    """Append one observation and flush it before returning."""
    path = Path(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class EmbeddingUpdateAudit:
    """Count token observations/gradients and summarize final embedding drift."""

    def __init__(self, token_ids_by_token, initial_embeddings):
        self.token_ids_by_token = {
            str(token): int(token_id) for token, token_id in token_ids_by_token.items()
        }
        if not self.token_ids_by_token:
            raise ValueError("at least one modifier token is required")
        if len(set(self.token_ids_by_token.values())) != len(self.token_ids_by_token):
            raise ValueError("modifier token IDs must be unique")

        self.initial_embeddings = initial_embeddings.detach().float().cpu().clone()
        vocab_size = self.initial_embeddings.shape[0]
        if min(self.token_ids_by_token.values()) < 0 or max(self.token_ids_by_token.values()) >= vocab_size:
            raise ValueError("modifier token ID is outside the embedding vocabulary")

        self.exposure_steps = {token: 0 for token in self.token_ids_by_token}
        self.nonzero_gradient_steps = {token: 0 for token in self.token_ids_by_token}
        self._pending_exposed_ids = set()
        self.observed_optimization_steps = 0

    def observe_input_ids(self, input_ids):
        """Accumulate modifier IDs exposed by microbatches in the current update."""
        observed_ids = {int(token_id) for token_id in input_ids.detach().cpu().reshape(-1).tolist()}
        modifier_ids = set(self.token_ids_by_token.values())
        self._pending_exposed_ids.update(observed_ids.intersection(modifier_ids))

    def complete_optimization_step(self, embedding_gradient):
        """Record the pending exposure and post-backward gradient observations."""
        present_tokens = []
        for token, token_id in self.token_ids_by_token.items():
            if token_id in self._pending_exposed_ids:
                self.exposure_steps[token] += 1
                present_tokens.append(token)
            if torch.count_nonzero(embedding_gradient[token_id].detach()).item() > 0:
                self.nonzero_gradient_steps[token] += 1
        self._pending_exposed_ids.clear()
        self.observed_optimization_steps += 1
        return present_tokens

    def finalize(self, final_embeddings):
        """Return drift evidence without enforcing any preservation threshold."""
        final_embeddings = final_embeddings.detach().float().cpu()
        if final_embeddings.shape != self.initial_embeddings.shape:
            raise ValueError("initial and final embedding tables must have the same shape")

        row_l2_delta = torch.linalg.vector_norm(final_embeddings - self.initial_embeddings, dim=1)
        modifier_ids = set(self.token_ids_by_token.values())
        non_modifier_mask = torch.ones(row_l2_delta.shape[0], dtype=torch.bool)
        non_modifier_mask[list(modifier_ids)] = False
        non_modifier_delta = row_l2_delta[non_modifier_mask]

        tokens = []
        for token, token_id in self.token_ids_by_token.items():
            tokens.append(
                {
                    "token": token,
                    "token_id": token_id,
                    "exposure_steps": self.exposure_steps[token],
                    "nonzero_gradient_steps": self.nonzero_gradient_steps[token],
                    "initial_final_l2_delta": float(row_l2_delta[token_id].item()),
                }
            )

        return {
            "schema_version": 1,
            "observed_optimization_steps": self.observed_optimization_steps,
            "modifier_tokens": tokens,
            "non_modifier_embedding_drift": {
                "mean_l2_delta": float(non_modifier_delta.mean().item()),
                "max_l2_delta": float(non_modifier_delta.max().item()),
                "changed_rows": int(torch.count_nonzero(non_modifier_delta).item()),
                "total_rows": int(non_modifier_delta.numel()),
                "enforced": False,
            },
        }


def _read_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def _artifact_size(path):
    path = Path(path)
    return path.stat().st_size if path.is_file() else 0


def validate_smoke_audit(config_path, run_dir):
    """Validate smoke evidence while treating non-modifier drift as report-only."""
    import yaml

    config_path = Path(config_path)
    run_dir = Path(run_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    evidence_dir = run_dir
    if not (evidence_dir / "training_metrics.jsonl").is_file():
        evidence_dir = run_dir / "checkpoints"
    metrics = _read_jsonl(evidence_dir / "training_metrics.jsonl")
    embedding_audit = json.loads(
        (evidence_dir / "embedding_update_audit.json").read_text(encoding="utf-8")
    )

    errors = []
    expected_steps = int(config["args"]["max_train_steps"])
    if len(metrics) != expected_steps:
        errors.append(f"metrics rows: expected {expected_steps}, got {len(metrics)}")
    observed_steps = int(embedding_audit.get("observed_optimization_steps", -1))
    if observed_steps != expected_steps:
        errors.append(
            f"embedding audit steps: expected {expected_steps}, got {observed_steps}"
        )

    for expected_step, metric in enumerate(metrics, start=1):
        if metric.get("step") != expected_step:
            errors.append(
                f"metrics step sequence: expected {expected_step}, got {metric.get('step')}"
            )
        for field in LOSS_METRIC_FIELDS:
            value = metric.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                errors.append(f"step {expected_step} field {field} is not finite: {value}")

    protocol = config.get("protocol", {})
    if "expected_exposure_counts" in protocol:
        expected_exposure = {
            str(token): int(count)
            for token, count in protocol["expected_exposure_counts"].items()
        }
    elif "expected_exposure_per_modifier_token" in protocol:
        count = int(protocol["expected_exposure_per_modifier_token"])
        expected_exposure = {
            token: count
            for token in ("<s1*>", "<s2*>", "<s3*>", "<c1*>", "<c2*>", "<c3*>")
        }
    else:
        raise ValueError(f"smoke exposure expectation is missing from {config_path}")

    expected_pairs = protocol.get("expected_modifier_token_pairs")
    if expected_pairs is None:
        raise ValueError(f"smoke modifier-token pair expectation is missing from {config_path}")
    actual_pairs = [metric.get("present_modifier_tokens") for metric in metrics]
    if actual_pairs != expected_pairs:
        errors.append(
            f"modifier token pair sequence: expected {expected_pairs}, got {actual_pairs}"
        )

    checkpoint_artifacts = []
    for name in CHECKPOINT_ARTIFACT_NAMES:
        path = evidence_dir / name
        size_bytes = _artifact_size(path)
        checkpoint_artifacts.append(
            {
                "name": name,
                "path": str(path),
                "size_bytes": size_bytes,
                "nonempty": size_bytes > 0,
            }
        )
        if size_bytes <= 0:
            errors.append(f"checkpoint artifact is missing or empty: {name}")

    token_evidence = {
        item.get("token"): item for item in embedding_audit.get("modifier_tokens", [])
    }
    for token, expected_count in expected_exposure.items():
        evidence = token_evidence.get(token)
        if evidence is None:
            errors.append(f"modifier token evidence is missing: {token}")
            continue
        actual_count = evidence.get("exposure_steps")
        if actual_count != expected_count:
            errors.append(
                f"{token} exposure_steps: expected {expected_count}, got {actual_count}"
            )
        # The two-step smoke intentionally makes no gradient/delta claim for
        # unseen tokens. Every token in the nine-step smoke has count > 0.
        if expected_count > 0:
            if evidence.get("nonzero_gradient_steps", 0) < 1:
                errors.append(f"{token} has no nonzero-gradient step")
            delta = evidence.get("initial_final_l2_delta")
            if not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta <= 0:
                errors.append(f"{token} initial-final delta is not positive and finite: {delta}")

    return {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "config": str(config_path),
        "run_dir": str(run_dir),
        "evidence_dir": str(evidence_dir),
        "expected_steps": expected_steps,
        "metrics_rows": len(metrics),
        "observed_optimization_steps": observed_steps,
        "expected_exposure_steps": expected_exposure,
        "expected_modifier_token_pairs": expected_pairs,
        "observed_modifier_token_pairs": actual_pairs,
        "checkpoint_artifacts": checkpoint_artifacts,
        "non_modifier_embedding_drift": embedding_audit.get(
            "non_modifier_embedding_drift"
        ),
        "non_modifier_drift_enforced": False,
        "errors": errors,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Validate ColorPeel smoke audit outputs")
    parser.add_argument("validate", nargs="?", default="validate", choices=("validate",))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = validate_smoke_audit(args.config, args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
