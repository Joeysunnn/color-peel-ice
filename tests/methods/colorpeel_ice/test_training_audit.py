import contextlib
import importlib.util
import itertools
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml


REPO_ROOT = Path(__file__).parents[3]
MODULE_PATH = REPO_ROOT / "src" / "train" / "training_audit.py"
SPEC = importlib.util.spec_from_file_location("training_audit", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CONFIG_ROOT = REPO_ROOT / "experiments" / "clevr_subject_color_3x3" / "configs"


TOKENS = {
    "<s1*>": 10,
    "<s2*>": 11,
    "<s3*>": 12,
    "<c1*>": 20,
    "<c2*>": 21,
    "<c3*>": 22,
}


class LossMetricsTests(unittest.TestCase):
    def test_loss_decomposition_is_numerically_equivalent(self):
        reconstruction = 1.25
        caa = 0.4
        weight = 0.2
        total = reconstruction + caa * weight
        metric = MODULE.build_training_metric(
            step=1,
            reconstruction_loss=reconstruction,
            caa_loss=caa,
            caa_weight=weight,
            total_loss=total,
            learning_rate=1.0e-5,
        )
        self.assertAlmostEqual(
            metric["total_loss"],
            metric["reconstruction_loss"] + metric["caa_weighted_loss"],
        )
        self.assertEqual(
            metric["losses_finite"],
            {"reconstruction": True, "caa": True, "total": True, "all": True},
        )

    def test_jsonl_writer_emits_one_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training_metrics.jsonl"
            record = MODULE.build_training_metric(
                step=1,
                reconstruction_loss=1.0,
                caa_loss=0.5,
                caa_weight=0.2,
                total_loss=1.1,
                learning_rate=1.0e-5,
            )
            MODULE.append_jsonl(path, record)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), record)


class EmbeddingAuditTests(unittest.TestCase):
    def make_audit(self):
        return MODULE.EmbeddingUpdateAudit(TOKENS, torch.zeros(30, 4))

    @staticmethod
    def gradient_for(*token_ids):
        gradient = torch.zeros(30, 4)
        for token_id in token_ids:
            gradient[token_id, 0] = 1.0
        return gradient

    def test_two_step_coverage_counts_only_observed_tokens(self):
        audit = self.make_audit()
        audit.observe_input_ids(torch.tensor([[0, TOKENS["<s1*>"] , TOKENS["<c1*>"]]]))
        audit.complete_optimization_step(self.gradient_for(TOKENS["<s1*>"] , TOKENS["<c1*>"]))
        audit.observe_input_ids(torch.tensor([[0, TOKENS["<s1*>"] , TOKENS["<c2*>"]]]))
        audit.complete_optimization_step(self.gradient_for(TOKENS["<s1*>"] , TOKENS["<c2*>"]))

        self.assertEqual(
            audit.exposure_steps,
            {"<s1*>": 2, "<s2*>": 0, "<s3*>": 0, "<c1*>": 1, "<c2*>": 1, "<c3*>": 0},
        )
        self.assertEqual(audit.nonzero_gradient_steps, audit.exposure_steps)

    def test_nine_step_cartesian_grid_exposes_each_token_three_times(self):
        audit = self.make_audit()
        for subject_id, color_id in itertools.product((10, 11, 12), (20, 21, 22)):
            audit.observe_input_ids(torch.tensor([[subject_id, color_id]]))
            audit.complete_optimization_step(self.gradient_for(subject_id, color_id))
        self.assertEqual(set(audit.exposure_steps.values()), {3})
        self.assertEqual(set(audit.nonzero_gradient_steps.values()), {3})

    def test_finalize_reports_drift_without_enforcement(self):
        audit = self.make_audit()
        final = torch.zeros(30, 4)
        final[TOKENS["<s1*>"] , 0] = 2.0
        final[0, 0] = 0.5
        result = audit.finalize(final)
        token = next(item for item in result["modifier_tokens"] if item["token"] == "<s1*>")
        self.assertEqual(token["initial_final_l2_delta"], 2.0)
        self.assertEqual(result["non_modifier_embedding_drift"]["changed_rows"], 1)
        self.assertFalse(result["non_modifier_embedding_drift"]["enforced"])

    def test_literal_adamw_weight_decay_changes_zero_gradient_ordinary_row(self):
        embeddings = torch.nn.Parameter(
            torch.tensor(
                [
                    [1.0, -2.0],
                    [0.5, 0.25],
                    [0.0, 0.0],
                ]
            )
        )
        learning_rate = 1.0e-2
        weight_decay = 0.01
        optimizer = torch.optim.AdamW(
            [embeddings],
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
            eps=1.0e-8,
        )
        before = embeddings.detach().clone()
        embeddings.grad = torch.zeros_like(embeddings)
        embeddings.grad[1] = 1.0

        optimizer.step()

        expected_ordinary = before[0] * (1.0 - learning_rate * weight_decay)
        self.assertTrue(torch.allclose(embeddings.detach()[0], expected_ordinary))
        self.assertFalse(torch.equal(embeddings.detach()[0], before[0]))
        self.assertFalse(torch.equal(embeddings.detach()[1], before[1]))
        self.assertTrue(torch.equal(embeddings.detach()[2], before[2]))


class SmokeConfigTests(unittest.TestCase):
    def test_smoke_configs_lock_steps_and_exposure_expectations(self):
        baseline = yaml.safe_load((CONFIG_ROOT / "baseline.yaml").read_text(encoding="utf-8"))
        self.assertEqual(baseline["args"]["concepts_list"], "${COLORPEEL_CONCEPTS_LIST}")
        expected = {
            "smoke_2step.yaml": {
                "max_train_steps": 2,
                "exposure": {"<s1*>": 2, "<s2*>": 0, "<s3*>": 0, "<c1*>": 1, "<c2*>": 1, "<c3*>": 0},
            },
            "smoke_9step.yaml": {
                "max_train_steps": 9,
                "exposure": {token: 3 for token in TOKENS},
            },
            "smoke_turquoise_2step.yaml": {
                "max_train_steps": 2,
                "exposure": {"<s1*>": 2, "<s2*>": 0, "<s3*>": 0, "<c1*>": 1, "<c2*>": 1, "<c3*>": 0},
            },
            "smoke_turquoise_9step.yaml": {
                "max_train_steps": 9,
                "exposure": {token: 3 for token in TOKENS},
            },
        }
        for filename, wanted in expected.items():
            config = yaml.safe_load((CONFIG_ROOT / filename).read_text(encoding="utf-8"))
            self.assertEqual(config["args"]["max_train_steps"], wanted["max_train_steps"])
            self.assertEqual(config["args"]["adam_weight_decay"], 0.01)
            self.assertEqual(config["args"]["concepts_list"], "${COLORPEEL_CONCEPTS_LIST}")
            if "turquoise" in filename:
                self.assertEqual(
                    config["args"]["initializer_token"],
                    "cube+sphere+cylinder+red+turquoise+gray",
                )
                self.assertEqual(config["protocol"]["scientific_change"], "cyan_initializer_only")
            if "2step" in filename:
                self.assertEqual(config["protocol"]["expected_exposure_counts"], wanted["exposure"])
            else:
                self.assertEqual(config["protocol"]["expected_exposure_per_modifier_token"], 3)
            pairs = config["protocol"]["expected_modifier_token_pairs"]
            self.assertEqual(len(pairs), wanted["max_train_steps"])
            self.assertEqual(len({tuple(pair) for pair in pairs}), wanted["max_train_steps"])

    def test_official_optimizer_scope_and_update_order_are_unchanged(self):
        source = (REPO_ROOT / "src" / "train" / "train_colorpeel.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "itertools.chain(text_encoder.get_input_embeddings().parameters(), custom_diffusion_layers.parameters())",
            source,
        )
        self.assertIn("weight_decay=args.adam_weight_decay", source)
        self.assertIn("loss += cos * args.cos_weight", source)
        ordered_operations = (
            "accelerator.backward(loss)",
            "grads_text_encoder.data[index_grads_to_zero, :]",
            "accelerator.clip_grad_norm_",
            "optimizer.step()",
            "lr_scheduler.step()",
            "optimizer.zero_grad",
        )
        positions = [source.index(operation) for operation in ordered_operations]
        self.assertEqual(positions, sorted(positions))

    def test_accelerate_0203_logging_uses_project_configuration(self):
        source = (REPO_ROOT / "src" / "train" / "train_colorpeel.py").read_text(
            encoding="utf-8"
        )
        project_config_start = source.index("accelerator_project_config = ProjectConfiguration(")
        accelerator_start = source.index("accelerator = Accelerator(", project_config_start)
        project_config_source = source[project_config_start:accelerator_start]
        accelerator_source = source[accelerator_start:source.index("\n\n", accelerator_start)]
        self.assertIn("logging_dir=logging_dir", project_config_source)
        self.assertNotIn("logging_dir=logging_dir", accelerator_source)


class SmokeAuditValidatorTests(unittest.TestCase):
    @staticmethod
    def fake_artifact_files(missing=None):
        return patch.object(
            MODULE,
            "_artifact_size",
            side_effect=lambda path: 0 if Path(path).name == missing else 128,
        )

    def write_evidence(self, run_dir, pairs, exposure, nonzero, delta, invalid_total=False):
        steps = len(pairs)
        metrics_path = run_dir / "training_metrics.jsonl"
        with metrics_path.open("w", encoding="utf-8") as handle:
            for step in range(1, steps + 1):
                metric = MODULE.build_training_metric(
                    step=step,
                    reconstruction_loss=1.0,
                    caa_loss=0.5,
                    caa_weight=0.2,
                    total_loss=1.1,
                    learning_rate=1.0e-5,
                    present_modifier_tokens=pairs[step - 1],
                )
                if invalid_total and step == steps:
                    metric["total_loss"] = math.nan
                handle.write(json.dumps(metric) + "\n")
        audit = {
            "observed_optimization_steps": steps,
            "modifier_tokens": [
                {
                    "token": token,
                    "token_id": token_id,
                    "exposure_steps": exposure[token],
                    "nonzero_gradient_steps": nonzero[token],
                    "initial_final_l2_delta": delta[token],
                }
                for token, token_id in TOKENS.items()
            ],
            "non_modifier_embedding_drift": {
                "mean_l2_delta": 0.01,
                "max_l2_delta": 0.02,
                "changed_rows": 24,
                "total_rows": 24,
                "enforced": False,
            },
        }
        (run_dir / "embedding_update_audit.json").write_text(
            json.dumps(audit), encoding="utf-8"
        )

    def test_two_step_validates_seen_tokens_only(self):
        exposure = {"<s1*>": 2, "<s2*>": 0, "<s3*>": 0, "<c1*>": 1, "<c2*>": 1, "<c3*>": 0}
        nonzero = {token: int(count > 0) for token, count in exposure.items()}
        delta = {token: float(count > 0) for token, count in exposure.items()}
        pairs = [["<s1*>", "<c1*>"], ["<s1*>", "<c2*>"]]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            self.write_evidence(run_dir, pairs, exposure, nonzero, delta)
            with self.fake_artifact_files():
                result = MODULE.validate_smoke_audit(CONFIG_ROOT / "smoke_2step.yaml", run_dir)
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["non_modifier_drift_enforced"])

    def test_cli_accepts_launcher_run_root(self):
        exposure = {"<s1*>": 2, "<s2*>": 0, "<s3*>": 0, "<c1*>": 1, "<c2*>": 1, "<c3*>": 0}
        nonzero = {token: int(count > 0) for token, count in exposure.items()}
        delta = {token: float(count > 0) for token, count in exposure.items()}
        pairs = [["<s1*>", "<c1*>"], ["<s1*>", "<c2*>"]]
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            evidence_dir = run_root / "checkpoints"
            evidence_dir.mkdir()
            self.write_evidence(evidence_dir, pairs, exposure, nonzero, delta)
            with self.fake_artifact_files(), contextlib.redirect_stdout(io.StringIO()):
                return_code = MODULE.main(
                    [
                        "validate",
                        "--config",
                        str(CONFIG_ROOT / "smoke_2step.yaml"),
                        "--run-dir",
                        str(run_root),
                    ]
                )
        self.assertEqual(return_code, 0)

    def test_nine_step_requires_three_exposures_and_positive_updates(self):
        exposure = {token: 3 for token in TOKENS}
        nonzero = {token: 1 for token in TOKENS}
        delta = {token: 0.5 for token in TOKENS}
        pairs = [
            [f"<s{subject}*>", f"<c{color}*>"]
            for subject, color in itertools.product(range(1, 4), range(1, 4))
        ]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            self.write_evidence(run_dir, pairs, exposure, nonzero, delta)
            with self.fake_artifact_files():
                result = MODULE.validate_smoke_audit(CONFIG_ROOT / "smoke_9step.yaml", run_dir)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(all(item["nonempty"] for item in result["checkpoint_artifacts"]))

    def test_missing_token_checkpoint_fails(self):
        exposure = {token: 3 for token in TOKENS}
        nonzero = {token: 1 for token in TOKENS}
        delta = {token: 0.5 for token in TOKENS}
        pairs = [
            [f"<s{subject}*>", f"<c{color}*>"]
            for subject, color in itertools.product(range(1, 4), range(1, 4))
        ]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            self.write_evidence(run_dir, pairs, exposure, nonzero, delta)
            with self.fake_artifact_files(missing="<c3*>.bin"):
                result = MODULE.validate_smoke_audit(CONFIG_ROOT / "smoke_9step.yaml", run_dir)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("<c3*>.bin" in error for error in result["errors"]))

    def test_nonfinite_loss_fails_without_enforcing_nonmodifier_drift(self):
        exposure = {token: 3 for token in TOKENS}
        nonzero = {token: 1 for token in TOKENS}
        delta = {token: 0.5 for token in TOKENS}
        pairs = [
            [f"<s{subject}*>", f"<c{color}*>"]
            for subject, color in itertools.product(range(1, 4), range(1, 4))
        ]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            self.write_evidence(run_dir, pairs, exposure, nonzero, delta, invalid_total=True)
            with self.fake_artifact_files():
                result = MODULE.validate_smoke_audit(CONFIG_ROOT / "smoke_9step.yaml", run_dir)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("total_loss" in error for error in result["errors"]))
        self.assertFalse(result["non_modifier_drift_enforced"])


if __name__ == "__main__":
    unittest.main()
