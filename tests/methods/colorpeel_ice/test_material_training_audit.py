import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml


REPO_ROOT = Path(__file__).parents[3]
SPEC = importlib.util.spec_from_file_location(
    "material_training_audit", REPO_ROOT / "src" / "train" / "training_audit.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
CONFIG_DIR = REPO_ROOT / "experiments" / "clevr_subject_color_material_3x3x2" / "configs"
TOKENS = ("<s1*>", "<s2*>", "<s3*>", "<c1*>", "<c2*>", "<c3*>", "<m1*>", "<m2*>")


class MaterialTrainingAuditTests(unittest.TestCase):
    def test_configs_lock_eight_tokens_and_exposure(self):
        base = yaml.safe_load((CONFIG_DIR / "material_base.yaml").read_text(encoding="utf-8"))
        self.assertEqual(tuple(base["args"]["modifier_token"].split("+")), TOKENS)
        self.assertEqual(base["args"]["initializer_token"],
                         "cube+sphere+cylinder+red+turquoise+gray+metal+rubber")
        smoke = yaml.safe_load((CONFIG_DIR / "smoke_18step.yaml").read_text(encoding="utf-8"))
        exposure = smoke["protocol"]["expected_exposure_counts"]
        self.assertEqual([exposure[token] for token in TOKENS[:6]], [6] * 6)
        self.assertEqual([exposure[token] for token in TOKENS[6:]], [9, 9])
        self.assertEqual(len(smoke["protocol"]["expected_modifier_token_pairs"]), 18)
        self.assertTrue(all(len(row) == 3 for row in smoke["protocol"]["expected_modifier_token_pairs"]))

    def test_eighteen_step_audit_requires_all_eight_artifacts(self):
        config_path = CONFIG_DIR / "smoke_18step.yaml"
        pairs = [[f"<s{s}*>", f"<c{c}*>", f"<m{m}*>"]
                 for s, c, m in itertools.product(range(1, 4), range(1, 4), range(1, 3))]
        exposure = {token: (6 if token[1] in {"s", "c"} else 9) for token in TOKENS}
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            with (run_dir / "training_metrics.jsonl").open("w", encoding="utf-8") as handle:
                for step, present in enumerate(pairs, start=1):
                    handle.write(json.dumps(AUDIT.build_training_metric(
                        step=step, reconstruction_loss=1.0, caa_loss=0.5, caa_weight=0.2,
                        total_loss=1.1, learning_rate=1e-5, present_modifier_tokens=present,
                    )) + "\n")
            (run_dir / "embedding_update_audit.json").write_text(json.dumps({
                "observed_optimization_steps": 18,
                "modifier_tokens": [{"token": token, "token_id": index, "exposure_steps": exposure[token],
                                     "nonzero_gradient_steps": 1, "initial_final_l2_delta": 0.1}
                                    for index, token in enumerate(TOKENS)],
                "non_modifier_embedding_drift": {"mean_l2_delta": 0.01, "max_l2_delta": 0.02,
                                                 "changed_rows": 2, "total_rows": 100, "enforced": False},
            }), encoding="utf-8")
            with patch.object(AUDIT, "_artifact_size", return_value=128):
                result = AUDIT.validate_smoke_audit(config_path, run_dir)
            with patch.object(AUDIT, "_artifact_size",
                              side_effect=lambda path: 0 if Path(path).name == "<m2*>.bin" else 128):
                missing = AUDIT.validate_smoke_audit(config_path, run_dir)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["checkpoint_artifacts"]), 9)
        self.assertEqual(missing["status"], "failed")
        self.assertTrue(any("<m2*>.bin" in error for error in missing["errors"]))

    def test_embedding_audit_is_token_count_agnostic(self):
        ids = {token: index + 1 for index, token in enumerate(TOKENS)}
        audit = AUDIT.EmbeddingUpdateAudit(ids, torch.zeros(20, 4))
        for s, c, m in itertools.product(range(3), range(3), range(2)):
            selected = (ids[TOKENS[s]], ids[TOKENS[3 + c]], ids[TOKENS[6 + m]])
            audit.observe_input_ids(torch.tensor([selected]))
            gradient = torch.zeros(20, 4)
            gradient[list(selected), 0] = 1.0
            audit.complete_optimization_step(gradient)
        self.assertEqual([audit.exposure_steps[token] for token in TOKENS[:6]], [6] * 6)
        self.assertEqual([audit.exposure_steps[token] for token in TOKENS[6:]], [9, 9])


if __name__ == "__main__":
    unittest.main()
