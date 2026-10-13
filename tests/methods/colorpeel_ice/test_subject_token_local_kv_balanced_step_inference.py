import copy
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
EVALUATOR_PATH = (
    ROOT / "experiments/subject_token_local_kv_balanced_step_ablation_v1/evaluate.py"
)
PROTOCOL_PATH = (
    ROOT
    / "experiments/subject_token_local_kv_balanced_step_ablation_v1/protocols/inference_v1.json"
)
SPEC = importlib.util.spec_from_file_location("balanced_subject_step_evaluate", EVALUATOR_PATH)
EVALUATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EVALUATOR)


def protocol():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


class BalancedSubjectStepInferenceTests(unittest.TestCase):
    def test_protocol_and_comparison_matrix_are_locked(self):
        value = protocol()
        EVALUATOR.validate_protocol(value)
        for step in EVALUATOR.SUBJECT_STEPS:
            rows = EVALUATOR.comparison_rows(value, step)
            self.assertEqual(len(rows), 36)
            self.assertEqual({row["subject_step"] for row in rows}, {step})
            self.assertEqual({row["group"] for row in rows}, {"red", "blue", "city_street"})
            self.assertEqual({row["condition"] for row in rows}, set(EVALUATOR.CONDITIONS))
            self.assertEqual({row["seed"] for row in rows}, {42, 43, 44})
            self.assertEqual(sum(row["material_step"] == 5000 for row in rows), 18)

    def test_comparison_prompts_match_original_protocol_byte_for_byte(self):
        value = protocol()
        source_path = ROOT / value["comparison"]["source_protocol"]
        self.assertEqual(
            EVALUATOR.sha256(source_path), value["comparison"]["source_protocol_sha256"])
        source = json.loads(source_path.read_text(encoding="utf-8"))
        self.assertEqual(value["comparison"]["groups"], source["groups"])
        self.assertEqual(value["comparison"]["sampling"], source["sampling"])

    def test_transfer_replay_preserves_all_140_source_rows(self):
        source = []
        for prompt_index in range(28):
            for seed in (42, 43, 44, 45, 46):
                source.append({
                    "id": f"source-{prompt_index}-{seed}",
                    "color": f"group-{prompt_index}",
                    "prompt": f"a photo of <S*> mailbox prompt {prompt_index}",
                    "seed": seed,
                    "num_inference_steps": 100,
                    "guidance_scale": 3.5,
                })
        rows = EVALUATOR.transfer_rows(source, 700)
        self.assertEqual(len(rows), 140)
        self.assertEqual([row["source_id"] for row in rows], [row["id"] for row in source])
        self.assertEqual([row["prompt"] for row in rows], [row["prompt"] for row in source])
        self.assertEqual([row["seed"] for row in rows], [row["seed"] for row in source])

    def test_protocol_rejects_material_or_sampling_drift(self):
        value = protocol()
        changed = copy.deepcopy(value)
        changed["material"]["snapshot_step"] = 1000
        with self.assertRaisesRegex(ValueError, "Material selection differs"):
            EVALUATOR.validate_protocol(changed)
        changed = copy.deepcopy(value)
        changed["comparison"]["sampling"]["seeds"].append(45)
        with self.assertRaisesRegex(ValueError, "comparison matrix differs"):
            EVALUATOR.validate_protocol(changed)


if __name__ == "__main__":
    unittest.main()
