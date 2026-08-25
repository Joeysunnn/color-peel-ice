import importlib.util
from pathlib import Path
import unittest

from src.methods.colorpeel_ice.material_evaluation_protocol import build_manifest, read_protocol


REPO_ROOT = Path(__file__).parents[3]
PROTOCOL_PATH = (REPO_ROOT / "experiments" / "clevr_subject_color_material_3x3x2" /
                 "manifests" / "clevr_material_heldout_eval.json")


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


QWEN = load("material_qwen", "scripts/methods/colorpeel_ice/predict_qwen_material.py")
SCORER = load("material_scorer", "scripts/methods/colorpeel_ice/score_material_multiview.py")


class MaterialQwenAndScoreTests(unittest.TestCase):
    def test_qwen_parser_requires_exact_three_keys(self):
        self.assertEqual(QWEN.parse_prediction('{"shape":"cube","color":"red","material":"metal"}'),
                         {"shape": "cube", "color": "red", "material": "metal"})
        with self.assertRaisesRegex(ValueError, "exactly"):
            QWEN.parse_prediction('{"shape":"cube","color":"red"}')
        with self.assertRaisesRegex(ValueError, "exactly"):
            QWEN.parse_prediction('{"shape":"cube","color":"red","material":"metal","extra":1}')

    def test_perfect_campaign_scores_all_axes_and_interventions(self):
        protocol = read_protocol(PROTOCOL_PATH); manifest = []
        for fold in "ABC":
            for seed in (42, 43, 44):
                manifest.extend({**row, "dtype": "float16", "safety_checker_disabled": True,
                                 "safety_risk_acknowledged": True,
                                 "protocol_fingerprint_sha256": protocol["_source_sha256"],
                                 "model_fingerprint_sha256": f"{fold}-{seed}"}
                                for row in build_manifest(protocol, fold_id=fold, training_seed=seed))
        predictions = [{"id": row["id"], "status": "ok", "predicted_shape": row["expected_shape"],
                        "predicted_color": row["expected_color"],
                        "predicted_material": row["expected_material"]} for row in manifest]
        metrics, scored, tables = SCORER.score(manifest, predictions, protocol)
        self.assertEqual(len(scored), 3240)
        self.assertEqual(metrics["overall"]["joint_accuracy_all_expected"], 1.0)
        self.assertEqual(len(tables["checkpoint_split"]), 18)
        self.assertEqual(len(tables["cell"]), 162)
        self.assertEqual(len(tables["material_intervention"]), 1620)
        self.assertEqual(len(tables["color_intervention"]), 1080)
        self.assertEqual(len(tables["shape_intervention"]), 1080)
        patterns = [row["split_pattern"] for row in tables["material_intervention"]]
        self.assertEqual(patterns.count("1seen+1heldout"), 1080)
        self.assertEqual(patterns.count("2seen+0heldout"), 540)


if __name__ == "__main__":
    unittest.main()
