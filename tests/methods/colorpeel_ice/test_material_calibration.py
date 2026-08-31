import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[3]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = load("material_calibration_prepare", "src/methods/colorpeel_ice/prepare_material_calibration.py")
QWEN = load("material_calibration_qwen", "scripts/methods/colorpeel_ice/predict_qwen_material_reference.py")
SCORE = load("material_calibration_score", "scripts/methods/colorpeel_ice/score_material_reference.py")
RUNNER = load("material_calibration_runner", "scripts/launch/colorpeel_run.py")


def reference_rows():
    rows = []
    index = 0
    for shape in PREPARE.SHAPES:
        for color in PREPARE.COLORS:
            for material in PREPARE.MATERIALS:
                for view in PREPARE.VIEWS:
                    rows.append({"shape": shape, "color": color, "material": material,
                                 "view_index": view, "image": f"image-{index}.jpg",
                                 "artifact_sha256": {"image": f"{index:064x}"},
                                 "render_seed": index, "cell_id": f"{shape}_{color}_{material}",
                                 "renderer_profile_id": "multiview_render_v3_material",
                                 "renderer_profile_sha256": "profile"})
                    index += 1
    return rows


class MaterialCalibrationTests(unittest.TestCase):
    def test_reference_manifest_is_locked_360_grid(self):
        rows = PREPARE.reference_manifest(reference_rows(), Path("/reference"))
        self.assertEqual(len(rows), 360)
        self.assertEqual(len({row["id"] for row in rows}), 360)
        self.assertEqual(QWEN.validate_items(rows), rows)

    def test_reference_manifest_rejects_missing_view(self):
        with self.assertRaisesRegex(ValueError, "locked"):
            PREPARE.reference_manifest(reference_rows()[:-1], Path("/reference"))

    def test_pair_review_selection_is_balanced_and_deterministic(self):
        pairs = []
        for fold in "ABC":
            for train_seed in (42, 43, 44):
                for generation_seed in range(42, 62):
                    for shape in PREPARE.SHAPES:
                        for color in PREPARE.COLORS:
                            group = (fold, train_seed, generation_seed, shape, color)
                            pairs.append({"group": group, "metal": {"id": "metal"},
                                          "rubber": {"id": "rubber"}})
        first = PREPARE.select_pair_review_pairs(pairs)
        second = PREPARE.select_pair_review_pairs(pairs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 162)
        self.assertEqual({pair["group"][2] for pair, _ in first}, {42, 43})
        strata = {(pair["group"][0], pair["group"][1], pair["group"][3], pair["group"][4])
                  for pair, _ in first}
        self.assertEqual(len(strata), 81)

    def test_reference_scorer_exposes_rubber_only_failure(self):
        manifest = PREPARE.reference_manifest(reference_rows(), Path("/reference"))
        predictions = []
        for row in manifest:
            predictions.append({"id": row["id"], "status": "ok",
                                "predicted_shape": row["expected_shape"],
                                "predicted_color": row["expected_color"],
                                "predicted_material": "metal"})
        metrics, scored = SCORE.score(manifest, predictions)
        self.assertEqual(len(scored), 360)
        self.assertEqual(metrics["shape_accuracy_all_expected"], 1.0)
        self.assertEqual(metrics["color_accuracy_all_expected"], 1.0)
        self.assertEqual(metrics["material_accuracy_all_expected"], 0.5)
        self.assertEqual(metrics["material_confusion"]["rubber->metal"], 180)

    def test_launcher_owns_calibration_outputs(self):
        run = Path("/runs/calibration")
        self.assertEqual(RUNNER.managed_output_args("prepare_material_calibration", run),
                         {"output-dir": str(run / "evaluation" / "calibration")})
        self.assertEqual(RUNNER.managed_output_args("predict_qwen_material_reference", run),
                         {"output": str(run / "evaluation" / "qwen_predictions.jsonl")})
        self.assertEqual(RUNNER.managed_output_args("score_material_reference", run),
                         {"output-dir": str(run / "evaluation" / "material_reference_metrics")})

    def test_tracked_calibration_configs_lock_expected_stages(self):
        configs = ROOT / "experiments" / "clevr_subject_color_material_3x3x2" / "configs"
        prepare = RUNNER.read_config(configs / "prepare_material_calibration.json")
        self.assertEqual(prepare["stage"], "prepare_material_calibration")
        self.assertEqual(prepare["args"]["realized-views"], "$COLORPEEL_MATERIAL_REALIZED_MANIFEST")
        self.assertEqual(prepare["args"]["reference-root"], "$COLORPEEL_MATERIAL_RENDER_ROOT")
        self.assertEqual(RUNNER.read_config(configs / "predict_qwen_material_reference.json")["stage"],
                         "predict_qwen_material_reference")
        self.assertEqual(RUNNER.read_config(configs / "score_material_reference.json")["stage"],
                         "score_material_reference")


if __name__ == "__main__":
    unittest.main()
