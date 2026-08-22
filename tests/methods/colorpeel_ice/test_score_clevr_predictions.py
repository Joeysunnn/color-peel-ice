import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[3]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


inference = load_module(
    "clevr_inference_for_scoring",
    ROOT / "scripts" / "methods" / "colorpeel_ice" / "generate.py",
)
scoring = load_module(
    "clevr_scoring",
    ROOT / "scripts" / "methods" / "colorpeel_ice" / "score_clevr_predictions.py",
)


def test_grid_accuracy_and_axis_contingency_counts():
    manifest = inference.build_manifest()
    predictions = {}
    for item in manifest:
        if item["category"] == "grid":
            predictions[item["id"]] = {
                "id": item["id"],
                "predicted_shape": item["subject_label"],
                "predicted_color": item["color_label"],
            }
        elif item["category"] == "subject_only":
            predictions[item["id"]] = {
                "id": item["id"],
                "predicted_shape": item["subject_label"],
                "predicted_color": "red",
            }
        elif item["category"] == "color_only":
            predictions[item["id"]] = {
                "id": item["id"],
                "predicted_shape": "cube",
                "predicted_color": item["color_label"],
            }

    metrics, merged = scoring.score(manifest, predictions)

    assert len(merged) == 900
    assert metrics["predicted_items"] == 300
    assert metrics["missing_prediction_items"] == 600
    assert metrics["grid"] == {
        "total": 180,
        "shape_correct": 180,
        "color_correct": 180,
        "joint_correct": 180,
        "shape_accuracy": 1.0,
        "color_accuracy": 1.0,
        "joint_accuracy": 1.0,
    }
    subject_table = metrics["axis_contingency"]["subject_token_by_predicted_color"]
    assert all(subject_table["counts"][shape]["red"] == 20 for shape in scoring.SHAPES)
    color_table = metrics["axis_contingency"]["color_token_by_predicted_shape"]
    assert all(color_table["counts"][color]["cube"] == 20 for color in scoring.COLORS)


def test_missing_and_open_vocab_predictions_have_explicit_buckets():
    manifest = [
        item
        for item in inference.build_manifest()
        if item["category"] in ("subject_only", "color_only")
    ]
    subject_item = next(item for item in manifest if item["category"] == "subject_only")
    color_item = next(item for item in manifest if item["category"] == "color_only")
    predictions = {
        subject_item["id"]: {
            "id": subject_item["id"],
            "predicted_shape": "cube",
            "predicted_color": "purple",
        },
        color_item["id"]: {
            "id": color_item["id"],
            "predicted_shape": "cone",
            "predicted_color": "red",
        },
    }

    metrics, _ = scoring.score(manifest, predictions)
    subject_table = metrics["axis_contingency"]["subject_token_by_predicted_color"]
    color_table = metrics["axis_contingency"]["color_token_by_predicted_shape"]

    assert subject_table["counts"][subject_item["subject_label"]]["other"] == 1
    assert color_table["counts"][color_item["color_label"]]["other"] == 1
    assert sum(row["missing"] for row in subject_table["counts"].values()) == 59
    assert sum(row["missing"] for row in color_table["counts"].values()) == 59
