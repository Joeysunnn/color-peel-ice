import importlib.util
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).parents[3]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generate = load_module(
    "generate_for_qwen",
    ROOT / "scripts" / "methods" / "colorpeel_ice" / "generate.py",
)
qwen = load_module(
    "predict_qwen",
    ROOT / "scripts" / "methods" / "colorpeel_ice" / "predict_qwen.py",
)


def test_locked_qwen_protocol_and_300_non_transfer_items():
    items = qwen.non_transfer_items(generate.build_manifest())

    assert len(items) == 300
    assert qwen.MODEL_ID == "Qwen/Qwen3-VL-8B-Instruct"
    assert qwen.MAX_NEW_TOKENS == 128
    assert qwen.LOCAL_FILES_ONLY is True
    assert set(item["category"] for item in items) == {
        "grid",
        "subject_only",
        "color_only",
    }


def test_fixed_json_parser_rejects_extra_or_unknown_values():
    assert qwen.parse_prediction('```json\n{"shape":"cube","color":"red"}\n```') == {
        "shape": "cube",
        "color": "red",
    }
    for invalid in (
        '{"shape":"cube","color":"red","note":"x"}',
        '{"shape":"cone","color":"red"}',
        "not json",
    ):
        try:
            qwen.parse_prediction(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid response accepted: {invalid}")


def test_prediction_success_and_failures_are_explicit(tmp_path):
    items = qwen.non_transfer_items(generate.build_manifest())[:3]
    image_dir = tmp_path / "images"
    for item in (items[0], items[2]):
        path = image_dir / item["image_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2), (173, 35, 35)).save(path)

    calls = 0

    def fake_predictor(image):
        nonlocal calls
        calls += 1
        if calls == 1:
            return '{"shape":"cube","color":"red"}'
        return '{"shape":"cone","color":"red"}'

    statuses = qwen.run_predictions(items, image_dir, fake_predictor)

    assert [status["status"] for status in statuses] == ["ok", "failure", "failure"]
    assert statuses[0]["predicted_shape"] == "cube"
    assert statuses[0]["predicted_color"] == "red"
    assert statuses[0]["do_sample"] is False
    assert statuses[0]["max_new_tokens"] == 128
    assert statuses[1]["failure_reason"] == "image_missing"
    assert statuses[2]["failure_reason"].startswith("image_or_response_error:")


def test_model_cache_failure_writes_300_failures_and_returns_nonzero(tmp_path):
    manifest = tmp_path / "generation.jsonl"
    output = tmp_path / "qwen_predictions.jsonl"
    generate.write_manifest(generate.build_manifest(), manifest)

    class MissingCache:
        def __init__(self, device):
            raise OSError("cache miss")

    original = qwen.QwenPredictor
    qwen.QwenPredictor = MissingCache
    try:
        result = qwen.main(
            [
                "--manifest",
                str(manifest),
                "--image-dir",
                str(tmp_path / "images"),
                "--output",
                str(output),
            ]
        )
    finally:
        qwen.QwenPredictor = original

    rows = qwen.read_jsonl(output)
    assert result == 1
    assert len(rows) == 300
    assert all(row["status"] == "failure" for row in rows)
    assert all(row["failure_reason"].startswith("model_load_error:") for row in rows)
