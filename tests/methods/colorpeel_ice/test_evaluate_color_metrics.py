import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).parents[3]
SPEC = importlib.util.spec_from_file_location(
    "color_metrics",
    ROOT / "scripts" / "methods" / "colorpeel_ice" / "evaluate_color_metrics.py",
)
color_metrics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(color_metrics)


def item(item_id="transfer-000-seed-42"):
    return {
        "id": item_id,
        "category": "transfer",
        "prompt": "a <c1*> bowl on the table",
        "seed": 42,
        "color_label": "red",
        "color_token": "<c1*>",
        "image_path": f"images/transfer/{item_id}.png",
    }


def test_exact_color_has_zero_error(tmp_path):
    image_root = tmp_path / "generated"
    mask_root = tmp_path / "masks"
    relative = Path(item()["image_path"])
    (image_root / relative).parent.mkdir(parents=True)
    (mask_root / relative).parent.mkdir(parents=True)
    pixels = np.full((2, 2, 3), [173, 35, 35], dtype=np.uint8)
    Image.fromarray(pixels).save(image_root / relative)
    Image.fromarray(np.full((2, 2), 255, dtype=np.uint8)).save(mask_root / relative)

    result = color_metrics.evaluate_item(
        item(), image_root, mask_root, {"red": (173, 35, 35)}
    )

    assert result["status"] == "ok"
    assert result["mask_pixels"] == 4
    assert result["hue_valid_pixels"] == 4
    assert result["hue_status"] == "ok"
    for metric in color_metrics.METRICS:
        for fraction in color_metrics.FRACTIONS:
            assert abs(result[f"{metric}_{int(fraction * 100)}pct"]) < 1e-12


def test_missing_and_empty_masks_are_failures(tmp_path):
    image_root = tmp_path / "generated"
    mask_root = tmp_path / "masks"
    relative = Path(item()["image_path"])
    (image_root / relative).parent.mkdir(parents=True)
    Image.fromarray(np.full((2, 2, 3), [173, 35, 35], dtype=np.uint8)).save(
        image_root / relative
    )

    missing = color_metrics.evaluate_item(
        item(), image_root, mask_root, {"red": (173, 35, 35)}
    )
    assert missing["status"] == "failure"
    assert missing["failure_reason"] == "mask_missing"

    (mask_root / relative).parent.mkdir(parents=True)
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(mask_root / relative)
    empty = color_metrics.evaluate_item(
        item(), image_root, mask_root, {"red": (173, 35, 35)}
    )
    assert empty["status"] == "failure"
    assert empty["failure_reason"] == "mask_empty"


def test_gray_hue_is_reported_as_undefined():
    errors = color_metrics.pixel_errors(
        np.array([[87, 87, 87]], dtype=np.uint8), (87, 87, 87)
    )

    assert np.isnan(errors["hue_angular_deg"][0])
    assert color_metrics.closest_fraction_mean(errors["hue_angular_deg"], 1.0) is None


def test_target_rgb_values_come_from_locked_clevr_manifest():
    colors = color_metrics.load_target_colors(
        ROOT
        / "experiments"
        / "clevr_subject_color_3x3"
        / "manifests"
        / "clevr_3x3_manifest.json"
    )

    assert colors == {
        "red": (173, 35, 35),
        "cyan": (41, 208, 208),
        "gray": (87, 87, 87),
    }
