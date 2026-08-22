import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).parents[3]
SPEC = importlib.util.spec_from_file_location(
    "rerun_black_images",
    ROOT / "scripts" / "methods" / "colorpeel_ice" / "rerun_black_images.py",
)
rerun_black = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(rerun_black)


def _write_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "id": "transfer-000-seed-42",
                "prompt": "a <c2*> bowl on the table",
                "seed": 42,
                "num_inference_steps": 100,
                "guidance_scale": 6.0,
                "image_path": "images/transfer/black.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_black_image(root: Path) -> None:
    path = root / "images" / "transfer" / "black.png"
    path.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8), mode="RGB").save(path)


def test_protocol_isolates_checker_change_from_fp32(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    source = tmp_path / "source"
    _write_manifest(manifest)
    _write_black_image(source)

    stage1 = rerun_black.parse_args(
        [
            "--manifest", str(manifest),
            "--image-dir", str(source),
            "--output-dir", str(tmp_path / "stage1"),
            "--diagnostic-stage", "safety_flag",
            "--dtype", "float16",
            "--dry-run",
        ]
    )
    assert stage1.dtype == "float16"
    assert stage1.disable_safety_checker is False

    with pytest.raises(SystemExit):
        rerun_black.parse_args(
            [
                "--manifest", str(manifest),
                "--image-dir", str(source),
                "--output-dir", str(tmp_path / "bad"),
                "--diagnostic-stage", "safety_flag",
                "--dtype", "float32",
                "--dry-run",
            ]
        )


def test_later_stages_require_prior_still_black_and_explicit_safety_ack(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    source = tmp_path / "source"
    _write_manifest(manifest)
    _write_black_image(source)
    prior = tmp_path / "prior.jsonl"
    prior.write_text(
        json.dumps(
            {
                "id": "transfer-000-seed-42",
                "status": "failure",
                "output_audit": {"is_black": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    stage2 = rerun_black.parse_args(
        [
            "--manifest", str(manifest),
            "--image-dir", str(source),
            "--output-dir", str(tmp_path / "stage2"),
            "--diagnostic-stage", "disable_safety",
            "--dtype", "float16",
            "--prior-status", str(prior),
            "--disable-safety-checker",
            "--acknowledge-safety-risk",
            "--dry-run",
        ]
    )
    assert stage2.dtype == "float16"
    assert stage2.disable_safety_checker is True

    stage3 = rerun_black.parse_args(
        [
            "--manifest", str(manifest),
            "--image-dir", str(source),
            "--output-dir", str(tmp_path / "stage3"),
            "--diagnostic-stage", "fp32_finite",
            "--dtype", "float32",
            "--prior-status", str(prior),
            "--disable-safety-checker",
            "--acknowledge-safety-risk",
            "--dry-run",
        ]
    )
    assert stage3.dtype == "float32"
    assert stage3.disable_safety_checker is True

    with pytest.raises(SystemExit):
        rerun_black.parse_args(
            [
                "--manifest", str(manifest),
                "--image-dir", str(source),
                "--output-dir", str(tmp_path / "bad"),
                "--diagnostic-stage", "fp32_finite",
                "--dtype", "float32",
                "--prior-status", str(prior),
                "--disable-safety-checker",
                "--dry-run",
            ]
        )

    prior.write_text(
        json.dumps(
            {
                "id": "transfer-000-seed-42",
                "status": "ok",
                "output_audit": {"is_black": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no still-black ids"):
        rerun_black.continuing_ids(prior)


def test_dry_run_selects_only_decoded_black_images(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    source = tmp_path / "source"
    output = tmp_path / "stage1"
    _write_manifest(manifest)
    _write_black_image(source)

    assert rerun_black.main(
        [
            "--manifest", str(manifest),
            "--image-dir", str(source),
            "--output-dir", str(output),
            "--diagnostic-stage", "safety_flag",
            "--dtype", "float16",
            "--dry-run",
        ]
    ) == 0
    rows = rerun_black.read_jsonl(output / "rerun_status.jsonl")
    assert rows[0]["source_audit"]["is_black"] is True
    assert rows[0]["diagnostic_stage"] == "safety_flag"
    assert rows[0]["torch_dtype"] == "float16"
    assert rows[0]["safety_checker_disabled"] is False
