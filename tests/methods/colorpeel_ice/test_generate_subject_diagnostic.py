import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest


ROOT = Path(__file__).parents[3]
SPEC = importlib.util.spec_from_file_location(
    "generate_subject_diagnostic",
    ROOT
    / "scripts"
    / "methods"
    / "colorpeel_ice"
    / "generate_subject_diagnostic.py",
)
subject = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(subject)


def test_manifest_is_fixed_75_image_subject_diagnostic():
    rows = subject.build_manifest()
    assert len(rows) == 75
    assert {row["seed"] for row in rows} == {42, 43, 44, 45, 46}
    assert {row["subject_label"] for row in rows} == {"cube", "sphere", "cylinder"}
    assert all(row["uses_trained_kv"] is True for row in rows)
    assert Counter(row["condition"] for row in rows) == {
        "learned_subject_only": 15,
        "natural_subject_only": 15,
        "learned_subject_literal_red": 15,
        "learned_subject_literal_cyan": 15,
        "learned_subject_literal_gray": 15,
    }
    groups = defaultdict(list)
    for row in rows:
        groups[row["pair_id"]].append(row)
        assert row["num_inference_steps"] == 100
        assert row["guidance_scale"] == 6.0
    assert len(groups) == 15
    assert all(len(group) == 5 for group in groups.values())


def test_prompts_change_only_the_approved_subject_or_literal_color_expression():
    rows = subject.build_manifest()
    cube = {
        row["condition"]: row["prompt"]
        for row in rows
        if row["subject_label"] == "cube" and row["seed"] == 42
    }
    assert cube == {
        "learned_subject_only": "a photo of <s1*> shape",
        "natural_subject_only": "a photo of cube shape",
        "learned_subject_literal_red": "a photo of <s1*> shape in red color",
        "learned_subject_literal_cyan": "a photo of <s1*> shape in cyan color",
        "learned_subject_literal_gray": "a photo of <s1*> shape in gray color",
    }


def test_dry_run_writes_manifest_without_model(tmp_path):
    output = tmp_path / "subject"
    assert subject.main(["--output-dir", str(output), "--dry-run"]) == 0
    lines = (output / "subject_diagnostic_manifest.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 75
    assert all(json.loads(line)["category"] == "subject_diagnostic" for line in lines)
    assert not (output / "subject_diagnostic_status.jsonl").exists()


def test_safety_checker_disable_requires_explicit_acknowledgement(tmp_path):
    default = subject.parse_args(["--output-dir", str(tmp_path), "--dry-run"])
    assert default.disable_safety_checker is False
    assert default.acknowledge_safety_risk is False

    with pytest.raises(SystemExit):
        subject.parse_args(
            [
                "--output-dir", str(tmp_path),
                "--disable-safety-checker",
                "--dry-run",
            ]
        )
    disabled = subject.parse_args(
        [
            "--output-dir", str(tmp_path),
            "--disable-safety-checker",
            "--acknowledge-safety-risk",
            "--dry-run",
        ]
    )
    assert disabled.disable_safety_checker is True
    assert disabled.acknowledge_safety_risk is True


def test_generation_writes_per_image_status(monkeypatch, tmp_path):
    class FakeGenerator:
        def __init__(self, device):
            self.device = device

        def manual_seed(self, _seed):
            return self

    class FakePipe:
        def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(
                images=[Image.new("RGB", (512, 512))],
                nsfw_content_detected=[False],
            )

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(Generator=FakeGenerator)
    )
    args = SimpleNamespace(
        output_dir=tmp_path,
        device="cpu",
        disable_safety_checker=True,
        acknowledge_safety_risk=True,
    )
    row = subject.build_manifest()[0]
    status = subject.generate([row], FakePipe(), args)[0]
    assert status == {
        "id": row["id"],
        "pair_id": row["pair_id"],
        "condition": row["condition"],
        "image_path": str(tmp_path / row["image_path"]),
        "status": "ok",
        "failure_reason": None,
        "nsfw_content_detected": False,
        "safety_checker_disabled": True,
        "safety_risk_acknowledged": True,
    }
