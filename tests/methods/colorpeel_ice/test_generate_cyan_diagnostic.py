import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[3]
SPEC = importlib.util.spec_from_file_location(
    "generate_cyan_diagnostic",
    ROOT
    / "scripts"
    / "methods"
    / "colorpeel_ice"
    / "generate_cyan_diagnostic.py",
)
cyan = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(cyan)


def test_manifest_is_locked_540_image_factorial():
    rows = cyan.build_manifest()
    assert len(rows) == 540
    assert {row["seed"] for row in rows} == {42, 43, 44}
    assert len({row["noun"] for row in rows}) == 10
    assert Counter(row["model_variant"] for row in rows) == {
        "trained": 300,
        "vanilla": 240,
    }
    assert Counter(row["template_family"] for row in rows) == {
        "adjective_transfer": 270,
        "training_suffix": 270,
    }
    assert Counter(row["color_candidate"] for row in rows) == {
        "learned_token": 60,
        "cyan": 120,
        "aqua": 120,
        "teal": 120,
        "turquoise": 120,
    }
    groups = defaultdict(list)
    for row in rows:
        groups[row["pair_id"]].append(row)
        assert row["num_inference_steps"] == 100
        assert row["guidance_scale"] == 6.0
    assert len(groups) == 30
    assert all(len(group) == 18 for group in groups.values())


def test_literal_candidates_are_paired_and_learned_token_is_trained_only():
    rows = cyan.build_manifest()
    literal = [row for row in rows if row["color_candidate"] != "learned_token"]
    learned = [row for row in rows if row["color_candidate"] == "learned_token"]
    comparisons = defaultdict(set)
    for row in literal:
        comparisons[row["comparison_id"]].add(row["model_variant"])
        assert row["color_expression"] in cyan.LITERAL_CANDIDATES
    assert len(comparisons) == 240
    assert all(variants == {"vanilla", "trained"} for variants in comparisons.values())
    assert len(learned) == 60
    assert all(row["model_variant"] == "trained" for row in learned)
    assert all("<c2*>" in row["prompt"] for row in learned)


def test_two_prompt_families_use_locked_text_rules():
    rows = cyan.build_manifest()
    pants = [
        row
        for row in rows
        if row["noun"] == "pants"
        and row["seed"] == 42
        and row["color_candidate"] == "cyan"
        and row["model_variant"] == "vanilla"
    ]
    by_family = {row["template_family"]: row["prompt"] for row in pants}
    assert by_family == {
        "adjective_transfer": "a women wearing cyan pants",
        "training_suffix": "a photo of a pair of pants in cyan color",
    }


def test_dry_run_writes_manifest_without_model(tmp_path):
    output = tmp_path / "diagnostic"
    assert cyan.main(["--output-dir", str(output), "--dry-run"]) == 0
    lines = (output / "cyan_diagnostic_manifest.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 540
    assert all(json.loads(line)["category"] == "cyan_diagnostic" for line in lines)
    assert not (output / "cyan_diagnostic_status.jsonl").exists()


def test_safety_checker_disable_requires_explicit_acknowledgement(tmp_path):
    default = cyan.parse_args(["--output-dir", str(tmp_path), "--dry-run"])
    assert default.disable_safety_checker is False
    assert default.acknowledge_safety_risk is False

    with pytest.raises(SystemExit):
        cyan.parse_args(
            [
                "--output-dir", str(tmp_path),
                "--disable-safety-checker",
                "--dry-run",
            ]
        )
    disabled = cyan.parse_args(
        [
            "--output-dir", str(tmp_path),
            "--disable-safety-checker",
            "--acknowledge-safety-risk",
            "--dry-run",
        ]
    )
    assert disabled.disable_safety_checker is True
    assert disabled.acknowledge_safety_risk is True


def test_pipeline_default_keeps_checker_and_disable_removes_it(monkeypatch, tmp_path):
    class FakePipe:
        def __init__(self):
            self.safety_checker = object()
            self.requires_safety_checker = True

        def to(self, _device):
            return self

    created = []

    class FakeDiffusionPipeline:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            pipe = FakePipe()
            created.append(pipe)
            return pipe

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(float16="fp16", float32="fp32"),
    )
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(DiffusionPipeline=FakeDiffusionPipeline),
    )
    common = {
        "dtype": "float16",
        "pretrained_model_name_or_path": cyan.MODEL_ID,
        "device": "cuda:0",
        "model_dir": tmp_path,
    }
    enabled = cyan.load_pipeline(
        SimpleNamespace(**common, disable_safety_checker=False), trained=False
    )
    assert enabled.safety_checker is not None
    assert enabled.requires_safety_checker is True

    disabled = cyan.load_pipeline(
        SimpleNamespace(**common, disable_safety_checker=True), trained=False
    )
    assert disabled.safety_checker is None
    assert disabled.requires_safety_checker is False


def test_generation_status_records_safety_policy(monkeypatch, tmp_path):
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

    from PIL import Image

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(Generator=FakeGenerator)
    )
    args = SimpleNamespace(
        output_dir=tmp_path,
        device="cpu",
        disable_safety_checker=True,
        acknowledge_safety_risk=True,
    )
    status = cyan.generate_condition([cyan.build_manifest()[0]], FakePipe(), args)[0]
    assert status["status"] == "ok"
    assert status["safety_checker_disabled"] is True
    assert status["safety_risk_acknowledged"] is True
