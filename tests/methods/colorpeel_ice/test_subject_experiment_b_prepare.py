import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from experiments.subject_multicolor_multimaterial_b import prepare, review


@pytest.mark.parametrize("fail", [False, True])
def test_preparation_runs_only_two_dry_runs_and_preserves_sources(tmp_path, monkeypatch, fail):
    monkeypatch.setenv("COLORPEEL_OUTPUT_ROOT", str(tmp_path))
    original, reviewed, output = (tmp_path / name for name in ("original", "reviewed", "new"))
    for source in (original, reviewed):
        source.mkdir()
        (source / "old.png").write_bytes(b"immutable old data")
    monkeypatch.setattr(prepare, "git_info", lambda: {"commit": "a" * 40, "branch": "codex/subject-experiment-b"})
    monkeypatch.setattr(prepare, "stage", lambda a, b, folder: folder.mkdir())
    monkeypatch.setattr(prepare, "verify_dataset", lambda _: {"status": "verified", "rows": 10})
    monkeypatch.setattr(prepare, "write_review", lambda *_: {"training_rows": 10})
    plans = []

    def plan_inference(dataset, folder, step):
        plans.append(step)
        return {"status": "dry_run", "generated_images": 0, "expected_images": 20}

    monkeypatch.setattr(prepare, "prepare_inference", plan_inference)

    def bundle(dataset, folder):
        folder.mkdir()
        configs = {method: {"stage": "train", "args": {},
                           "run": {"study": prepare.STUDY, "variant": method, "seed": 42}}
                   for method in ("fullrank", "lora_r64")}
        paths = {}
        for method, config in configs.items():
            path = folder / (method + ".json")
            path.write_text(json.dumps(config))
            paths[method] = str(path)
        return {"configs": configs, "paths": paths}

    monkeypatch.setattr(prepare, "build_configuration_bundle", bundle)
    calls = []

    def dry_run(command, **kwargs):
        assert command[-1] == "--dry-run" and "--execute-training" not in command
        config = json.loads(Path(command[command.index("--config") + 1]).read_text())
        folder = Path(command[command.index("--run-dir") + 1])
        folder.mkdir()
        (folder / "config.yaml").write_text(json.dumps(config))
        (folder / "manifest.json").write_text(json.dumps({"status": "dry-run"}))
        calls.append(command)
        return SimpleNamespace(returncode=int(fail))

    monkeypatch.setattr(prepare.subprocess, "run", dry_run)
    if fail:
        with pytest.raises(RuntimeError):
            prepare.run(original, reviewed, output)
    else:
        assert prepare.run(original, reviewed, output)["status"] == "data_and_configuration_verified"
    state = json.loads((output / "status.json").read_text())
    assert state["training_started"] is state["inference_executed"] is False
    assert len(calls) == (1 if fail else 2)
    assert plans == ([] if fail else [500, 750, 1000])
    assert not list(output.rglob("checkpoints"))
    assert all((source / "old.png").read_bytes() == b"immutable old data" for source in (original, reviewed))
    with pytest.raises(ValueError):
        prepare.run(original, reviewed, output)


def test_review_contains_all_actual_images_and_exact_captions(tmp_path, monkeypatch):
    dataset, output = tmp_path / "dataset", tmp_path / "review"
    dataset.mkdir()
    rows, colors = [], {}
    sources = ("red", "green", "cyan", "blue", "magenta")
    for col, (color, source) in enumerate(zip(review.COLORS, sources)):
        for row_index, material in enumerate(review.MATERIALS):
            rgb = (30 + col * 35, 40 + row_index * 100, 80)
            path = dataset / f"{color}_{material}.png"
            Image.new("RGB", (512, 512), rgb).save(path)
            colors[col, row_index] = rgb
            rows.append({"id": path.stem, "color": color, "source_color": source,
                "material": material, "image": str(path), "mask": "paired-mask.png",
                "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "mask_sha256": "a" * 64,
                "caption": f"a photo of <S*> mailbox in {color} color with a {material} surface"})
    (dataset / "training_assets_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(review, "verify_dataset", lambda _: {"status": "verified", "rows": 10})
    result = review.write_review(dataset, output)
    with Image.open(result["contact_sheet"]) as board:
        assert board.size == (1920, 1110)
        for (col, row_index), expected in colors.items():
            actual = board.getpixel((col * 384 + 190, 100 + row_index * 500 + 200))
            assert max(abs(a - b) for a, b in zip(actual, expected)) <= 4
    with Path(result["captions"]).open(encoding="utf-8-sig", newline="") as handle:
        captions = list(csv.DictReader(handle))
    assert [row["caption"] for row in captions] == [row["caption"] for row in rows]
    assert captions[0]["source_color"] == "red" and captions[0]["color"] == "pink"
    assert captions[-1]["source_color"] == "magenta" and captions[-1]["color"] == "purple"
    assert all(hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest() == row["image_sha256"] for row in rows)
