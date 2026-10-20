import builtins
import itertools
import json
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from experiments.subject_albedo_pbr_c import inference as module
from experiments.subject_multicolor_multimaterial_b import inference as baseline


def fake_dataset(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    rows = []
    factors = itertools.product(("pink", "green", "cyan", "blue", "purple"),
                                ("non-metallic", "metallic"), ("glossy", "moderately rough", "rough"))
    for number, (color, material, roughness) in enumerate(factors):
        image, mask = dataset / f"{number}.png", dataset / f"{number}_mask.png"
        Image.new("RGB", (32, 32), (number * 7, 50, 100)).save(image)
        Image.new("L", (32, 32), 255).save(mask)
        rows.append({"image": str(image), "mask": str(mask),
                     "caption": f"a photo of <S*> mailbox in {color} color with a {roughness} {material} surface",
                     "image_sha256": module.sha256(image), "mask_sha256": module.sha256(mask)})
    for name in ("concepts.json", "source_locks.json", "dataset_audit.json", "visual_review.json"):
        (dataset / name).write_text("{}", encoding="utf8")
    (dataset / "training_assets_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
    calls = []
    monkeypatch.setattr(module.data, "verify_dataset", lambda path: calls.append(path))
    monkeypatch.setattr(module, "git_info", lambda: {"commit": "abc", "branch": "codex/test"})
    return dataset, rows, calls


@pytest.mark.parametrize("step", (500, 750, 1000))
def test_same_prompts_sampling_and_paired_export_as_b(step):
    assert module.prompt_rows(step) == baseline.prompt_rows(step)
    assert module.SAMPLING is baseline.SAMPLING
    assert module.export_subject_step is baseline.export_subject_step


def test_dry_run_displays_every_actual_training_image_and_caption_without_gpu(tmp_path, monkeypatch):
    dataset, references, calls = fake_dataset(tmp_path, monkeypatch)
    imported = builtins.__import__
    def no_gpu(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "diffusers"}:
            raise AssertionError("dry run imported a GPU runtime")
        return imported(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_gpu)
    output = tmp_path / "step500"
    result = module.run(dataset, output, 500)
    assert result["status"] == "dry_run" and result["generated_images"] == 0
    assert result["training_images"] == 30 and len(calls) == 2
    assert json.loads((output / "training_references.json").read_text())["images"] == references
    with Image.open(output / "training_contact_sheet.jpg") as sheet:
        assert sheet.size == (1280, 1745)
        for number in range(30):
            actual = sheet.getpixel(((number % 5) * 256 + 10, (number // 5) * 285 + 45))
            assert max(abs(a - b) for a, b in zip(actual, (number * 7, 50, 100))) <= 5
    assert not (output / "images").exists()


def test_selections_reaudit_c_recipe_and_reject_b_config(tmp_path, monkeypatch):
    dataset, _, _ = fake_dataset(tmp_path, monkeypatch)
    selections, audits = [], []
    for method in module.METHODS:
        run = tmp_path / method
        run.mkdir()
        config = run / "config.yaml"
        module.training._write(config, module.training.build_config(method, dataset))
        selections.append({"method": method, "run": str(run), "status": "verified", "token": "<S*>",
                           "branch": "subject", "concept_id": "mailbox", "steps": 1000,
                           "hashes": {str(config): module.sha256(config)}})
    def audit(recipe, run, expected):
        audits.append(expected)
        return next(row for row in selections if row["run"] == str(run))
    monkeypatch.setattr(module.shared, "audit_run", audit)
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"git": module.git_info(), "runs": selections}))
    assert module._selections(index, dataset) == selections and len(audits) == 2
    run = tmp_path / module.METHODS[0]
    module.training._write(run / "config.yaml", baseline.training.build_config(module.METHODS[0], dataset))
    with pytest.raises(ValueError, match="differs from Experiment C"):
        module._selections(index, dataset)


def test_changed_training_rgb_and_old_result_directory_are_rejected(tmp_path, monkeypatch):
    dataset, rows, _ = fake_dataset(tmp_path, monkeypatch)
    Image.new("RGB", (32, 32), "red").save(rows[0]["image"])
    output = tmp_path / "step1000"
    with pytest.raises(ValueError, match="artifact changed"):
        module.run(dataset, output, 1000)
    assert not output.exists()
    output.mkdir()
    marker = output / "old.txt"
    marker.write_text("existing result")
    with pytest.raises(ValueError, match="must be new"):
        module.run(dataset, output, 1000)
    assert marker.read_text() == "existing result"


@pytest.mark.parametrize("missing_safety", (False, True))
def test_inference_uses_exact_b_sampling_and_all_c_references(tmp_path, monkeypatch, missing_safety):
    dataset, _, _ = fake_dataset(tmp_path, monkeypatch)
    index = tmp_path / "index.json"
    index.write_text("{}")
    output = tmp_path / "step750"
    selections = [{"method": method, "run": str(tmp_path / method)} for method in module.METHODS]
    monkeypatch.setattr(module, "_selections", lambda *args: selections)
    monkeypatch.setattr(module, "export_subject_step", lambda selection, step, path:
                        {**selection, "step": step, "hashes": {str(index): module.sha256(index)}})
    generators, loads, calls, masks = [], [], [], []
    class Generator:
        def __init__(self, device):
            self.device = device
            generators.append(self)
        def manual_seed(self, seed):
            self.seed = seed
            return self
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(Generator=Generator, cuda=SimpleNamespace(empty_cache=lambda: None)))
    def load(selected, device):
        loads.append((selected, device))
        def pipe(prompt, **kwargs):
            calls.append((prompt, kwargs))
            return SimpleNamespace(images=[Image.new("RGB", (32, 32), "blue")],
                nsfw_content_detected=None if missing_safety else [False])
        return pipe
    monkeypatch.setattr(module.evaluate, "load_pipeline", load)
    def attention(pipe, prompt, tokens, guidance):
        masks.append((prompt, tokens, guidance))
        return {"modifier_token_mask": "S mask"}
    monkeypatch.setattr(module.evaluate, "attention_masks", attention)
    if missing_safety:
        with pytest.raises(ValueError, match="safety checker"):
            module.run(dataset, output, 750, index, True)
        assert json.loads((output / "provenance.json").read_text())["status"] == "failed"
        return
    result = module.run(dataset, output, 750, index, True)
    assert result["status"] == "succeeded" and result["generated_images"] == 20
    assert len(loads) == 2 and all(len(selected) == 1 for selected, _ in loads)
    assert len(calls) == len(generators) == len(masks) == 20
    assert len(list((output / "images").rglob("*.png"))) == 20
    assert len(result["contact_sheets"]) == 3
    for (_, kwargs), generator, (_, tokens, guidance) in zip(calls, generators, masks):
        assert tokens == ["<S*>"] and guidance == 3.5
        assert kwargs["num_inference_steps"] == 100 and kwargs["guidance_scale"] == 3.5
        assert kwargs["width"] == kwargs["height"] == 512
        assert generator.seed in (42, 43) and generator.device == "cuda:0"
        assert kwargs["cross_attention_kwargs"] == {"modifier_token_mask": "S mask"}
    for sheet in result["contact_sheets"][1:]:
        with Image.open(output / sheet) as image:
            assert image.size == (1280, 2457)
