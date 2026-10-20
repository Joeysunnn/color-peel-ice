import builtins
import itertools
import json
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from experiments.subject_multicolor_multimaterial_b import inference as module


def fake_dataset(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    rows = []
    for number, (color, material) in enumerate(itertools.product(module.data.COLORS, ("metallic", "matte"))):
        image, mask = dataset / f"{number}.png", dataset / f"{number}_mask.png"
        Image.new("RGB", (32, 32), (number * 20, 50, 100)).save(image)
        Image.new("L", (32, 32), 255).save(mask)
        rows.append({"image": str(image), "mask": str(mask), "color": color, "material": material,
                     "caption": f"a photo of <S*> mailbox in {color} color with a {material} surface",
                     "image_sha256": module.sha256(image), "mask_sha256": module.sha256(mask)})
    for name in ("concepts.json", "source_locks.json", "dataset_audit.json"):
        (dataset / name).write_text("{}", encoding="utf8")
    (dataset / "training_assets_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
    calls = []
    monkeypatch.setattr(module.data, "verify_dataset", lambda path: calls.append(path))
    monkeypatch.setattr(module, "git_info", lambda: {"commit": "abc", "branch": "codex/test"})
    return dataset, rows, calls


@pytest.mark.parametrize("step", module.STEPS)
def test_fixed_material_matrix_is_twenty_subject_only_prompts(step):
    rows = module.prompt_rows(step)
    assert len(rows) == len({row["id"] for row in rows}) == 20
    assert {(row["method"], row["material"], row["seed"]) for row in rows} == set(
        itertools.product(module.METHODS, module.MATERIALS, (42, 43)))
    for row in rows:
        assert row["step"] == step and row["prompt"].count("<S*>") == 1
        assert not any(token in row["prompt"] for token in ("<M*>", "<T*>", "<C*>"))
        assert row["prompt"] == f"a photo of <S*> mailbox with a {row['material']} surface"
    assert module.SAMPLING == {"scheduler": "PNDM", "num_inference_steps": 100,
        "guidance_scale": 3.5, "width": 512, "height": 512, "dtype": "float16",
        "safety_checker": True, "generator": "fresh_device_generator_per_image", "seeds": [42, 43]}


@pytest.mark.parametrize("step", (0, 700, True, "500"))
def test_rejects_undeclared_snapshot(step):
    with pytest.raises(ValueError):
        module.prompt_rows(step)


def test_default_prepares_manifest_and_ten_training_references_without_gpu(tmp_path, monkeypatch):
    dataset, references, calls = fake_dataset(tmp_path, monkeypatch)
    imported = builtins.__import__
    def no_gpu(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "diffusers"}:
            raise AssertionError("dry run imported a GPU runtime")
        return imported(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_gpu)
    monkeypatch.setattr(module.evaluate, "load_pipeline", lambda *args: pytest.fail("loaded pipeline"))
    output = tmp_path / "step500"
    result = module.run(dataset, output, 500)
    assert result["status"] == "dry_run" and result["generated_images"] == 0
    assert result["expected_images"] == 20 and result["training"] == []
    assert len(calls) == 2
    assert not (output / "images").exists() and not (output / "checkpoints").exists()
    assert len((output / "generation_manifest.jsonl").read_text().splitlines()) == 20
    assert json.loads((output / "training_references.json").read_text())["images"] == references
    with Image.open(output / "training_contact_sheet.jpg") as sheet:
        assert sheet.size == (1280, 620)
        for number in range(10):
            x, y = (number % 5) * 256 + 10, (number // 5) * 285 + 45
            actual = sheet.getpixel((x, y))
            assert max(abs(a - b) for a, b in zip(actual, (number * 20, 50, 100))) <= 5


def test_inference_requires_completed_training_before_creating_output(tmp_path, monkeypatch):
    dataset, _, _ = fake_dataset(tmp_path, monkeypatch)
    output = tmp_path / "step750"
    with pytest.raises(ValueError, match="completed training index"):
        module.run(dataset, output, 750, execute_inference=True)
    assert not output.exists()


def test_dry_run_refuses_changed_reference_and_existing_output(tmp_path, monkeypatch):
    dataset, rows, _ = fake_dataset(tmp_path, monkeypatch)
    Image.new("RGB", (32, 32), "red").save(rows[0]["image"])
    output = tmp_path / "step1000"
    with pytest.raises(ValueError, match="artifact changed"):
        module.run(dataset, output, 1000)
    assert not output.exists()
    output.mkdir()
    marker = output / "old.txt"
    marker.write_text("old result")
    with pytest.raises(ValueError, match="must be new"):
        module.run(dataset, output, 1000)
    assert marker.read_text() == "old result"


def test_completed_index_is_reaudited_against_exact_experiment_recipe(tmp_path, monkeypatch):
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
        audits.append((recipe, run, expected))
        return next(row for row in selections if row["run"] == str(run))
    monkeypatch.setattr(module.shared, "audit_run", audit)
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"git": module.git_info(), "runs": selections}))
    assert module._selections(index, dataset) == selections and len(audits) == 2
    selections[0]["branch"] = "material"
    index.write_text(json.dumps({"git": module.git_info(), "runs": selections}))
    with pytest.raises(ValueError, match="Subject selection"):
        module._selections(index, dataset)


def test_subject_export_rejects_material_selection_without_touching_disk(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="Subject-only"):
        module.export_subject_step({"token": "<M*>", "branch": "material"}, 500, tmp_path / "derived")
    assert not (tmp_path / "derived").exists()


@pytest.mark.parametrize("missing_safety", (False, True))
def test_explicit_inference_uses_two_subject_only_adapters_and_fixed_sampling(tmp_path, monkeypatch, missing_safety):
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
        return {"modifier_token_mask": "Boolean S mask sentinel"}
    monkeypatch.setattr(module.evaluate, "attention_masks", attention)
    if missing_safety:
        with pytest.raises(ValueError, match="safety checker"):
            module.run(dataset, output, 750, index, True, "cuda:0")
        assert json.loads((output / "provenance.json").read_text())["status"] == "failed"
        return
    result = module.run(dataset, output, 750, index, True, "cuda:0")
    assert result["status"] == "succeeded" and result["generated_images"] == 20
    assert len(loads) == 2 and all(len(selected) == 1 for selected, _ in loads)
    assert len(calls) == len(generators) == len(masks) == 20
    assert len(list((output / "images").rglob("*.png"))) == 20
    assert len(result["contact_sheets"]) == 3
    for (prompt, kwargs), generator, (_, tokens, guidance) in zip(calls, generators, masks):
        assert tokens == ["<S*>"] and guidance == 3.5
        assert kwargs["num_inference_steps"] == 100 and kwargs["guidance_scale"] == 3.5
        assert kwargs["width"] == kwargs["height"] == 512
        assert generator.seed in (42, 43) and generator.device == "cuda:0"
        assert kwargs["cross_attention_kwargs"] == {"modifier_token_mask": "Boolean S mask sentinel"}


@pytest.mark.parametrize("method,step", itertools.product(module.METHODS, (500, 1000)))
def test_export_pairs_s_row_with_same_step_adapter_including_empty_self_entries(tmp_path, monkeypatch, method, step):
    torch = pytest.importorskip("torch")
    run = tmp_path / "training"
    checkpoint = run / "checkpoints"
    directory = checkpoint / f"checkpoint-{step}"
    directory.mkdir(parents=True)
    module.training._write(run / "config.yaml", module.training.build_config(method, tmp_path / "dataset"))
    weight = "pytorch_token_local_kv_weights.bin" if method == "fullrank" else "pytorch_lora_kv_weights.bin"
    empty = {"self.processor": {}} if method == "fullrank" else {}
    shapes = {"cross.delta": (2, 768)}
    monkeypatch.setattr(module.snapshots, "adapter_schema", lambda requested: (shapes, empty))
    for name in module.snapshots.STATE_FILES:
        (directory / name).write_bytes(b"state")
    adapter = {"cross.delta": torch.full((2, 768), step / 1000)}
    torch.save(adapter, directory / "pytorch_model.bin")
    torch.save({module.snapshots.EMBEDDING_KEY: torch.stack((torch.zeros(768), torch.full((768,), step / 1000)))},
               directory / "pytorch_model_1.bin")
    torch.save({"cross.delta": torch.ones(2, 768), **empty}, checkpoint / weight)
    torch.save({"<S*>": torch.ones(768)}, checkpoint / "<S*>.bin")
    (checkpoint / "adaptation_config.json").write_text(json.dumps({"modifier_tokens": ["<S*>"]}))
    (checkpoint / "embedding_update_audit.json").write_text(json.dumps({"observed_optimization_steps": 1000,
        "modifier_tokens": [{"token": "<S*>", "token_id": 1, "exposure_steps": 1000,
                             "nonzero_gradient_steps": 1000, "initial_final_l2_delta": 1.0}]}))
    selection = {"token": "<S*>", "branch": "subject", "method": method, "status": "verified", "steps": 1000,
                 "weight": weight, "run": str(run), "checkpoint": str(checkpoint),
                 "hashes": {str(run / "config.yaml"): module.sha256(run / "config.yaml")}}
    output = tmp_path / "derived"
    derived = module.export_subject_step(selection, step, output)
    actual = torch.load(output / weight, map_location="cpu")
    assert set(actual) == set(shapes) | set(empty)
    assert torch.equal(actual["cross.delta"], adapter["cross.delta"])
    assert torch.equal(torch.load(output / "<S*>.bin")["<S*>"], torch.full((768,), step / 1000))
    assert derived["step"] == derived["steps"] == step and derived["training_steps"] == 1000
    assert set(derived["source_snapshot_hashes"]) == {str(directory / name) for name in ("pytorch_model.bin", "pytorch_model_1.bin")}
    assert json.loads((output / "receipt.json").read_text())["paired_embedding_finite"] is True
