import json
import math
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
import yaml
from unittest.mock import patch

from experiments.sd15_token_local_kv_v1 import snapshots
from experiments.sd15_token_local_kv_v1.contract import build_config, sha256


class FakeTensor:
    """Shape/value tensor double: checkpoint tests need no local torch installation."""

    def __init__(self, shape, value=1.0):
        self.shape = tuple(shape)
        self.value = value

    def __getitem__(self, index):
        if index >= self.shape[0]:
            raise IndexError(index)
        return FakeTensor(self.shape[1:], self.value)

    def clone(self):
        return FakeTensor(self.shape, self.value)

    def contiguous(self):
        return self


class FakeFinite:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self.value


def filesystem_path(path):
    # Linux training filenames contain characters unavailable on Windows.
    path = Path(path)
    return path.with_name(path.name.replace("<S*>", "subject_token").replace("<M*>", "material_token"))


def write_state(value, path):
    with filesystem_path(path).open("wb") as handle:
        pickle.dump(value, handle)


def fake_torch():
    module = types.ModuleType("torch")
    module.Tensor = FakeTensor
    module.loads = []

    def load(path, map_location=None):
        if map_location != "cpu":
            raise AssertionError("snapshot tensors must stay on CPU")
        module.loads.append(Path(path))
        with filesystem_path(path).open("rb") as handle:
            return pickle.load(handle)

    module.load = load
    module.save = write_state
    module.isfinite = lambda tensor: FakeFinite(math.isfinite(tensor.value))
    module.equal = lambda left, right: left.shape == right.shape and left.value == right.value
    return module


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.torch = fake_torch()
        self.patcher = patch.dict(sys.modules, {"torch": self.torch})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        for target in ("experiments.sd15_token_local_kv_v1.snapshots.sha256",
                       "experiments.sd15_token_local_kv_v1.evaluate.sha256"):
            hash_patch = patch(target, side_effect=lambda path: sha256(filesystem_path(path)))
            hash_patch.start()
            self.addCleanup(hash_patch.stop)

    def write_json(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def fixture(self, recipe="subject_fullrank"):
        run = self.root / recipe
        final = run / "checkpoints"
        final.mkdir(parents=True)
        config = build_config(recipe)
        args = config["args"]
        token, maximum = args["modifier_token"], args["max_train_steps"]
        method = "fullrank" if args.get("token_local_kv") else "lora_r64"
        weight = "pytorch_token_local_kv_weights.bin" if method == "fullrank" else "pytorch_lora_kv_weights.bin"
        (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        self.write_json(run / "manifest.json", {"status": "succeeded", "returncode": 0,
                        "stage": "train", "run": config["run"],
                        "sd15_migration_inputs": {"assets_verified": True}})
        adaptation = {"modifier_tokens": [token], "weight_name": weight,
                      "adaptation_mode": "token_local_kv" if method == "fullrank" else (
                          "lora_subject_kv" if token == "<S*>" else "lora_material_kv")}
        if method == "lora_r64":
            adaptation.update(mode="token_local_kv", rank=64, alpha=64)
        self.write_json(final / "adaptation_config.json", adaptation)
        shapes, empty = snapshots.adapter_schema(method)
        adapter = {key: FakeTensor(shape) for key, shape in shapes.items()}
        write_state({**adapter, **empty}, final / weight)
        write_state({token: FakeTensor((768,))}, final / (token + ".bin"))
        audit = {"observed_optimization_steps": maximum, "modifier_tokens": [{
            "token": token, "token_id": 49408, "exposure_steps": maximum,
            "nonzero_gradient_steps": maximum, "initial_final_l2_delta": 0.2}]}
        self.write_json(final / "embedding_update_audit.json", audit)
        rows = [{"step": step, **dict.fromkeys(snapshots.LOSS_FIELDS, 0.5),
                 "losses_finite": {"all": True}, "present_modifier_tokens": [token]}
                for step in range(1, maximum + 1)]
        (final / "training_metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        for step in snapshots.checkpoint_plan(args):
            directory = final / f"checkpoint-{step}"
            directory.mkdir()
            write_state(adapter, directory / "pytorch_model.bin")
            write_state({snapshots.EMBEDDING_KEY: FakeTensor((49409, 768))},
                        directory / "pytorch_model_1.bin")
            for name in snapshots.STATE_FILES[2:]:
                (directory / name).write_bytes(b"state")
        return run, final

    def test_all_six_frozen_checkpoint_plans_and_schema_counts(self):
        expected = {"subject_fullrank": [300, 500, 700, 1000, 2000],
                    "subject_lora_r64": [600, 1000, 2000, 3000],
                    "material_ground_reflection_fullrank": [1000, 2000, 3000, 4000, 5000],
                    "material_mailbox_lora_r64": [600, 1000, 2000, 3000],
                    "material_metal_spoon_lora_r64": [600, 1000, 2000, 3000],
                    "material_wood_spoon_lora_r64": [600, 1000, 2000, 3000]}
        for recipe, steps in expected.items():
            self.assertEqual(snapshots.checkpoint_plan(build_config(recipe)["args"]), steps)
        for method, count, empty_count in (("fullrank", 32, 16), ("lora_r64", 64, 0)):
            shapes, empty = snapshots.adapter_schema(method)
            self.assertEqual((len(shapes), len(empty)), (count, empty_count))

    def test_audit_exports_each_paired_snapshot_once_and_preserves_source(self):
        for recipe in ("subject_fullrank", "material_mailbox_lora_r64"):
            with self.subTest(recipe=recipe):
                run, final = self.fixture(recipe)
                original = {str(path): sha256(path) for path in run.rglob("*") if path.is_file()}
                result = snapshots.audit_training(recipe, run, self.root / (recipe + "_derived"))
                self.assertEqual(result["status"], "verified")
                self.assertTrue(result["final_snapshot_matches_export"])
                self.assertTrue(result["embedding_audit"]["nonzero_gradient_observed"])
                for step, record in result["checkpoints"].items():
                    path = Path(record["checkpoint"])
                    self.assertTrue((path / "receipt.json").is_file())
                    self.assertEqual(record["source_text_encoder_sha256"],
                                     sha256(final / f"checkpoint-{step}/pytorch_model_1.bin"))
                    self.assertEqual(self.torch.loads.count(final / f"checkpoint-{step}/pytorch_model_1.bin"), 1)
                    with (path / result["weight"]).open("rb") as handle:
                        adapter = pickle.load(handle)
                    self.assertEqual(sum(value == {} for value in adapter.values()),
                                     16 if result["method"] == "fullrank" else 0)
                self.assertEqual(original, {str(path): sha256(path) for path in run.rglob("*") if path.is_file()})

    def test_export_single_snapshot_still_audits_all_steps(self):
        run, final = self.fixture()
        record = snapshots.export_snapshot("subject_fullrank", run, 500, self.root / "derived")
        self.assertEqual(record["step"], 500)
        self.assertEqual(len(list(self.root.glob("derived*"))), 1)
        self.assertEqual(sum(path.name == "pytorch_model_1.bin" for path in self.torch.loads), 5)

    def test_rejects_existing_output_and_training_directory(self):
        run, _ = self.fixture()
        for target in (run, run / "new", self.root):
            with self.subTest(target=target), self.assertRaises((ValueError, FileExistsError)):
                snapshots.audit_training("subject_fullrank", run, target)
        self.assertFalse((run / "new").exists())

    def test_metrics_reject_duplicates_nonfinite_values_and_missing_tokens(self):
        path = self.root / "metrics.jsonl"
        base = {"step": 1, **dict.fromkeys(snapshots.LOSS_FIELDS, 0.5),
                "losses_finite": {"all": True}, "present_modifier_tokens": ["<S*>"]}
        for rows in ([base, base], [{**base, "total_loss": float("nan")}],
                     [{**base, "present_modifier_tokens": []}],
                     [{**base, "losses_finite": {"all": False}}]):
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                snapshots._metrics(path, 1, "<S*>")

    def test_rejects_missing_training_state(self):
        run, final = self.fixture()
        (final / "checkpoint-300/optimizer.bin").unlink()
        with self.assertRaisesRegex(ValueError, "training state"):
            snapshots.audit_training("subject_fullrank", run)

    def test_rejects_final_snapshot_mismatch(self):
        run, final = self.fixture()
        path = final / "checkpoint-2000/pytorch_model.bin"
        with path.open("rb") as handle:
            adapter = pickle.load(handle)
        adapter[next(iter(adapter))].value = 2.0
        write_state(adapter, path)
        with self.assertRaisesRegex(ValueError, "final snapshot"):
            snapshots.audit_training("subject_fullrank", run)

    def test_rejects_nonfinite_snapshot_adapter_and_embedding(self):
        run, final = self.fixture()
        adapter_path = final / "checkpoint-300/pytorch_model.bin"
        with adapter_path.open("rb") as handle:
            adapter = pickle.load(handle)
        key = next(iter(adapter))
        adapter[key].value = float("nan")
        write_state(adapter, adapter_path)
        with self.assertRaisesRegex(ValueError, "finiteness"):
            snapshots.audit_training("subject_fullrank", run)
        adapter[key].value = 1.0
        write_state(adapter, adapter_path)
        write_state({snapshots.EMBEDDING_KEY: FakeTensor((49409, 768), float("inf"))},
                    final / "checkpoint-300/pytorch_model_1.bin")
        with self.assertRaisesRegex(ValueError, "nonfinite snapshot token"):
            snapshots.audit_training("subject_fullrank", run)

    def test_rejects_rank_and_token_audit_drift(self):
        run, final = self.fixture("subject_lora_r64")
        path = final / "pytorch_lora_kv_weights.bin"
        with path.open("rb") as handle:
            adapter = pickle.load(handle)
        adapter[next(iter(adapter))].shape = (4, 768)
        write_state(adapter, path)
        with self.assertRaisesRegex(ValueError, "shape"):
            snapshots.audit_training("subject_lora_r64", run)
        audit_path = final / "embedding_update_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["modifier_tokens"][0]["token"] = "<M*>"
        self.write_json(audit_path, audit)
        with self.assertRaisesRegex(ValueError, "modifier token"):
            snapshots.audit_training("subject_lora_r64", run)

    def test_rejects_unplanned_snapshot_and_wrong_recipe(self):
        run, final = self.fixture()
        with self.assertRaisesRegex(ValueError, "requested snapshot"):
            snapshots.export_snapshot("subject_fullrank", run, 42, self.root / "derived")
        with self.assertRaises(ValueError):
            snapshots.audit_training("subject_lora_r64", run)
        (final / "checkpoint-42").mkdir()
        with self.assertRaisesRegex(ValueError, "directories"):
            snapshots.audit_training("subject_fullrank", run)


if __name__ == "__main__":
    unittest.main()
