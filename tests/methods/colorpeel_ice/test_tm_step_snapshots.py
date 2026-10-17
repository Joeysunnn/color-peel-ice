import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.sd15_token_local_kv_v1.contract import sha256
from experiments.texture_material_disentangled_v1 import step_snapshots as snapshots
from tests.methods.colorpeel_ice import test_sd15_snapshots as doubles


class StepSnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        filesystem_path = doubles.filesystem_path
        self.physical = lambda path: filesystem_path(Path(path).with_name(
            Path(path).name.replace("<T*>", "texture_token")))
        for target, value in (("tests.methods.colorpeel_ice.test_sd15_snapshots.filesystem_path", self.physical),
                              ("experiments.texture_material_disentangled_v1.step_snapshots.sha256",
                               lambda path: sha256(self.physical(path))),
                              ("experiments.sd15_token_local_kv_v1.snapshots.sha256",
                               lambda path: sha256(self.physical(path)))):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.torch = doubles.fake_torch()
        patcher = patch.dict(sys.modules, {"torch": self.torch})
        patcher.start()
        self.addCleanup(patcher.stop)

    def fixture(self, method="fullrank", branch="texture", maximum=1000, name="new"):
        run = self.root / name
        checkpoint = run / "checkpoints"
        checkpoint.mkdir(parents=True)
        token = "<T*>" if branch == "texture" else "<M*>"
        weight = "pytorch_token_local_kv_weights.bin" if method == "fullrank" else "pytorch_lora_kv_weights.bin"
        config = {"stage": "train", "run": {"study": "fixture", "variant": "fixture", "seed": 42},
                  "args": {"max_train_steps": maximum, "checkpoint_steps":
                           [300, 500, 750, 1000] if maximum == 1000 else [300]}}
        (run / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
        (run / "manifest.json").write_text('{}', encoding="utf-8")
        adaptation = {"modifier_tokens": [token], "adaptation_mode": "token_local_kv" if method == "fullrank"
                      else "lora_material_kv" if branch == "material" else "lora_subject_kv"}
        if method == "lora_r64":
            adaptation.update(rank=64, alpha=64, mode="token_local_kv")
        (checkpoint / "adaptation_config.json").write_text(json.dumps(adaptation), encoding="utf-8")
        audit = {"observed_optimization_steps": maximum,
                 "modifier_tokens": [{"token": token, "token_id": 2}]}
        (checkpoint / "embedding_update_audit.json").write_text(json.dumps(audit), encoding="utf-8")
        shapes, empty = snapshots.adapter_schema(method)
        adapter = {key: doubles.FakeTensor(shape) for key, shape in shapes.items()}
        doubles.write_state({**adapter, **empty}, checkpoint / weight)
        doubles.write_state({token: doubles.FakeTensor((768,))}, checkpoint / (token + ".bin"))
        for step in config["args"]["checkpoint_steps"]:
            directory = checkpoint / f"checkpoint-{step}"
            directory.mkdir()
            doubles.write_state(adapter, directory / "pytorch_model.bin")
            doubles.write_state({snapshots.EMBEDDING_KEY: doubles.FakeTensor((3, 768))},
                                directory / "pytorch_model_1.bin")
        files = [run / "config.yaml", run / "manifest.json"] + list(checkpoint.glob("*.*"))
        # The exported token file has a Linux name in the selection receipt.
        files = [path for path in files if path.name != self.physical(Path(token + ".bin")).name]
        files.append(checkpoint / (token + ".bin"))
        return {"method": method, "branch": branch, "concept_id": "example", "token": token,
                "weight": weight, "steps": maximum, "status": "verified", "run": str(run),
                "checkpoint": str(checkpoint),
                "hashes": {str(path.resolve()): sha256(self.physical(path)) for path in files}}

    def test_exports_all_steps_for_both_methods_and_tokens_without_source_mutation(self):
        for method, branch in (("fullrank", "texture"), ("fullrank", "material"),
                               ("lora_r64", "texture"), ("lora_r64", "material")):
            name = method + branch
            selection = self.fixture(method, branch, name=name)
            before = {str(path): sha256(path) for path in Path(selection["run"]).rglob("*") if path.is_file()}
            for step in (500, 750, 1000):
                result = snapshots.export_step(selection, step, self.root / f"{name}_{step}")
                self.assertEqual(result["steps"], step)
                self.assertEqual(result["training_steps"], 1000)
                self.assertTrue(result["paired_embedding_finite"])
                adapter = self.torch.load(Path(result["checkpoint"]) / result["weight"], map_location="cpu")
                self.assertEqual(sum(value == {} for value in adapter.values()), 16 if method == "fullrank" else 0)
                self.assertTrue(set(selection["hashes"]) <= set(result["hashes"]))
                self.assertTrue((Path(result["checkpoint"]) / "receipt.json").is_file())
            self.assertEqual(before, {str(path): sha256(path) for path in Path(selection["run"]).rglob("*") if path.is_file()})

    def test_prefix_equivalence_and_mismatch(self):
        new = self.fixture()
        baseline = self.fixture(maximum=300, name="old")
        self.assertEqual(snapshots.compare_baseline(new, baseline)["status"], "verified")
        directory = Path(new["checkpoint"]) / "checkpoint-300"
        doubles.write_state({snapshots.EMBEDDING_KEY: doubles.FakeTensor((3, 768), 2)},
                            directory / "pytorch_model_1.bin")
        with self.assertRaisesRegex(ValueError, "prefix differs"):
            snapshots.compare_baseline(new, baseline)

    def test_rejects_invalid_step_existing_output_and_source_output(self):
        selection = self.fixture()
        for step, output in ((600, self.root / "bad"), (500, self.root),
                             (500, Path(selection["run"]) / "derived")):
            with self.subTest(step=step, output=output), self.assertRaises((ValueError, FileExistsError)):
                snapshots.export_step(selection, step, output)

    def test_rejects_changed_source_and_nonfinite_snapshot(self):
        selection = self.fixture()
        snapshot = Path(selection["checkpoint"]) / "checkpoint-500/pytorch_model_1.bin"
        doubles.write_state({snapshots.EMBEDDING_KEY: doubles.FakeTensor((3, 768), float("nan"))}, snapshot)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            snapshots.export_step(selection, 500, self.root / "bad")
        Path(selection["run"], "manifest.json").write_text('{"changed":true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            snapshots.export_step(selection, 500, self.root / "bad")

    def test_rejects_final_pair_mismatch_and_rank_drift(self):
        selection = self.fixture()
        doubles.write_state({snapshots.EMBEDDING_KEY: doubles.FakeTensor((3, 768), 2)},
                            Path(selection["checkpoint"]) / "checkpoint-1000/pytorch_model_1.bin")
        with self.assertRaisesRegex(ValueError, "final snapshot differs"):
            snapshots.export_step(selection, 1000, self.root / "bad")
        lora = self.fixture("lora_r64", name="lora")
        path = Path(lora["checkpoint"]) / "adaptation_config.json"
        value = json.loads(path.read_text())
        value["rank"] = 4
        path.write_text(json.dumps(value))
        lora["hashes"][str(path.resolve())] = sha256(path)
        with self.assertRaisesRegex(ValueError, "mode or rank"):
            snapshots.export_step(lora, 500, self.root / "bad")


if __name__ == "__main__":
    unittest.main()
