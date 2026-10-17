import contextlib
import csv
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from experiments.texture_material_disentangled_v1 import material_probe as probe


class MaterialProbeTests(unittest.TestCase):
    def test_matched_three_arm_matrix_has_no_texture_cues(self):
        protocol = probe.data.read_protocol()
        rows = probe.build_rows(protocol, 500)
        self.assertEqual(len(rows), 96)
        groups, bases = {}, {}
        for row in rows:
            self.assertNotIn("texture", row["prompt"])
            self.assertNotIn("strip", row["prompt"])
            self.assertNotIn("checker", row["prompt"])
            self.assertNotIn("<T*>", row["prompt"])
            self.assertEqual(row["prompt"].count("<M*>"), int(row["condition"] == "learned_material"))
            key = tuple(row[field] for field in ("method", "material", "context", "seed"))
            groups.setdefault(key, []).append(row)
            if row["condition"] == "base":
                bases.setdefault((row["context"], row["seed"]), set()).add(row["prompt"])
        self.assertEqual(len(groups), 32)
        self.assertTrue(all(len(prompts) == 1 for prompts in bases.values()))
        for group in groups.values():
            self.assertEqual({row["condition"] for row in group}, set(probe.CONDITIONS))
            for field in ("shape", "color", "light", "seed"):
                self.assertEqual(len({row[field] for row in group}), 1)
            literal = next(row for row in group if row["condition"] == "literal")
            self.assertIn(protocol["materials"][literal["material"]]["literal"], literal["prompt"])
        for step in (750, 1000):
            self.assertEqual([{**row, "training_steps": 500} for row in probe.build_rows(protocol, step)], rows)
        with self.assertRaises(ValueError):
            probe.build_rows(protocol, 300)

    def source_fixture(self, root):
        protocol = probe.data.read_protocol()
        (root / "step500/inference").mkdir(parents=True)
        runs = []
        for method in probe.infer.METHODS:
            for branch in ("texture", "material"):
                for concept in probe.infer.target_ids(protocol, branch):
                    checkpoint = root / "adapters" / f"{method}_{branch}_{concept}"
                    checkpoint.mkdir(parents=True)
                    config = {"modifier_tokens": ["<M*>" if branch == "material" else "<T*>"],
                              "adaptation_mode": "token_local_kv" if method == "fullrank" else "lora_material_kv",
                              "rank": 64, "alpha": 64, "mode": "token_local_kv"}
                    probe.write_json(checkpoint / "adaptation_config.json", config)
                    runs.append({"method": method, "branch": branch, "concept_id": concept,
                                 "checkpoint": str(checkpoint)})
        index = {"training_steps": 500, "source_training_steps": 1000, "runs": runs}
        probe.write_json(root / "status.json", {"status": "execution_verified"})
        probe.write_json(root / "step500/training_index.json", index)
        provenance = {"status": "succeeded", "training_steps": 500,
                      "sampling": protocol["inference"],
                      "training_index_sha256": probe.sha256(root / "step500/training_index.json")}
        probe.write_json(root / "step500/inference/provenance.json", provenance)
        return index, provenance

    def test_source_preserves_step_checks_and_selects_only_four_materials(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index, provenance = self.source_fixture(root)
            with patch.object(probe.infer, "verify_inputs") as verify:
                actual, selections, locks = probe.verify_source(root, 500)
                self.assertEqual(actual, index)
                self.assertEqual(len(selections), 4)
                self.assertTrue(all(item["branch"] == "material" for item in selections))
                self.assertEqual({(item["method"], item["concept_id"]) for item in selections},
                                 {(method, material) for method in probe.infer.METHODS
                                  for material in probe.infer.target_ids(probe.data.read_protocol(), "material")})
                verify.assert_called_once_with(index, probe.data.read_protocol(), 500)
                self.assertEqual(len(locks), 3)
                probe.write_json(root / "status.json", {"status": "running"})
                with self.assertRaisesRegex(ValueError, "completed source"):
                    probe.verify_source(root, 500)
                probe.write_json(root / "status.json", {"status": "execution_verified"})
                for field, replacement in (("training_steps", 750), ("status", "failed"),
                                           ("sampling", {**provenance["sampling"], "guidance_scale": 7.5})):
                    probe.write_json(root / "step500/inference/provenance.json", {**provenance, field: replacement})
                    with self.assertRaisesRegex(ValueError, "step or sampling"):
                        probe.verify_source(root, 500)
                for field, replacement in (("training_steps", 750), ("source_training_steps", 750)):
                    probe.write_json(root / "step500/training_index.json", {**index, field: replacement})
                    probe.write_json(root / "step500/inference/provenance.json", {
                        **provenance, "training_index_sha256": probe.sha256(root / "step500/training_index.json")})
                    with self.assertRaisesRegex(ValueError, "step or sampling"):
                        probe.verify_source(root, 500)
                probe.write_json(root / "step500/training_index.json", index)
                probe.write_json(root / "step500/inference/provenance.json", provenance)
                path = Path(selections[-1]["checkpoint"]) / "adaptation_config.json"
                config = probe.read_json(path)
                for field, replacement in (("modifier_tokens", ["<T*>"]), ("rank", 32), ("alpha", 32)):
                    probe.write_json(path, {**config, field: replacement})
                    with self.assertRaisesRegex(ValueError, "Material adapter"):
                        probe.verify_source(root, 500)

    def test_run_routes_native_and_single_material_with_unchanged_sampling(self):
        protocol = probe.data.read_protocol()
        selections = [{"method": method, "branch": "material", "concept_id": material, "token": "<M*>"}
                      for method in probe.infer.METHODS for material in probe.infer.target_ids(protocol, "material")]
        calls, loaded, generators = [], [], []
        mask = object()

        class Generator:
            def __init__(self, device):
                self.device = device
                generators.append(self)

            def manual_seed(self, seed):
                self.seed = seed
                return self

        def load_pipeline(selection, device):
            loaded.append(selection)

            def pipeline(prompt, **kwargs):
                calls.append((selection, prompt, kwargs))
                return types.SimpleNamespace(images=[Image.new("RGB", (2, 2))], nsfw_content_detected=[False])
            return pipeline

        index = {"training_git": {"commit": "training"}, "git": {"commit": "old_inference"}}
        torch = types.SimpleNamespace(Generator=Generator, no_grad=contextlib.nullcontext,
                                      cuda=types.SimpleNamespace(empty_cache=Mock()))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new"
            with (patch.dict(sys.modules, {"torch": torch}),
                  patch.object(probe, "git_info", return_value={"commit": "probe"}),
                  patch.object(probe, "resolve_run_dir", side_effect=lambda path: path),
                  patch.object(probe, "verify_source", return_value=(index, selections, {})),
                  patch.object(probe.infer, "verify_inputs"),
                  patch.object(probe, "package_versions", return_value={}),
                  patch.object(probe, "load_pipeline", side_effect=load_pipeline),
                  patch.object(probe, "attention_masks", return_value={"modifier_token_mask": mask}) as masks,
                  patch.object(probe, "review_bundle", return_value=[])):
                probe.run(Path(temporary) / "source", output, 500, "cuda:2")
            self.assertEqual(loaded.count([]), 1)
            self.assertEqual([item[0] for item in loaded if item], selections)
            self.assertEqual(len(calls), 96)
            self.assertEqual(len(generators), 96)
            self.assertEqual(len({id(generator) for generator in generators}), 96)
            self.assertEqual(masks.call_count, 32)
            for selected, prompt, kwargs in calls:
                self.assertEqual(kwargs["num_inference_steps"], 100)
                self.assertEqual(kwargs["guidance_scale"], 3.5)
                self.assertEqual((kwargs["width"], kwargs["height"]), (512, 512))
                self.assertEqual(kwargs["generator"].device, "cuda:2")
                self.assertIn(kwargs["generator"].seed, (42, 43))
                self.assertEqual(bool(selected), "<M*>" in prompt)
                if selected:
                    self.assertEqual(len(selected), 1)
                    self.assertIs(kwargs["cross_attention_kwargs"]["modifier_token_mask"], mask)
                else:
                    self.assertNotIn("cross_attention_kwargs", kwargs)
            for call in masks.call_args_list:
                self.assertEqual(call.args[2:], (["<M*>"], 3.5))
            summary = probe.read_json(output / "summary.json")
            self.assertEqual(summary["status"], "succeeded")
            self.assertEqual(summary["by_condition"], {condition: 32 for condition in probe.CONDITIONS})
            self.assertEqual(summary["by_method"], {method: 48 for method in probe.infer.METHODS})

    def test_review_has_three_columns_and_excludes_filtered_from_blind_ratings(self):
        protocol = probe.data.read_protocol()
        rows = [{**row, "status": "ok"} for row in probe.build_rows(protocol, 750)]
        rows[0]["status"] = "safety_filtered"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "images").mkdir()
            for row in rows:
                Image.new("RGB", (2, 2)).save(output / row["image_path"])
            sheets = probe.review_bundle(output, rows, protocol)
            self.assertEqual(len(sheets), 8)
            with Image.open(output / sheets[0]) as image:
                self.assertEqual(image.width, 3 * 272)
            with (output / "blinded_ratings.csv").open(newline="", encoding="utf-8") as handle:
                ratings = list(csv.DictReader(handle))
            self.assertEqual(len(ratings), 95)
            blind_key = probe.read_json(output / "blinded_key.json")["samples"]
            filtered_id = next(row["blind_id"] for row in blind_key if row["status"] == "safety_filtered")
            self.assertNotIn(filtered_id, {row["blind_id"] for row in ratings})
            html = (output / "blinded_review.html").read_text(encoding="utf-8")
            self.assertEqual(html.count("<figure>"), 95)
            self.assertNotIn(rows[0]["image_path"], html)


if __name__ == "__main__":
    unittest.main()
