import contextlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from experiments.texture_material_disentangled_v1 import texture_probe as probe


class TextureProbeTests(unittest.TestCase):
    def test_matched_three_arm_matrix_omits_material_and_preserves_contexts(self):
        protocol = probe.data.read_protocol()
        rows = probe.build_rows(protocol, 500)
        self.assertEqual(len(rows), 96)
        self.assertEqual(len({row["id"] for row in rows}), 96)
        groups, bases = {}, {}
        for row in rows:
            prompt = row["prompt"]
            for cue in ("<M*>", "made of", "metal", "plastic", "rough", "smooth", "polished"):
                self.assertNotIn(cue, prompt)
            context = next(item for item in protocol["dataset"]["heldout_contexts"]
                           if item["id"] == row["context"])
            expected = f"a photo of a {context['color']} {context['shape']}"
            if row["condition"] == "literal":
                expected += f" with {protocol['textures'][row['texture']]['literal']} texture"
            elif row["condition"] == "learned_texture":
                expected += " with <T*> texture"
            expected += f", {protocol['dataset']['light_phrases'][context['light']]}"
            self.assertEqual(prompt, expected)
            self.assertIn(row["seed"], protocol["inference"]["seeds"])
            key = tuple(row[field] for field in ("method", "texture", "context", "seed"))
            groups.setdefault(key, []).append(row)
            if row["condition"] == "base":
                bases.setdefault((row["context"], row["seed"]), set()).add(prompt)
        self.assertEqual(len(groups), 32)
        self.assertTrue(all(len(prompts) == 1 for prompts in bases.values()))
        for group in groups.values():
            self.assertEqual({row["condition"] for row in group}, set(probe.CONDITIONS))
            for field in ("shape", "color", "light", "seed"):
                self.assertEqual(len({row[field] for row in group}), 1)
        for step in (750, 1000):
            self.assertEqual([{**row, "training_steps": 500}
                              for row in probe.build_rows(protocol, step)], rows)
        with self.assertRaises(ValueError):
            probe.build_rows(protocol, 300)

    def mocked_run(self, root, mutate_source=False):
        protocol = probe.data.read_protocol()
        selections = [{"method": method, "branch": "texture", "concept_id": texture, "token": "<T*>"}
                      for method in probe.infer.METHODS
                      for texture in probe.infer.target_ids(protocol, "texture")]
        source = root / "old_source"
        source.mkdir()
        marker = source / "preserved.txt"
        marker.write_text("old source", encoding="utf-8")
        locks = {str(marker): probe.data.sha256(marker)}
        references = {(item["method"], item["concept_id"]): {"images": ["actual staged image"]}
                      for item in selections}
        index = {"training_git": {"commit": "training"}, "git": {"commit": "old_inference"}}
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
                if mutate_source and len(calls) == 96:
                    marker.write_text("changed externally", encoding="utf-8")
                return types.SimpleNamespace(images=[Image.new("RGB", (2, 2))],
                                             nsfw_content_detected=[False])
            return pipeline

        torch = types.SimpleNamespace(Generator=Generator, no_grad=contextlib.nullcontext,
                                      cuda=types.SimpleNamespace(empty_cache=Mock()))
        output = root / "new"
        with (patch.dict(sys.modules, {"torch": torch}),
              patch.object(probe, "git_info", return_value={"commit": "probe"}),
              patch.object(probe, "resolve_run_dir", side_effect=lambda path: path),
              patch.object(probe, "verify_source", return_value=(index, selections, locks, references)) as verify,
              patch.object(probe, "package_versions", return_value={}),
              patch.object(probe, "load_pipeline", side_effect=load_pipeline),
              patch.object(probe, "attention_masks", return_value={"modifier_token_mask": mask}) as masks,
              patch.object(probe, "review_bundle", return_value=[]) as review):
            if mutate_source:
                with self.assertRaisesRegex(ValueError, "source checkpoint, training images or checkout changed"):
                    probe.run(source, output, 500, "cuda:2")
                review.assert_not_called()
            else:
                probe.run(source, output, 500, "cuda:2")
                self.assertEqual(review.call_args.args[3], references)
                self.assertEqual(marker.read_text(encoding="utf-8"), "old source")
            self.assertEqual(verify.call_count, 2)
        return output, selections, calls, loaded, generators, masks, mask

    def test_run_loads_only_texture_and_keeps_sampler_and_seed_pairing(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, selections, calls, loaded, generators, masks, mask = self.mocked_run(Path(temporary))
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
                self.assertEqual(bool(selected), "<T*>" in prompt)
                if selected:
                    self.assertEqual(len(selected), 1)
                    self.assertEqual(selected[0]["branch"], "texture")
                    self.assertIs(kwargs["cross_attention_kwargs"]["modifier_token_mask"], mask)
                else:
                    self.assertNotIn("cross_attention_kwargs", kwargs)
            for call in masks.call_args_list:
                self.assertEqual(call.args[2:], (["<T*>"], 3.5))
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "succeeded")
            self.assertEqual(summary["by_condition"], {condition: 32 for condition in probe.CONDITIONS})
            self.assertEqual(summary["by_method"], {method: 48 for method in probe.infer.METHODS})
            ledger = [json.loads(line) for line in (output / "generation_status.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["material_adapter_loaded"] is False for row in ledger))
            self.assertTrue(all(row["texture_adapter_loaded"] == (row["condition"] == "learned_texture")
                                for row in ledger))
            provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
            self.assertFalse(provenance["material_adapter_loaded"])
            self.assertFalse(provenance["retraining"])
            self.assertEqual(provenance["sampling"]["scheduler"], "PNDM")

    def test_source_change_fails_before_publishing_review_or_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, *_ = self.mocked_run(Path(temporary), mutate_source=True)
            self.assertFalse((output / "summary.json").exists())
            provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["status"], "failed")

    def test_existing_output_is_rejected_without_writes_or_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "existing"
            output.mkdir()
            marker = output / "old.txt"
            marker.write_text("preserve", encoding="utf-8")
            with (patch.object(probe, "resolve_run_dir", side_effect=lambda path: path),
                  patch.object(probe, "verify_source") as verify,
                  patch.object(probe, "load_pipeline") as load):
                with self.assertRaisesRegex(ValueError, "new external output"):
                    probe.run(root / "source", output, 500)
            verify.assert_not_called()
            load.assert_not_called()
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")
            self.assertEqual(list(output.iterdir()), [marker])


if __name__ == "__main__":
    unittest.main()
