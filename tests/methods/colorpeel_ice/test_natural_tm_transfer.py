"""Ordinary-object transfer keeps the trained branches and matched controls isolated."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from experiments.natural_texture_material_v1 import probe, probe_review, run_probe
from tests.methods.colorpeel_ice import test_natural_tm_probe as neutral
from tests.methods.colorpeel_ice import test_natural_tm_probe_review as neutral_review


OBJECTS = ("chair", "mug", "vase", "sofa")


class NaturalTransferTests(unittest.TestCase):
    def test_exact_three_arm_object_prompts_and_checkpoint_matrix_preserve_neutral_protocol(self):
        settings = neutral.protocol()
        original = copy.deepcopy(settings)
        neutral_rows = probe.build_rows(settings, 500)
        contexts = probe.prompt_contexts(settings, transfer=True)
        self.assertEqual([item["id"] for item in contexts], list(OBJECTS))
        self.assertTrue(all(item["shape"] == item["id"] and item["color"] is None
                            and item["light"] is None for item in contexts))
        transfer = json.loads(probe.TRANSFER_PROMPTS.read_text(encoding="utf8"))
        for branch in ("texture", "material"):
            self.assertEqual(transfer["literal_" + branch], neutral.LITERALS[branch])
        rows = probe.build_rows(settings, 500, transfer=True)
        self.assertEqual(len(rows), 288)
        self.assertEqual(len({row["id"] for row in rows}), 288)
        groups = {}
        for row in rows:
            self.assertIn(row["context"], OBJECTS)
            self.assertEqual(row["shape"], row["context"])
            self.assertIsNone(row["color"])
            self.assertIsNone(row["light"])
            branch, condition = row["branch"], row["condition"]
            expected = "a photo of a " + row["context"]
            if condition != "base":
                value = (neutral.LITERALS[branch][row["source_id"]] if condition == "literal"
                         else "<T*>" if branch == "texture" else "<M*>")
                expected += f" with {value} texture" if branch == "texture" else f" made of {value}"
            self.assertEqual(row["prompt"], expected)
            forbidden = ("<M*>", "made of", "metal", "wood", "polished") if branch == "texture" else (
                "<T*>", "texture", "speckled", "grain", "striped")
            for cue in forbidden:
                self.assertNotIn(cue, row["prompt"])
            key = tuple(row[field] for field in ("method", "concept_id", "context", "seed"))
            groups.setdefault(key, []).append(row)
        self.assertEqual(len(groups), 96)
        self.assertTrue(all({item["condition"] for item in group} == {
            "base", "literal", "learned_" + group[0]["branch"]} for group in groups.values()))
        for step in (750, 1000):
            self.assertEqual([{**row, "training_steps": 500}
                              for row in probe.build_rows(settings, step, transfer=True)], rows)
        self.assertEqual(settings, original)
        self.assertEqual(probe.build_rows(settings, 500), neutral_rows)
        self.assertEqual(probe.build_rows(settings, 500, transfer=False), neutral_rows)

    def test_transfer_run_uses_single_branch_native_controls_fresh_seeds_and_actual_references(self):
        fixture = neutral.NaturalProbeTests()
        original_run = probe.run
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(probe, "run", side_effect=lambda *args, **kwargs:
                              original_run(*args, **kwargs, transfer=True)):
                output, selections, calls, loaded, generators, masks, mask = fixture.mocked_run(Path(temporary))
            rows = [json.loads(line) for line in (output / "generation_status.jsonl").read_text().splitlines()]
            expected = probe.build_rows(neutral.protocol(), 500, transfer=True)
            self.assertEqual({row["id"]: row["prompt"] for row in rows},
                             {row["id"]: row["prompt"] for row in expected})
            self.assertEqual(loaded.count([]), 1)
            self.assertEqual([item[0] for item in loaded if item], selections)
            self.assertEqual(len(generators), 288)
            self.assertEqual(len({id(item) for item in generators}), 288)
            self.assertEqual(masks.call_count, 96)
            for selection, prompt, kwargs in calls:
                self.assertEqual((kwargs["num_inference_steps"], kwargs["guidance_scale"],
                                  kwargs["width"], kwargs["height"]), (100, 3.5, 512, 512))
                self.assertIn(kwargs["generator"].seed, (42, 43))
                if selection:
                    self.assertEqual(len(selection), 1)
                    self.assertEqual(prompt.count(selection[0]["token"]), 1)
                    self.assertIs(kwargs["cross_attention_kwargs"]["modifier_token_mask"], mask)
                else:
                    self.assertNotIn("cross_attention_kwargs", kwargs)
            provenance = json.loads((output / "provenance.json").read_text())
            self.assertEqual(provenance["protocol_sha256"], probe.schema.protocol_sha256(neutral.protocol()))
            self.assertEqual(provenance["prompt_protocol_sha256"], probe.schema.sha256(probe.TRANSFER_PROMPTS))
            self.assertEqual(provenance["prompt_protocol"], json.loads(probe.TRANSFER_PROMPTS.read_text()))
            self.assertEqual(provenance["probe_suite"], "ordinary_object_transfer")
            self.assertFalse(provenance["retraining"])

    def test_parent_dispatches_transfer_to_every_step_and_freezes_selected_prompt_protocol(self):
        original_run = run_probe.run
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(run_probe, "run", side_effect=lambda *args, **kwargs:
                              original_run(*args, **kwargs, transfer=True)):
                output, source, calls, audit = neutral.NaturalProbeBatchTests().mocked_batch(Path(temporary))
            for (command, kwargs), step in zip(calls, (500, 750, 1000)):
                self.assertIn("--transfer", command)
                self.assertEqual(command[command.index("--step") + 1], str(step))
                self.assertEqual(command[command.index("--source-run") + 1], str(source))
            selected = json.loads(probe.TRANSFER_PROMPTS.read_text(encoding="utf8"))
            receipts = [json.loads(path.read_text(encoding="utf8"))
                        for path in output.glob("*protocol.json")]
            self.assertIn(selected, receipts)
            state = json.loads((output / "status.json").read_text())
            self.assertEqual(state["status"], "execution_verified")
            self.assertEqual(state["generated_images"], 864)
            self.assertEqual(state["probe_suite"], "ordinary_object_transfer")
            self.assertEqual(state["prompt_protocol_sha256"], probe.schema.sha256(probe.TRANSFER_PROMPTS))
            self.assertEqual(json.loads((output / "training_audit.json").read_text()), audit)

    def test_transfer_contact_sheets_keep_all_twelve_actual_training_reference_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings, _, references, _ = neutral_review.NaturalProbeReviewTests().fixture(root)
            rows = probe.build_rows(settings, 750, transfer=True)
            for row in rows:
                row["status"] = "ok"
                Image.new("RGB", (16, 16), (10, 20, 30)).save(root / row["image_path"])
            drawn_text, original_text = [], ImageDraw.ImageDraw.text

            def draw_text(draw, position, text, *args, **kwargs):
                drawn_text.append(text)
                return original_text(draw, position, text, *args, **kwargs)

            with patch.object(ImageDraw.ImageDraw, "text", new=draw_text):
                sheets = probe_review.review_bundle(root, rows, settings, references,
                                                    contexts=probe.prompt_contexts(settings, transfer=True))
            self.assertEqual(len(sheets), 24)
            self.assertTrue(all((root / sheet).is_file() for sheet in sheets))
            self.assertTrue(set(OBJECTS) <= set(drawn_text))
            groups = json.loads((root / "training_references.json").read_text())["groups"]
            self.assertEqual(len(groups), 12)
            for group in groups:
                self.assertEqual(group["images"], references[group["method"], group["concept_id"]]["images"])
                self.assertEqual(len(group["images"]), 16 if group["branch"] == "texture" else 8)
            blinded = json.loads((root / "blinded_key.json").read_text())["samples"]
            self.assertEqual({row["context"] for row in blinded}, set(OBJECTS))


if __name__ == "__main__":
    unittest.main()
