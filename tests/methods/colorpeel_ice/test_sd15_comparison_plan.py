from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.sd15_token_local_kv_v1.compare import (
    INFERENCE_PROTOCOL, plan_jobs, reference_rows, selection_key,
)
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256


class ComparisonPlanTests(unittest.TestCase):
    def test_fullrank_uses_its_own_subject_step_and_fixed_material_5000(self):
        for step in (300, 500, 700, 1000, 2000):
            for condition in ("subject_only", "material_only", "subject_material", "literal_metal"):
                with self.subTest(step=step, condition=condition):
                    # Row metadata must never override the locked historical M selection.
                    self.assertEqual(selection_key(f"fullrank_comparison_{step}", {
                        "condition": condition, "material_step": 3000}),
                        ("subject_fullrank", step, "material_ground_reflection_fullrank", 5000))
            self.assertEqual(selection_key(f"fullrank_transfer_{step}", {}),
                             ("subject_fullrank", step, None, None))
        self.assertEqual(selection_key("fullrank_material", {"material_step": 1000}),
                         (None, None, "material_ground_reflection_fullrank", 5000))

    def test_lora_subject_excludes_the_historical_rank4_material_arm(self):
        self.assertIsNone(selection_key("lora_subject", {
            "condition": "subject_material_token", "step": 1000}))
        for condition in ("subject_only", "subject_literal_material"):
            self.assertEqual(selection_key("lora_subject", {"condition": condition, "step": 2000}),
                             ("subject_lora_r64", 2000, None, None))
        self.assertEqual(selection_key("lora_transfer_3000", {"subject_step": 3000}),
                         ("subject_lora_r64", 3000, None, None))
        with self.assertRaises(ValueError):
            selection_key("lora_subject", {"condition": "unknown", "step": 1000})

    def test_material_transfer_and_composition_preserve_distinct_adapter_steps(self):
        for material in ("mailbox", "metal_spoon", "wood_spoon"):
            source = f"lora_material_{material}"
            recipe = f"material_{material}_lora_r64"
            self.assertEqual(selection_key(source, {
                "family": "transfer", "material_step": 600, "subject_step": 3000}),
                (None, None, recipe, 600))
            self.assertEqual(selection_key(source, {
                "family": "composition", "material_step": 1000, "subject_step": 2000}),
                ("subject_lora_r64", 2000, recipe, 1000))
        with self.assertRaises(ValueError):
            selection_key("lora_material_mailbox", {"family": "unknown", "material_step": 600})

    def fixture(self, root):
        directory = root / "historical"
        (directory / "images").mkdir(parents=True)
        image = directory / "images/old.png"
        image.write_bytes(b"immutable historical image")
        rows = [{"id": "old-1", "prompt": "a photo of <S*> mailbox", "seed": 42,
                 "num_inference_steps": 100, "guidance_scale": 3.5,
                 "image_path": "images/old.png"}]
        ledger = [{**rows[0], "status": "ok", "nsfw_content_detected": False,
                   "image_sha256": sha256(image)}]
        self.write_jsonl(directory / "generation_manifest.jsonl", rows)
        self.write_jsonl(directory / "generation_status.jsonl", ledger)
        (directory / "provenance.json").write_text('{"status":"succeeded"}', encoding="utf-8")
        return directory, rows, ledger

    def write_jsonl(self, path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def reference(self, directory):
        return {"id": "fixture", "directory": directory.name, "files": {
            name: sha256(directory / name) for name in (
                "generation_manifest.jsonl", "generation_status.jsonl", "provenance.json")}}

    def test_historical_rows_capture_exact_pairing_without_writing_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory, rows, _ = self.fixture(root)
            reference = self.reference(directory)
            result = reference_rows(reference, root)
            self.assertEqual(result[0]["prompt"], rows[0]["prompt"])
            self.assertEqual(result[0]["seed"], rows[0]["seed"])
            self.assertEqual(result[0]["reference_image"], str(directory / "images/old.png"))
            self.assertEqual(result[0]["reference_sha256"], sha256(directory / "images/old.png"))
            self.assertIs(result[0]["reference_filtered"], False)
            self.assertEqual(self.reference(directory), reference)

    def test_historical_hash_ledger_and_prompt_drift_are_rejected(self):
        for failure in ("file_hash", "image_hash", "missing_ledger", "duplicate_ledger",
                        "prompt_mismatch", "sampling_drift", "missing_safety", "failed_provenance"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                directory, rows, ledger = self.fixture(root)
                reference = self.reference(directory)
                if failure == "file_hash":
                    (directory / "provenance.json").write_text('{"status":"succeeded","extra":1}')
                elif failure == "image_hash":
                    (directory / "images/old.png").write_bytes(b"changed image")
                else:
                    if failure == "missing_ledger":
                        ledger = []
                    elif failure == "duplicate_ledger":
                        ledger *= 2
                    elif failure == "prompt_mismatch":
                        ledger[0]["prompt"] = "a different mailbox prompt"
                    elif failure == "sampling_drift":
                        rows[0]["num_inference_steps"] = 50
                        ledger[0]["num_inference_steps"] = 50
                    elif failure == "missing_safety":
                        del ledger[0]["nsfw_content_detected"]
                    elif failure == "failed_provenance":
                        (directory / "provenance.json").write_text('{"status":"failed"}')
                    self.write_jsonl(directory / "generation_manifest.jsonl", rows)
                    self.write_jsonl(directory / "generation_status.jsonl", ledger)
                    # Refresh outer file locks to exercise the independent row/ledger checks.
                    reference = self.reference(directory)
                with self.assertRaises(ValueError):
                    reference_rows(reference, root)

    def synthetic_history(self, protocol):
        histories = {}

        def add(source, prompt, seed, **metadata):
            rows = histories.setdefault(source, [])
            rows.append({"id": f"{source}-{len(rows)}", "prompt": prompt, "seed": seed,
                         "image_path": f"old/{len(rows):04d}.png", "num_inference_steps": 100,
                         "guidance_scale": 3.5, **metadata})

        historical = {item["payload"]["schema"]: item["payload"]
                      for item in protocol["historical_protocols"]}
        fullrank = historical["subject_token_local_kv_balanced_step_ablation_inference/v1"]
        for step in fullrank["subject"]["snapshot_steps"]:
            for group in fullrank["comparison"]["groups"]:
                for condition, prompt in group["prompts"].items():
                    for seed in fullrank["comparison"]["sampling"]["seeds"]:
                        add(f"fullrank_comparison_{step}", prompt, seed, condition=condition,
                            subject_step=step, material_step=5000)
            for prompt_index in range(fullrank["transfer"]["prompt_count"]):
                for seed in fullrank["transfer"]["sampling"]["seeds"]:
                    add(f"fullrank_transfer_{step}", f"<S*> transfer prompt {prompt_index}", seed,
                        subject_step=step)
        material = historical["material_token_local_pilot_evaluation/v1"]
        for group in material["groups"]:
            for prompt in group["prompts"]:
                for seed in material["sampling"]["seeds"]:
                    add("fullrank_material", prompt, seed)
        subject = historical["lora_kv_subject_r64_a64_kvlr5em5_step_ablation/v1"]
        for step in subject["snapshot_steps"]:
            for group in subject["groups"]:
                for condition in subject["conditions"]:
                    suffix = {"subject_only": "", "subject_literal_material": " made of metal",
                              "subject_material_token": " made of <M*>"}[condition]
                    for seed in subject["sampling"]["seeds"]:
                        add("lora_subject", group["subject_prompt"] + suffix, seed,
                            condition=condition, step=step)
        lora = historical["natural_material_fullmaps_lora_r64_a64_inference/v1"]
        for step in lora["subject"]["snapshot_steps"]:
            for prompt_index in range(fullrank["transfer"]["prompt_count"]):
                for seed in fullrank["transfer"]["sampling"]["seeds"]:
                    add(f"lora_transfer_{step}", f"<S*> transfer prompt {prompt_index}", seed,
                        subject_step=step)
        for material, literal in lora["material"]["literal_material"].items():
            for step in lora["material"]["snapshot_steps"]:
                for object_name in lora["transfer"]["objects"]:
                    for arm in lora["transfer"]["arms"]:
                        suffix = {"base": "", "literal": f" made of {literal}",
                                  "token": " made of <M*>"}[arm]
                        for seed in lora["transfer"]["sampling"]["seeds"]:
                            add(f"lora_material_{material}", f"a photo of a {object_name}" + suffix,
                                seed, family="transfer", material_step=step)
                for subject_step in lora["subject"]["snapshot_steps"]:
                    for group in lora["composition"]["groups"]:
                        for seed in lora["composition"]["sampling"]["seeds"]:
                            add(f"lora_material_{material}", group["subject_prompt"] + " made of <M*>",
                                seed, family="composition", material_step=step, subject_step=subject_step)
        return histories

    def test_full_historical_matrix_builds_66_jobs_and_2560_exact_replays(self):
        protocol = read_json(INFERENCE_PROTOCOL)
        histories = self.synthetic_history(protocol)
        audits = {}
        for recipe in ("subject_fullrank", "material_ground_reflection_fullrank", "subject_lora_r64",
                       "material_mailbox_lora_r64", "material_metal_spoon_lora_r64",
                       "material_wood_spoon_lora_r64"):
            steps = ((300, 500, 700, 1000, 2000) if recipe == "subject_fullrank"
                     else (5000,) if recipe == "material_ground_reflection_fullrank"
                     else (600, 1000, 2000, 3000))
            audits[recipe] = {"checkpoints": {str(step): {
                "recipe": recipe, "step": step, "token": "<S*>" if recipe.startswith("subject") else "<M*>",
                "method": "fullrank" if recipe.endswith("fullrank") else "lora_r64"}
                for step in steps}}
        original = deepcopy(histories)
        with patch("experiments.sd15_token_local_kv_v1.compare.reference_rows",
                   side_effect=lambda reference, root: histories[reference["id"]]):
            jobs = plan_jobs(protocol, audits, Path("new-output"))
        self.assertEqual((len(jobs), sum(len(job["rows"]) for job in jobs)), (66, 2560))
        self.assertEqual(histories, original)
        self.assertEqual(len({job["id"] for job in jobs}), 66)
        actual_counts = {}
        for job in jobs:
            method = job["selections"][0]["method"]
            actual_counts[method] = actual_counts.get(method, 0) + len(job["rows"])
            source = {row["id"]: row for row in histories[job["reference"]["id"]]}
            for index, row in enumerate(job["rows"]):
                self.assertEqual(row["image_path"], f"images/{index:04d}.png")
                self.assertEqual({key: value for key, value in row.items()
                                  if key not in ("image_path", "source_id")},
                                 {key: value for key, value in source[row["source_id"]].items()
                                  if key != "image_path"})
                if job["reference"]["id"] == "lora_subject":
                    self.assertNotEqual(row["condition"], "subject_material_token")
        self.assertEqual(actual_counts, {"fullrank": 940, "lora_r64": 1620})
        histories["lora_subject"][0]["prompt"] += " made of <M*>"
        with patch("experiments.sd15_token_local_kv_v1.compare.reference_rows",
                   side_effect=lambda reference, root: histories[reference["id"]]):
            with self.assertRaisesRegex(ValueError, "unselected adapter"):
                plan_jobs(protocol, audits, Path("new-output"))


if __name__ == "__main__":
    unittest.main()
