import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from src.methods.colorpeel_ice import prepare_material_token_local_pilot as prepare
from src.methods.colorpeel_ice.multiview_render_contract import (
    EXPECTED_PROFILE, EXPECTED_PROFILE_V2, EXPECTED_PROFILE_V3, EXPECTED_PROFILE_V4,
    EXPECTED_PROFILE_V5, canonical_sha256,
)
from scripts.launch import colorpeel_run


ROOT = Path(__file__).parents[3]
EXPERIMENT = ROOT / "experiments" / "material_token_local_pilot_v1"
PROTOCOL = EXPERIMENT / "protocols" / "material_token_local_pilot_v1.json"
PROFILE = EXPERIMENT / "configs" / "render_profile.json"
RENDERER_PATH = ROOT / "scripts" / "methods" / "colorpeel_ice" / "render_clevr_multiview.py"
GENERATOR_PATH = ROOT / "scripts" / "methods" / "colorpeel_ice" / "generate_material_token_local_pilot.py"
EVALUATION = EXPERIMENT / "protocols" / "material_token_local_pilot_v1_evaluation.json"

RENDERER_SPEC = importlib.util.spec_from_file_location("material_pilot_renderer", RENDERER_PATH)
RENDERER = importlib.util.module_from_spec(RENDERER_SPEC)
assert RENDERER_SPEC.loader is not None
RENDERER_SPEC.loader.exec_module(RENDERER)

GENERATOR_SPEC = importlib.util.spec_from_file_location("material_pilot_generator", GENERATOR_PATH)
GENERATOR = importlib.util.module_from_spec(GENERATOR_SPEC)
assert GENERATOR_SPEC.loader is not None
GENERATOR_SPEC.loader.exec_module(GENERATOR)


class MaterialTokenLocalPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = prepare.validate_protocol(prepare.read_json(PROTOCOL))

    def test_versioned_profile_is_exact_and_preserves_historical_profiles(self):
        self.assertEqual(json.loads(PROFILE.read_text(encoding="utf-8")), EXPECTED_PROFILE_V5)
        self.assertEqual(RENDERER.validate_profile(EXPECTED_PROFILE_V5), EXPECTED_PROFILE_V5)
        for historical in (EXPECTED_PROFILE, EXPECTED_PROFILE_V2, EXPECTED_PROFILE_V3, EXPECTED_PROFILE_V4):
            self.assertEqual(RENDERER.validate_profile(historical), historical)

    def test_preview_and_full_grids_are_complete_and_renderer_valid(self):
        preview = prepare.build_render_requests(self.protocol, "preview")
        full = prepare.build_render_requests(self.protocol, "full")
        self.assertEqual(len(preview), 36)
        self.assertEqual(len(full), 72)
        self.assertEqual(preview, prepare.build_render_requests(self.protocol, "preview"))
        self.assertEqual({row["color"] for row in preview}, {"red", "blue"})
        self.assertEqual({row["lighting_condition"] for row in full}, set(prepare.LIGHTING))
        self.assertEqual({row["viewpoint"] for row in full}, set(prepare.VIEWPOINTS))
        self.assertTrue(all(row["material"] == "metal" and row["material_token"] == "<M*>" for row in full))
        self.assertTrue(all(row["renderer_profile_sha256"] == canonical_sha256(EXPECTED_PROFILE_V5) for row in full))
        RENDERER.validate_requests(preview, EXPECTED_PROFILE_V5)
        RENDERER.validate_requests(full, EXPECTED_PROFILE_V5)

    def test_preview_plan_stays_training_blocked_and_has_review_checklist(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "preview"
            status = prepare.plan(self.protocol, "preview", output)
            rows = prepare.read_jsonl(output / "render_requests.jsonl")
            checklist = (output / "preview_review_checklist.md").read_text(encoding="utf-8")
        self.assertEqual(status["status"], "planned_pending_human_preview")
        self.assertEqual(status["training_authorization"], "blocked_pending_preview_human_review")
        self.assertEqual(len(rows), 36)
        self.assertIn("warm_top", checklist)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(prepare.ProtocolError):
                prepare.plan(self.protocol, "full", Path(temporary) / "full")

    def test_launcher_refuses_blocked_material_training_config(self):
        config = {"stage": "train", "status": "blocked_pending_preview_human_review",
                  "run": {"study": "material_token_local_pilot_v1", "variant": "pilot", "seed": 42},
                  "args": {}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            prepare.write_json(path, config)
            with self.assertRaisesRegex(ValueError, "remains blocked"):
                colorpeel_run.read_config(path)

    def mock_render(self, root, requests):
        root.mkdir()
        contract = {"profile_id": EXPECTED_PROFILE_V5["profile_id"],
                    "requests_sha256": canonical_sha256(requests)}
        prepare.write_json(root / "render_contract.json", contract)
        records = []
        for index, request in enumerate(requests):
            image = root / "images" / f"{index:03d}.jpg"
            mask = root / "masks" / f"{index:03d}.png"
            background_mask = root / "background_masks" / f"{index:03d}.png"
            scene = root / "scenes" / f"{index:03d}.json"
            for path in (image, mask, background_mask, scene):
                path.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"mock image")
            mask.write_bytes(b"mock mask")
            background_mask.write_bytes(b"mock background mask")
            prepare.write_json(scene, {"index": index})
            records.append({**request, "image": image.relative_to(root).as_posix(),
                            "mask": mask.relative_to(root).as_posix(),
                            "background_mask": background_mask.relative_to(root).as_posix(),
                            "scene_json": scene.relative_to(root).as_posix(),
                            "render_contract_sha256": canonical_sha256(contract),
                            "artifact_sha256": {"image": prepare.sha256(image),
                                                "mask": prepare.sha256(mask),
                                                "background_mask": prepare.sha256(background_mask),
                                                "scene_json": prepare.sha256(scene)}})
        prepare.write_jsonl(root / "renderer_realization.jsonl", records)
        return records

    def test_full_staging_writes_one_material_only_concept_and_hashes_each_source(self):
        requests = prepare.build_render_requests(self.protocol, "full")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preview_root, render_root, staging = root / "preview", root / "render", root / "staging"
            self.mock_render(preview_root, prepare.build_render_requests(self.protocol, "preview"))
            approval = root / "review.json"
            prepare.write_json(approval, {"verdict": "pass", "reviewer": "human", "reviewed_at": "2026-09-23",
                                          "renderer_realization_sha256": prepare.sha256(preview_root / "renderer_realization.jsonl")})
            bad_approval = root / "bad_review.json"
            prepare.write_json(bad_approval, {"verdict": "pass", "reviewer": "human", "reviewed_at": "2026-09-23",
                                              "renderer_realization_sha256": "0" * 64})
            with self.assertRaises(prepare.ProtocolError):
                prepare.plan(self.protocol, "full", root / "unapproved_full", preview_root, bad_approval)
            self.assertEqual(prepare.plan(self.protocol, "full", root / "approved_full", preview_root, approval)["request_count"], 72)
            self.mock_render(render_root, requests)
            mask_module = types.ModuleType("src.train.instance_mask_utils")
            mask_module.load_latent_instance_mask = lambda *args: None
            with patch.dict(sys.modules, {"src.train.instance_mask_utils": mask_module}):
                status = prepare.stage_training_assets(self.protocol, render_root, staging, preview_root, approval)
            train_config = {
                "stage": "train", "status": "authorized_after_preview_review",
                "run": {"study": "material_token_local_pilot_v1", "variant": "pilot", "seed": 42},
                "args": {"concepts_list": str(staging / "concepts.json")},
                "data_manifest": str(staging / "training_assets_manifest.jsonl"),
                "material_pilot_authorization": {
                    "preview_root": str(preview_root), "review_record": str(approval),
                    "staging_root": str(staging),
                    "staging_provenance_sha256": prepare.sha256(staging / "staging_provenance.json"),
                },
            }
            colorpeel_run.validate_material_pilot_train_inputs(train_config, {})
            concepts = prepare.read_json(staging / "concepts.json")
            manifest = prepare.read_jsonl(staging / "training_assets_manifest.jsonl")
            Path(manifest[0]["staged_image"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "staged image differs"):
                colorpeel_run.validate_material_pilot_train_inputs(train_config, {})
        self.assertEqual(status["image_count"], 72)
        self.assertEqual(concepts[0]["instance_prompt"], ["a photo of an object made of <M*>"])
        self.assertIn("instance_mask_dir", concepts[0])
        self.assertEqual(len(manifest), 72)
        self.assertTrue(all(len(row["image_sha256"]) == 64 and len(row["mask_sha256"]) == 64 for row in manifest))

    def test_evaluation_manifest_covers_the_four_required_interventions(self):
        protocol = GENERATOR.validate_protocol(GENERATOR.read_json(EVALUATION))
        rows = GENERATOR.build_manifest(protocol)
        self.assertEqual(len(rows), 60)
        self.assertEqual(
            {row["group"] for row in rows},
            {"seen_reconstruction", "color_invariance", "unseen_object_transfer", "lighting_robustness"},
        )
        self.assertTrue(all(row["modifier_token"] == "<M*>" for row in rows))

    def test_evaluation_checkpoint_is_bound_to_completed_material_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            model = run / "checkpoints"
            model.mkdir()
            manifest = run / "manifest.json"
            prepare.write_json(manifest, {"status": "succeeded", "stage": "train", "run": {
                "study": "material_token_local_pilot_v1", "variant": "standalone_metal_token_local_kv_5000",
            }})
            hashes = {"weights": "a" * 64, "adaptation": "b" * 64, "token": "c" * 64}
            lock = run / "checkpoint_lock.json"
            prepare.write_json(lock, {"schema": "material_token_local_checkpoint_lock/v1",
                                      "evaluation_protocol_sha256": GENERATOR.sha256(EVALUATION),
                                      "run_manifest_path": str(manifest),
                                      "run_manifest_sha256": GENERATOR.sha256(manifest),
                                      "model_dir": str(model), "checkpoint_sha256": hashes,
                                      "base_model": "CompVis/stable-diffusion-v1-4"})
            self.assertEqual(len(GENERATOR.validate_checkpoint_lock(
                lock, EVALUATION, model, hashes, "CompVis/stable-diffusion-v1-4")), 64)
            with self.assertRaisesRegex(ValueError, "checkpoint lock"):
                GENERATOR.validate_checkpoint_lock(
                    lock, EVALUATION, model, {**hashes, "weights": "d" * 64},
                    "CompVis/stable-diffusion-v1-4")


if __name__ == "__main__":
    unittest.main()
