import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from src.methods.colorpeel_ice.material_evaluation_protocol import (
    build_manifest,
    read_protocol,
    validate_campaign,
)


REPO_ROOT = Path(__file__).parents[3]
PROTOCOL_PATH = (REPO_ROOT / "experiments" / "clevr_subject_color_material_3x3x2" /
                 "manifests" / "clevr_material_heldout_eval.json")
GENERATOR_PATH = REPO_ROOT / "scripts" / "methods" / "colorpeel_ice" / "generate_material_multiview.py"
SPEC = importlib.util.spec_from_file_location("material_generator", GENERATOR_PATH)
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class MaterialEvaluationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = read_protocol(PROTOCOL_PATH)

    def test_checkpoint_manifest_counts_and_intervention_groups(self):
        rows = build_manifest(self.protocol, fold_id="A", training_seed=42)
        self.assertEqual(len(rows), 360)
        self.assertEqual(sum(row["held_out"] for row in rows), 120)
        self.assertEqual(len({row["fixed_shape_color_group"] for row in rows}), 180)
        self.assertEqual(len({row["fixed_shape_material_group"] for row in rows}), 120)
        self.assertEqual(len({row["fixed_color_material_group"] for row in rows}), 120)

    def test_campaign_is_3240_with_locked_seen_heldout_counts(self):
        rows = []
        for fold in "ABC":
            for seed in (42, 43, 44):
                rows.extend({**row, "dtype": "float16", "safety_checker_disabled": True,
                             "safety_risk_acknowledged": True,
                             "protocol_fingerprint_sha256": self.protocol["_source_sha256"],
                             "model_fingerprint_sha256": f"{fold}-{seed}"}
                            for row in build_manifest(self.protocol, fold_id=fold, training_seed=seed))
        validated = validate_campaign(rows, self.protocol)
        self.assertEqual(len(validated), 3240)
        self.assertEqual(sum(row["held_out"] for row in validated), 1080)
        changed = json.loads(json.dumps(rows)); changed[0]["expected_material"] = "rubber"
        with self.assertRaisesRegex(ValueError, "locked field"):
            validate_campaign(changed, self.protocol)

    def test_generator_model_dir_requires_all_eight_tokens(self):
        self.assertEqual(len(GENERATOR.MODEL_ARTIFACTS), 9)
        self.assertIn("<m2*>.bin", GENERATOR.MODEL_ARTIFACTS)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(GENERATOR, "MODEL_ARTIFACTS", ("weights.bin", "m2.bin")):
                (root / "weights.bin").write_bytes(b"x")
                (root / "m2.bin").write_bytes(b"x")
                GENERATOR.validate_model_dir(root)
                (root / "m2.bin").unlink()
                with self.assertRaisesRegex(FileNotFoundError, "m2.bin"):
                    GENERATOR.validate_model_dir(root)

    def test_resume_rejects_protocol_fingerprint_mismatch(self):
        row = {**build_manifest(self.protocol, fold_id="A", training_seed=42, generation_seeds=[42])[0],
               "model_fingerprint_sha256": "model", "protocol_fingerprint_sha256": "protocol"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); image_path = root / row["image_path"]
            image_path.parent.mkdir(parents=True); Image.new("RGB", (512, 512)).save(image_path)
            status = {"id": row["id"], "status": "ok", "image_sha256": GENERATOR.sha256_file(image_path),
                      "model_fingerprint_sha256": "model", "protocol_fingerprint_sha256": "wrong"}
            status_path = root / "generation_status.jsonl"; GENERATOR.write_jsonl([status], status_path)
            args = type("Args", (), {"skip_existing": True, "output_dir": root, "status_path": status_path})()
            with self.assertRaisesRegex(RuntimeError, "protocol_fingerprint"):
                GENERATOR.pending_rows([row], args)


if __name__ == "__main__":
    unittest.main()
