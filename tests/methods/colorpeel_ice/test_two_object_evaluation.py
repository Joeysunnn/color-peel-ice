import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.methods.colorpeel_ice.two_object_evaluation_protocol import (
    build_manifest,
    file_sha256,
    read_protocol,
)


REPO_ROOT = Path(__file__).parents[3]


def load_script(name):
    path = REPO_ROOT / "scripts" / "methods" / "colorpeel_ice" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATION = load_script("generate_two_object")
BUNDLE = load_script("bundle_two_object_evaluation")


class TwoObjectEvaluationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = read_protocol()
        cls.rows = build_manifest(cls.protocol)

    def test_manifest_has_balanced_seen_and_unseen_position_swaps(self):
        self.assertEqual(len(self.rows), 720)
        self.assertEqual(sum(row["pair_group"] == "seen" for row in self.rows), 360)
        self.assertEqual(sum(row["pair_group"] == "unseen" for row in self.rows), 360)
        self.assertEqual(len({row["scene_id"] for row in self.rows}), 36)
        seen_pairs, unseen_pairs = set(), set()
        appearances = {}
        for row in self.rows:
            self.assertIn(" on the left and ", row["prompt"])
            self.assertIn(" on the right", row["prompt"])
            self.assertEqual(row["prompt"].count("<"), 6)
            if row["generation_seed"] != 42:
                continue
            pair = frozenset((row["left"]["state_id"], row["right"]["state_id"]))
            {"seen": seen_pairs, "unseen": unseen_pairs}[row["pair_group"]].add(pair)
            for side in ("left", "right"):
                key = (row["pair_group"], row[side]["state_id"], side)
                appearances[key] = appearances.get(key, 0) + 1
        self.assertEqual(len(seen_pairs), 9)
        self.assertEqual(len(unseen_pairs), 9)
        self.assertFalse(seen_pairs & unseen_pairs)
        self.assertEqual(set(appearances.values()), {1})

    def test_smoke_dry_run_covers_seen_unseen_and_both_orientations(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "smoke"
            GENERATION.main([
                "--output-dir", str(output),
                "--evaluation-protocol", str(Path(self.protocol["_source_path"])),
                "--disable-safety-checker", "--acknowledge-safety-risk",
                "--smoke", "--dry-run",
            ])
            rows = GENERATION.read_jsonl(output / "generation_manifest.jsonl")
            self.assertEqual(len(rows), 4)
            self.assertEqual({(row["pair_group"], row["orientation"]) for row in rows}, {
                ("seen", "forward"), ("seen", "swapped"),
                ("unseen", "forward"), ("unseen", "swapped"),
            })
            self.assertEqual({row["generation_seed"] for row in rows}, {42})

    def test_resume_skips_only_hash_and_fingerprint_matched_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows = [{
                **row,
                "model_fingerprint_sha256": "m" * 64,
                "protocol_fingerprint_sha256": self.protocol["_source_sha256"],
            } for row in self.rows[:2]]
            path = output / rows[0]["image_path"]
            path.parent.mkdir(parents=True)
            Image.new("RGB", (512, 512), (1, 2, 3)).save(path)
            status_path = output / "generation_status.jsonl"
            GENERATION.write_jsonl([{
                "id": rows[0]["id"],
                "status": "ok",
                "image_sha256": file_sha256(path),
                "model_fingerprint_sha256": "m" * 64,
                "protocol_fingerprint_sha256": self.protocol["_source_sha256"],
            }], status_path)
            args = argparse.Namespace(output_dir=output, status_path=status_path, skip_existing=True)
            self.assertEqual(GENERATION.pending_rows(rows, args), [rows[1]])
            rows[0]["model_fingerprint_sha256"] = "x" * 64
            with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                GENERATION.pending_rows(rows, args)

    def test_model_provenance_accepts_versioned_joint_binding_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "training"
            model = run / "checkpoints"
            model.mkdir(parents=True)
            (run / "manifest.json").write_text(json.dumps({
                "status": "succeeded",
                "returncode": 0,
                "run": {"variant": "joint_binding_seed42"},
                "git": {"commit": "a" * 40, "branch": "test"},
            }), encoding="utf-8")
            (run / "config.yaml").write_text("stage: train\n", encoding="utf-8")
            for name in GENERATION.MODEL_ARTIFACTS:
                (model / name).write_bytes(name.encode())
            provenance = GENERATION.model_provenance(model, run)
            self.assertEqual(provenance["parent_training_variant"], "joint_binding_seed42")
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            manifest["run"]["variant"] = "unapproved"
            (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "approved two-object"):
                GENERATION.model_provenance(model, run)


class TwoObjectEvaluationBundleTests(unittest.TestCase):
    def test_full_synthetic_ledgers_validate_and_detect_hash_tamper(self):
        protocol = read_protocol()
        expected = build_manifest(protocol)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_rows, status_rows = [], []
            for scene_index, row in enumerate(expected):
                path = root / row["image_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                color = ((row["generation_seed"] - 42) * 11 % 256, scene_index % 256, 97)
                Image.new("RGB", (512, 512), color).save(path)
                enriched = {
                    **row,
                    "safety_checker_disabled": True,
                    "safety_risk_acknowledged": True,
                    "model_fingerprint_sha256": "m" * 64,
                    "protocol_fingerprint_sha256": protocol["_source_sha256"],
                }
                manifest_rows.append(enriched)
                status_rows.append({
                    "id": row["id"],
                    "status": "ok",
                    "failure_reason": None,
                    "image_sha256": file_sha256(path),
                    "model_fingerprint_sha256": "m" * 64,
                    "protocol_fingerprint_sha256": protocol["_source_sha256"],
                })
            GENERATION.write_jsonl(manifest_rows, root / "generation_manifest.jsonl")
            GENERATION.write_jsonl(status_rows, root / "generation_status.jsonl")
            self.assertEqual(len(BUNDLE.validate(root, Path(protocol["_source_path"]))), 720)
            status_rows[0]["image_sha256"] = "0" * 64
            GENERATION.write_jsonl(status_rows, root / "generation_status.jsonl")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                BUNDLE.validate(root, Path(protocol["_source_path"]))


if __name__ == "__main__":
    unittest.main()
