import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from src.methods.colorpeel_ice.prepare_clevr_two_object import build_scenes, build_states
from src.methods.colorpeel_ice.prepare_joint_two_object import build_package
from src.train.instance_mask_utils import pair_joint_instance_images_and_masks
from src.train import joint_binding_utils as joint


REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = (
    REPO_ROOT / "experiments" / "clevr_two_object_subject_color_material" /
    "manifests" / "clevr_two_object_manifest.json"
)


class JointBindingLossTests(unittest.TestCase):
    def test_balanced_reconstruction_normalizes_each_instance(self):
        per_pixel = torch.zeros(1, 4, 4, 4)
        per_pixel[:, :, :2, :] = 1
        masks = torch.zeros(1, 2, 4, 4)
        masks[:, 0, :2, :] = 1
        masks[:, 1, 2:, :] = 1
        self.assertEqual(joint.balanced_instance_masked_mse(per_pixel, masks).item(), 2.0)

    def test_caa_is_scoped_within_object_groups(self):
        attention = torch.zeros(2, 2, 6)
        attention[0, 0, :3] = 1
        attention[1, 1, 3:] = 1
        loss = joint.grouped_caa_loss(attention, [[0, 1, 2], [3, 4, 5]])
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_modifier_positions_require_exactly_one_occurrence(self):
        input_ids = torch.tensor([99, 10, 11, 12, 20, 21, 22, 99])
        self.assertEqual(
            joint.modifier_group_positions(input_ids, [[10, 11, 12], [20, 21, 22]]),
            [[1, 2, 3], [4, 5, 6]],
        )
        with self.assertRaisesRegex(ValueError, "exactly once"):
            joint.modifier_group_positions(torch.tensor([10, 10, 11, 12]), [[10, 11, 12]])

    def test_ice_attention_loss_is_finite_and_differentiable(self):
        attention = torch.full((2, 2, 6), 0.01, requires_grad=True)
        with torch.no_grad():
            attention[0, 0, :3] = 1
            attention[1, 1, 3:] = 1
        masks = torch.zeros(2, 4, 4)
        masks[0, :2, :2] = 1
        masks[1, 2:, 2:] = 1
        with mock.patch.object(joint, "ICE_SINKHORN_ITERATIONS", 5):
            loss = joint.ice_wasserstein_attention_loss(
                attention, [[0, 1, 2], [3, 4, 5]], masks
            )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(attention.grad).all())

    def test_joint_pairing_rejects_stem_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, left, right = root / "images", root / "left", root / "right"
            for directory in (images, left, right):
                directory.mkdir()
            (images / "a.jpg").write_bytes(b"rgb")
            (left / "a.png").write_bytes(b"left")
            (right / "a.png").write_bytes(b"right")
            self.assertEqual(len(pair_joint_instance_images_and_masks(images, left, right)), 1)
            (right / "extra.png").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "right mask directory"):
                pair_joint_instance_images_and_masks(images, left, right)


class JointBindingPreparationTests(unittest.TestCase):
    def _fake_prepared_root(self, root: Path) -> Path:
        prepared = root / "prepared"
        (prepared / "training").mkdir(parents=True)
        (prepared / "protocol_status.json").write_text(
            json.dumps({"status": "validated_pending_human_review"}), encoding="utf-8"
        )
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        states = build_states(manifest)
        scenes = build_scenes(states)
        state_roots = {}
        concepts = []
        for state in states:
            image_dir = prepared / "source_images" / state["state_id"]
            mask_dir = prepared / "source_masks" / state["state_id"]
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            state_roots[state["state_id"]] = (image_dir, mask_dir)
            concepts.append({
                "instance_prompt": state["instance_prompt"],
                "instance_data_dir": str(image_dir),
                "instance_mask_dir": str(mask_dir),
            })
        (prepared / "training" / "concepts.json").write_text(json.dumps(concepts), encoding="utf-8")
        rows = []
        for scene in scenes:
            for view_index in range(20):
                row = {**scene, "view_index": view_index, "split": "train" if view_index < 16 else "audit"}
                rows.append(row)
                if view_index >= 16:
                    continue
                stem = f"{scene['scene_id']}__view_{view_index:02d}"
                rgb = f"rgb:{stem}".encode()
                for obj in scene["objects"]:
                    image_dir, mask_dir = state_roots[obj["state_id"]]
                    (image_dir / f"{stem}.jpg").write_bytes(rgb)
                    (mask_dir / f"{stem}.png").write_bytes(f"mask:{obj['side']}".encode())
        with (prepared / "realized_scenes.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return prepared

    def test_package_has_288_unique_joint_samples_and_locked_smokes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self._fake_prepared_root(root)
            output = root / "joint"
            result = build_package(prepared, output)
            self.assertEqual(result["joint_training_image_count"], 288)
            concepts = json.loads((output / "full_training" / "concepts.json").read_text())
            self.assertEqual(len(concepts), 18)
            self.assertEqual(sum(len(list(Path(row["instance_data_dir"]).glob("*.jpg"))) for row in concepts), 288)
            smoke18 = json.loads((output / "smokes" / "smoke_18step" / "train_config.json").read_text())
            self.assertEqual(smoke18["protocol"]["expected_exposure_counts"], {
                "<s1*>": 12, "<s2*>": 12, "<s3*>": 12,
                "<c1*>": 12, "<c2*>": 12, "<c3*>": 12,
                "<m1*>": 18, "<m2*>": 18,
            })
            self.assertTrue(smoke18["args"]["joint_two_object_binding"])
            self.assertEqual(smoke18["args"]["lambda_attention"], joint.ICE_ATTENTION_WEIGHT)


if __name__ == "__main__":
    unittest.main()
