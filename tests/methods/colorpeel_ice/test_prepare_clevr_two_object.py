import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.methods.colorpeel_ice.multiview_render_contract import (
    EXPECTED_PROFILE_V1,
    EXPECTED_PROFILE_V2,
    EXPECTED_PROFILE_V3,
    EXPECTED_PROFILE_V4,
    canonical_sha256,
)
from src.methods.colorpeel_ice import prepare_clevr_two_object as prepare


REPO_ROOT = Path(__file__).parents[3]
PROFILE_PATH = (
    REPO_ROOT / "experiments" / "clevr_two_object_subject_color_material" /
    "configs" / "multiview_render_v4_two_object.json"
)
RENDERER_PATH = REPO_ROOT / "scripts" / "methods" / "colorpeel_ice" / "render_clevr_two_object.py"
SPEC = importlib.util.spec_from_file_location("render_clevr_two_object", RENDERER_PATH)
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)
MASK_UTILS_PATH = REPO_ROOT / "src" / "train" / "instance_mask_utils.py"
MASK_SPEC = importlib.util.spec_from_file_location("instance_mask_utils", MASK_UTILS_PATH)
MASK_UTILS = importlib.util.module_from_spec(MASK_SPEC)
MASK_SPEC.loader.exec_module(MASK_UTILS)


class TwoObjectProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.protocol, cls.states, cls.scenes = prepare.load_inputs(
            prepare.DEFAULT_MANIFEST, prepare.DEFAULT_PROTOCOL
        )
        cls.requests = prepare.build_render_requests(cls.manifest, cls.protocol)

    def test_historical_profiles_and_new_profile_have_locked_fingerprints(self):
        self.assertEqual(canonical_sha256(EXPECTED_PROFILE_V1),
                         "246cb06778a74f994311c3e1e3a8a4aa973ce7a308d3cd2732dcdcc021bf8529")
        self.assertEqual(canonical_sha256(EXPECTED_PROFILE_V2),
                         "4890b5a481b7f903f383beeee97607c7f8fd410708925eea923c54cab2b3ece8")
        self.assertEqual(canonical_sha256(EXPECTED_PROFILE_V3),
                         "0db0afe3f5697c7c7a41b12b4b463331ff3d0e4e6ea248096f3145795eab076a")
        self.assertEqual(canonical_sha256(EXPECTED_PROFILE_V4),
                         "75af0818d2e18a033878f050a1cf1c4791dc519c1c187eddbe7fa6a639d3f14a")
        self.assertEqual(json.loads(PROFILE_PATH.read_text(encoding="utf-8")), EXPECTED_PROFILE_V4)
        self.assertEqual(RENDERER.validate_profile(EXPECTED_PROFILE_V4), EXPECTED_PROFILE_V4)

    def test_pairing_balances_every_state_across_left_and_right(self):
        self.assertEqual(len(self.states), 18)
        self.assertEqual(len(self.scenes), 18)
        counts = {(state["state_id"], side): 0 for state in self.states for side in ("left", "right")}
        for scene in self.scenes:
            left, right = scene["objects"]
            self.assertNotEqual(left["shape"], right["shape"])
            self.assertNotEqual(left["color"], right["color"])
            self.assertNotEqual(left["material"], right["material"])
            for obj in scene["objects"]:
                counts[(obj["state_id"], obj["side"])] += 1
                modifiers = re.findall(r"<[^>]+>", obj["instance_prompt"][0])
                self.assertEqual(len(modifiers), 3)
        self.assertEqual(set(counts.values()), {1})

    def test_requests_are_360_and_swapped_orientations_share_seed(self):
        self.assertEqual(len(self.requests), 360)
        self.assertEqual(canonical_sha256(self.requests),
                         "e9deb85f28a38cbceddf3bfc55db7527533a6c79abceef7dcb96f1069031211a")
        self.assertEqual(
            [(row["scene_id"], row["view_index"]) for row in self.requests[:2]],
            [("pair_00_forward", 0), ("pair_00_swapped", 0)],
        )
        side_counts = {(state["state_id"], side): 0 for state in self.states for side in ("left", "right")}
        paired = {}
        for row in self.requests:
            paired.setdefault((row["pair_index"], row["view_index"]), []).append(row)
            for obj in row["objects"]:
                side_counts[(obj["state_id"], obj["side"])] += 1
        self.assertEqual(set(side_counts.values()), {20})
        for rows in paired.values():
            self.assertEqual({row["orientation"] for row in rows}, {"forward", "swapped"})
            self.assertEqual(len({row["render_seed"] for row in rows}), 1)

    def test_background_and_scientific_parameters_are_locked(self):
        self.assertFalse(EXPECTED_PROFILE_V4["background"]["varied"])
        self.assertEqual(EXPECTED_PROFILE_V4["objects"]["positions_xy"],
                         {"left": [-1.6, 0.0], "right": [1.6, 0.0]})
        changed = json.loads(json.dumps(EXPECTED_PROFILE_V4))
        changed["objects"]["positions_xy"]["left"][0] = -1.5
        with self.assertRaisesRegex(RENDERER.RendererError, "differs from locked"):
            RENDERER.validate_profile(changed)


class InstanceMaskTests(unittest.TestCase):
    def test_pairing_is_optional_and_requires_exact_stems(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, masks = root / "images", root / "masks"
            images.mkdir(); masks.mkdir()
            (images / "a.jpg").write_bytes(b"a")
            self.assertEqual(MASK_UTILS.pair_instance_images_and_masks(images, None),
                             [(images / "a.jpg", None)])
            Image.new("L", (512, 512), 255).save(masks / "a.png")
            pairs = MASK_UTILS.pair_instance_images_and_masks(images, masks)
            self.assertEqual(pairs, [(images / "a.jpg", masks / "a.png")])
            Image.new("L", (512, 512), 255).save(masks / "extra.png")
            with self.assertRaisesRegex(ValueError, "exactly one matching"):
                MASK_UTILS.pair_instance_images_and_masks(images, masks)

    def test_binary_instance_mask_downsamples_and_rejects_invalid_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.png"
            array = np.zeros((512, 512), dtype=np.uint8)
            array[128:384, 128:384] = 255
            Image.fromarray(array).save(valid)
            latent = MASK_UTILS.load_latent_instance_mask(valid, 512, 64)
            self.assertEqual(latent.shape, (64, 64))
            self.assertEqual(set(np.unique(latent)), {0.0, 1.0})
            invalid = root / "invalid.png"
            array[0, 0] = 127
            Image.fromarray(array).save(invalid)
            with self.assertRaisesRegex(ValueError, "strictly binary"):
                MASK_UTILS.load_latent_instance_mask(invalid, 512, 64)

    def test_two_object_masks_are_disjoint_and_partition_background(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (512, 512), (10, 20, 30)).save(root / "img.jpg")
            left = Image.new("L", (512, 512), 0)
            right = Image.new("L", (512, 512), 0)
            for x in range(70, 190):
                for y in range(180, 320):
                    left.putpixel((x, y), 255)
            for x in range(320, 440):
                for y in range(180, 320):
                    right.putpixel((x, y), 255)
            background = Image.fromarray(
                255 - np.maximum(np.asarray(left), np.asarray(right)), mode="L"
            )
            left.save(root / "left.png"); right.save(root / "right.png"); background.save(root / "background.png")
            counts = prepare._validate_artifacts(
                root / "img.jpg", {"left": root / "left.png", "right": root / "right.png"},
                root / "background.png",
            )
            self.assertEqual(counts, {"left": 120 * 140, "right": 120 * 140})
            right.putpixel((100, 200), 255); right.save(root / "right.png")
            with self.assertRaisesRegex(prepare.ProtocolError, "overlap"):
                prepare._validate_artifacts(
                    root / "img.jpg", {"left": root / "left.png", "right": root / "right.png"},
                    root / "background.png",
                )


class TwoObjectStagingTests(unittest.TestCase):
    @unittest.skipUnless(prepare.yaml is not None, "PyYAML is required")
    def test_staging_creates_576_image_mask_pairs_and_image_dirs_are_clean(self):
        manifest, protocol, states, _ = prepare.load_inputs(prepare.DEFAULT_MANIFEST, prepare.DEFAULT_PROTOCOL)
        requests = prepare.build_render_requests(manifest, protocol)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); render_root = root / "render"; output = root / "output"
            realized = []
            for row in requests:
                view_dir = render_root / row["scene_id"] / f"view_{row['view_index']:02d}"
                view_dir.mkdir(parents=True)
                (view_dir / "img.jpg").write_bytes(f"{row['scene_id']}:{row['view_index']}".encode())
                (view_dir / "mask_left.png").write_bytes(b"left")
                (view_dir / "mask_right.png").write_bytes(b"right")
                realized.append({**row, "image": (view_dir / "img.jpg").relative_to(render_root).as_posix(),
                                 "masks": {side: (view_dir / f"mask_{side}.png").relative_to(render_root).as_posix()
                                           for side in ("left", "right")}})
            output.mkdir()
            summary = prepare.build_training_outputs(
                render_root, realized, states, output, prepare.DEFAULT_CONFIG
            )
            concepts = json.loads(Path(summary["concepts"]).read_text(encoding="utf-8"))
            self.assertEqual(len(concepts), 18)
            self.assertEqual(summary["object_record_count"], 576)
            images = list((output / "training" / "train_assets").glob("*/*"))
            masks = list((output / "training" / "train_masks").glob("*/*"))
            self.assertEqual(len(images), 576)
            self.assertEqual(len(masks), 576)
            self.assertTrue(all(path.suffix == ".jpg" for path in images))
            self.assertTrue(all(path.suffix == ".png" for path in masks))

    @unittest.skipUnless(prepare.yaml is not None, "PyYAML is required")
    def test_human_authorization_builds_masked_smokes_and_full_config(self):
        manifest, protocol, states, _ = prepare.load_inputs(prepare.DEFAULT_MANIFEST, prepare.DEFAULT_PROTOCOL)
        requests = prepare.build_render_requests(manifest, protocol)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            render_root, prepared, authorized = root / "render", root / "prepared", root / "authorized"
            realized = []
            for row in requests:
                view_dir = render_root / row["scene_id"] / f"view_{row['view_index']:02d}"
                view_dir.mkdir(parents=True)
                (view_dir / "img.jpg").write_bytes(f"{row['scene_id']}:{row['view_index']}".encode())
                (view_dir / "mask_left.png").write_bytes(b"left")
                (view_dir / "mask_right.png").write_bytes(b"right")
                realized.append({
                    **row,
                    "image": (view_dir / "img.jpg").relative_to(render_root).as_posix(),
                    "masks": {
                        side: (view_dir / f"mask_{side}.png").relative_to(render_root).as_posix()
                        for side in ("left", "right")
                    },
                })
            prepared.mkdir()
            prepare.build_training_outputs(render_root, realized, states, prepared, prepare.DEFAULT_CONFIG)
            prepare._write_jsonl(prepared / "realized_scenes.jsonl", realized)
            prepare._write_json(prepared / "protocol_status.json", {
                "status": "validated_pending_human_review",
                "training_object_record_count": 576,
            })

            result = prepare.build_authorized_training_package(prepared, authorized, states)

            self.assertEqual(result["status"], "ready_for_training_smokes")
            self.assertTrue(result["human_gate"]["training_authorized"])
            smoke2 = json.loads(Path(result["smoke_configs"]["smoke_2step"]).read_text(encoding="utf-8"))
            smoke18 = json.loads(Path(result["smoke_configs"]["smoke_18step"]).read_text(encoding="utf-8"))
            self.assertEqual(smoke2["args"]["max_train_steps"], 2)
            self.assertEqual(smoke18["args"]["max_train_steps"], 18)
            self.assertTrue(smoke2["protocol"]["gt_instance_masks_in_training"])
            self.assertTrue(smoke18["protocol"]["gt_instance_masks_in_training"])
            self.assertEqual(len(json.loads(Path(smoke2["args"]["concepts_list"]).read_text())), 2)
            self.assertEqual(len(json.loads(Path(smoke18["args"]["concepts_list"]).read_text())), 18)
            for smoke_name, count in (("smoke_2step", 2), ("smoke_18step", 18)):
                images = list((authorized / "smokes" / smoke_name / "train_assets").glob("*/*.jpg"))
                masks = list((authorized / "smokes" / smoke_name / "train_masks").glob("*/*.png"))
                self.assertEqual(len(images), count)
                self.assertEqual([path.stem for path in images], [path.stem for path in masks])


class TwoObjectRendererResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest, protocol, _, _ = prepare.load_inputs(prepare.DEFAULT_MANIFEST, prepare.DEFAULT_PROTOCOL)
        cls.requests = prepare.build_render_requests(manifest, protocol)

    def _write_record(self, output: Path, contract: dict, request: dict):
        view_dir = output / request["scene_id"] / f"view_{request['view_index']:02d}"
        view_dir.mkdir(parents=True)
        files = {
            "image": view_dir / "img.jpg", "scene_json": view_dir / "scene.json",
            "background_mask": view_dir / "background.png", "mask_left": view_dir / "mask_left.png",
            "mask_right": view_dir / "mask_right.png",
        }
        for name, path in files.items():
            path.write_bytes(name.encode("ascii"))
        relative = {name: path.relative_to(output).as_posix() for name, path in files.items()}
        record = {
            **request, "image": relative["image"], "scene_json": relative["scene_json"],
            "background_mask": relative["background_mask"],
            "masks": {"left": relative["mask_left"], "right": relative["mask_right"]},
            "render_contract_sha256": canonical_sha256(contract),
            "artifact_sha256": {name: RENDERER.BASE.file_sha256(path) for name, path in files.items()},
        }
        RENDERER.BASE.write_json(view_dir / ".record.json", record)
        RENDERER.BASE.append_jsonl(output / "renderer_realization.jsonl", record)
        return record

    def test_resume_accepts_exact_record_and_rejects_tamper(self):
        contract = RENDERER.BASE.stable_contract(
            self.requests, EXPECTED_PROFILE_V4, {"base_scene_blendfile": "a" * 64}
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "rendered"
            RENDERER.BASE.prepare_output_root(output, contract, resume=False)
            request = self.requests[0]
            record = self._write_record(output, contract, request)
            expected = {(request["scene_id"], request["view_index"]): request}
            completed = RENDERER.load_completed_records(output, expected, contract)
            self.assertEqual(completed[(request["scene_id"], request["view_index"])], record)
            (output / record["masks"]["left"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(RENDERER.RendererError, "hash changed"):
                RENDERER.load_completed_records(output, expected, contract)


if __name__ == "__main__":
    unittest.main()
