"""CPU checks for reference-only cropping and read-only source packages."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from experiments.sd15_token_local_kv_v1.contract import sha256
from experiments.zest_material_counterfactual_v1 import focus_reference as focus


class ReferenceFocusTest(unittest.TestCase):
    def test_mask_scaling_square_padding_and_reference_only_operation(self):
        reference = Image.new("RGB", (1024, 1024), (210, 90, 30))
        target = Image.new("RGB", reference.size, (20, 140, 230))
        mask = Image.new("L", (512, 512), 0)
        mask.paste(255, (100, 150, 200, 250))
        before = reference.tobytes(), target.tobytes(), mask.tobytes()
        focused, metadata = focus.focus_reference(reference, mask)
        self.assertEqual(metadata["mask_bbox1024"], [200, 300, 400, 500])
        self.assertEqual(metadata["crop_box1024"], [180, 280, 420, 520])
        self.assertEqual(focused.size, target.size)
        self.assertGreater(metadata["foreground_fraction_focused"], metadata["foreground_fraction_full"])
        self.assertEqual((reference.tobytes(), target.tobytes(), mask.tobytes()), before)

    def test_diagnostic_preserves_sources_and_passes_uncropped_target_conditions(self):
        with tempfile.TemporaryDirectory(prefix="zest_focus_") as temporary:
            root = Path(temporary).resolve()
            run, asset_root = root / "original", root / "assets"
            run.mkdir()
            asset_root.mkdir()
            (asset_root / "assets.json").write_text("{}", encoding="utf8")
            def write(path, value):
                path.write_text(json.dumps(value), encoding="utf8")
            files = {"target": "source/target.png", "object_mask": "source/object_mask.png",
                     "edit_mask": "source/edit_mask.png", "control_depth": "source/control_depth.png"}
            (run / "source").mkdir()
            target = Image.new("RGB", (1024, 1024), (210, 90, 30))
            target.save(run / files["target"])
            mask = Image.new("L", (512, 512), 0)
            mask.paste(255, (100, 150, 200, 250))
            mask.save(run / files["object_mask"])
            Image.new("RGB", target.size, "white").save(run / files["edit_mask"])
            Image.new("RGB", target.size, (70, 70, 70)).save(run / files["control_depth"])
            plan = {"subjects": [{"id": "mailbox", "source_files": files}], "inference": {"strength": .9999}}
            for name, value in (("plan.json", plan), ("config.json", {}), ("bank_plan.json", {})):
                write(run / name, value)
            hashes = {path.relative_to(run).as_posix(): sha256(path) for path in run.rglob("*") if path.is_file()}
            write(run / "prepare_receipt.json", {"output_hashes": hashes})
            (run / "bank").mkdir()
            target.save(run / "bank/sphere.png")
            mask.save(run / "bank/mask.png")
            write(run / "bank/manifest.json", {"records": [{"id": "estimated_original", "mask": str(run/"bank/mask.png"),
                                                             "mask_sha256": sha256(run/"bank/mask.png")}]})
            records = []
            for reference_id in ("self", "estimated_original"):
                reference = files["target"] if reference_id == "self" else "bank/sphere.png"
                raw = f"full_{reference_id}.png"
                target.save(run / raw)
                records.append({"id": "identity__mailbox__baseline__"+reference_id, "subject_id": "mailbox",
                    "variant": "baseline", "stage": "identity", "reference_id": reference_id,
                    "reference_image": reference, "reference_sha256": sha256(run/reference),
                    "raw_image": raw, "raw_sha256": sha256(run/raw)})
            write(run / "generation_manifest.json", {"records": records})
            originals = {path: sha256(path) for path in run.rglob("*") if path.is_file()}
            adapter = SimpleNamespace(pipe=SimpleNamespace(num_timesteps=29))
            generated = Image.new("RGB", target.size, (0, 170, 230))
            with patch.object(focus.assets, "read", return_value={}), patch.object(focus.assets, "verify"), \
                 patch.object(focus.official, "setup", return_value=adapter), \
                 patch.object(focus.official, "compatibility", return_value={}), \
                 patch.object(focus.official, "generate", return_value=generated) as generate, \
                 patch.object(focus, "git_info", return_value={}), \
                 patch.dict(sys.modules, torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=Mock()))):
                output = focus.run(run, asset_root, 0, ("baseline",))
            self.assertEqual({path: sha256(path) for path in originals}, originals)
            self.assertEqual((output/"prepare_receipt.json").read_bytes(), (run/"prepare_receipt.json").read_bytes())
            focus.load_plan(output)  # The unchanged prepared hashes remain valid for assess.py.
            self.assertEqual(generate.call_count, 2)
            for call in generate.call_args_list:
                self.assertEqual(call.args[1].tobytes(), target.tobytes())
                self.assertEqual(call.args[2].getpixel((0, 0)), (255, 255, 255))
                self.assertEqual(call.args[3].getpixel((0, 0)), (70, 70, 70))
                self.assertIsNone(call.kwargs["init_image"])
            saved = focus.read(output/"generation_manifest.json")
            self.assertFalse(saved["training_eligible"])
            for row in saved["records"]:
                self.assertEqual(focus.picture(output/row["raw_image"]).getpixel((0, 0)), (0, 170, 230))
            self.assertFalse((output/"quality").exists())
            self.assertFalse((output/"estimates").exists())


if __name__ == "__main__":
    unittest.main()
