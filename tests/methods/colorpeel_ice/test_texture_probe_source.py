import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.texture_material_disentangled_v1 import texture_probe_source as source


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class TextureProbeSourceTests(unittest.TestCase):
    def fixture(self, root, step=500):
        protocol = source.data.read_protocol()
        staging = root / "staging"
        audit = {"datasets": {}}
        for texture in source.infer.target_ids(protocol, "texture"):
            folder, rows, groups = staging / texture, [], []
            for record in source.data._records(protocol):
                if record["branch"] != "texture" or record["concept_id"] != texture:
                    continue
                image = folder / "images" / record["id"] / "image.png"
                mask = folder / "masks" / record["id"] / "image.png"
                for path, mode in ((image, "RGB"), (mask, "L")):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new(mode, (2, 2), 128).save(path)
                prompt = source.data.caption(record, protocol)
                rows.append({**record, "prompt": prompt, "image": str(image), "mask": str(mask),
                             "image_sha256": source.sha256(image), "mask_sha256": source.sha256(mask)})
                groups.append({"instance_prompt": [prompt], "instance_data_dir": str(image.parent),
                               "instance_mask_dir": str(mask.parent)})
            write_json(folder / "concepts.json", groups)
            ledger = folder / "training_assets_manifest.jsonl"
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            write_json(folder / "dataset_receipt.json", {"branch": "texture", "token": "<T*>",
                "concept_id": texture, "image_count": 16, "concepts_sha256": source.sha256(folder / "concepts.json"),
                "training_assets_manifest_sha256": source.sha256(ledger)})
            audit["datasets"][texture] = {"dataset_receipt_sha256": source.sha256(folder / "dataset_receipt.json")}
        write_json(staging / "audit.json", audit)
        runs = []
        for method in source.infer.METHODS:
            for branch in ("texture", "material"):
                for concept in source.infer.target_ids(protocol, branch):
                    run = root / "training" / f"{method}_{concept}"
                    checkpoint = run / "adapter"
                    hashes = {}
                    if branch == "texture":
                        folder = staging / concept
                        config = {"stage": "train", "run": {"study": "fixture", "variant": concept, "seed": 42},
                                  "args": {"concepts_list": str(folder / "concepts.json")},
                                  "data_manifest": str(folder / "training_assets_manifest.jsonl")}
                        write_json(run / "config.yaml", config)
                        write_json(run / "manifest.json", {"data_manifest": config["data_manifest"]})
                        adaptation = {"modifier_tokens": ["<T*>"], "adaptation_mode": "token_local_kv"
                            if method == "fullrank" else "lora_subject_kv", "rank": 64, "alpha": 64,
                            "mode": "token_local_kv"}
                        write_json(checkpoint / "adaptation_config.json", adaptation)
                        hashes = {str(path): source.sha256(path) for path in
                                  (run / "config.yaml", run / "manifest.json", checkpoint / "adaptation_config.json")}
                    runs.append({"method": method, "branch": branch, "concept_id": concept,
                                 "token": "<T*>" if branch == "texture" else "<M*>", "run": str(run),
                                 "checkpoint": str(checkpoint), "hashes": hashes})
        index = {"training_steps": step, "source_training_steps": 1000, "runs": runs}
        index_path = root / f"step{step}/training_index.json"
        write_json(index_path, index)
        write_json(root / "status.json", {"status": "execution_verified"})
        provenance = {"status": "succeeded", "training_steps": step, "sampling": protocol["inference"],
                      "training_index_sha256": source.sha256(index_path)}
        write_json(root / f"step{step}/inference/provenance.json", provenance)
        return index, provenance, audit

    def test_selects_only_texture_and_locks_actual_training_rgb_masks_and_groups(self):
        for step in source.STEPS:
            with self.subTest(step=step), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                index, _, audit = self.fixture(root, step)
                with (patch.object(source.infer, "verify_inputs") as verify,
                      patch.object(source.data, "verify_staging", return_value=audit) as staging):
                    actual, selections, locks, refs = source.verify_source(root, step)
                self.assertEqual(actual, index)
                verify.assert_called_once_with(index, source.data.read_protocol(), step)
                staging.assert_called_once_with(root / "staging")
                self.assertEqual(len(selections), 4)
                self.assertEqual(set(refs), {(method, texture) for method in source.infer.METHODS
                    for texture in source.infer.target_ids(source.data.read_protocol(), "texture")})
                for ref in refs.values():
                    self.assertEqual(ref["step"], step)
                    self.assertEqual(len(ref["images"]), 16)
                    for image in ref["images"]:
                        path = Path(image["image"])
                        self.assertEqual(locks[str(path)], source.sha256(path))
                        self.assertIn("images", path.parts)
                        self.assertIn("<T*>", image["prompt"])
                        with Image.open(path) as rgb:
                            self.assertEqual(rgb.mode, "RGB")
                self.assertTrue(all(source.sha256(Path(path)) == digest for path, digest in locks.items()))

    def test_rejects_sampling_step_and_incomplete_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, provenance, audit = self.fixture(root)
            with (patch.object(source.infer, "verify_inputs"),
                  patch.object(source.data, "verify_staging", return_value=audit)):
                for field, value in (("status", "failed"), ("training_steps", 750),
                                     ("training_index_sha256", "changed"), ("sampling", {})):
                    write_json(root / "step500/inference/provenance.json", {**provenance, field: value})
                    with self.assertRaisesRegex(ValueError, "step or sampling"):
                        source.verify_source(root, 500)
                with self.assertRaisesRegex(ValueError, "completed source"):
                    source.verify_source(root, 300)
                write_json(root / "status.json", {"status": "running"})
                with self.assertRaisesRegex(ValueError, "completed source"):
                    source.verify_source(root, 500)

    def test_rejects_material_adapter_or_wrong_rank_for_texture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index, _, audit = self.fixture(root)
            selection = next(item for item in index["runs"] if item["branch"] == "texture"
                             and item["method"] == "lora_r64")
            path = Path(selection["checkpoint"]) / "adaptation_config.json"
            original = source.read_json(path)
            with (patch.object(source.infer, "verify_inputs"),
                  patch.object(source.data, "verify_staging", return_value=audit)):
                for field, value in (("modifier_tokens", ["<M*>"]), ("rank", 32),
                                     ("adaptation_mode", "lora_material_kv")):
                    write_json(path, {**original, field: value})
                    with self.assertRaisesRegex(ValueError, "Texture adapter"):
                        source.verify_source(root, 500)

    def test_rejects_changed_rgb_and_config_dataset_substitution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index, _, audit = self.fixture(root)
            item = next(item for item in index["runs"] if item["branch"] == "texture")
            folder = root / "staging" / item["concept_id"]
            row = json.loads((folder / "training_assets_manifest.jsonl").read_text().splitlines()[0])
            image = Path(row["image"])
            original = image.read_bytes()
            with (patch.object(source.infer, "verify_inputs"),
                  patch.object(source.data, "verify_staging", return_value=audit)):
                Image.new("RGB", (2, 2), "red").save(image)
                with self.assertRaisesRegex(ValueError, "RGB training"):
                    source.verify_source(root, 500)
                image.write_bytes(original)
                config_path = Path(item["run"]) / "config.yaml"
                config = source.read_config(config_path)
                config["data_manifest"] = str(root / "unrelated_manifest.jsonl")
                write_json(config_path, config)
                with self.assertRaisesRegex(ValueError, "dataset paths"):
                    source.verify_source(root, 500)

    def test_rejects_sampling_groups_that_do_not_match_training_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index, _, audit = self.fixture(root)
            item = next(item for item in index["runs"] if item["branch"] == "texture")
            folder = root / "staging" / item["concept_id"]
            concepts = source.read_json(folder / "concepts.json")
            concepts[0]["instance_prompt"] = ["wrong training caption"]
            write_json(folder / "concepts.json", concepts)
            receipt = source.read_json(folder / "dataset_receipt.json")
            receipt["concepts_sha256"] = source.sha256(folder / "concepts.json")
            write_json(folder / "dataset_receipt.json", receipt)
            audit["datasets"][item["concept_id"]]["dataset_receipt_sha256"] = source.sha256(folder / "dataset_receipt.json")
            with (patch.object(source.infer, "verify_inputs"),
                  patch.object(source.data, "verify_staging", return_value=audit)):
                with self.assertRaisesRegex(ValueError, "sampling groups"):
                    source.verify_source(root, 500)


if __name__ == "__main__":
    unittest.main()
