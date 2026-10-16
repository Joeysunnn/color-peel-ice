import json
from pathlib import Path
import tempfile
import unittest

from experiments.sd_native_comparison_v1.generate import PROTOCOL, read_protocol, rows, sha256
from experiments.sd_native_comparison_v1.bundle import verify_pair


def fixture(root):
    paths = []
    for model in ("sd14", "sd15"):
        directory = root / model
        directory.mkdir()
        paths.append(directory)
        ledger = []
        for row in rows(read_protocol(), model):
            image = directory / row["image_path"]
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes((model + row["pair_id"]).encode())
            ledger.append({**row, "status": "succeeded", "latent_sha256": row["pair_id"],
                           "safety_filtered": False, "image_sha256": sha256(image)})
        (directory / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ledger))
        inventory = directory / "model_inventory.json"
        inventory.write_text(json.dumps({"hashes_complete": True, "safety_components": {"same": True}}))
        (directory / "manifest.json").write_text(json.dumps({
            "status": "succeeded", "returncode": 0, "model": model, "adapters": False,
            "protocol_sha256": sha256(PROTOCOL), "generated_rows": 20,
            "model_inventory_sha256": sha256(inventory), "git": {"commit": "same"},
            "script_sha256": "same", "safety_source": "same",
            "runtime": {"scheduler_config": {"beta_start": 0.00085, "_diffusers_version": model}},
        }))
    return paths


class NativeBundleTests(unittest.TestCase):
    def test_verified_pairs_allow_model_config_metadata_to_differ(self):
        with tempfile.TemporaryDirectory() as temporary:
            left, right = fixture(Path(temporary))
            a, b = verify_pair(left, right)
            self.assertEqual(len(a), 20)
            self.assertEqual(len(b), 20)

    def test_rejects_different_actual_noise_even_with_the_same_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            left, right = fixture(Path(temporary))
            ledger = right / "ledger.jsonl"
            values = [json.loads(line) for line in ledger.read_text().splitlines()]
            values[0]["latent_sha256"] = "different"
            ledger.write_text("".join(json.dumps(r) + "\n" for r in values))
            with self.assertRaisesRegex(ValueError, "noise differs"):
                verify_pair(left, right)

    def test_rejects_modified_output_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            left, right = fixture(Path(temporary))
            (right / "images/chair/seed42.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "ledger differs"):
                verify_pair(left, right)
