"""CPU contracts for ZeST case construction and immutable prepared/cache inputs."""

import json
from pathlib import Path
import tempfile
import unittest

from experiments.zest_material_counterfactual_v1 import assess, run
from experiments.sd15_token_local_kv_v1.contract import sha256


class RunTest(unittest.TestCase):
    def setUp(self):
        self.plan = {"subjects": [{"id": "mailbox", "source_files": {"target": "source/mailbox/target.png"}}],
                     "variants": ["baseline", "color", "texture", "combined"]}
        self.bank = {"status": "complete", "records": [
            {"id": f"M{m}_R{r}", "metallic": m, "roughness": r,
             "image": f"/external/bank/M{m}_R{r}.png", "image_sha256": "frozen"}
            for m in (0., .5, 1.) for r in (.2, .5, .8)] + [
            {"id": "estimated_original", "metallic": .04, "roughness": .35,
             "image": "/external/bank/anchor.png", "image_sha256": "anchor"}]}

    def test_identity_controls_separate_self_and_estimated_sphere(self):
        rows = run.jobs(self.plan, self.bank, "identity", self.plan["variants"])
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["reference_id"], "self")
        self.assertIsNone(rows[0]["reference_target"])
        self.assertEqual(rows[1]["reference_target"], {"metallic": .04, "roughness": .35})
        self.assertEqual(rows[0]["reference_image"], "source/mailbox/target.png")

    def test_grid_batches_carriers_with_same_references_without_training(self):
        self.plan["subjects"].append({"id": "another_carrier", "source_files": {"target": "source/other/target.png"}})
        rows = run.jobs(self.plan, self.bank, "grid", self.plan["variants"])
        self.assertEqual(len(rows), 72)
        self.assertEqual(len({r["id"] for r in rows}), 72)
        self.assertNotIn("estimated_original", {r["reference_id"] for r in rows})
        self.assertEqual(rows[0]["reference_image"], rows[36]["reference_image"])
        self.assertNotEqual(rows[0]["subject_id"], rows[36]["subject_id"])

    def test_unknown_variant_or_incomplete_bank_is_rejected(self):
        with self.assertRaises(ValueError):
            run.jobs(self.plan, self.bank, "grid", ["invented"])
        self.bank["status"] = "failed"
        with self.assertRaises(ValueError):
            run.jobs(self.plan, self.bank, "grid", ["baseline"])

    def test_prepared_hash_and_cached_estimate_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.bin"
            image.write_bytes(b"fixed")
            (root / "plan.json").write_text(json.dumps(self.plan))
            (root / "prepare_receipt.json").write_text(json.dumps({"output_hashes": {"image.bin": sha256(image)}}))
            self.assertEqual(run.load_plan(root), self.plan)
            cache = {"records": {"generated:case": {"input": {"id": "case"}, "hashes": {"image.bin": sha256(image)}}}}
            assess.validate_cache(root, cache, {"generated:case": {"id": "case"}})
            with self.assertRaises(ValueError):
                assess.validate_cache(root, cache, {"generated:case": {"id": "different"}})
            image.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                run.load_plan(root)
            with self.assertRaises(ValueError):
                assess.validate_cache(root, cache, {})


if __name__ == "__main__":
    unittest.main()
