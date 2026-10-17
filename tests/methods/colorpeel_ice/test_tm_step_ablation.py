import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.texture_material_disentangled_v1 import data, infer, train, step_ablation


class StepAblationTests(unittest.TestCase):
    def test_only_training_horizon_and_snapshot_list_change(self):
        self.assertEqual(step_ablation.STEPS, (500, 750, 1000))
        for recipe in train.recipes():
            original = train.build_config(recipe, Path("fixed_staging"), "pilot")
            changed = step_ablation.build_config(recipe, Path("fixed_staging"))
            expected = copy.deepcopy(original)
            expected["args"].update(max_train_steps=1000, checkpoint_steps=[300, 500, 750, 1000])
            expected["run"]["variant"] += "_steps1000"
            self.assertEqual(changed, expected)
            self.assertEqual(changed["args"]["checkpointing_steps"], 300)
            self.assertNotIn("resume_from_checkpoint", changed["args"])
        self.assertEqual(data.read_protocol()["training"]["steps"], 300)

    def test_explicit_step_requires_all_eight_matching_paired_exports(self):
        protocol = data.read_protocol()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            runs = [{**recipe, "run": str(root), "checkpoint": str(root), "status": "verified",
                     "weight": infer.WEIGHTS[recipe["method"]], "steps": 500,
                     "hashes": {str(root / name): "hash" for name in (
                         infer.WEIGHTS[recipe["method"]], recipe["token"] + ".bin", "adaptation_config.json")}}
                    for recipe in train.recipes()]
            index = {"mode": "pilot", "protocol_sha256": data.protocol_sha256(), "runs": runs}
            with patch.object(infer, "sha256", return_value="hash"):
                infer.verify_inputs(index, protocol, 500)
                for selected in (300, 750, 1000):
                    with self.assertRaisesRegex(ValueError, "pilot steps"):
                        infer.verify_inputs(index, protocol, selected)
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    infer.verify_inputs(index, protocol, 600)
                with self.assertRaisesRegex(ValueError, "eight distinct"):
                    infer.verify_inputs({**index, "runs": runs[:-1]}, protocol, 500)


if __name__ == "__main__":
    unittest.main()
