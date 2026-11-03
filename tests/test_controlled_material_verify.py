"""Regression: smoke run paths must satisfy the real launcher contract."""

from pathlib import Path
import unittest

from experiments.sd21_controlled_material_v2.verify import training_directory
from scripts.launch.colorpeel_run import validate_run_id


class VerificationRunTests(unittest.TestCase):
    def test_path_preserves_recipe_study_variant_and_seed(self):
        config = {'run': {'study': 'sd21_qwen_initializers_v1', 'variant': 'material_smoke2', 'seed': 42}}
        revision = {'commit': '6b5a263c13ce5d48875b43f5dbb041d4eaea0122'}
        path = training_directory(Path('/runs/verification'), config, revision)
        validate_run_id(path, config, revision['commit'])
        self.assertEqual(path.parent, Path('/runs/verification/training'))
        with self.assertRaises(ValueError):
            validate_run_id(Path('20261103-230116__controlled_material_smoke2__6b5a263__42'), config, revision['commit'])


if __name__ == '__main__':
    unittest.main()
