import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).parents[3] / "src" / "train" / "token_gradient_utils.py"
SPEC = importlib.util.spec_from_file_location("token_gradient_utils", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ModifierGradientMaskTests(unittest.TestCase):
    def test_all_six_modifier_rows_are_preserved(self):
        modifier_ids = [10, 11, 12, 20, 21, 22]
        rows_to_zero = MODULE.modifier_rows_to_zero(30, modifier_ids)

        self.assertEqual([i for i, zero in enumerate(rows_to_zero) if not zero], modifier_ids)
        self.assertTrue(all(rows_to_zero[i] for i in range(30) if i not in modifier_ids))

    def test_rejects_invalid_or_empty_ids(self):
        with self.assertRaises(ValueError):
            MODULE.modifier_rows_to_zero(10, [])
        with self.assertRaises(ValueError):
            MODULE.modifier_rows_to_zero(10, [10])


if __name__ == "__main__":
    unittest.main()
