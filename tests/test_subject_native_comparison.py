import unittest

import numpy as np

from experiments.subject_pink_input_repro_v1.compare import array_comparison, plain


class SubjectNativeComparisonTests(unittest.TestCase):
    def test_small_difference_is_not_hidden_by_a_tolerance(self):
        original = np.array([1, 2], dtype=np.float32)
        changed = original.copy()
        changed[0] = np.nextafter(changed[0], np.float32(2))
        self.assertFalse(array_comparison(original, changed)['exactly_equal'])
        self.assertTrue(array_comparison(original, original.copy())['exactly_equal'])

    def test_shape_or_dtype_changes_are_not_silently_aligned(self):
        original = np.arange(6, dtype=np.float32).reshape(2, 3)
        resized = array_comparison(original, original.reshape(3, 2))
        self.assertFalse(resized['exactly_equal'])
        self.assertNotIn('max_absolute_difference', resized)
        self.assertFalse(array_comparison(original, original.astype(np.float64))['exactly_equal'])

    def test_voxel_layout_slice_bounds_are_preserved(self):
        self.assertEqual(plain({'color': slice(0, 3), 'size': (2, 3, 4)}),
                         {'color': {'slice': [0, 3, None]}, 'size': [2, 3, 4]})


if __name__ == '__main__':
    unittest.main()
