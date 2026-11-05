import unittest

import numpy as np

from experiments.wood_spoon_faceting_audit_v1.render import align_vertex_hemisphere, normal_jumps


class SpoonNormalOrientationTests(unittest.TestCase):
    def test_opposite_face_winding_does_not_create_an_antipodal_vertex_jump(self):
        flat = np.array([[[0., 0., 1.], [0., 0., -1.]]], dtype=np.float32)
        vertex = np.array([[[0., 0., 1.], [0., 0., 1.]]], dtype=np.float32)
        position = np.array([[[0., 0., 2.], [0., 0., 2.]]], dtype=np.float32)
        mask = np.ones((1, 2), dtype=bool)
        sign = np.where((flat * position).sum(-1, keepdims=True) > 0, -1., 1.)
        flat, raw_vertex = flat * sign, vertex * sign
        self.assertEqual(normal_jumps(raw_vertex, mask, (0, 0, 2, 1))['max'], 180.)
        aligned, qc = align_vertex_hemisphere(flat, raw_vertex, mask)
        self.assertEqual(normal_jumps(aligned, mask, (0, 0, 2, 1))['max'], 0.)
        self.assertEqual(qc['aligned_pixels'], 1)
        np.testing.assert_array_equal(aligned, flat)
        np.testing.assert_array_equal(raw_vertex[0, 1], [0., 0., 1.])

    def test_coherent_normals_and_lengths_are_preserved(self):
        flat = np.array([[[0., 0., -1.], [0., 0., -1.]]])
        vertex = np.array([[[.3, 0., -.4], [-.3, 0., -.4]]])
        aligned, qc = align_vertex_hemisphere(flat, vertex, np.ones((1, 2), bool))
        np.testing.assert_array_equal(aligned, vertex)
        np.testing.assert_array_equal(np.linalg.norm(aligned, axis=-1), np.linalg.norm(vertex, axis=-1))
        self.assertEqual(qc['aligned_pixels'], 0)

    def test_zero_and_orthogonal_averages_are_reported_without_fabricated_normals(self):
        flat = np.array([[[0., 0., -1.], [0., 0., -1.], [0., 0., -1.]]])
        vertex = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 0., 1.]]])
        aligned, qc = align_vertex_hemisphere(flat, vertex, np.ones((1, 3), bool))
        self.assertEqual(qc['zero_vertex_normal_pixels'], 1)
        self.assertEqual(qc['near_orthogonal_pixels'], 1)
        self.assertEqual(qc['aligned_pixels'], 1)
        np.testing.assert_array_equal(aligned[0, :2], vertex[0, :2])
        self.assertTrue(((flat * aligned).sum(-1) >= 0).all())


if __name__ == '__main__':
    unittest.main()
