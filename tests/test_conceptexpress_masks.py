"""Focused contract checks for the inference-only ConceptExpress adapter."""

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from src.methods.colorpeel_ice import conceptexpress_masks as masks


class ConceptExpressMasksTests(unittest.TestCase):
    def test_precluster_uses_last_level_above_ten_not_fixed_object_count(self):
        self.assertEqual(masks.select_precluster_index([100, 27, 12, 4]), 2)
        self.assertEqual(masks.select_precluster_index([100, 10, 4]), 1)
        for counts in ([9, 4], [100, 27], [10], []):
            with self.subTest(counts=counts), self.assertRaisesRegex(ValueError, 'crossing'):
                masks.select_precluster_index(counts)

    def test_native_mapping_keeps_asymmetric_position_without_flip_or_crop(self):
        latent = np.zeros((64, 64), dtype=bool)
        latent[8:24, 40:56] = True
        native = masks.mask_to_native(latent, (128, 256))
        expected = np.zeros((256, 128), dtype=bool)
        expected[32:96, 80:112] = True
        np.testing.assert_array_equal(native, expected)
        for invalid in (np.zeros((32, 64)), np.full((64, 64), .5)):
            with self.assertRaises(ValueError):
                masks.mask_to_native(invalid, (128, 256))

    def test_background_filter_and_postcluster_keep_separate_concept_masks(self):
        labels = np.minimum(np.arange(4096) // 410, 9)
        eot = np.zeros((64, 64), dtype=np.float32)
        eot.ravel()[labels < 2] = 1
        features = np.ones((4096, 3), dtype=np.float32) / 3
        calls = []

        def finch(data, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return np.column_stack([labels, labels, labels // 4]), [25, 10, 3], None, [None, .2, .4]
            self.assertEqual(data.shape, (2, 3))
            self.assertAlmostEqual(kwargs['min_sim'], .2)
            np.testing.assert_array_equal(kwargs['mask_candidate'][0], (labels == 0).reshape(64, 64))
            return np.array([[0], [1]]), [2], None, [.2]

        found, info = masks.cluster_attention(features, eot, finch)
        self.assertEqual(len(found), 2)
        self.assertFalse((found[0] & found[1]).any())
        np.testing.assert_array_equal(found[1], (labels == 1).reshape(64, 64))
        self.assertEqual(info['precluster_index'], 1)
        self.assertEqual([row['label'] for row in info['candidate_scores'] if row['retained']], [0, 1])
        self.assertEqual(calls[0]['distance'], 'kld')
        self.assertFalse(calls[0]['ensure_early_exit'])
        self.assertTrue(calls[1]['ensure_early_exit'])

    def test_equal_global_saliency_rejects_all_without_fallback(self):
        labels = np.minimum(np.arange(4096) // 410, 9)
        def finch(*args, **kwargs):
            return np.column_stack([labels, labels // 4]), [10, 3], None, [None, .2]
        with self.assertRaisesRegex(ValueError, 'no foreground'):
            masks.cluster_attention(np.ones((4096, 3)), np.ones((64, 64)), finch)

    def test_wrong_or_dirty_upstream_is_rejected_before_code_execution(self):
        for responses in (['wrongcommit'], [masks.UPSTREAM_COMMIT, ' M clustering/finch.py']):
            with self.subTest(responses=responses), patch.object(
                    masks.subprocess, 'check_output', side_effect=responses):
                with self.assertRaisesRegex(ValueError, 'pinned, clean'):
                    masks.load_finch('unused_checkout')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch test runs in server environment')
    def test_capture_uses_last_half_heads_but_preserves_full_attention_output(self):
        import torch

        probs = torch.stack([torch.eye(4), torch.eye(4),
                             torch.ones((4, 4)) / 4, torch.ones((4, 4)) / 4])
        class Attention:
            to_q = to_k = to_v = staticmethod(lambda x: x)
            to_out = [lambda x: x, lambda x: x]
            prepare_attention_mask = staticmethod(lambda *args: None)
            head_to_batch_dim = staticmethod(lambda x: x.repeat(4, 1, 1))
            get_attention_scores = staticmethod(lambda *args: probs)
            batch_to_head_dim = staticmethod(lambda x: x.mean(0, keepdim=True))

        store = []
        hidden = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
        result = masks.AttentionCapture(store, 'down')(Attention(), hidden)
        self.assertEqual(store[0][:3], ('down', False, 2))
        torch.testing.assert_close(store[0][3], torch.ones((4, 4)) / 4)
        expected = torch.bmm(probs, hidden.repeat(4, 1, 1)).mean(0, keepdim=True)
        torch.testing.assert_close(result, expected)


if __name__ == '__main__':
    unittest.main()
