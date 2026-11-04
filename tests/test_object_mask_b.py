import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from experiments.object_mask_ab_v1.run import refine, sample_points


class SchemeBTests(unittest.TestCase):
    def test_points_are_distinct_seeded_foreground_xy(self):
        mask = np.zeros((12, 40), dtype=bool)
        mask[2:6, 25:34] = True
        points = sample_points(mask, np.random.default_rng(20))
        self.assertEqual(points, sample_points(mask, np.random.default_rng(20)))
        self.assertEqual(len(set(map(tuple, points))), 3)
        self.assertTrue(all(mask[y, x] for x, y in points))

    def test_insufficient_foreground_fails_without_fabricated_points(self):
        with self.assertRaises(ValueError):
            sample_points(np.array([[True, False, True]]), np.random.default_rng(20))

    def test_refinement_keeps_concepts_separate_and_records_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / 'coarse').mkdir()
            image = Image.new('RGB', (96, 48), 'orange')
            first = np.zeros((64, 64), dtype=bool)
            first[5:25, 5:25] = True
            second = np.flip(first, axis=1).copy()
            third = np.ones((64, 64), dtype=bool)
            calls = []

            def backend(image, points):
                calls.append(points)
                if len(calls) == 2:
                    raise RuntimeError('injected SAM failure')
                mask = np.zeros((image.height, image.width), dtype=bool)
                x, y = points[0]
                mask[max(0, y-1):y+2, max(0, x-1):x+2] = True
                return mask, [0.4, 0.8, 0.2], 1

            rows, failures = refine(image, [first, second, third], output, 20, backend)
            self.assertEqual([row['id'] for row in rows], ['concept_001', 'concept_003'])
            self.assertEqual(failures[0]['concept_id'], 'concept_002')
            self.assertEqual(len(calls), 3)
            for row in rows:
                self.assertEqual(row['selected_candidate'], 1)
                self.assertEqual(row['sam_candidate_scores'], [0.4, 0.8, 0.2])
                with Image.open(row['object_mask']) as mask_image:
                    self.assertEqual(mask_image.size, (96, 48))
                self.assertEqual(json.loads((output / 'coarse' /
                    f"{row['id']}_points.json").read_text()), row['positive_points_xy'])
            self.assertTrue((output / 'coarse/concept_002_points.json').exists())


if __name__ == '__main__':
    unittest.main()
