import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.subject_trellis_counterfactual_v1 import review_asset as review


class SubjectCameraContinuationTests(unittest.TestCase):
    def optimizer(self, callback):
        optimize = ModuleType('scipy.optimize')
        optimize.minimize = Mock(side_effect=callback)
        scipy = ModuleType('scipy')
        scipy.optimize = optimize
        context = patch.dict('sys.modules', {'scipy': scipy, 'scipy.optimize': optimize})
        return context, optimize.minimize

    def test_default_retains_72_candidates_and_original_single_optimizer_budget(self):
        seen = []

        def objective(point):
            seen.append(list(point))
            return sum(float(value) ** 2 for value in point)

        def minimize(fun, initial, **kwargs):
            self.assertEqual(len(seen), 72)
            self.assertEqual(len({tuple(point) for point in seen}), 72)
            self.assertTrue(all(point[3:] == [2.8, .6, .2] for point in seen))
            self.assertEqual(kwargs, {'method': 'Powell', 'bounds': review.BOUNDS,
                'options': {'maxfev': 350, 'maxiter': 6, 'xtol': .005, 'ftol': .001}})
            fun(initial)
            return SimpleNamespace(success=False, message='test budget reached')

        context, optimizer = self.optimizer(minimize)
        with context:
            _, _, _, trajectory, metadata = review.fit_camera(objective, [.6, .2])
        optimizer.assert_called_once()
        self.assertEqual(len(trajectory), 73)
        self.assertEqual(metadata['fitting_mode'], 'coarse72_then_powell')

    def test_continuation_skips_coarse_and_retains_baseline_if_refinement_worse(self):
        initial = [1., .2, 0., 1.5, .6, .2]
        objective = Mock(side_effect=lambda point: float(np.square(np.array(point) - initial).sum()))

        def minimize(fun, start, **kwargs):
            self.assertEqual(objective.call_count, 1)
            self.assertEqual(start, initial)
            self.assertEqual(kwargs, {'method': 'Powell', 'bounds': review.BOUNDS,
                'options': {'maxfev': 1000, 'maxiter': 10, 'xtol': .005, 'ftol': .001}})
            worse = [1.1, *initial[1:]]
            fun(worse)
            fun(start)
            return SimpleNamespace(success=False, message='test budget reached', x=worse)

        context, optimizer = self.optimizer(minimize)
        with context:
            loss, best, _, trajectory, metadata = review.fit_camera(objective, None, initial)
        optimizer.assert_called_once()
        self.assertEqual((loss, best), (0., initial))
        self.assertEqual(len(trajectory), 3)
        self.assertEqual([point for _, point in trajectory],
                         [initial, [1.1, *initial[1:]], initial])
        self.assertEqual(metadata['initial_parameters'], dict(zip(review.PARAMETERS, initial)))
        self.assertEqual(metadata['fitting_mode'], 'single_warm_start_powell')

    def test_bad_initial_parameters_fail_before_renderer_import(self):
        invalid = ([0] * 5, [[0] * 6], [0., 0., 0., 1.5, np.nan, .5],
                   [0., 0., 0., np.inf, .5, .5], [0., 0., 0., 1., .5, .5])
        for parameters in invalid:
            with self.subTest(parameters=parameters), patch.dict('sys.modules', {'torch': None}), self.assertRaises(ValueError):
                review.review(None, None, None, None, initial_parameters=parameters)

    def test_initial_review_is_bound_to_asset_directory_and_identical_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / 'native_asset.pt'
            asset.write_bytes(b'fixture')
            source = root / 'source_cutout512.png'
            source.write_bytes(b'source fixture')
            supplied_source = root / 'same_source.png'
            supplied_source.write_bytes(source.read_bytes())
            initial_review = root / 'review/review.json'
            initial_review.parent.mkdir()
            initial = [1., .2, .1, 1.5, .6, .2]
            # JSON key order deliberately differs from the six optimizer parameters.
            parameters = dict(reversed(list(zip(review.PARAMETERS, initial))))
            initial_review.write_text(json.dumps({'camera': {'parameters': parameters}}))
            self.assertEqual(review.read_initial_parameters(initial_review, asset, supplied_source), initial)
            wrong_review = root / 'another_review.json'
            wrong_review.write_bytes(initial_review.read_bytes())
            with self.assertRaisesRegex(ValueError, 'existing review'):
                review.read_initial_parameters(wrong_review, asset, source)
            supplied_source.write_bytes(b'another image')
            with self.assertRaisesRegex(ValueError, 'source differs'):
                review.read_initial_parameters(initial_review, asset, supplied_source)


if __name__ == '__main__':
    unittest.main()
