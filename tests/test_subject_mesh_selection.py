import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.subject_trellis_counterfactual_v1.run import OFFICIAL
from src.methods.colorpeel_ice import subject_mesh_selection as selector
from src.methods.colorpeel_ice.single_image_subject import prepare_inputs, sha256

HAS_SCIPY = importlib.util.find_spec('scipy') is not None
VERTICES = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
FACES = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)  # Open/nonmanifold is permitted.


def row(name, iou=.9, distance=.01, eligible=True):
    return {'name': name, 'eligible': eligible, 'metrics': {
        'mask_iou': iou, 'normalized_boundary_distance': distance}}


class GeometryTests(unittest.TestCase):
    def test_open_mesh_and_some_degenerate_faces_are_allowed(self):
        faces = np.vstack([FACES, [0, 0, 0]])
        actual = selector.geometry_checks(VERTICES, faces, np.ones((3, 6)))
        self.assertTrue(actual['passed'])
        self.assertEqual(actual['degenerate_faces'], 1)
        self.assertEqual(actual['positive_area_faces'], 2)

    def test_invalid_arrays_rejected_before_indexing(self):
        broken = [(VERTICES * np.nan, FACES, np.ones(3)),
                  (VERTICES, FACES.astype(float), np.ones(3)),
                  (VERTICES, np.array([[0, 1, 8]]), np.ones(3)),
                  (VERTICES, np.array([[-1, 1, 2]]), np.ones(3)),
                  (VERTICES, FACES, np.array([np.inf])),
                  (VERTICES, np.zeros((0, 3), dtype=int), np.ones(3)),
                  (np.ones((4, 3)), FACES, np.ones(3)),
                  (VERTICES, np.array([[0, 0, 0]]), np.ones(3))]
        for values in broken:
            with self.subTest(values=values):
                self.assertFalse(selector.geometry_checks(*values)['passed'])

    def test_planar_positive_area_mesh_is_valid_but_collinear_geometry_is_not(self):
        planar = VERTICES.copy()
        planar[:, 2] = 0
        self.assertTrue(selector.geometry_checks(planar, FACES, np.ones(3))['passed'])
        collinear = np.array([[0., 0, 0], [1, 1, 1], [2, 2, 2]])
        self.assertFalse(selector.geometry_checks(collinear, np.array([[0, 1, 2]]), np.ones(3))['passed'])


@unittest.skipUnless(HAS_SCIPY, 'SciPy EDT is not installed on this host')
class MaskMetricsTests(unittest.TestCase):
    def test_identical_all_components_holes_and_canvas_edges(self):
        mask = np.zeros((512, 512), bool)
        mask[:20, :30] = True
        mask[3:6, 3:6] = False
        mask[100:115, 200:230] = True
        result = selector.mask_metrics(mask, mask)
        self.assertEqual(result['mask_iou'], 1)
        self.assertEqual(result['boundary_distance_pixels'], 0)
        self.assertEqual(result['sam_bbox_xyxy_inclusive'], [0, 0, 229, 114])
        self.assertAlmostEqual(result['sam_bbox_diagonal_pixels'], np.hypot(230, 115))
        # The hole contributes16 boundary pixels; the two outer rectangles contribute182.
        self.assertEqual(result['sam_boundary_pixels'], 198)

    def test_directional_means_have_equal_weights_and_sam_normalization(self):
        first, second = np.zeros((512, 512), bool), np.zeros((512, 512), bool)
        first[20:80, 30:90] = True
        second[20:80, 30:90] = True
        second[100:103, 100:103] = True
        forward = selector.mask_metrics(first, second)
        reverse = selector.mask_metrics(second, first)
        self.assertGreater(forward['boundary_distance_pixels'], 0)
        self.assertNotEqual(forward['render_to_sam_boundary_pixels'], forward['sam_to_render_boundary_pixels'])
        expected = .5 * (forward['render_to_sam_boundary_pixels'] + forward['sam_to_render_boundary_pixels'])
        self.assertAlmostEqual(forward['boundary_distance_pixels'], expected)
        self.assertAlmostEqual(forward['boundary_distance_pixels'], reverse['boundary_distance_pixels'])
        self.assertAlmostEqual(forward['normalized_boundary_distance'], expected / np.hypot(73, 83))

    def test_empty_or_unaligned_masks_fail(self):
        mask = np.ones((512, 512), bool)
        for other in (np.zeros_like(mask), np.ones((128, 128), bool)):
            with self.assertRaisesRegex(ValueError, 'nonempty aligned'):
                selector.mask_metrics(mask, other)


class ChoiceTests(unittest.TestCase):
    def choose(self, rows, **kwargs):
        return selector.choose(rows, 'source', 'sam', 'object_003', **kwargs)

    def test_none_and_one_valid(self):
        result = self.choose([row('original', eligible=False), row('pink', eligible=False)])
        self.assertEqual(result['status'], 'blocked')
        self.assertIsNone(result['selected'])
        result = self.choose([row('original', eligible=False), row('pink')])
        self.assertEqual(result['selected'], 'pink')
        self.assertFalse(result['quality_accepted_by_user'])

    def test_highest_score_can_differ_from_highest_iou(self):
        rows = [row('original', .95, .1), row('pink', .92, .01)]
        before = copy.deepcopy(rows)
        result = self.choose(rows)
        self.assertEqual(result['selected'], 'pink')
        self.assertEqual(rows, before)

    def test_close_requires_both_thresholds(self):
        for iou, distance in ((.88, .01), (.9, .02)):
            result = self.choose([row('original'), row('pink', iou, distance)])
            self.assertFalse(result['close'])
        result = self.choose([row('original'), row('pink', .89, .015)])
        self.assertTrue(result['close'])

    def test_close_draw_is_reproducible_order_independent_and_source_salted(self):
        rows = [row('original'), row('pink')]
        result = self.choose(rows)
        self.assertEqual(result, self.choose(rows))
        self.assertEqual(result['selected'], self.choose(list(reversed(rows)))['selected'])
        changed_source = selector.choose(rows, 'other source', 'sam', 'object_003')
        changed_sam = selector.choose(rows, 'source', 'other sam', 'object_003')
        self.assertNotEqual(result['random']['derived_seed'], changed_source['random']['derived_seed'])
        self.assertNotEqual(result['random']['derived_seed'], changed_sam['random']['derived_seed'])
        self.assertTrue(result['random']['draw_used'])

    def test_invalid_eligible_metrics_and_seed_rejected(self):
        for invalid in (row('original', .79), row('original', distance=float('nan'))):
            with self.assertRaisesRegex(ValueError, 'invalid metrics'):
                self.choose([invalid, row('pink')])
        with self.assertRaisesRegex(ValueError, 'seed'):
            self.choose([row('original'), row('pink')], seed=-1)


class SelectionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        image, mask = self.root / 'image.png', self.root / 'sam.png'
        Image.new('RGB', (512, 512), (180, 100, 40)).save(image)
        self.sam = np.zeros((512, 512), np.uint8)
        self.sam[100:350, 150:370] = 255
        Image.fromarray(self.sam).save(mask)
        self.semantic = prepare_inputs(image, mask, self.root / 'input')
        self.semantic_path = self.root / 'input/source_receipt.json'
        alpha = self.root / 'soft_alpha.png'
        Image.fromarray(np.where(self.sam > 0, 254, 0).astype(np.uint8)).save(alpha)
        self.runs, self.receipts, self.states = {}, {}, {}
        for name in ('original', 'pink'):
            condition = self.root / ('condition_' + name)
            condition.mkdir()
            rgba = np.zeros((512, 512, 4), np.uint8)
            rgba[..., :3] = np.where(self.sam[..., None] > 0, 100 if name == 'original' else 200, 0)
            rgba[..., 3] = np.where(self.sam > 0, 254, 0)
            cutout = condition / 'cutout.png'
            Image.fromarray(rgba).save(cutout)
            receipt = {**self.semantic, 'semantic_receipt_path': str(self.semantic_path),
                'semantic_receipt_sha256': sha256(self.semantic_path),
                'conditioning_alpha_path': str(alpha), 'conditioning_alpha_sha256': sha256(alpha),
                'cutout_path': str(cutout), 'cutout_sha256': sha256(cutout)}
            if name == 'pink':
                receipt['conditioning_recolor'] = {'hue_degrees': 0}
            receipt_path = condition / 'source_receipt.json'
            self.write(receipt_path, receipt)
            run = self.root / name
            (run / 'repro_outputs').mkdir(parents=True)
            (run / 'review').mkdir()
            (run / 'provenance').mkdir()
            copied = run / 'provenance/source_receipt.json'
            copied.write_bytes(receipt_path.read_bytes())
            (run / 'native_asset.pt').write_bytes(b'mocked native tensor asset')
            (run / 'source_cutout512.png').write_bytes(cutout.read_bytes())
            review = {'camera': {'extrinsics': np.eye(4).tolist(), 'intrinsics': np.eye(3).tolist(),
                                 'refinement_success': True}, 'metrics': {'original512_mask_iou': .9}}
            self.write(run / 'review/review.json', review)
            state = {'status': 'generated_pending_user_review', 'generated_assets': 1,
                'source_receipt': {'path': str(receipt_path), 'copied_path': str(copied),
                                   'sha256': sha256(receipt_path)}, 'review': review,
                'official_commit': OFFICIAL, 'seed': 42, 'pipeline_type': '1024_cascade',
                'encoder_receipt_sha256': 'encoder', 'flow_dtype': {'flow': 'torch.float16'},
                'sampling': {'shape': {'steps': 12}}, 'models': {'sha256': {'model': 'digest'}},
                'sparse_algorithm': 'explicit_gemm', 'git': {'commit': 'a' * 40}}
            self.write(run / 'pipeline.json', {'args': {
                'shape_slat_normalization': {'mean': [0], 'std': [1]},
                'tex_slat_normalization': {'mean': [0], 'std': [1]}}})
            self.write(run / 'repro_outputs/status.json', state)
            self.runs[name], self.receipts[name], self.states[name] = run, receipt_path, state

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding='utf8')

    def run_selection(self, **kwargs):
        native = {'vertices': VERTICES, 'faces': FACES, 'attrs': np.ones((4, 6))}
        with patch.object(selector, '_load_native', return_value=native), \
                patch.object(selector, '_render_opaque', return_value=(self.sam > 0, {'mock': True})) as render, \
                patch.object(selector, 'mask_metrics', return_value={'mask_iou': 1., 'normalized_boundary_distance': 0.}):
            result = selector.select_mesh(self.semantic_path, self.runs, 'object_003', self.root / 'selection', **kwargs)
        self.render_calls = render.call_count
        self.assertEqual(json.loads((self.root / 'selection/selection.json').read_text()), result)
        return result

    def test_completed_pair_keeps_all_evidence_and_semantic_mask(self):
        result = self.run_selection()
        self.assertTrue(result['close'])
        self.assertEqual(self.render_calls, 2)
        self.assertEqual(result['random']['salt']['native_sam_sha256'], self.semantic['original_mask_sha256'])
        for run in self.runs.values():
            for filename in ('native_asset.pt', 'review/review.json', 'repro_outputs/status.json'):
                self.assertIn(str(run / filename), result['source_locks'])
        self.assertIn(self.semantic['mask_path'], result['source_locks'])
        self.assertIn(self.semantic['original_mask_path'], result['source_locks'])
        self.assertEqual(sha256(self.semantic['mask_path']), self.semantic['mask_sha256'])

    def test_missing_or_failed_generation_preserves_other_candidate(self):
        self.states['original']['status'] = 'failed'
        self.write(self.runs['original'] / 'repro_outputs/status.json', self.states['original'])
        result = self.run_selection()
        self.assertEqual(result['selected'], 'pink')
        self.assertIn('did not complete', result['candidates'][0]['reason'])
        self.assertIn(str(self.runs['original'] / 'repro_outputs/status.json'), result['source_locks'])

    def test_both_missing_write_blocked_without_loading_or_rendering(self):
        self.runs = {name: self.root / ('missing_' + name) for name in self.runs}
        result = self.run_selection()
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(self.render_calls, 0)

    def test_mismatched_sampler_blocks_completed_pair(self):
        self.states['pink']['sampling']['shape']['steps'] = 13
        self.write(self.runs['pink'] / 'repro_outputs/status.json', self.states['pink'])
        result = self.run_selection()
        self.assertEqual(result['status'], 'blocked')
        self.assertFalse(result['completed_candidates_comparable'])
        self.assertEqual(self.render_calls, 0)

    def test_mismatched_normalization_blocks_completed_pair(self):
        pipeline_path = self.runs['pink'] / 'pipeline.json'
        pipeline = json.loads(pipeline_path.read_text())
        pipeline['args']['shape_slat_normalization']['mean'] = [1]
        self.write(pipeline_path, pipeline)
        result = self.run_selection()
        self.assertEqual(result['status'], 'blocked')
        self.assertFalse(result['completed_candidates_comparable'])

    def test_mismatched_generation_commit_blocks_completed_pair(self):
        self.states['pink']['git']['commit'] = 'b' * 40
        self.write(self.runs['pink'] / 'repro_outputs/status.json', self.states['pink'])
        result = self.run_selection()
        self.assertEqual(result['status'], 'blocked')
        self.assertFalse(result['completed_candidates_comparable'])

    def test_different_soft_alpha_blocks_pair_even_with_identical_sam(self):
        receipt_path = self.receipts['pink']
        receipt = json.loads(receipt_path.read_text())
        alpha_path = receipt_path.parent / 'different_alpha.png'
        alpha = np.where(self.sam > 0, 253, 0).astype(np.uint8)
        Image.fromarray(alpha).save(alpha_path)
        with Image.open(receipt['cutout_path']) as opened:
            rgba = np.array(opened)
        rgba[..., 3] = alpha
        Image.fromarray(rgba).save(receipt['cutout_path'])
        receipt.update(conditioning_alpha_path=str(alpha_path), conditioning_alpha_sha256=sha256(alpha_path),
                       cutout_sha256=sha256(receipt['cutout_path']))
        self.write(receipt_path, receipt)
        self.states['pink']['source_receipt']['sha256'] = sha256(receipt_path)
        Path(self.states['pink']['source_receipt']['copied_path']).write_bytes(receipt_path.read_bytes())
        (self.runs['pink'] / 'source_cutout512.png').write_bytes(Path(receipt['cutout_path']).read_bytes())
        self.write(self.runs['pink'] / 'repro_outputs/status.json', self.states['pink'])
        result = self.run_selection()
        self.assertIsNone(result['selected'])
        self.assertFalse(result['completed_candidates_comparable'])

    def test_native_evaluation_failure_does_not_skip_valid_sibling(self):
        native = {'vertices': VERTICES, 'faces': FACES, 'attrs': np.ones((4, 6))}
        with patch.object(selector, '_load_native', side_effect=[ValueError('broken native asset'), native]), \
                patch.object(selector, '_render_opaque', return_value=(self.sam > 0, {'mock': True})) as render, \
                patch.object(selector, 'mask_metrics', return_value={'mask_iou': 1., 'normalized_boundary_distance': 0.}):
            result = selector.select_mesh(self.semantic_path, self.runs, 'object_003', self.root / 'selection')
        self.assertEqual(result['selected'], 'pink')
        self.assertIn('broken native asset', result['candidates'][0]['reason'])
        self.assertEqual(render.call_count, 1)

    def test_cli_returns_normally_with_blocked_missing_candidates(self):
        with patch.object(selector.sys, 'argv', ['selector', '--semantic-receipt', str(self.semantic_path),
                '--candidate', 'original=' + str(self.root / 'missing_original'),
                '--candidate', 'pink=' + str(self.root / 'missing_pink'),
                '--object-id', 'object_003', '--output', str(self.root / 'selection')]):
            self.assertIsNone(selector.main())
        result = json.loads((self.root / 'selection/selection.json').read_text())
        self.assertEqual(result['status'], 'blocked')
        self.assertIsNone(result['selected'])

    def test_low_saved_review_rejects_before_render_without_camera_refit(self):
        self.states['original']['review']['metrics']['original512_mask_iou'] = .7
        self.write(self.runs['original'] / 'repro_outputs/status.json', self.states['original'])
        self.write(self.runs['original'] / 'review/review.json', self.states['original']['review'])
        result = self.run_selection()
        self.assertEqual(result['selected'], 'pink')
        self.assertEqual(self.render_calls, 1)

    def test_changed_semantic_binding_is_not_comparable_evidence(self):
        receipt = json.loads(self.receipts['original'].read_text())
        receipt['semantic_receipt_sha256'] = 'changed'
        self.write(self.receipts['original'], receipt)
        Path(self.states['original']['source_receipt']['copied_path']).write_bytes(self.receipts['original'].read_bytes())
        self.states['original']['source_receipt']['sha256'] = sha256(self.receipts['original'])
        self.write(self.runs['original'] / 'repro_outputs/status.json', self.states['original'])
        result = self.run_selection()
        self.assertEqual(result['selected'], 'pink')
        self.assertIn('semantic receipt binding', result['candidates'][0]['reason'])

    def test_actual_frame_clipping_is_rejected_even_with_good_saved_review(self):
        clipped = self.sam > 0
        clipped[0, 200] = True
        with patch.object(selector, '_load_native', return_value={
                'vertices': VERTICES, 'faces': FACES, 'attrs': np.ones((4, 6))}), \
                patch.object(selector, '_render_opaque', return_value=(clipped, {'mock': True})), \
                patch.object(selector, 'mask_metrics', return_value={'mask_iou': .99, 'normalized_boundary_distance': .001}):
            result = selector.select_mesh(self.semantic_path, self.runs, 'object_003', self.root / 'selection')
        self.assertIsNone(result['selected'])
        self.assertIn('border clipping', result['candidates'][0]['reason'])
        self.assertIsNotNone(result['candidates'][0]['metrics'])


if __name__ == '__main__':
    unittest.main()
