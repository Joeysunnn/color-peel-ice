import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from src.methods.colorpeel_ice import subject_alpha as alpha_module
from src.methods.colorpeel_ice.single_image_subject import prepare_inputs, sha256

HAS_SCIPY = importlib.util.find_spec('scipy') is not None


class CachedBiRefNetTests(unittest.TestCase):
    def test_cached_files_verified_offline_and_missing_or_changed_files_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / 'models/birefnet/pinned'
            bundle.mkdir(parents=True)
            code, weight = bundle / 'birefnet.py', bundle / 'model.safetensors'
            code.write_bytes(b'fixture code')
            weight.write_bytes(b'fixture weights')
            blob = hashlib.sha1(b'blob 12\0fixture code').hexdigest()
            spec = root / 'inputs.json'
            spec.write_text(json.dumps({'official_commit': 'a' * 40, 'segmentation': {
                'revision': 'pinned', 'files': {
                    code.name: {'size': code.stat().st_size, 'git_blob_sha1': blob},
                    weight.name: {'size': weight.stat().st_size, 'sha256': sha256(weight)}}}}))
            with patch.object(alpha_module, 'ASSETS', root), patch.object(alpha_module, 'SPEC', spec):
                actual, _, locks = alpha_module.cached_bundle()
                self.assertEqual(actual, bundle)
                self.assertEqual(set(locks), {str(spec), str(code), str(weight)})
                code.write_bytes(b'Fixture code')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    alpha_module.cached_bundle()
                code.write_bytes(b'fixture code')
                weight.unlink()
                with self.assertRaises(FileNotFoundError):
                    alpha_module.cached_bundle()
                self.assertFalse(weight.exists())

    def test_wrong_official_commit_rejected_before_loading_model(self):
        with patch.object(alpha_module, 'cached_bundle', return_value=(Path('/fixture'),
                {'official_commit': 'a' * 40}, {})), \
                patch.object(alpha_module.subprocess, 'check_output', return_value='b' * 40):
            with self.assertRaisesRegex(ValueError, 'pinned clean'):
                alpha_module.load_segmenter()

    def test_invalid_alpha_or_masks_rejected_without_label_dependency(self):
        with self.assertRaisesRegex(ValueError, 'uint8'):
            alpha_module.associate_components(np.ones((4, 4)), {'object': np.ones((4, 4))}, 'object')
        with self.assertRaisesRegex(ValueError, 'missing'):
            alpha_module.associate_components(np.ones((4, 4), dtype=np.uint8), {}, 'object')
        with self.assertRaisesRegex(ValueError, 'aligned'):
            alpha_module.associate_components(np.ones((4, 4), dtype=np.uint8),
                                               {'object': np.zeros((4, 4))}, 'object')


@unittest.skipUnless(HAS_SCIPY, 'SciPy label is not installed on this host')
class ComponentAssociationTests(unittest.TestCase):
    def test_full_component_preserves_254_body_and_outside_sam_soft_halo(self):
        alpha = np.zeros((10, 12), dtype=np.uint8)
        alpha[2:8, 2:8] = 1
        alpha[3:7, 3:7] = 254
        sam = np.zeros_like(alpha, dtype=bool)
        sam[3:7, 3:7] = True
        before = alpha.copy()
        selected, report = alpha_module.associate_components(alpha, {'object': sam}, 'object')
        np.testing.assert_array_equal(selected, before)
        np.testing.assert_array_equal(alpha, before)
        self.assertEqual(report['sam_coverage'], 1)
        self.assertEqual(selected[2, 2], 1)
        self.assertEqual(selected[4, 4], 254)

    def test_diagonal_support_is_eight_connected_and_whole_components_are_selected(self):
        alpha = np.zeros((12, 14), dtype=np.uint8)
        alpha[1:3, 1:3] = 254
        alpha[3, 3] = 1  # Diagonal-only connection to the selected body's component.
        alpha[6:8, 2:4] = 128  # A second disconnected piece of the selected instance.
        alpha[2:4, 9:11] = 253  # Another instance.
        alpha[9, 12] = 10  # Unassigned component.
        first, second = np.zeros_like(alpha, bool), np.zeros_like(alpha, bool)
        first[1:3, 1:3] = True
        first[6:8, 2:4] = True
        second[2:4, 9:11] = True
        selected, report = alpha_module.associate_components(alpha, {'first': first, 'second': second}, 'first')
        expected = alpha.copy()
        expected[2:4, 9:11] = 0
        expected[9, 12] = 0
        np.testing.assert_array_equal(selected, expected)
        self.assertEqual(selected[3, 3], 1)
        self.assertEqual(len(report['selected_components']), 2)

    def test_shared_component_is_rejected_without_clipping_at_sam_boundary(self):
        alpha = np.full((8, 10), 254, dtype=np.uint8)
        first, second = np.zeros_like(alpha, bool), np.zeros_like(alpha, bool)
        first[:, :4], second[:, 6:] = True, True
        with self.assertRaisesRegex(ValueError, 'multiple SAM instances'):
            alpha_module.associate_components(alpha, {'first': first, 'second': second}, 'first')
        self.assertTrue((alpha == 254).all())

    def test_absent_or_low_coverage_is_rejected(self):
        sam = np.zeros((10, 10), bool)
        sam[2:8, 2:8] = True
        for box in ((slice(0, 1), slice(0, 1)), (slice(2, 4), slice(2, 4))):
            with self.subTest(box=box):
                alpha = np.zeros(sam.shape, dtype=np.uint8)
                alpha[box] = 254
                with self.assertRaisesRegex(ValueError, 'coverage'):
                    alpha_module.associate_components(alpha, {'object': sam}, 'object')


@unittest.skipUnless(HAS_SCIPY, 'SciPy label is not installed on this host')
class SubjectAlphaPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.segmentation = self.root / 'segmentation'
        self.segmentation.mkdir()
        self.source = self.segmentation / 'source.png'
        self.image = Image.linear_gradient('L').resize((80, 60)).convert('RGB')
        self.image.save(self.source)
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[14:46, 22:58] = 255
        self.mask = self.segmentation / 'sam_001.png'
        Image.fromarray(mask).save(self.mask)
        self.original = prepare_inputs(self.source, self.mask, self.root / 'semantic_input')
        self.parent = self.root / 'semantic_input/source_receipt.json'
        self.objects = self.segmentation / 'objects.json'
        self.manifest = {'status': 'succeeded', 'source': {'canonical_image': str(self.source),
            'canonical_image_sha256': sha256(self.source)}, 'objects': [{
                'id': 'object_001', 'source_image': str(self.source),
                'object_mask': str(self.mask), 'object_mask_sha256': sha256(self.mask)}]}
        alpha_module.write_json(self.objects, self.manifest)
        alpha = np.zeros(mask.shape, dtype=np.uint8)
        alpha[10:50, 18:62], alpha[12:48, 20:60] = 1, 254
        self.segmented = self.image.convert('RGBA')
        self.segmented.putalpha(Image.fromarray(alpha))

    def test_single_component_exactly_reproduces_historical_joint_rgba_resize(self):
        other = self.segmentation / 'sam_002.png'
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[20:30, 70:75] = 255
        Image.fromarray(mask).save(other)
        self.manifest['objects'].append({'id': 'object_002', 'source_image': str(self.source),
            'object_mask': str(other), 'object_mask_sha256': sha256(other)})
        alpha_module.write_json(self.objects, self.manifest)
        before = {path: Path(path).read_bytes() for path in self.original['source_locks']}
        before[str(self.parent)] = self.parent.read_bytes()
        before[str(self.objects)] = self.objects.read_bytes()
        before[str(other)] = other.read_bytes()
        predictor = Mock(return_value=self.segmented.copy())
        receipt = alpha_module.prepare(self.parent, self.objects, 'object_001', self.root / 'alpha', predictor)
        observed = predictor.call_args.args[0]
        self.assertEqual(observed.size, (80, 60))
        self.assertEqual(observed.tobytes(), self.image.tobytes())
        expected, transform = alpha_module.square_cutout(self.segmented)
        with Image.open(receipt['cutout_path']) as actual:
            self.assertEqual(actual.tobytes(), expected.tobytes())
        for key in ('source_path', 'source_sha256', 'image512_path', 'image512_sha256',
                    'mask_path', 'mask_sha256', 'original_mask_path', 'original_mask_sha256'):
            self.assertEqual(receipt[key], self.original[key])
        self.assertFalse(receipt['alpha_matches_mask'])
        self.assertEqual(receipt['semantic_receipt_sha256'], sha256(self.parent))
        self.assertEqual(receipt['conditioning_alpha']['transform'], transform)
        self.assertIn(str(self.objects), receipt['source_locks'])
        self.assertIn(str(self.mask), receipt['source_locks'])
        self.assertEqual(receipt['source_locks'][str(other)], sha256(other))
        self.assertEqual({path: Path(path).read_bytes() for path in before}, before)
        alpha_module.check_source_receipt(self.root / 'alpha/source_receipt.json')

    def test_other_sam_instance_tampering_is_rejected_before_inference(self):
        other = self.segmentation / 'sam_002.png'
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[20:30, 70:75] = 255
        Image.fromarray(mask).save(other)
        self.manifest['objects'].append({'id': 'object_002', 'source_image': str(self.source),
            'object_mask': str(other), 'object_mask_sha256': sha256(other)})
        alpha_module.write_json(self.objects, self.manifest)
        mask[0, 0] = 255
        Image.fromarray(mask).save(other)
        predictor = Mock(return_value=self.segmented.copy())
        with self.assertRaisesRegex(ValueError, 'SAM instance mask changed'):
            alpha_module.prepare(self.parent, self.objects, 'object_001', self.root / 'alpha', predictor)
        predictor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
