import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.methods.colorpeel_ice import object_segmentation as segmentation


def candidate(mask, category='mug', score=0.8):
    height, width = mask.shape
    return {'mask': mask, 'category': category, 'box': [0, 0, width, height],
            'detection_score': score, 'sam_iou_score': 0.95}


class ObjectSegmentationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / 'input.png'
        pixels = np.zeros((20, 30, 3), dtype=np.uint8)
        pixels[:, :15] = [220, 40, 10]
        pixels[:, 15:] = [10, 40, 220]
        Image.fromarray(pixels).save(self.image)

    def run_stage(self, categories, backend, output='run', image=None):
        path = segmentation.run(image or self.image, categories, self.root / output, segmenter=backend)
        manifest = json.loads(path.read_text())
        self.assertEqual(json.loads((path.parent / 'status.json').read_text())['status'], manifest['status'])
        return manifest

    def test_same_category_instances_remain_separate_and_are_not_unioned(self):
        first = np.zeros((20, 30), dtype=bool)
        second = first.copy()
        first[3:17, 2:11] = True
        second[2:18, 20:28] = True
        manifest = self.run_stage(['mug'], lambda image, category: [candidate(first), candidate(second)])
        self.assertEqual(manifest['status'], 'succeeded')
        self.assertEqual(len(manifest['objects']), 2)
        for row, expected in zip(manifest['objects'], (first, second)):
            with Image.open(row['object_mask']) as image:
                self.assertEqual(image.mode, 'L')
                self.assertEqual(image.size, (30, 20))
                np.testing.assert_array_equal(np.asarray(image), expected.astype(np.uint8) * 255)
            self.assertEqual(row['object_mask_sha256'], segmentation.sha256(row['object_mask']))
        self.assertFalse(manifest['visual_quality_verified'])

    def test_cross_category_duplicate_keeps_highest_detector_score(self):
        mask = np.zeros((20, 30), dtype=bool)
        mask[4:16, 4:13] = True
        manifest = self.run_stage(['mug', 'cup'], lambda image, category:
            [candidate(mask, category, 0.9 if category == 'cup' else 0.6)])
        self.assertEqual(manifest['status'], 'succeeded')
        self.assertEqual([row['category'] for row in manifest['objects']], ['cup'])
        self.assertEqual(manifest['rejected'][0]['reason'], 'duplicate_mask')
        self.assertEqual(manifest['rejected'][0]['mask_iou'], 1.0)
        self.assertEqual(manifest['rejected'][0]['duplicate_criterion'], 'mask_iou')

    def test_contained_lower_score_part_is_suppressed_with_overlap_evidence(self):
        outer = np.zeros((20, 30), dtype=bool)
        outer[2:18, 2:28] = True
        inner = np.zeros_like(outer)
        inner[5:10, 5:10] = True
        manifest = self.run_stage(['mug'], lambda image, category:
                                  [candidate(inner, score=0.6), candidate(outer, score=0.9)])
        self.assertEqual(manifest['status'], 'succeeded')
        self.assertEqual(len(manifest['objects']), 1)
        self.assertEqual(manifest['objects'][0]['mask_pixels'], int(outer.sum()))
        rejected = manifest['rejected'][0]
        self.assertEqual(rejected['duplicate_criterion'], 'mask_overlap_min')
        self.assertEqual(rejected['mask_overlap_min'], 1.0)
        self.assertEqual(rejected['mask_pixels'], int(inner.sum()))
        self.assertEqual(rejected['mask_intersection_pixels'], int(inner.sum()))
        self.assertLess(rejected['mask_iou'], segmentation.DEDUPLICATION_IOU)
        self.assertEqual(rejected['detection_score'], 0.6)

    def test_lower_score_surrounding_region_is_suppressed_across_categories(self):
        surrounding = np.ones((20, 30), dtype=bool)
        object_mask = np.zeros_like(surrounding)
        object_mask[5:15, 5:15] = True
        manifest = self.run_stage(['door', 'mug'], lambda image, category:
            [candidate(surrounding, category, 0.3)] if category == 'door'
            else [candidate(object_mask, category, 0.8)])
        self.assertEqual([row['category'] for row in manifest['objects']], ['mug'])
        self.assertEqual(manifest['rejected'][0]['category'], 'door')
        self.assertEqual(manifest['rejected'][0]['duplicate_criterion'], 'mask_overlap_min')

    def test_separate_objects_with_small_mask_overlap_are_retained(self):
        first = np.zeros((20, 30), dtype=bool)
        second = first.copy()
        first[3:15, 3:16] = True
        second[3:15, 14:27] = True
        manifest = self.run_stage(['mug'], lambda image, category: [candidate(first), candidate(second)])
        self.assertEqual(manifest['status'], 'succeeded')
        self.assertEqual(len(manifest['objects']), 2)
        self.assertEqual(manifest['rejected'], [])

    def test_crop_hides_other_objects_and_cutout_has_zero_transparent_rgb(self):
        mask = np.zeros((20, 30), dtype=bool)
        mask[5:15, 10:20] = True
        mask[8:10, 14:16] = False
        manifest = self.run_stage(['mug'], lambda image, category: [candidate(mask)])
        row = manifest['objects'][0]
        original = np.asarray(Image.open(self.image))
        crop = row['crop_box_xyxy']
        expected = np.where(mask[..., None], original, 127).astype(np.uint8)
        np.testing.assert_array_equal(np.asarray(Image.open(row['vlm_input'])),
                                      expected[crop[1]:crop[3], crop[0]:crop[2]])
        cutout = np.asarray(Image.open(row['cutout']))
        self.assertTrue((cutout[~mask] == 0).all())
        np.testing.assert_array_equal(cutout[mask, :3], original[mask])
        self.assertTrue((cutout[mask, 3] == 255).all())

    def test_exif_orientation_uses_canonical_coordinates(self):
        image = self.root / 'rotated.jpg'
        exif = Image.Exif()
        exif[274] = 6
        Image.new('RGB', (30, 20), 'red').save(image, exif=exif)

        def backend(canonical, category):
            self.assertEqual(canonical.size, (20, 30))
            mask = np.zeros((30, 20), dtype=bool)
            mask[2:20, 3:10] = True
            return [candidate(mask)]

        manifest = self.run_stage(['mug'], backend, image=image)
        self.assertEqual(manifest['source']['original_size'], [30, 20])
        self.assertEqual(manifest['source']['input_size'], [20, 30])
        self.assertEqual(Image.open(manifest['source']['canonical_image']).size, (20, 30))
        self.assertEqual(Image.open(manifest['objects'][0]['object_mask']).size, (20, 30))

    def test_empty_and_invalid_masks_fail_without_usable_object_records(self):
        for index, mask in enumerate((np.zeros((20, 30), dtype=bool), np.ones((10, 10)),
                                     np.full((20, 30), np.nan), np.full((20, 30), 0.5))):
            with self.subTest(index=index):
                manifest = self.run_stage(['mug'], lambda image, category: [candidate(mask)], output=f'bad{index}')
                self.assertEqual(manifest['status'], 'failed')
                self.assertEqual(manifest['objects'], [])
                self.assertEqual(len(manifest['rejected']), 1)
        manifest = self.run_stage(['mug'], lambda image, category: [], output='empty')
        self.assertEqual(manifest['failures'][0]['reason'], 'no_detection')
        self.assertEqual(manifest['status'], 'failed')

    def test_some_valid_instances_and_a_failed_category_are_partial(self):
        mask = np.zeros((20, 30), dtype=bool)
        mask[5:15, 10:20] = True

        def backend(image, category):
            if category == 'cup':
                raise RuntimeError('inference failed')
            return [candidate(mask)]

        manifest = self.run_stage(['mug', 'cup'], backend)
        self.assertEqual(manifest['status'], 'partial')
        self.assertEqual(len(manifest['objects']), 1)
        self.assertEqual(manifest['failures'][0]['category'], 'cup')

    def test_existing_and_repository_output_are_rejected_without_mutation(self):
        existing = self.root / 'existing'
        existing.mkdir()
        marker = existing / 'marker.txt'
        marker.write_text('preserve')
        repo_output = Path(segmentation.__file__).resolve().parents[3] / 'segmentation_test_output'
        for output in (existing, repo_output):
            with self.assertRaises(ValueError):
                segmentation.run(self.image, ['mug'], output, segmenter=lambda image, category: [])
        self.assertEqual(marker.read_text(), 'preserve')
        self.assertFalse(repo_output.exists())

    def test_model_load_failure_keeps_failed_receipt_without_downloading_or_gt(self):
        with patch.object(segmentation, 'ObjectSegmenter', side_effect=OSError('model cache miss')):
            path = segmentation.run(self.image, ['mug'], self.root / 'missing_cache')
        manifest = json.loads(path.read_text())
        self.assertEqual(manifest['status'], 'failed')
        self.assertEqual(manifest['objects'], [])
        self.assertTrue(manifest['models']['local_files_only'])
        self.assertEqual(manifest['models']['sam_model'], 'facebook/sam-vit-base')
        self.assertIn('model cache miss', manifest['failures'][0]['reason'])
        self.assertEqual(manifest['source']['image_sha256'], segmentation.sha256(self.image))


if __name__ == '__main__':
    unittest.main()
