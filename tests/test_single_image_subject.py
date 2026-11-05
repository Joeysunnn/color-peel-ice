import copy
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from src.methods.colorpeel_ice import single_image_subject as subject


class SingleImageSubjectTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def source(self, size=(512, 512)):
        width, height = size
        rgb = np.full((height, width, 3), [180, 100, 40], dtype=np.uint8)
        rgb[:, :width // 4] = [20, 40, 60]
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[height // 4:3 * height // 4, width // 2:3 * width // 4] = 255
        image_path, mask_path = self.root / 'source.png', self.root / 'mask.png'
        Image.fromarray(rgb).save(image_path)
        Image.fromarray(mask).save(mask_path)
        return image_path, mask_path, rgb, mask

    def test_full_canvas_pair_preserves_asymmetric_coordinates_and_padding(self):
        image_path, mask_path, rgb, mask = self.source((200, 100))
        receipt = subject.prepare_inputs(image_path, mask_path, self.root / 'inputs')
        self.assertEqual(receipt['transform']['resized_size'], [512, 256])
        self.assertEqual(receipt['transform']['pad_left_top'], [0, 128])
        self.assertFalse(receipt['transform']['crop'])
        with Image.open(receipt['mask_path']) as opened:
            actual = np.asarray(opened)
        expected = np.zeros((512, 512), dtype=np.uint8)
        expected[192:320, 256:384] = 255
        np.testing.assert_array_equal(actual, expected)
        with Image.open(receipt['cutout_path']) as opened:
            rgba = np.asarray(opened)
        np.testing.assert_array_equal(rgba[..., 3], actual)
        self.assertTrue((rgba[actual == 0, :3] == 0).all())
        self.assertEqual(receipt['source_sha256'], subject.sha256(image_path))
        self.assertEqual(receipt['original_mask_sha256'], subject.sha256(mask_path))
        subject.check_locks(receipt['source_locks'])

    def test_512_original_pixels_and_background_are_unchanged_in_photo_rows(self):
        image_path, mask_path, original, mask = self.source()
        receipt = subject.prepare_inputs(image_path, mask_path, self.root / 'inputs')
        rows = subject.photo_rows(receipt, self.root / 'photos', 'number plate')
        self.assertEqual([row['id'] for row in rows],
                         ['photo_original', 'photo_lch_blue', 'photo_lch_green'])
        for row in rows:
            with Image.open(row['image']) as opened, Image.open(row['mask']) as opened_mask:
                actual, actual_mask = np.asarray(opened), np.asarray(opened_mask)
            np.testing.assert_array_equal(actual_mask, mask)
            np.testing.assert_array_equal(actual[mask == 0], original[mask == 0])
            self.assertEqual(row['caption'].count('<S*>'), 1)
            self.assertIn('number plate', row['caption'])
            if row['id'] == 'photo_original':
                np.testing.assert_array_equal(actual, original)
            else:
                self.assertTrue(np.any(actual[mask > 0] != original[mask > 0]))
                self.assertEqual(row['recolor_metrics']['outside_mask_changed_pixel_count'], 0)
        self.assertEqual([row['hue_degrees'] for row in rows[1:]], [240., 120.])

    def test_invalid_masks_and_reused_directory_are_rejected(self):
        image_path, mask_path, rgb, mask = self.source()
        image = Image.fromarray(rgb)
        for invalid in (mask[:128], np.full(mask.shape, 128), np.zeros_like(mask)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                subject.square_pair(image, invalid)
        subject.prepare_inputs(image_path, mask_path, self.root / 'inputs')
        with self.assertRaisesRegex(ValueError, 'fresh'):
            subject.prepare_inputs(image_path, mask_path, self.root / 'inputs')

    def test_source_mutation_breaks_receipt_locks(self):
        image_path, mask_path, _, _ = self.source()
        receipt = subject.prepare_inputs(image_path, mask_path, self.root / 'inputs')
        image_path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            subject.check_locks(receipt['source_locks'])

    def test_exact_eight_counterfactuals_with_dynamic_subject_category(self):
        rows = subject.appearance_rows('vase')
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({row['id'] for row in rows}), 8)
        self.assertEqual({(row['color'], row['metallic'], row['roughness']) for row in rows},
                         {(c, m, r) for c in ('blue', 'green') for m in (0, 1) for r in (.15, .85)})
        self.assertTrue(all(row['caption'].count('<S*>') == 1 and 'vase' in row['caption'] for row in rows))
        with self.assertRaises(ValueError):
            subject.appearance_rows('vase <C*>')

    def test_foreign_instance_asset_cannot_be_attached(self):
        image_path, mask_path, _, _ = self.source()
        subject.prepare_inputs(image_path, mask_path, self.root / 'inputs')
        trellis = self.root / 'trellis/repro_outputs'
        trellis.mkdir(parents=True)
        (trellis / 'status.json').write_text(json.dumps({'status': 'generated_pending_user_review',
            'generated_assets': 1, 'source_receipt': {'sha256': 'wrong-instance'}}))
        with self.assertRaisesRegex(ValueError, 'this instance'):
            subject.render_dataset(self.root / 'inputs', self.root / 'trellis', self.root / 'render', 'vase')
        self.assertFalse((self.root / 'render').exists())

    def review(self, iou=.814):
        return {'metrics': {'original512_mask_iou': iou}, 'camera': {
            'extrinsics': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2.174], [0, 0, 0, 1]],
            'intrinsics': [[1.374, 0, .543], [0, 1.374, .171], [0, 0, 1]],
            'refinement_success': False, 'refinement_message': 'Maximum function evaluations exceeded.'}}

    def trusted_asset_fixture(self, iou):
        image_path, mask_path, _, _ = self.source()
        receipt = subject.prepare_inputs(image_path, mask_path, self.root / 'inputs')
        trellis = self.root / 'trellis'
        (trellis / 'repro_outputs').mkdir(parents=True)
        (trellis / 'review').mkdir()
        (trellis / 'source_cutout512.png').write_bytes(Path(receipt['cutout_path']).read_bytes())
        (trellis / 'native_asset.pt').write_bytes(b'fake asset for render-interface test')
        (trellis / 'review/review.json').write_text(json.dumps(self.review(iou)))
        (trellis / 'repro_outputs/status.json').write_text(json.dumps({
            'status': 'generated_pending_user_review', 'generated_assets': 1,
            'source_receipt': {'sha256': subject.sha256(self.root / 'inputs/source_receipt.json')}}))
        return trellis

    def test_saved_camera_preserves_source_frame_despite_optimizer_budget(self):
        review = self.review()
        original = copy.deepcopy(review)
        camera, quality = subject.reviewed_source_camera(review)
        self.assertEqual(camera['extrinsics'], original['camera']['extrinsics'])
        self.assertEqual(camera['intrinsics'], original['camera']['intrinsics'])
        self.assertEqual(camera['framing'], 'exact_saved_source_camera')
        self.assertFalse(quality['refinement_success'])
        self.assertTrue(quality['warnings'])
        self.assertFalse(quality['quality_accepted_by_user'])
        self.assertEqual(review, original)

    def test_invalid_saved_projection_or_nonrigid_rotation_is_rejected(self):
        for field, row, column, value in [('extrinsics', 0, 0, 2), ('intrinsics', 1, 1, .4),
                                          ('extrinsics', 0, 3, float('nan'))]:
            review = self.review()
            review['camera'][field][row][column] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                subject.reviewed_source_camera(review)
        review = self.review()
        review['camera']['extrinsics'].pop()
        with self.assertRaisesRegex(ValueError, 'shape'):
            subject.reviewed_source_camera(review)

    def test_known_bad_review_rejected_before_render_or_photo_generation(self):
        from experiments.subject_trellis_counterfactual_v1.run import OFFICIAL
        trellis = self.trusted_asset_fixture(.661872)
        with patch.object(subject.subprocess, 'check_output', side_effect=[OFFICIAL, '']), \
                patch.object(subject, 'check_locks'), patch.object(subject, 'photo_rows') as photos, \
                self.assertRaisesRegex(ValueError, 'review mask IoU'):
            subject.render_dataset(self.root / 'inputs', trellis, self.root / 'render', 'mail box')
        photos.assert_not_called()
        self.assertFalse((self.root / 'render').exists())

    def test_source_border_crop_is_preserved_but_new_edges_are_rejected(self):
        source = np.zeros((512, 512), dtype=bool)
        source[300:, 100:400] = True
        quality = subject.check_source_framing(source.copy(), source)
        self.assertEqual(quality['source_mask_iou'], 1)
        self.assertTrue(quality['render_border_contact']['bottom'])
        for y, x in ((0, 200), (400, 0), (400, 511)):
            rendered = source.copy()
            rendered[y, x] = True  # IoU remains high; an unrelated clipping edge must still fail.
            with self.subTest(edge=(y, x)), self.assertRaisesRegex(ValueError, 'border clipping'):
                subject.check_source_framing(rendered, source)
        rendered = source.copy()
        source[-2:] = False
        subject.check_source_framing(rendered, source)  # Two-pixel rasterization tolerance.
        source[-3:] = False
        with self.assertRaisesRegex(ValueError, 'border clipping'):
            subject.check_source_framing(rendered, source)

    def test_actual_render_iou_is_checked_even_when_review_reports_high_iou(self):
        from experiments.subject_trellis_counterfactual_v1.run import OFFICIAL
        trellis = self.trusted_asset_fixture(.95)

        class Tensor:
            def __init__(self, array, **kwargs): self.array = np.asarray(array)
            def __len__(self): return len(self.array)
            def cpu(self): return self
            def numpy(self): return self.array
            def new_tensor(self, array): return Tensor(array)

        actual_mask = np.zeros((512, 512))
        actual_mask[20:60, 20:60] = 1  # Nonempty and in-frame, but disjoint from the source object.
        # A mesh can align with geometry alpha while failing the semantic SAM guard.
        receipt_path = self.root / 'inputs/source_receipt.json'
        receipt = json.loads(receipt_path.read_text())
        soft_path = self.root / 'inputs/conditioning_alpha.png'
        soft = (actual_mask * 230).astype(np.uint8)
        Image.fromarray(soft).save(soft_path)
        rgba = np.zeros((512, 512, 4), dtype=np.uint8)
        rgba[actual_mask > 0, :3] = [180, 100, 40]
        rgba[..., 3] = soft
        Image.fromarray(rgba).save(receipt['cutout_path'])
        receipt.update(conditioning_alpha_path=str(soft_path), conditioning_alpha_sha256=subject.sha256(soft_path),
                       cutout_sha256=subject.sha256(receipt['cutout_path']))
        receipt['source_locks'].update({str(soft_path): subject.sha256(soft_path),
                                       receipt['cutout_path']: receipt['cutout_sha256']})
        subject.write_json(receipt_path, receipt)
        (trellis / 'source_cutout512.png').write_bytes(Path(receipt['cutout_path']).read_bytes())
        state_path = trellis / 'repro_outputs/status.json'
        state = json.loads(state_path.read_text())
        state['source_receipt']['sha256'] = subject.sha256(receipt_path)
        subject.write_json(state_path, state)
        torch = ModuleType('torch')
        torch.load = MagicMock(return_value={'vertices': Tensor(np.eye(3)), 'faces': Tensor([[0, 1, 2]])})
        torch.zeros = MagicMock(return_value=Tensor([0]))
        torch.tensor, torch.long = Tensor, 'long'
        torch.inference_mode, torch.manual_seed = nullcontext, MagicMock()
        cv2 = ModuleType('cv2')
        cv2.IMREAD_UNCHANGED, cv2.COLOR_BGR2RGB = 1, 2
        cv2.imread = MagicMock(return_value=np.ones((2, 2, 3)))
        cv2.cvtColor = lambda array, _: array
        mesh = ModuleType('trellis2.representations.mesh')
        mesh.PbrMaterial, mesh.MeshWithPbrMaterial = MagicMock(), MagicMock()
        renderers = ModuleType('trellis2.renderers')
        renderers.EnvMap = MagicMock()
        renderers.PbrMeshRenderer = MagicMock()
        render = renderers.PbrMeshRenderer.return_value.render
        render.return_value = {'mask': Tensor(actual_mask)}
        modules = {'torch': torch, 'cv2': cv2, 'trellis2': ModuleType('trellis2'),
                   'trellis2.representations': ModuleType('trellis2.representations'),
                   'trellis2.representations.mesh': mesh, 'trellis2.renderers': renderers}
        with patch.dict(sys.modules, modules), patch.object(sys, 'path', list(sys.path)), \
                patch.dict(subject.os.environ), patch.object(subject, 'check_locks'), \
                patch.object(subject.subprocess, 'check_output', side_effect=[OFFICIAL, '']), \
                patch.object(subject, 'photo_rows') as photos, self.assertRaisesRegex(ValueError, 'actual Subject render'):
            subject.render_dataset(self.root / 'inputs', trellis, self.root / 'render', 'mail box')
        self.assertEqual(render.call_count, 1)
        np.testing.assert_array_equal(render.call_args.args[1].array, self.review(.95)['camera']['extrinsics'])
        np.testing.assert_array_equal(render.call_args.args[2].array, self.review(.95)['camera']['intrinsics'])
        photos.assert_not_called()
        self.assertFalse((self.root / 'render/subject.json').exists())


if __name__ == '__main__':
    unittest.main()
