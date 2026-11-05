import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.mailbox_alpha_control_v1.prepare import controlled_pair
from experiments.mailbox_alpha_control_v1 import prepare as preparation
from experiments.subject_trellis_counterfactual_v1.run import check_source_receipt
from experiments.subject_trellis_counterfactual_v1.run import fixed_crop_condition


class MailboxAlphaControlTests(unittest.TestCase):
    def fixtures(self):
        hard = np.zeros((512, 512, 4), dtype=np.uint8)
        hard[291:512, 145:440] = [210, 120, 160, 255]
        soft = hard.copy()
        soft[290:512, 144:441] = [1, 2, 3, 254]  # RGB must be ignored, expanded support retained.
        return hard, soft

    def test_only_alpha_changes_and_black_halo_cannot_change_fixed_condition(self):
        hard, soft = self.fixtures()
        before = hard.copy()
        variants, crop = controlled_pair(hard, soft)
        self.assertEqual(crop, [145, 254, 439, 548])
        for rgba in variants.values():
            np.testing.assert_array_equal(rgba[..., :3], hard[..., :3])
        np.testing.assert_array_equal(variants['hard_replay'], hard)
        np.testing.assert_array_equal(variants['soft_only'][..., 3], soft[..., 3])
        full = fixed_crop_condition(Image.fromarray(variants['soft_only']), crop)
        restricted = variants['soft_only'].copy()
        restricted[hard[..., 3] == 0, 3] = 0
        self.assertEqual(full.tobytes(), fixed_crop_condition(Image.fromarray(restricted), crop).tobytes())
        self.assertNotEqual(full.tobytes(), fixed_crop_condition(Image.fromarray(hard), crop).tobytes())
        np.testing.assert_array_equal(hard, before)

    def test_off_canvas_padding_and_float32_composite(self):
        hard, _ = self.fixtures()
        hard[291, 145, 3] = 254
        actual = np.asarray(fixed_crop_condition(Image.fromarray(hard), [145, 254, 439, 548]))
        self.assertEqual(actual.shape, (294, 294, 3))
        np.testing.assert_array_equal(actual[37, 0], [209, 119, 159])
        self.assertFalse(actual[258:].any())

    def test_reject_confounded_inputs_and_invalid_crops(self):
        hard, soft = self.fixtures()
        soft[300, 200, 3] = 0
        with self.assertRaisesRegex(ValueError, 'cover old SAM'):
            controlled_pair(hard, soft)
        for crop in ([0, 0, 5, 6], [0, 0, 0, 0], [0., 0, 5, 5]):
            with self.subTest(crop=crop), self.assertRaises(ValueError):
                fixed_crop_condition(Image.fromarray(hard), crop)

    def test_receipts_preserve_sam_and_lock_both_parents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_folder = root / 'sources'
            source_folder.mkdir()
            hard, soft = self.fixtures()
            source, mask = source_folder / 'photo.png', source_folder / 'sam.png'
            Image.fromarray(hard[..., :3]).save(source)
            Image.fromarray(hard[..., 3]).save(mask)
            parents = []
            for name, rgba in [('hard', hard), ('soft', soft)]:
                cutout, alpha = source_folder / (name + '.png'), source_folder / (name + '_alpha.png')
                Image.fromarray(rgba).save(cutout)
                Image.fromarray(rgba[..., 3]).save(alpha)
                record = {}
                for key, path in [('source', source), ('mask', mask), ('cutout', cutout), ('conditioning_alpha', alpha)]:
                    record[key + '_path'], record[key + '_sha256'] = str(path), preparation.sha256(path)
                parent = source_folder / (name + '.json')
                parent.write_text(json.dumps(record))
                parents.append(parent)
            with patch.object(preparation, 'git_info', return_value={'commit': 'fixture'}):
                audit = preparation.prepare(*parents, root / 'prepared')
                with self.assertRaisesRegex(ValueError, 'fresh'):
                    preparation.prepare(*parents, root / 'prepared')
            for name in ('hard_replay', 'soft_only'):
                receipt = Path(audit['variants'][name]['receipt'])
                self.assertEqual(json.loads(receipt.read_text())['mask_path'], str(mask))
                checked = check_source_receipt(receipt)
                self.assertTrue(all(str(parent) in checked['sha256'] for parent in parents))
            parents[1].write_text('{}')
            with self.assertRaisesRegex(ValueError, 'source lock hash mismatch'):
                check_source_receipt(receipt)


if __name__ == '__main__':
    unittest.main()
