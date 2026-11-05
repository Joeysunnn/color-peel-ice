import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.wood_spoon_input_cause_v1 import prepare as diagnostic
from experiments.subject_trellis_counterfactual_v1.run import check_source_receipt


class WoodSpoonInputCauseTests(unittest.TestCase):
    def test_hard_alpha_keeps_foreground_rgb_and_official_crop(self):
        rgba = np.zeros((512, 512, 4), dtype=np.uint8)
        rgba[19:484, 209:302] = [189, 123, 67, 254]
        rgba[19, 209, 3] = 205  # Strict >204 includes this crop boundary.
        rgba[18, 208] = [200, 150, 100, 204]  # Excluded by the official crop.
        rgba[16, 206] = [100, 80, 60, 1]  # Soft halo removed, not retained as RGB.
        before = rgba.copy()
        actual = diagnostic.harden_alpha(rgba)
        support = before[..., 3] > 204
        np.testing.assert_array_equal(actual[support, :3], before[support, :3])
        np.testing.assert_array_equal(actual[..., 3], support.astype(np.uint8) * 255)
        self.assertTrue((actual[~support, :3] == 0).all())
        self.assertEqual(diagnostic.official_crop_box(actual[..., 3]), [23., 19., 487., 483.])
        np.testing.assert_array_equal(rgba, before)

    def test_receipt_locks_parent_native_and_generated_alpha(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / 'historical/inputs/wood_spoon'
            inputs.mkdir(parents=True)
            native = root / 'historical/wood_spoon/native_asset.pt'
            native.parent.mkdir()
            native.write_bytes(b'fixture native provenance only; never loaded')
            rgba = np.zeros((512, 512, 4), dtype=np.uint8)
            rgba[20:480, 210:300] = [180, 100, 50, 254]
            source, cutout, mask = [inputs / name for name in ('source.png', 'cutout512.png', 'mask512.png')]
            Image.fromarray(rgba[..., :3]).save(source)
            Image.fromarray(rgba).save(cutout)
            Image.fromarray(rgba[..., 3]).save(mask)
            receipt = {'segmentation': {'method': 'fixture BiRefNet'}, 'soft_alpha': True}
            for name, path in [('source', source), ('cutout', cutout), ('mask', mask)]:
                receipt[name + '_path'], receipt[name + '_sha256'] = str(path), diagnostic.sha256(path)
            parent = inputs / 'source_receipt.json'
            parent.write_text(json.dumps(receipt))
            output = root / 'diagnostic'
            expected = {name: receipt[name + '_sha256'] for name in ('source', 'cutout', 'mask')}
            with patch.object(diagnostic, 'EXPECTED', expected), patch.object(diagnostic, 'NATIVE_SHA', diagnostic.sha256(native)), patch.object(diagnostic, 'git_info', return_value={'commit': 'fixture'}):
                actual = diagnostic.prepare(parent, output)
                with self.assertRaisesRegex(ValueError, 'fresh'):
                    diagnostic.prepare(parent, output)
            self.assertEqual(actual['source_path'], str(source))
            self.assertFalse(actual['soft_alpha'])
            self.assertEqual((output / 'original_alpha512.png').read_bytes(), mask.read_bytes())
            self.assertIn(str(native), actual['source_locks'])
            check_source_receipt(output / 'source_receipt.json')
            native.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'source lock hash mismatch'):
                check_source_receipt(output / 'source_receipt.json')


if __name__ == '__main__':
    unittest.main()
