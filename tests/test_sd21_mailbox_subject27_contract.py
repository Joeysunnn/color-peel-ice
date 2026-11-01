"""Verify recipe preservation and immutable24+3 staging, without GPU work."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.sd21_mailbox_subject27_v1 import contract
from experiments.subject_trellis_dataset_v1.build import PROTOCOL, rows


class Mailbox27ContractTest(unittest.TestCase):
    def test_only_authorized_science_arguments_change(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {contract.DATASET_ENV: temporary}):
            for method in contract.METHODS:
                old = contract.baseline.build_config(method, Path(temporary))
                new = contract.build_config('subject', method)
                changed = {key for key in old['args'].keys() | new['args'].keys()
                           if old['args'].get(key) != new['args'].get(key)}
                self.assertEqual(changed, {'pretrained_model_name_or_path', 'max_train_steps', 'checkpoint_steps'})
                self.assertEqual(new['args']['initializer_token'], 'mailbox')
                self.assertEqual(new['args']['modifier_token'], '<S*>')
                self.assertEqual(new['args']['max_train_steps'], 600)
                self.assertEqual(new['args']['checkpoint_steps'], [200, 400, 600])
                self.assertEqual(new['args']['learning_rate'], 1e-5)
                self.assertEqual(new['args']['kv_learning_rate'], 1e-5 if method == 'fullrank' else 5e-5)

    def test_stages24_plus3_and_detects_changed_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, photos, out = root / 'source72', root / 'photos', root / 'new'
            (source / 'dataset').mkdir(parents=True)
            out.mkdir()
            config = json.loads(PROTOCOL.read_text())
            config['colors'] = {key: config['colors'][key] for key in ('blue', 'green')}
            records = rows(config)
            mask = np.zeros((512, 512), np.uint8)
            mask[200:, 140:440] = 255
            for i, row in enumerate(contract.select24(records)):
                for key, pixels in (('image', np.full((512, 512, 3), 30 + i, np.uint8)), ('mask', mask)):
                    path = source / 'dataset' / key / (row['id'] + '.png')
                    path.parent.mkdir(exist_ok=True)
                    Image.fromarray(pixels).save(path)
                    row[key], row[key + '_sha256'] = path.relative_to(source / 'dataset').as_posix(), contract.sha256(path)
            manifest = source / 'dataset/manifest.json'
            manifest.write_text(json.dumps({'records': records}))
            status = source / 'dataset_status.json'
            status.write_text(json.dumps({'status': 'completed_pending_review', 'images_completed': 72, 'native_inputs_unchanged': True}))
            original = root / 'original.jpg'
            Image.fromarray(np.full((512, 512, 3), [200, 80, 30], np.uint8)).save(original)
            locks = {str(p): contract.sha256(p) for p in (manifest, status, original)}
            for color, rgb in (('blue', [30, 50, 200]), ('green', [30, 200, 50])):
                for folder, pixels in (('images', np.full((512, 512, 3), rgb, np.uint8)), ('masks', mask)):
                    path = photos / color / folder / 'image.png'
                    path.parent.mkdir(parents=True)
                    Image.fromarray(pixels).save(path)
                    locks[str(path)] = contract.sha256(path)
            with patch.object(contract, 'SOURCE_ROOT', source), patch.object(contract, 'PHOTO_ROOT', photos), \
                    patch.object(contract, 'ORIGINAL', original), patch.object(contract, 'SOURCE_HASHES', locks), \
                    patch.dict(os.environ, {contract.DATASET_ENV: str(out / 'dataset')}):
                before = {path: Path(path).read_bytes() for path in locks}
                result = contract.stage_dataset(out / 'dataset')
                self.assertEqual(result['rows'], 27)
                refs = result['references']
                self.assertEqual([r['id'] for r in refs[-3:]], ['photo_original', 'photo_lch_blue', 'photo_lch_green'])
                self.assertTrue(all(r['light'] == 'studio' and r['controls']['roughness'] != .5 for r in refs[:24]))
                self.assertTrue(np.array_equal(np.asarray(Image.open(refs[24]['image'])), np.asarray(Image.open(original).convert('RGB'))))
                for row in refs[:24] + refs[25:]:
                    self.assertEqual(contract.sha256(row['image']), contract.sha256(row['parent_image']))
                self.assertEqual(before, {path: Path(path).read_bytes() for path in locks})
                Image.fromarray(np.full((512, 512, 3), 0, np.uint8)).save(photos / 'blue/images/image.png')
                with self.assertRaises(ValueError):
                    contract.verify_dataset('subject')

    def test_missing_or_duplicate_endpoint_rejected(self):
        config = json.loads(PROTOCOL.read_text())
        selected = contract.select24(rows(config))
        for invalid in (selected[:-1], selected[:-1] + [selected[0]]):
            with self.assertRaises(ValueError):
                contract.select24(invalid)


if __name__ == '__main__':
    unittest.main()
