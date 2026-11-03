"""CPU checks for fixed source identity, selection, and non-mutating preparation."""

import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.sd21_texture_external_preserve_v1 import assets


class ExternalPreservationAssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        buffer = io.BytesIO()
        Image.new('RGB', (512, 512), (80, 140, 200)).save(buffer, format='JPEG')
        self.payload = buffer.getvalue()
        digest = hashlib.sha256(self.payload).hexdigest()
        self.records = []
        for image_id in reversed(range(1000)):
            path = self.source / f'{image_id}-fixture.jpg'
            path.write_bytes(self.payload)
            self.records.append({'path': str(path), 'sha256': digest, 'id': image_id,
                                 'size': [512, 512], 'mode': 'RGB'})
        self.output = self.root / 'output' / 'manifest.json'
        for name, value in (('SOURCE_ROOT', self.source),
                            ('EXPECTED_INVENTORY_SHA256', assets.inventory_digest(self.records))):
            context = patch.object(assets, name, value)
            context.start()
            self.addCleanup(context.stop)

    def test_complete_locks_and_fixed_numeric_sampling_without_source_changes(self):
        rng_state = random.getstate()
        result = assets.prepare(self.source, self.output)
        self.assertEqual(random.getstate(), rng_state)
        manifest = json.loads(self.output.read_text())
        inventory = json.loads(Path(result['inventory_path']).read_text())
        ids = [row['id'] for row in manifest['records']]
        self.assertEqual(ids, random.Random(42).sample(list(range(1000)), 200))
        self.assertEqual(ids[:5], [654, 114, 25, 759, 281])
        self.assertEqual(len(set(ids)), 200)
        self.assertEqual([row['id'] for row in inventory['records']], list(range(1000)))
        self.assertEqual(manifest['prompt'], 'a photo at the beach')
        self.assertEqual(manifest['schema'], 1)
        self.assertIsNone(manifest['generation_model'])
        self.assertIsNone(manifest['generation_seed'])
        self.assertEqual(len(result['source_locks']), 1002)
        for filename, digest in result['source_locks'].items():
            self.assertEqual(hashlib.sha256(Path(filename).read_bytes()).hexdigest(), digest)
        self.assertEqual(len(list(self.source.iterdir())), 1000)
        with self.assertRaisesRegex(ValueError, 'must be new'):
            assets.prepare(self.source, self.output)

    def test_unselected_image_change_is_rejected_before_outputs(self):
        self.assertNotIn(0, random.Random(42).sample(list(range(1000)), 200))
        (self.source / '0-fixture.jpg').write_bytes(self.payload + b'changed')
        with self.assertRaisesRegex(ValueError, 'pre-registered'):
            assets.prepare(self.source, self.output)
        self.assertFalse(self.output.parent.exists())

    def test_missing_or_duplicate_ids_are_rejected(self):
        path = self.source / '999-fixture.jpg'
        path.unlink()
        with self.assertRaisesRegex(ValueError, '0..999'):
            assets.prepare(self.source, self.output)
        path.write_bytes(self.payload)
        (self.source / '0-second.jpg').write_bytes(self.payload)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            assets.prepare(self.source, self.output)

    def test_wrong_size_mode_and_nonimage_entries_are_rejected(self):
        path = self.source / '0-fixture.jpg'
        for mode, size in (('RGB', (256, 512)), ('L', (512, 512))):
            Image.new(mode, size).save(path)
            with self.assertRaisesRegex(ValueError, '512x512 RGB'):
                assets.prepare(self.source, self.output)
        path.write_bytes(self.payload)
        (self.source / 'captions.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            assets.prepare(self.source, self.output)

    def test_root_and_output_boundaries(self):
        with self.assertRaisesRegex(ValueError, 'authorized'):
            assets.prepare(self.root, self.output)
        with self.assertRaisesRegex(ValueError, 'outside'):
            assets.prepare(self.source, self.source / 'manifest.json')
        with self.assertRaisesRegex(ValueError, 'distinct'):
            assets.prepare(self.source, self.root / 'externalinventory.json')


if __name__ == '__main__':
    unittest.main()
