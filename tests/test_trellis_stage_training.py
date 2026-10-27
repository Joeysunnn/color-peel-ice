"""CPU validation of immutable-source Subject staging and exact pairing."""

import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import numpy as np
from PIL import Image

from experiments.subject_trellis_dataset_v1 import stage_training as staging
from experiments.subject_trellis_dataset_v1.build import rows, sha256, write


class TrellisStageTrainingTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / ('test-trellis-stage-' + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(self.cleanup)
        self.source, self.destination = self.root / 'source', self.root / 'staged'
        self.candidates = self.source / 'dataset'
        self.candidates.mkdir(parents=True)
        config = json.loads(staging.PROTOCOL.read_text())
        config['asset'], config['camera_source'] = str(self.root / 'native.bin'), str(self.root / 'camera.json')
        Path(config['asset']).write_bytes(b'frozen native asset')
        Path(config['camera_source']).write_text('{}')
        config['asset_sha256'], config['camera_source_sha256'] = sha256(config['asset']), sha256(config['camera_source'])
        protocol = self.root / 'frozen_protocol.json'
        write(protocol, config)
        shutil.copyfile(protocol, self.source / 'protocol.json')
        records, concepts = [], []
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:400, 120:360] = 255
        for i, row in enumerate(rows(config)):
            image = self.candidates / 'images' / row['id'] / 'image.png'
            region = self.candidates / 'masks' / row['id'] / 'image.png'
            image.parent.mkdir(parents=True)
            region.parent.mkdir(parents=True)
            rgb = np.full((512, 512, 3), 127, dtype=np.uint8)
            rgb[mask == 255] = [i + 1, 40, 200]
            Image.fromarray(rgb).save(image)
            Image.fromarray(mask).save(region)
            records.append({**row, 'image': image.relative_to(self.candidates).as_posix(),
                'mask': region.relative_to(self.candidates).as_posix(),
                'image_sha256': sha256(image), 'mask_sha256': sha256(region)})
            concepts.append({'instance_prompt': [row['caption']], 'instance_data_dir': str(image.parent),
                             'instance_mask_dir': str(region.parent)})
        self.records = records
        write(self.candidates / 'manifest.json', {'schema': config['schema'], 'records': records,
            'training_started': False, 'training_eligible_images': 0,
            'source_hashes': {path: sha256(path) for path in (config['asset'], config['camera_source'], str(protocol))}})
        write(self.candidates / 'concepts.json', concepts)
        manifest_sha, concepts_sha = sha256(self.candidates / 'manifest.json'), sha256(self.candidates / 'concepts.json')
        verification = {'status': 'technical_checks_passed_scientific_review_pending', 'images': 180,
            'pair_groups': 30, 'single_factor_pairs': 210, 'manifest_sha256': manifest_sha,
            'concepts_sha256': concepts_sha, 'training_started': False, 'training_eligible_images': 0}
        write(self.source / 'verification.json', verification)
        (self.source / 'repro_outputs').mkdir()
        write(self.source / 'repro_outputs/status.json', {'status': 'dataset180_constructed_pending_user_review',
            'images_completed': 180, 'training_started': False, 'verification': verification})
        for name in ('runtime.json', 'cameras.json', 'dataset/pairs.json'):
            write(self.source / name, {})
        self.patches = patch.multiple(staging, CANDIDATES=self.candidates, MANIFEST_SHA=manifest_sha,
            CONCEPTS_SHA=concepts_sha, PROTOCOL=protocol, ROOT=self.root / 'project')
        self.patches.start()
        self.addCleanup(self.patches.stop)

    def cleanup(self):
        if not self.root.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
            raise ValueError('cleanup must stay inside test workspace')
        shutil.rmtree(self.root)

    def source_hashes(self):
        return {str(path.relative_to(self.source)): sha256(path) for path in self.source.rglob('*') if path.is_file()}

    def test_complete_byte_copy_keeps_source_pending_and_unchanged(self):
        before = self.source_hashes()
        with patch('experiments.subject_trellis_dataset_v1.verify.verify', side_effect=AssertionError('old receipt must stay read-only')):
            audit = staging.stage(self.candidates, self.destination)
            self.assertEqual(staging.verify_dataset(self.destination), audit)
        self.assertEqual(self.source_hashes(), before)
        self.assertEqual(audit['rows'], 180)
        self.assertEqual(audit['source_mode'], 'synthetic_native_geometry')
        self.assertFalse(audit['sampling']['shuffle'])
        self.assertEqual((audit['sampling']['complete_cycles'], audit['sampling']['extra_prefix_images']), (5, 100))
        actual = [json.loads(line) for line in (self.destination / 'training_assets_manifest.jsonl').read_text().splitlines()]
        self.assertEqual([row['id'] for row in actual], [row['id'] for row in self.records])
        self.assertEqual([row['caption'] for row in actual], [row['caption'] for row in self.records])
        self.assertTrue(all(row['training_eligible'] and row['review_status'] == 'user_authorized_training' for row in actual))

    def test_source_pixel_change_is_rejected_without_creating_staging(self):
        image = self.candidates / self.records[0]['image']
        with Image.open(image) as handle:
            changed = handle.copy()
        changed.putpixel((150, 150), (1, 2, 3))
        changed.save(image)
        with self.assertRaisesRegex(ValueError, 'immutable'):
            staging.stage(self.candidates, self.destination)
        self.assertFalse(self.destination.exists())

    def test_staged_mask_pair_swap_is_rejected_even_for_identical_mask_bytes(self):
        staging.stage(self.candidates, self.destination)
        path = self.destination / 'training_assets_manifest.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0]['mask'], records[1]['mask'] = records[1]['mask'], records[0]['mask']
        path.write_text(''.join(json.dumps(row) + '\n' for row in records))
        with self.assertRaisesRegex(ValueError, 'pairing'):
            staging.verify_dataset(self.destination)


if __name__ == '__main__':
    unittest.main()
