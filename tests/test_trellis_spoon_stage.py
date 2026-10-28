"""Verify independent spoon staging without changing historical Mailbox defaults."""

import json
from pathlib import Path
import shutil
from unittest.mock import patch

from experiments.subject_trellis_dataset_v1 import stage_training as shared
from experiments.subject_trellis_dataset_v1.build import rows, sha256, write
from experiments.subject_trellis_spoons_v1 import stage_training as spoons
from tests import test_trellis_stage_training as fixtures


class TrellisSpoonStageTest(fixtures.TrellisStageTrainingTest):
    def setUp(self):
        super().setUp()
        self.spec = self.subject_source('metal_spoon')
        self.registry = patch.object(spoons, 'SOURCES', {'metal_spoon': self.spec})
        self.registry.start()
        self.addCleanup(self.registry.stop)

    def subject_source(self, subject):
        config = shared.read(shared.PROTOCOL)
        config.update(subject_id=subject, subject_noun='spoon')
        write(shared.PROTOCOL, config)
        shutil.copyfile(shared.PROTOCOL, self.source / 'protocol.json')
        definitions = {row['id']: row for row in rows(config)}
        for row in self.records:
            row.update(definitions[row['id']])
        manifest = shared.read(self.candidates / 'manifest.json')
        manifest['records'] = self.records
        manifest['source_hashes'][str(shared.PROTOCOL)] = sha256(shared.PROTOCOL)
        write(self.candidates / 'manifest.json', manifest)
        write(self.candidates / 'concepts.json', [
            {'instance_prompt': [row['caption']],
             'instance_data_dir': str((self.candidates / row['image']).parent),
             'instance_mask_dir': str((self.candidates / row['mask']).parent)} for row in self.records])
        spec = {'subject_id': subject, 'candidates': self.candidates, 'protocol': shared.PROTOCOL,
            'manifest_sha256': sha256(self.candidates / 'manifest.json'),
            'concepts_sha256': sha256(self.candidates / 'concepts.json'), 'request': spoons.REQUEST}
        self.refresh_verification(spec)
        # The inherited default tests still exercise the unchanged default API.
        shared.MANIFEST_SHA, shared.CONCEPTS_SHA = spec['manifest_sha256'], spec['concepts_sha256']
        return spec

    def refresh_verification(self, spec):
        verification = shared.read(self.source / 'verification.json')
        verification.update(manifest_sha256=spec['manifest_sha256'], concepts_sha256=spec['concepts_sha256'])
        write(self.source / 'verification.json', verification)
        status = shared.read(self.source / 'repro_outputs/status.json')
        status['verification'] = verification
        write(self.source / 'repro_outputs/status.json', status)

    def test_spoon_full_and_blue_green_stage_keep_identity_bytes_and_current_approval(self):
        before = self.source_hashes()
        for subset in (False, True):
            destination = self.root / ('spoon72' if subset else 'spoon180')
            audit = spoons.stage(self.candidates, destination, subset)
            self.assertEqual(spoons.verify_dataset(destination), audit)
            self.assertEqual(audit['rows'], 72 if subset else 180)
            self.assertEqual(audit['subject_id'], 'metal_spoon')
            approval = shared.read(destination / 'approval.json')
            self.assertEqual(approval['request'], spoons.REQUEST)
            self.assertEqual(approval['subject_id'], 'metal_spoon')
            self.assertNotIn('four requested transfers', approval['scope'])
            records = [json.loads(line) for line in (destination / 'training_assets_manifest.jsonl').read_text().splitlines()]
            self.assertEqual({row['subject_id'] for row in records}, {'metal_spoon'})
            self.assertEqual({row['color'] for row in records}, {'blue', 'green'} if subset else {'pink', 'green', 'cyan', 'blue', 'purple'})
            for row in records:
                self.assertIn('<S*> spoon', row['caption'])
                for key in ('image', 'mask'):
                    self.assertEqual(sha256(row[key]), row['source_' + key + '_sha256'])
                    self.assertTrue(Path(row['source_' + key]).is_relative_to(self.candidates))
        self.assertEqual(self.source_hashes(), before)

    def test_wood_subject_is_independent_and_unknown_or_changed_source_is_rejected(self):
        wood = self.subject_source('wood_spoon')
        with patch.object(spoons, 'SOURCES', {'wood_spoon': wood}):
            audit = spoons.stage(self.candidates, self.destination, True)
            self.assertEqual(audit['subject_id'], 'wood_spoon')
            self.assertEqual(spoons.verify_dataset(self.destination), audit)
            with self.assertRaisesRegex(ValueError, 'frozen complete'):
                spoons.stage(self.root / 'other_source', self.root / 'not_created')
        with self.assertRaisesRegex(ValueError, 'unknown'):
            spoons.verify_dataset(self.destination)

    def test_custom_source_hash_mismatch_is_rejected_before_staging(self):
        self.spec['manifest_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'immutable'):
            spoons.stage(self.candidates, self.destination)
        self.assertFalse(self.destination.exists())

    def test_mixed_subject_manifest_is_rejected_even_when_its_manifest_hash_is_updated(self):
        manifest = shared.read(self.candidates / 'manifest.json')
        manifest['records'][0]['subject_id'] = 'wood_spoon'
        write(self.candidates / 'manifest.json', manifest)
        self.spec['manifest_sha256'] = sha256(self.candidates / 'manifest.json')
        self.refresh_verification(self.spec)
        with self.assertRaisesRegex(ValueError, 'caption, controls'):
            spoons.stage(self.candidates, self.destination)
        self.assertFalse(self.destination.exists())
