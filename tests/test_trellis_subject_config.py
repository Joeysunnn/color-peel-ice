"""CPU checks for subject-neutral captions and explicitly pinned protocols."""

from collections import defaultdict
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from experiments.subject_trellis_dataset_v1 import build, verify as verification


class TrellisSubjectConfigTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='trellis-subject-config-', dir=os.environ.get('TMP'))
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.config = json.loads(build.PROTOCOL.read_text())

    def test_mailbox_default_rows_keep_original_caption_and_fields(self):
        records = build.rows(self.config)
        self.assertEqual(len(records), 180)
        for row in records:
            adjective = {0.15: 'glossy', 0.5: 'moderately rough', 0.85: 'rough'}[row['controls']['roughness']]
            material = 'metallic' if row['controls']['metallic'] else 'non-metallic'
            self.assertEqual(row['caption'], f"a photo of <S*> mailbox in {row['color']} color with a {adjective} {material} surface")
            self.assertNotIn('subject_id', row)

    def test_spoon_subject_ids_do_not_leak_into_captions(self):
        for subject in ('metal_spoon', 'wood_spoon'):
            config = {**self.config, 'subject_id': subject, 'subject_noun': 'spoon'}
            records = build.rows(config)
            self.assertEqual({row['subject_id'] for row in records}, {subject})
            for row in records:
                self.assertIn('a photo of <S*> spoon in ', row['caption'])
                self.assertNotIn('mailbox', row['caption'])
                self.assertNotIn('metal_spoon', row['caption'])
                self.assertNotIn('wood_spoon', row['caption'])
                self.assertEqual(row['caption'].count('<S*>'), 1)
        metal = build.rows({**self.config, 'subject_id': 'metal_spoon', 'subject_noun': 'spoon'})
        wood = build.rows({**self.config, 'subject_id': 'wood_spoon', 'subject_noun': 'spoon'})
        self.assertEqual([row['caption'] for row in metal], [row['caption'] for row in wood])
        self.assertNotEqual(metal[0]['subject_id'], wood[0]['subject_id'])

    def test_all_subjects_keep_180_rows_30_groups_and_210_matched_edges(self):
        for subject in (None, 'metal_spoon', 'wood_spoon'):
            config = self.config if subject is None else {**self.config, 'subject_id': subject, 'subject_noun': 'spoon'}
            groups = defaultdict(list)
            for row in build.rows(config):
                row['geometry_buffer_sha256'] = {'mask': 'fixed', 'normal': 'fixed'}
                groups[row['pair_group']].append(row)
            self.assertEqual(sum(map(len, groups.values())), 180)
            self.assertEqual(len(groups), 30)
            edges = [edge for group in groups.values() for edge in verification.one_factor_pairs(group)]
            self.assertEqual(len(edges), 210)
            if subject is not None:
                self.assertEqual({edge['matched_controls']['subject_id'] for edge in edges}, {subject})

    def test_pair_cannot_change_subject_id(self):
        config = {**self.config, 'subject_id': 'metal_spoon', 'subject_noun': 'spoon'}
        group = [row for row in build.rows(config) if row['pair_group'] == 'left__pink__studio']
        for row in group:
            row['geometry_buffer_sha256'] = {'mask': 'fixed', 'normal': 'fixed'}
        group[-1]['subject_id'] = 'wood_spoon'
        with self.assertRaisesRegex(ValueError, 'fixed control'):
            verification.one_factor_pairs(group)

    def test_verification_uses_explicit_protocol_hash_before_artifact_checks(self):
        custom = self.root / 'spoon_protocol.json'
        config = {**self.config, 'subject_id': 'metal_spoon', 'subject_noun': 'spoon'}
        build.write(custom, config)
        output = self.root / 'run'
        (output / 'dataset').mkdir(parents=True)
        shutil.copyfile(custom, output / 'protocol.json')
        records = build.rows(config)
        records[-1] = copy.deepcopy(records[0])
        build.write(output / 'dataset/manifest.json', {'records': records, 'schema': config['schema'],
            'training_started': False, 'training_eligible_images': 0})
        # The custom file is accepted, then the deliberately duplicated grid is rejected.
        with self.assertRaisesRegex(ValueError, 'complete unique'):
            verification.verify(output, protocol_path=custom)
        # The default Mailbox protocol must not validate the custom frozen copy.
        with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
            verification.verify(output)
        config['subject_id'] = 'wood_spoon'
        build.write(custom, config)
        with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
            verification.verify(output, protocol_path=custom)
        self.assertFalse((output / 'verification.json').exists())

    def test_build_reads_explicit_protocol_for_preflight(self):
        custom = self.root / 'spoon_protocol.json'
        source = self.root / 'native_source'
        source.mkdir()
        asset, camera = source / 'native.pt', source / 'camera.json'
        asset.write_bytes(b'own generated geometry fixture')
        camera.write_text('{}')
        config = {**self.config, 'subject_id': 'wood_spoon', 'subject_noun': 'spoon',
                  'asset': str(asset), 'asset_sha256': 'wrong-on-purpose',
                  'camera_source': str(camera), 'camera_source_sha256': build.sha256(camera)}
        build.write(custom, config)
        output = self.root / 'not-created'
        with patch.object(build, 'resolve_run_dir', side_effect=lambda path: Path(path).resolve()), \
             patch.object(build, 'git_info', return_value={'commit': 'cpu-preflight'}), \
             self.assertRaisesRegex(ValueError, 'immutable input changed'):
            build.run(output, protocol_path=custom)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
