"""Qualitative conflicts must not silently rewrite the locked rendered target."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.sd21_qwen_initializers_v1 import data, material_target as target


def semantics(metal='dielectric', rough='low', confidence='high'):
    return {'metallicity': {'label': metal, 'confidence': confidence, 'evidence': 'visible reflection'},
            'roughness': {'label': rough, 'confidence': confidence, 'evidence': 'highlight width'}}


class ComparisonTests(unittest.TestCase):
    def test_high_confidence_opposite_extremes_are_flagged_without_changing_target(self):
        for semantic, roughness, metallic in ((semantics(metal='metallic'), .35, .2),
                (semantics(), .35, .8), (semantics(), .65, .04), (semantics(rough='high'), .25, .04)):
            original = copy.deepcopy(semantic)
            result = target.compare_material_target(semantic, roughness, metallic)
            self.assertEqual(result['status'], 'conflict')
            self.assertTrue(result['training_blocked'])
            self.assertFalse(result['physical_truth_verified'])
            self.assertFalse(result['target_values_changed'])
            self.assertFalse(result['initializer_reselected'])
            self.assertEqual(semantic, original)

    def test_intermediate_uncertain_mixed_or_nonhigh_claims_do_not_block(self):
        for semantic in (semantics(confidence='medium'), semantics(confidence='low'),
                         semantics(metal='mixed'), semantics(metal='uncertain'), semantics(rough='medium')):
            result = target.compare_material_target(semantic, .35, .04)
            self.assertEqual(result['status'], 'inconclusive')
            self.assertFalse(result['training_blocked'])
        result = target.compare_material_target(semantics(), .353, .039)
        self.assertEqual(result['status'], 'no_clear_conflict')
        self.assertFalse(result['physical_truth_verified'])

    def test_invalid_scalars_and_classes_are_not_accepted(self):
        for value in (True, float('nan'), float('inf'), -0.1, 1.1, '0.4'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                target.compare_material_target(semantics(), value, .04)
        with self.assertRaises(ValueError):
            target.compare_material_target(semantics(rough='glossy'), .35, .04)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.locks = {}
        self.image = self.root / 'photo.png'
        self.input = self.root / 'supermat.png'
        Image.new('RGB', (8, 8), 'orange').save(self.image)
        Image.new('RGBA', (8, 8), (255, 165, 0, 0)).save(self.input)
        self.locks[str(self.input)] = target.sha256(self.input)
        self.manifest = {'schema': 'colorpeel_vlm_initializers/v2',
            'source': {'image': str(self.image), 'image_sha256': target.sha256(self.image)},
            'response': {'material': semantics()}}
        values = {'roughness': .353, 'metallic': .039}
        material = self.root / 'material.json'
        self.write(material, {**values, 'selected_region': 'lid'})
        self.write(self.root / 'plan/protocol.json', {
            'sources': {'mailbox': {**values, 'input': str(self.input), 'material': str(material)}},
            'materials': {'mailbox_material': values}})
        self.rows, rendered = [], []
        for index in range(8):
            row = {**values, 'id': str(index), 'source_id': 'mailbox', 'branch': 'material',
                'material_id': 'mailbox_material', 'image': str(self.image),
                'image_sha256': target.sha256(self.image), 'texture_image_nodes': 0,
                'normal_bump_displacement': False, 'roughness_input_linked': False,
                'metallic_input_linked': False, 'albedo_mode': 'uniform'}
            metadata = self.root / 'renders' / str(index) / 'metadata.json'
            self.write(metadata, row)
            self.locks.pop(str(metadata))  # metadata hashes are bound by the render manifest.
            rendered.append({**row, 'metadata': str(metadata), 'metadata_sha256': target.sha256(metadata)})
            self.rows.append({**row, 'source_image': str(self.image)})
        self.write(self.root / 'renders/manifest.json', {'records': rendered})

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf8')
        self.locks[str(path)] = target.sha256(path)

    def audit(self):
        return target.audit_material_target(self.manifest, self.rows, self.locks, self.root)

    def test_valid_target_locks_metadata_and_checks_decoded_rgb_instead_of_file_bytes(self):
        self.assertNotEqual(target.sha256(self.image), target.sha256(self.input))
        result = self.audit()
        self.assertTrue(result['source_rgb_equal'])
        self.assertTrue(result['target_provenance_verified'])
        self.assertEqual(result['numeric_target'], {'roughness': .353, 'metallic': .039})
        self.assertEqual(result['status'], 'no_clear_conflict')
        self.assertEqual(sum(name.endswith('metadata.json') for name in self.locks), 8)

    def test_staged_target_drift_is_rejected_even_if_source_files_are_unchanged(self):
        self.rows[0]['roughness'] = .9
        with self.assertRaisesRegex(ValueError, 'M/R target differs'):
            self.audit()

    def test_relative_render_image_and_metadata_paths_resolve_under_render_root(self):
        path = self.root / 'renders/manifest.json'
        manifest = json.loads(path.read_text())
        row = manifest['records'][0]
        row.update(image='relative.png', metadata='0/metadata.json')
        image = self.root / 'renders/relative.png'
        image.write_bytes(self.image.read_bytes())
        self.rows[0]['source_image'] = str(image)
        metadata = self.root / 'renders/0/metadata.json'
        self.write(metadata, {key: value for key, value in row.items() if key not in ('metadata', 'metadata_sha256')})
        row['metadata_sha256'] = target.sha256(metadata)
        self.write(path, manifest)
        self.assertTrue(self.audit()['target_provenance_verified'])

    def test_metadata_bytes_must_match_render_manifest(self):
        path = self.root / 'renders/0/metadata.json'
        path.write_text('{}', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'metadata hash differs'):
            self.audit()

    def test_different_original_photo_cannot_inherit_mailbox_material(self):
        Image.new('RGB', (8, 8), 'blue').save(self.image)
        self.manifest['source']['image_sha256'] = target.sha256(self.image)
        with self.assertRaisesRegex(ValueError, 'source RGB differ'):
            self.audit()

    def test_conflict_report_survives_blocked_preparation(self):
        self.manifest['source']['image_sha256'] = data.contract.SOURCE_SHA
        result = target.compare_material_target(semantics(metal='metallic'), .35, .04)
        output = self.root / 'blocked_dataset'
        with patch.object(data, 'source_rows', return_value=({'material': self.rows}, self.locks)), \
                patch.object(target, 'audit_material_target', return_value=result):
            with self.assertRaisesRegex(ValueError, 'VLM/material target conflict'):
                data.prepare(output, self.manifest, None)
        self.assertEqual(json.loads((output / 'material_target_consistency.json').read_text()), result)
        self.assertFalse((output / 'receipt.json').exists())


if __name__ == '__main__':
    unittest.main()
