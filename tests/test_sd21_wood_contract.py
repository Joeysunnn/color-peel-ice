"""Check inherited science arguments and immutable data independently of model weights."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.sd21_wood_branches_v1 import contract


class SD21WoodContractTest(unittest.TestCase):
    def old_config(self, branch, method):
        spec = contract.dataset_sources()[branch]
        if branch == 'subject':
            return contract.subject_training.build_config(method, Path(spec['dataset']))
        recipe = next(row for row in contract.natural_training.recipes()
                      if row['concept_id'] == 'wood_spoon_' + branch and row['method'] == method)
        baseline = contract.natural_training.build_config(recipe, contract.NATURAL_DATASET / 'staging', {})
        return (contract.material_config(baseline, method, 'wood_spoon', 'material')
                if branch == 'material' else baseline)

    def test_all_six_recipes_change_only_declared_model_dose_and_save_arguments(self):
        with patch.object(contract.subject_training, 'read_json', return_value={'subject_id': 'wood_spoon', 'rows': 72}), \
             patch.object(contract, '_natural_protocol', return_value={}):
            for branch in contract.BRANCHES:
                for method in contract.METHODS:
                    for smoke in (False, True):
                        with self.subTest(branch=branch, method=method, smoke=smoke):
                            old = self.old_config(branch, method)
                            new = contract.build_config(branch, method, smoke=smoke)
                            differences = {key for key in old['args'].keys() | new['args'].keys()
                                if old['args'].get(key) != new['args'].get(key)}
                            expected = {'pretrained_model_name_or_path', 'checkpoint_steps'}
                            if smoke:
                                expected.add('max_train_steps')
                            self.assertEqual(differences, expected)
                            self.assertEqual(new['args']['initializer_token'], contract.INITIALIZERS[branch])
                            self.assertEqual(new['args']['modifier_token'], contract.TOKENS[branch])
                            self.assertEqual(new['args']['checkpoint_steps'], [2] if smoke else [400, 600, 800, 1000])
                            self.assertEqual(new['args']['max_train_steps'], 2 if smoke else 1000)
                            self.assertEqual(new['run']['study'], contract.STUDY)
                            self.assertIn('smoke2' if smoke else 'steps1000', new['run']['variant'])
                            self.assertFalse(new['data_lineage']['old_weights_reused'])

    def test_frozen_source_counts_and_initializers_do_not_select180_or_metal(self):
        sources = contract.dataset_sources()
        self.assertEqual({key: row['rows'] for key, row in sources.items()},
                         {'subject': 72, 'material': 8, 'texture': 16})
        self.assertIn('subject_trellis_spoons600_v1', sources['subject']['dataset'])
        self.assertEqual(Path(sources['material']['dataset']).parts[-2:], ('staging', 'wood_spoon_material'))
        self.assertEqual(Path(sources['texture']['dataset']).parts[-2:], ('staging', 'wood_spoon_texture'))
        self.assertEqual(sources['material']['initializer'], 'material')
        self.assertEqual(sources['texture']['initializer'], 'texture')
        for spec in sources.values():
            self.assertEqual(spec['concepts_sha256'], spec['hashes'][spec['concepts']])
            self.assertEqual(spec['manifest_sha256'], spec['hashes'][spec['manifest']])

    def test_invalid_branch_method_and_smoke_are_rejected_before_source_reads(self):
        for branch, method, smoke in (('color', 'fullrank', False), ('subject', 'rank4', False),
                                      ('material', 'lora_r64', 1)):
            with self.subTest(branch=branch, method=method), self.assertRaises(ValueError):
                contract.build_config(branch, method, smoke=smoke)

    def subject_fixture(self):
        temporary = tempfile.TemporaryDirectory(prefix='sd21-wood-data-', dir=os.environ.get('TMP'))
        self.addCleanup(temporary.cleanup)
        dataset = Path(temporary.name).resolve()
        rows = []
        for index in range(72):
            image, mask = dataset / f'image{index}.png', dataset / f'mask{index}.png'
            image.write_bytes(b'unchanged source RGB')
            mask.write_bytes(b'unchanged source mask')
            rows.append({'id': str(index), 'image': str(image), 'mask': str(mask),
                'image_sha256': contract.sha256(image), 'mask_sha256': contract.sha256(mask),
                'caption': 'a photo of <S*> spoon in blue color'})
        (dataset / 'training_assets_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        for name in ('concepts.json', 'source_locks.json', 'approval.json', 'dataset_audit.json'):
            (dataset / name).write_text('{}')
        hashes = {str(dataset / name): contract.sha256(dataset / name) for name in
                  ('concepts.json', 'training_assets_manifest.jsonl', 'dataset_audit.json')}
        spec = {'branch': 'subject', 'dataset': str(dataset), 'concepts': str(dataset / 'concepts.json'),
            'manifest': str(dataset / 'training_assets_manifest.jsonl'), 'rows': 72, 'hashes': hashes}
        return dataset, rows, spec

    def test_dataset_verifier_preserves_all72_rows_and_rejects_rgb_change(self):
        dataset, rows, spec = self.subject_fixture()
        with patch.object(contract, 'dataset_sources', return_value={'subject': spec}), \
             patch.object(contract, 'verify_subject', return_value={
                 'subject_id': 'wood_spoon', 'rows': 72, 'selected_colors': ['green', 'blue']}):
            audit = contract.verify_dataset('subject')
            self.assertEqual(audit['references'], rows)
            Path(rows[0]['image']).write_bytes(b'changed RGB')
            with self.assertRaisesRegex(ValueError, 'RGB or mask bytes'):
                contract.verify_dataset('subject')

    def test_dataset_verifier_rejects_caption_or_order_edits_before_legacy_audit(self):
        dataset, rows, spec = self.subject_fixture()
        path = dataset / 'training_assets_manifest.jsonl'
        with patch.object(contract, 'dataset_sources', return_value={'subject': spec}), \
             patch.object(contract, 'verify_subject') as legacy:
            path.write_text(''.join(json.dumps(row) + '\n' for row in reversed(rows)))
            with self.assertRaisesRegex(ValueError, 'metadata bytes changed'):
                contract.verify_dataset('subject')
            legacy.assert_not_called()
            rows[0]['caption'] = 'a different subject caption'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'metadata bytes changed'):
                contract.verify_dataset('subject')
            legacy.assert_not_called()


if __name__ == '__main__':
    unittest.main()
