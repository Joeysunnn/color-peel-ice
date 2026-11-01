"""Check the approved27-image recipe and frozen-image/data-only receipt guards."""

from itertools import product
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.sd21_wood_subject27_v1 import contract


class Wood27ContractTest(unittest.TestCase):
    def test_science_arguments_change_only_concepts_for_both_methods_and_smoke(self):
        with patch.object(contract.previous.subject_training, 'read_json',
                          return_value={'subject_id': 'wood_spoon', 'rows': 72}):
            for method in contract.METHODS:
                for smoke in (False, True):
                    with self.subTest(method=method, smoke=smoke):
                        old = contract.previous.build_config('subject', method, smoke=smoke)
                        new = contract.build_config('subject', method, smoke=smoke)
                        changed = {key for key in old['args'].keys() | new['args'].keys()
                                   if old['args'].get(key) != new['args'].get(key)}
                        self.assertEqual(changed, {'concepts_list'})
                        self.assertEqual(new['args']['initializer_token'], 'spoon')
                        self.assertEqual(new['args']['checkpoint_steps'], [2] if smoke else [400, 600, 800, 1000])
                        self.assertEqual(new['run']['study'], contract.STUDY)
                        self.assertEqual(new['data_lineage']['source']['rows'], 27)
                        self.assertFalse(new['data_lineage']['old_weights_reused'])
        with self.assertRaises(ValueError):
            contract.build_config('texture', 'fullrank')

    def fixture(self):
        temporary = tempfile.TemporaryDirectory(prefix='sd21-subject27-', dir=os.environ.get('TMP'))
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name).resolve()
        root = folder / 'frozen'
        rows = []
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[128:384, 128:384] = 255
        combinations = list(product(('left', 'center', 'right'), ('blue', 'green'), (0, 1), (.15, .85)))
        for index in range(27):
            if index < 24:
                view, color, metal, rough = combinations[index]
                row = {'id': f'{view}__{color}__M{metal}_R{rough}__studio',
                    'color': color, 'view': view, 'light': 'studio',
                    'controls': {'metallic': metal, 'roughness': rough},
                    'caption': f'a photo of <S*> spoon in {color} color', 'kind': 'synthetic'}
            else:
                name = ('photo_original', 'photo_lch_blue', 'photo_lch_green')[index - 24]
                row = {'id': name, 'color': ('original', 'blue', 'green')[index - 24],
                    'view': 'original_photo', 'light': 'original_photo', 'controls': None,
                    'caption': 'a photo of <S*> spoon with a wood surface',
                    'kind': 'original_photograph' if index == 24 else 'lch_recolor'}
            row['training_eligible'] = False
            for field, pixels in (('image', np.full((512, 512, 3), 40 + index, dtype=np.uint8)), ('mask', mask)):
                path = root / 'dataset/assets' / row['id'] / ('images' if field == 'image' else 'masks') / 'image.png'
                path.parent.mkdir(parents=True)
                Image.fromarray(pixels).save(path)
                row[field], row[field + '_sha256'] = str(path), contract.sha256(path)
            rows.append(row)
        ledger = root / 'dataset/training_assets_manifest.jsonl'
        ledger.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        concepts = [{'instance_prompt': [row['caption']], 'instance_data_dir': str(Path(row['image']).parent),
                     'instance_mask_dir': str(Path(row['mask']).parent)} for row in rows]
        (root / 'dataset/concepts.json').write_text(json.dumps(concepts))
        original = folder / 'original.png'
        original.write_bytes(b'old immutable source')
        source_locks = {str(original): contract.sha256(original)}
        (root / 'source_locks.json').write_text(json.dumps(source_locks))
        (root / 'protocol.json').write_text(json.dumps({'training_authorized': False, 'training_started': False}))
        for index in range(11):
            (root / f'review{index}.json').write_text('{}')
        files = {path.relative_to(root).as_posix(): contract.sha256(path) for path in root.rglob('*') if path.is_file()}
        self.assertEqual(len(files), 69)
        audit = {'status': 'dataset27_constructed_pending_user_review', 'rows': 27,
            'synthetic_rows': 24, 'photo_rows': 3, 'synthetic_copies_byte_identical': True,
            'training_started': False, 'monitoring': False, 'files': files, 'source_locks': source_locks}
        (root / 'dataset_audit.json').write_text(json.dumps(audit))
        for name, value in (('SOURCE_ROOT', root), ('DATASET', root / 'dataset'),
            ('AUDIT_SHA256', contract.sha256(root / 'dataset_audit.json')),
            ('CONCEPTS_SHA256', files['dataset/concepts.json']),
            ('MANIFEST_SHA256', files['dataset/training_assets_manifest.jsonl'])):
            patcher = patch.object(contract, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return root, rows, original

    def test_real27_verifier_checks_all_files_and_keeps_dataset_flags_false(self):
        root, rows, original = self.fixture()
        before = (root / 'dataset_audit.json').read_bytes()
        result = contract.verify_dataset('subject')
        self.assertEqual(result['rows'], 27)
        self.assertEqual(result['references'], rows)
        self.assertTrue(all(row['training_eligible'] is False for row in result['references']))
        self.assertEqual(len(result['source_locks']), 71)  #69 outputs + receipt + upstream source
        self.assertIn(str(original), result['source_locks'])
        self.assertEqual((root / 'dataset_audit.json').read_bytes(), before)

    def test_photo_bytes_and_old_source_changes_are_rejected(self):
        root, rows, original = self.fixture()
        photo = Path(rows[24]['image'])
        before = photo.read_bytes()
        photo.write_bytes(b'changed original photo')
        with self.assertRaisesRegex(ValueError, 'output path or bytes'):
            contract.verify_dataset('subject')
        photo.write_bytes(before)
        original.write_bytes(b'changed upstream input')
        with self.assertRaisesRegex(ValueError, 'upstream source bytes'):
            contract.verify_dataset('subject')

    def test_factor_metadata_and_dataset_policy_changes_are_rejected(self):
        root, rows, _ = self.fixture()
        ledger = root / 'dataset/training_assets_manifest.jsonl'
        before = ledger.read_bytes()
        rows[0]['controls']['roughness'] = .50
        ledger.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaisesRegex(ValueError, 'metadata bytes'):
            contract.verify_dataset('subject')
        # Even a coherently rehashed candidate must obey the inherited24-cell factorial.
        with self.assertRaisesRegex(ValueError, 'exactly24'):
            contract.prepare.verify_output(root)
        ledger.write_bytes(before)
        audit_path = root / 'dataset_audit.json'
        audit = json.loads(audit_path.read_text())
        audit['training_started'] = True
        audit_path.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError, 'metadata bytes'):
            contract.verify_dataset('subject')


if __name__ == '__main__':
    unittest.main()
