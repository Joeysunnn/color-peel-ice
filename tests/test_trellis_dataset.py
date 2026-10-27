"""CPU checks for provenance-sensitive image pairing and single-factor edges."""

import copy
import json
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
from PIL import Image

from experiments.subject_trellis_dataset_v1.build import PROTOCOL, array_hash, rows, sha256
from experiments.subject_trellis_dataset_v1.verify import one_factor_pairs, verify, verify_record


class TrellisDatasetTest(unittest.TestCase):
    def setUp(self):
        self.output = Path(__file__).resolve().parents[1] / ('test-trellis-' + uuid.uuid4().hex)
        self.output.mkdir()
        self.addCleanup(self.cleanup)
        self.config = json.loads(PROTOCOL.read_text())

    def cleanup(self):
        if not self.output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
            raise ValueError('test cleanup must stay inside its workspace')
        shutil.rmtree(self.output)

    def fixture(self):
        row = rows(self.config)[0]
        mask = np.zeros((512, 512), dtype=bool)
        mask[100:400, 120:360] = True
        normal = np.zeros((3, 512, 512), dtype=np.float32)
        normal[2, mask] = 1
        raw = np.zeros((512, 512, 3), dtype=np.uint8)
        raw[mask] = [80, 40, 120]
        locations = {'image': f'dataset/images/{row["id"]}/image.png',
                     'mask': f'dataset/masks/{row["id"]}/image.png',
                     'raw_image': 'raw/source.png', 'foreground_rgba': 'cutouts/source.png', 'maps': 'maps/source.npz'}
        for location in locations.values():
            (self.output / location).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.where(mask[..., None], raw, 127).astype(np.uint8)).save(self.output / locations['image'])
        Image.fromarray(mask.astype(np.uint8) * 255).save(self.output / locations['mask'])
        Image.fromarray(raw).save(self.output / locations['raw_image'])
        Image.fromarray(np.concatenate([raw, mask[..., None].astype(np.uint8) * 255], axis=-1)).save(
            self.output / locations['foreground_rgba'])
        base_color = np.zeros((3, 512, 512), dtype=np.float32)
        base_color[:, mask] = np.array(self.config['colors'][row['color']])[:, None]
        np.savez_compressed(self.output / locations['maps'], base_color=base_color,
            metallic=np.zeros((512, 512), dtype=np.float32),
            roughness=mask.astype(np.float32) * .15, alpha=mask.astype(np.float32))
        row.update(image=locations['image'].removeprefix('dataset/'), image_sha256=sha256(self.output / locations['image']),
            mask=locations['mask'].removeprefix('dataset/'), mask_sha256=sha256(self.output / locations['mask']),
            raw_image=locations['raw_image'], raw_sha256=sha256(self.output / locations['raw_image']),
            foreground_rgba=locations['foreground_rgba'], maps=locations['maps'], maps_sha256=sha256(self.output / locations['maps']),
            geometry_buffer_sha256={'mask': array_hash(mask), 'normal': array_hash(normal)})
        return row, {'mask': mask, 'normal': normal}

    def test_valid_rgb_mask_and_material_map_pair(self):
        row, geometry = self.fixture()
        rgb, mask = verify_record(self.output, row, self.config, geometry)
        self.assertEqual(rgb.shape, (512, 512, 3))
        self.assertEqual(int(mask.sum()), 300 * 240)

    def test_wrong_mask_rejected_even_when_its_file_hash_is_updated(self):
        row, geometry = self.fixture()
        path = self.output / 'dataset' / row['mask']
        Image.fromarray(np.roll(geometry['mask'], 1, axis=0).astype(np.uint8) * 255).save(path)
        row['mask_sha256'] = sha256(path)
        with self.assertRaisesRegex(ValueError, 'geometry buffer'):
            verify_record(self.output, row, self.config, geometry)

    def test_wrong_declared_material_rejected_against_actual_maps(self):
        row, geometry = self.fixture()
        row['controls']['metallic'] = 1.0
        with self.assertRaisesRegex(ValueError, 'declared intervention'):
            verify_record(self.output, row, self.config, geometry)

    def test_pairs_change_exactly_one_factor_and_use_only_adjacent_roughness(self):
        group = [row for row in rows(self.config) if row['pair_group'] == 'left__pink__studio']
        for row in group:
            row['geometry_buffer_sha256'] = {'mask': 'fixed-mask', 'normal': 'fixed-normal'}
        pairs = one_factor_pairs(group)
        lookup = {row['id']: row for row in group}
        self.assertEqual(len(pairs), 7)
        self.assertEqual(sum(edge['axis'] == 'metallic' for edge in pairs), 3)
        self.assertEqual(sum(edge['axis'] == 'roughness' for edge in pairs), 4)
        self.assertEqual(len({(edge['from'], edge['to']) for edge in pairs}), 7)
        for edge in pairs:
            a, b = lookup[edge['from']], lookup[edge['to']]
            changed = {key for key in a['controls'] if a['controls'][key] != b['controls'][key]}
            self.assertEqual(changed, {edge['axis']})
            if edge['axis'] == 'roughness':
                self.assertAlmostEqual(edge['to_value'] - edge['from_value'], .35)
        group[-1]['light'] = 'different-light'
        with self.assertRaisesRegex(ValueError, 'fixed control'):
            one_factor_pairs(group)

    def test_duplicate_factorial_is_rejected_before_reading_uncreated_images(self):
        shutil.copyfile(PROTOCOL, self.output / 'protocol.json')
        records = rows(self.config)
        records[-1] = copy.deepcopy(records[0])
        (self.output / 'dataset').mkdir()
        manifest = {'records': records, 'schema': self.config['schema'],
                    'training_started': False, 'training_eligible_images': 0}
        (self.output / 'dataset/manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'complete unique'):
            verify(self.output)
        self.assertFalse((self.output / 'verification.json').exists())


if __name__ == '__main__':
    unittest.main()
