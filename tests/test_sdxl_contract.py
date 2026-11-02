"""CPU checks for SDXL independent probes and immutable source-data reuse."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.sdxl_mailbox_v1 import contract, data


class SDXLContractTest(unittest.TestCase):
    def test_transfer_matrix_is_single_branch_and_keeps_literal_controls(self):
        counts = {'subject': 396, 'color': 220, 'material': 528, 'texture': 528}
        for branch, spec in contract.TRAINING.items():
            rows = contract.transfer_cases(branch, spec['steps'])
            self.assertEqual(len(rows), counts[branch])
            self.assertEqual(len({row['id'] for row in rows}), len(rows))
            self.assertEqual({row['seed'] for row in rows}, set(range(30, 41)))
            for row in rows:
                for other, other_spec in contract.TRAINING.items():
                    if other != branch:
                        self.assertNotIn(other_spec['token'], row['prompt'])
                self.assertEqual(row['prompt'].count(spec['token']),
                                 int(row['condition'].startswith('learned_')))
        material = contract.transfer_cases('material', 400)
        texture = contract.transfer_cases('texture', 400)
        self.assertIn('a photo of a chair made of painted metal', [row['prompt'] for row in material])
        self.assertIn('a photo of a chair with fine speckled texture', [row['prompt'] for row in texture])
        self.assertEqual(contract.TRAINING['material']['initializer'], 'metal')
        self.assertEqual(2 * sum(counts.values()), 3344)

    def test_smoke_contains_only_learned_token_and_unknown_checkpoints_fail(self):
        for branch in contract.BRANCHES:
            rows = contract.transfer_cases(branch, 2, smoke=True)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['seed'], 30)
            self.assertTrue(rows[0]['condition'].startswith('learned_'))
            with self.assertRaises(ValueError):
                contract.transfer_cases(branch, 2)
        with self.assertRaises(ValueError):
            contract.transfer_cases('subject', [400, 400])

    def test_training_resize_preserves_valid_border_and_object_intersection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgb, mask = root / 'image.png', root / 'mask.png'
            Image.new('RGB', (512, 512), (80, 90, 100)).save(rgb)
            foreground = Image.new('L', (512, 512), 0)
            foreground.paste(255, (0, 0, 256, 512))
            foreground.save(mask)
            image, valid = data.load_training_image({'image': str(rgb), 'mask': None})
            self.assertEqual(image.size, (1024, 1024))
            self.assertEqual(valid.size, (128, 128))
            self.assertEqual(valid.histogram()[255], 126 * 126)
            self.assertEqual(valid.getpixel((0, 30)), 0)
            self.assertEqual(valid.getpixel((127, 30)), 0)
            _, clipped = data.load_training_image({'image': str(rgb), 'mask': str(mask)})
            self.assertEqual(clipped.histogram()[255], 63 * 126)
            self.assertEqual(clipped.getpixel((63, 30)), 255)
            self.assertEqual(clipped.getpixel((64, 30)), 0)

    def test_byte_copy_order_hash_verification_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, output = root / 'source', root / 'new'
            source.mkdir()
            image, mask = source / 'image.png', source / 'mask.png'
            Image.new('RGB', (512, 512), (30, 80, 100)).save(image)
            Image.new('L', (512, 512), 255).save(mask)
            locks = {str(path): data.sha256(path) for path in (image, mask)}
            inputs = {}
            for branch, spec in contract.TRAINING.items():
                inputs[branch] = [{'id': f'{branch}_{index:02d}', 'prompt': 'a photo of ' + spec['token'],
                    'image': str(image), 'image_sha256': locks[str(image)],
                    'mask': None if branch == 'color' else str(mask),
                    'mask_sha256': None if branch == 'color' else locks[str(mask)]}
                    for index in range(spec['images'])]
            with patch.object(data, 'source_rows', return_value=(inputs, locks)):
                specs = data.prepare_data(output)
                with self.assertRaises(ValueError):
                    data.prepare_data(output)
            for branch, spec in specs.items():
                copied = data.load_rows(spec['manifest'])
                self.assertEqual([row['id'] for row in copied], [row['id'] for row in inputs[branch]])
                self.assertEqual(Path(copied[0]['image']).read_bytes(), image.read_bytes())
            self.assertEqual(locks, {path: data.sha256(path) for path in locks})
            copied_image = Path(data.rows(output, 'subject')[0]['image'])
            Image.new('RGB', (512, 512), (1, 2, 3)).save(copied_image)
            with self.assertRaises(ValueError):
                data.verify(output)
            with self.assertRaises(ValueError):
                data.load_rows(specs['subject']['manifest'])

    def test_staging_detects_source_change_after_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            file = root / 'source.txt'
            file.write_text('original')
            original = data.sha256(file)
            file.write_text('changed')
            with self.assertRaises(ValueError):
                data._check_locks({str(file): original})


if __name__ == '__main__':
    unittest.main()
