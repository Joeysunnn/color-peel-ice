import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from experiments.icbench_five_image_mesh_v1 import run as batch


POOL = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 17, 18, 19, 24, 30, 31, 43, 46, 53, 58, 71, 74, 76, 77, 81, 86, 88]


class ICBenchFiveImageMeshTests(unittest.TestCase):
    def selection(self, root):
        rows = []
        for sample in random.Random(42).sample(POOL, 5):
            image = root / f'{sample}.jpg'
            image.write_bytes(str(sample).encode())
            rows.append({'id': sample, 'path': str(image), 'sha256': batch.sha256(image)})
        selection = {'sampling_seed': 42, 'eligible_image_ids': POOL, 'images': rows}
        path = root / 'selection.json'
        batch.write_json(path, selection)
        return path, selection

    def test_fixed_sampling_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, selection = self.selection(root)
            batch.validate_selection(selection)
            selection['images'].reverse()
            with self.assertRaisesRegex(ValueError, 'recorded single'):
                batch.validate_selection(selection)
            selection['images'].reverse()
            Path(selection['images'][0]['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                batch.validate_selection(selection)

    def test_intake_preserves_failed_images_and_objects_and_continues(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection_path, selection = self.selection(root)
            commands = []

            def execute(command, log, gpu):
                commands.append(command)
                output = Path(command[command.index('--output') + 1])
                if '--worker' in command:
                    action = command[command.index('--worker') + 1]
                    source = command[command.index('--image') + 1]
                    sample = int(Path(source).stem)
                    if action == 'discover' and sample == 53:
                        raise RuntimeError('fixture discovery failure')
                    output.mkdir()
                    if action == 'segment':
                        records = [{'id': f'object_{i+1:03d}', 'category': 'object',
                                    'source_image': source, 'object_mask': 'fixture'}
                                   for i in range(2 if sample == 3 else 1)]
                        batch.write_json(output / 'objects.json', {'source': {'image_sha256': batch.sha256(source)},
                            'status': 'succeeded', 'objects': records, 'failures': [], 'rejected': []})
                else:
                    self.assertIn('src.methods.colorpeel_ice.subject_alpha', command)
                    if 'object_002' in str(output):
                        raise RuntimeError('fixture BiRef association failure')
                    output.mkdir()
                    batch.write_json(output / 'source_receipt.json', {})

            def inputs(image, mask, output):
                output.mkdir()
                batch.write_json(output / 'source_receipt.json', {})

            with patch.object(batch, 'execute', side_effect=execute), \
                    patch.object(batch, 'prepare_inputs', side_effect=inputs), \
                    patch.object(batch, 'git_info', return_value={'commit': 'fixture'}):
                state = batch.intake(selection_path, root / 'intake', [0, 1])
            self.assertEqual(state['status'], 'partial')
            self.assertEqual(state['retained_objects'], 5)
            self.assertEqual(state['ready_objects'], 4)
            self.assertEqual(state['expected_meshes'], 40)
            self.assertEqual(state['images'][0]['status'], 'failed')
            objects = batch.read(root / 'intake/factorial_objects.json')
            self.assertIn('blocked_reason', objects['image_003__object_002'])
            self.assertTrue(all(row['source_rgb'] and row['reference_review'] is None for row in objects.values()))
            self.assertTrue(all(not {'initialize', 'train-object', '--fixed-crop', 'SuperMat'} & set(cmd) for cmd in commands))
            batch.write_json(root / 'intake/factorial_objects.json', {})
            with patch.object(batch, 'git_info', return_value={'commit': 'fixture'}):
                with self.assertRaisesRegex(ValueError, 'changed'):
                    batch.meshes(root / 'intake', root / 'meshes', [0, 1])


if __name__ == '__main__':
    unittest.main()
