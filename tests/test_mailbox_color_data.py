"""Exercise real18-render selection, copied RGB data and immutable source checks."""

from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.mailbox_color_sd15_v1 import data


class MailboxColorDataTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='mailbox-color-data-', dir=os.environ.get('TMP'))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / 'source18'
        self.staging = self.root / 'old_orange9'
        self.outputs = self.root / 'outputs'
        self.source.mkdir()
        self.staging.mkdir()
        protocol = copy.deepcopy(data.emission.protocol())
        records = []
        for color_index, stable_id in enumerate(('D1GT:13/30.png', 'D1GT:81/130.png')):
            for shape_index, shape in enumerate(('cube', 'sphere', 'cylinder')):
                for view in (0, 8, 16):
                    request_id = 'emission__' + stable_id.replace('D1GT:', 'D1GT_').replace('/', '_').replace('.png', '') + f'__{shape}__v{view:02d}'
                    image = self.source / (request_id + '.png')
                    Image.new('RGB', (512, 512), (100 + color_index * 100, 40 + shape_index * 20, 10 + view)).save(image)
                    metadata = self.source / (request_id + '.json')
                    data.write_json(metadata, {'request': {'stable_id': stable_id, 'shape': shape,
                        'view_index': view, 'material': 'Emission', 'emission_strength': 1.0}})
                    records.append({'request_id': request_id, 'image_relative_path': image.name,
                        'metadata_relative_path': metadata.name, 'image_sha256': data.sha256(image),
                        'metadata_sha256': data.sha256(metadata)})
        data.write_json(self.source / data.emission.MANIFEST_NAME,
            {'schema': 'd1_emission_color_branch_pilot_manifest/v1', 'request_count': 18, 'records': records})
        data.write_json(self.source / data.emission.ANALYSIS_NAME,
            {'status': 'passed', 'overall_pass': True, 'request_count': 18})
        protocol['source_pilot']['analysis_sha256'] = data.sha256(self.source / data.emission.ANALYSIS_NAME)
        protocol_path = self.root / 'orange_protocol.json'
        data.write_json(protocol_path, protocol)
        selected = data.emission.verified_source(self.source, protocol)
        staged_records, concepts = [], []
        for request_id, row in selected.items():
            folder = self.staging / request_id
            folder.mkdir()
            shutil.copyfile(row['image'], folder / 'img.png')
            concepts.append({'instance_prompt': [row['prompt']], 'instance_data_dir': str(folder)})
            staged_records.append({'request_id': request_id, 'shape': row['shape'], 'view_index': row['view_index'],
                'prompt': row['prompt'], 'source_image_sha256': row['image_sha256'], 'staging_method': 'copy_fallback'})
        data.write_json(self.staging / 'concepts.json', concepts)
        data.write_json(self.staging / 'staging_manifest.json', {'record_count': 9, 'records': staged_records,
            'source_analysis_sha256': protocol['source_pilot']['analysis_sha256'],
            'protocol_sha256': data.sha256(protocol_path)})
        legacy = data.read_config(data.LEGACY_CONFIG)
        legacy['color_material_source'].update(protocol=str(protocol_path), protocol_sha256=data.sha256(protocol_path),
            source_root=str(self.source), staging_root=str(self.staging),
            staging_manifest_sha256=data.sha256(self.staging / 'staging_manifest.json'),
            concepts_sha256=data.sha256(self.staging / 'concepts.json'))
        legacy_path = self.root / 'legacy.json'
        data.write_json(legacy_path, legacy)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(data, 'LEGACY_CONFIG', legacy_path))
        self.stack.enter_context(patch.dict(os.environ, {'COLORPEEL_OUTPUT_ROOT': str(self.outputs)}))

    def test_exact_orange9_rgb_copy_caption_alignment_and_no_masks(self):
        source_hashes = {str(path): data.sha256(path) for path in self.source.iterdir()}
        output = self.outputs / 'training_dataset'
        audit = data.stage_dataset(output)
        self.assertEqual(data.verify_dataset(output), audit)
        rows = [json.loads(line) for line in (output / 'training_assets_manifest.jsonl').read_text().splitlines()]
        self.assertEqual(audit['rows'], 9)
        self.assertEqual(audit['target']['stable_id'], 'D1GT:81/130.png')
        self.assertFalse(audit['uses_masks'])
        self.assertEqual(audit['source_order'], [row['id'] for row in rows])
        self.assertEqual([(row['shape'], row['view']) for row in rows],
            [(shape, view) for shape in ('cube', 'sphere', 'cylinder') for view in (0, 8, 16)])
        concepts = data.read_json(output / 'concepts.json')
        for row, concept in zip(rows, concepts):
            self.assertTrue(row['id'].startswith('emission__D1GT_81_130__'))
            self.assertEqual(Path(row['image']).read_bytes(), Path(row['source_image']).read_bytes())
            self.assertEqual(row['caption'], f'a photo of {row["shape"]} shape in <C*> color')
            self.assertEqual(concept, {'instance_prompt': [row['caption']], 'instance_data_dir': str(Path(row['image']).parent)})
            self.assertEqual(list(Path(concept['instance_data_dir']).iterdir()), [Path(row['image'])])
        self.assertEqual(len(list(output.rglob('*.png'))), 9)
        self.assertEqual(data.read_json(output / 'approval.json')['checkpoint_steps'], [400, 600, 800, 1000])
        self.assertEqual(data.read_json(output / 'approval.json')['inference_seeds'], list(range(30, 61)))
        self.assertEqual(source_hashes, {str(path): data.sha256(path) for path in self.source.iterdir()})

    def test_modified_training_image_caption_and_approval_are_rejected(self):
        for index, kind in enumerate(('image', 'caption', 'approval', 'extra_image')):
            with self.subTest(kind=kind):
                output = self.outputs / f'dataset{index}'
                data.stage_dataset(output)
                manifest = output / 'training_assets_manifest.jsonl'
                rows = [json.loads(line) for line in manifest.read_text().splitlines()]
                if kind == 'image':
                    Image.new('RGB', (512, 512), (0, 0, 0)).save(rows[0]['image'])
                elif kind == 'caption':
                    rows[0]['caption'] = 'a photo of <C*> mailbox'
                    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                elif kind == 'approval':
                    approval = data.read_json(output / 'approval.json')
                    approval['max_train_steps'] = 600
                    data.write_json(output / 'approval.json', approval)
                else:
                    shutil.copyfile(rows[0]['image'], Path(rows[0]['image']).parent / 'extra.png')
                with self.assertRaises(ValueError):
                    data.verify_dataset(output)

    def test_modified_historical_source_is_rejected(self):
        output = self.outputs / 'training_dataset'
        data.stage_dataset(output)
        image = next(self.source.glob('*D1GT_81_130*.png'))
        image.write_bytes(b'corrupted historical image')
        with self.assertRaises((ValueError, RuntimeError)):
            data.verify_dataset(output)

    def test_existing_outputs_repo_paths_and_source_overlap_are_rejected(self):
        output = self.outputs / 'training_dataset'
        data.stage_dataset(output)
        with self.assertRaises(ValueError):
            data.stage_dataset(output)
        with patch.dict(os.environ, {'COLORPEEL_OUTPUT_ROOT': str(data.ROOT.parent)}):
            with self.assertRaises(ValueError):
                data.stage_dataset(data.ROOT / 'must_not_create')
        with patch.dict(os.environ, {'COLORPEEL_OUTPUT_ROOT': str(self.root)}):
            with self.assertRaises(ValueError):
                data.stage_dataset(self.source / 'must_not_create')


if __name__ == '__main__':
    unittest.main()
