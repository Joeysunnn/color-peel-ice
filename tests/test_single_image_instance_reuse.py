import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from src.methods.colorpeel_ice import single_image_pipeline as pipeline


class SingleImageInstanceReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'original.jpg'
        Image.new('RGB', (512, 512), (200, 100, 20)).save(self.image)
        source = self.root / 'source.png'
        with Image.open(self.image) as opened:
            opened.save(source)
        self.record = {'id': 'object_003', 'category': 'mailbox',
                       'source_image': str(source)}
        for key in ('object_mask', 'vlm_input'):
            path = self.root / (key + '.png')
            Image.new('L' if key == 'object_mask' else 'RGB', (512, 512), 255).save(path)
            self.record.update({key: str(path), key + '_sha256': pipeline.sha256(path)})
        self.segmented = {'schema': 'colorpeel_object_segmentation/v1', 'status': 'partial',
                          'failures': ['unrelated instance'], 'source': {
                              'image': '/old/location/original.jpg',
                              'image_sha256': pipeline.sha256(self.image),
                              'canonical_image': str(source),
                              'canonical_image_sha256': pipeline.sha256(source)},
                          'objects': [{'id': 'object_001'}, self.record]}
        self.manifest = self.root / 'objects.json'
        pipeline.write_json(self.manifest, self.segmented)
        self.revision = {'commit': 'a' * 40, 'branch': 'codex/instance-reuse'}
        for context in (patch.object(pipeline, 'git_info', return_value=self.revision),
                        patch.dict(os.environ, {'COLORPEEL_RUN_ROOT': str(self.root),
                                                'COLORPEEL_OUTPUT_ROOT': str(self.root)})):
            context.start()
            self.addCleanup(context.stop)

    def test_selects_one_instance_and_locks_old_provenance(self):
        selected, provenance = pipeline.select_segmented_object(self.image, self.manifest, 'object_003')
        self.assertEqual([row['id'] for row in selected['objects']], ['object_003'])
        self.assertEqual(selected['status'], 'succeeded')
        self.assertEqual(provenance['source_locks'][str(self.manifest)], pipeline.sha256(self.manifest))
        self.assertEqual(provenance['source_locks'][str(self.image)], pipeline.sha256(self.image))

    def test_changed_or_ambiguous_sources_rejected_before_gpu_stages(self):
        mutations = [
            lambda data: data['source'].update(image_sha256='0' * 64),
            lambda data: data['source'].update(canonical_image_sha256='0' * 64),
            lambda data: data.update(objects=[self.record, self.record]),
            lambda data: data.update(objects=[]),
            lambda data: data['objects'][1].update(object_mask_sha256='0' * 64),
            lambda data: data['objects'][1].update(vlm_input_sha256='0' * 64),
            lambda data: data['objects'][1].update(source_image=str(self.image)),
        ]
        for index, mutation in enumerate(mutations):
            data = copy.deepcopy(self.segmented)
            mutation(data)
            pipeline.write_json(self.manifest, data)
            with self.subTest(index=index), patch.object(pipeline, 'execute') as execute:
                with self.assertRaises(ValueError):
                    pipeline.prepare(self.image, self.root / f'bad_{index}', ['material', 'texture'],
                                     0, pipeline.RUNTIME, self.manifest, 'object_003')
                execute.assert_not_called()
                self.assertFalse((self.root / f'bad_{index}').exists())

    def test_reuse_refreshes_vlm_and_cmt_and_keeps_source_files_unchanged(self):
        original = {path: path.read_bytes() for path in self.root.iterdir() if path.is_file()}
        calls = []

        def execute(command, log, gpu):
            command = [str(value) for value in command]
            calls.append(command)
            output = Path(command[command.index('--output') + 1])
            output.mkdir(parents=True)
            if '--worker' in command:
                self.assertEqual(command[command.index('--worker') + 1], 'initialize')
                self.assertEqual(command[command.index('--image') + 1], self.record['vlm_input'])
                pipeline.write_json(output / 'initializers.json', {
                    'selected_branches': ['color'], 'subject_category': 'mailbox', 'unavailable_branches': []})
            elif command[command.index('--stage') + 1] == 'finalize':
                pipeline.write_json(output / 'datasets.json', {'branches': {'color': {}}, 'failures': {}})

        with patch.object(pipeline, 'execute', side_effect=execute):
            result = pipeline.prepare(self.image, self.root / 'prepared', ['color'], 0,
                                      pipeline.RUNTIME, self.manifest, 'object_003')
        state = pipeline.read(result)
        self.assertEqual(state['status'], 'prepared')
        self.assertEqual([item['id'] for item in state['objects']], ['object_003'])
        self.assertEqual(len(calls), 5)  # initialize, CMT inputs/plan/render/finalize
        receipt = pipeline.read(result.parent / 'objects/object_003/input/source_receipt.json')
        self.assertEqual(receipt['source_locks'][str(self.manifest)], pipeline.sha256(self.manifest))
        self.assertEqual(receipt['source_locks'][str(self.image)], pipeline.sha256(self.image))
        self.assertTrue(all(path.read_bytes() == value for path, value in original.items()))

    def test_reuse_requires_paired_selection_and_excludes_subject(self):
        for manifest, object_id, branches in ((self.manifest, None, ['material']),
                                              (None, 'object_003', ['material']),
                                              (self.manifest, 'object_003', ['subject'])):
            with self.subTest(branches=branches), self.assertRaises(ValueError):
                pipeline.prepare(self.image, self.root / 'invalid', branches, 0,
                                 pipeline.RUNTIME, manifest, object_id)


if __name__ == '__main__':
    unittest.main()
