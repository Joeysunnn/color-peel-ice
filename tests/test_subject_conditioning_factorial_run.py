import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.subject_conditioning_factorial_v1 import prepare as preparation
from experiments.subject_conditioning_factorial_v1 import run as runner


class FactorialRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.revision = {'commit': 'a' * 40, 'branch': 'fixture'}
        self.pinned = {'model': 'b' * 40}
        self.models = self.root / 'model_revisions.json'
        self.write(self.models, self.pinned)
        self.commands, self.prepare_calls = [], []
        self.fail_variant = None

    @staticmethod
    def write(path, value):
        Path(path).write_text(json.dumps(value), encoding='utf8')

    def sources(self, name, reference=False):
        matte = self.root / (name + '.json')
        self.write(matte, {'fixture': name})
        review = self.root / (name + '_review.json')
        if reference:
            self.write(review, {'camera': 'old frozen camera'})
        return {'matte_receipt': matte, 'source_rgb': not reference,
                'reference_review': review if reference else None}

    def fake_prepare(self, matte, output, source_rgb=False):
        self.prepare_calls.append((matte, source_rgb))
        if Path(matte).stem == 'bad_prepare':
            raise ValueError('BiRef fixture preparation failure')
        output.mkdir(parents=True)
        variants = []
        for name in runner.VARIANT_ORDER:
            folder = output / name
            folder.mkdir()
            receipt, expected = folder / 'receipt.json', folder / 'expected.png'
            self.write(receipt, {'variant': name})
            Image.new('RGB', (4, 4), (100, 120, 140)).save(expected)
            variants.append({'name': name, 'source_receipt': str(receipt),
                             'expected_condition': str(expected), 'auto_crop_xyxy': [0, 0, 4, 4]})
        return {'git': self.revision, 'source_locks': {str(matte): runner.sha256(matte)},
                'crop_policy': 'official automatic crop', 'variants': variants}

    def fake_execute(self, command, log, gpu):
        self.commands.append(command)
        output = Path(command[command.index('--output') + 1])
        if command[3] == 'experiments.subject_trellis_counterfactual_v1.run':
            receipt = Path(command[command.index('--source-receipt') + 1])
            if receipt.parent.name == self.fail_variant:
                raise RuntimeError('fixture generation failed')
            (output / 'repro_outputs').mkdir(parents=True)
            (output / 'review').mkdir()
            (output / 'native_asset.pt').write_bytes(b'native mesh fixture')
            (output / 'actual_condition.png').write_bytes((receipt.parent / 'expected.png').read_bytes())
            # Pink has a higher reported score, but must never become the reference.
            self.write(output / 'review/review.json', {'camera': receipt.parent.name,
                'metrics': {'original512_mask_iou': 1 if receipt.parent.name.startswith('pink') else .1}})
            self.write(output / 'pipeline.json', {'args': {'shape_slat_normalization': [0, 1],
                                                         'tex_slat_normalization': [0, 1]}})
            self.write(output / 'repro_outputs/status.json', {
                'status': 'generated_pending_user_review', 'generated_assets': 1,
                'git': self.revision, 'official_commit': runner.OFFICIAL, 'seed': 42,
                'models': {'revisions': self.pinned}, 'source_receipt': {'sha256': runner.sha256(receipt)},
                'pipeline_type': '1024_cascade', 'encoder_receipt_sha256': 'encoder',
                'sampling': {'steps': 12}, 'flow_dtype': {'flow': 'float16'}, 'sparse_algorithm': 'explicit_gemm'})
        else:
            self.assertEqual(command[3], 'experiments.mailbox_alpha_control_v1.render')
            reference = Path(command[command.index('--reference-review') + 1])
            self.assertTrue(reference.is_file())
            output.mkdir()
            self.write(output / 'status.json', {'status': 'completed_pending_visual_review',
                                               'git': self.revision, 'completed_renders': 32})

    def run_case(self, objects, partial=False, default=False):
        output = self.root / 'output'
        def fresh(path):
            path.mkdir()
            return path
        with patch.object(runner, 'git_info', return_value=self.revision), \
                patch.object(runner, 'fresh', side_effect=fresh), \
                patch.object(runner, 'MODEL_REVISIONS', self.models), \
                patch.object(preparation, 'prepare', side_effect=self.fake_prepare), \
                patch.object(runner, 'execute', side_effect=self.fake_execute), \
                patch.object(runner, 'OBJECTS', objects):
            if partial:
                with self.assertRaisesRegex(RuntimeError, 'partial'):
                    runner.run(output, [0], **({} if default else {'objects': objects}))
            else:
                result = runner.run(output, [0], **({} if default else {'objects': objects}))
                self.assertEqual(result['status'], 'completed_pending_user_review')
        return json.loads((output / 'status.json').read_text())

    def test_default_objects_preserve_sixteen_jobs_and_supplied_references(self):
        objects = {name: self.sources(name, reference=True) for name in ('mailbox', 'wood_spoon')}
        for source in objects.values():
            del source['source_rgb']  # Existing OBJECTS does not have this field.
        state = self.run_case(objects, default=True)
        self.assertEqual(state['expected_jobs'], 16)
        self.assertEqual(state['generated_jobs'], 16)
        self.assertTrue(all(not use_source for _, use_source in self.prepare_calls))
        for name, source in objects.items():
            self.assertEqual(state['objects'][name]['reference_review'], str(source['reference_review']))

    def test_variable_object_count_freezes_predetermined_new_camera_without_score_selection(self):
        objects = {name: self.sources(name) for name in ('first', 'second', 'third')}
        state = self.run_case(objects)
        self.assertEqual(state['expected_jobs'], 24)
        self.assertEqual(state['generated_jobs'], 24)
        for name in objects:
            reference = str(self.root / 'output' / name / 'variants/original__biref__biref/trellis/review/review.json')
            self.assertEqual(state['objects'][name]['reference_review'], reference)
            self.assertEqual(state['source_locks'][reference], runner.sha256(reference))
        for command in self.commands:
            self.assertNotIn('--fixed-crop', command)
            self.assertNotIn('train', command)
        self.assertTrue(all(use_source for _, use_source in self.prepare_calls))

    def test_intake_and_prepare_failures_block_eight_jobs_each_and_continue(self):
        objects = {'intake_failed': {'blocked_reason': 'SAM found no usable mask'},
                   'bad': self.sources('bad_prepare'), 'healthy': self.sources('healthy')}
        state = self.run_case(objects, partial=True)
        self.assertEqual(state['expected_jobs'], 24)
        self.assertEqual(state['generated_jobs'], 8)
        self.assertEqual(len(state['jobs']), 24)
        blocked = [job for job in state['jobs'] if job['status'] == 'blocked']
        self.assertEqual(len(blocked), 16)
        self.assertTrue(all('error' in job for job in blocked))
        self.assertEqual(len(self.prepare_calls), 2)
        self.assertEqual([row['status'] for row in state['renders']], ['blocked', 'blocked', 'completed'])
        self.assertEqual(len(self.commands), 9)

    def test_failed_reference_generation_blocks_comparison_without_replacement(self):
        self.fail_variant = 'original__biref__biref'
        state = self.run_case({'only': self.sources('only')}, partial=True)
        self.assertEqual(state['expected_jobs'], 8)
        self.assertEqual(state['generated_jobs'], 7)
        self.assertEqual(state['renders'][0]['status'], 'blocked')
        self.assertIn('original__biref__biref', state['objects']['only']['reference_review'])
        self.assertEqual(len(self.commands), 8)


if __name__ == '__main__':
    unittest.main()
