"""CPU checks for immutable original images and fresh, exclusive seed appends."""

import copy
from io import BytesIO
import itertools
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import uuid

from PIL import Image

from experiments.subject_trellis_dataset_v1 import append_prompt_seeds as append
from experiments.sd15_token_local_kv_v1.contract import sha256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


class TrellisAppendSeedsTest(unittest.TestCase):
    def setUp(self):
        self.temporary_root = Path(tempfile.gettempdir()).resolve()
        self.root = self.temporary_root / ('trellis-append-' + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(self.cleanup)
        self.experiment = self.root / 'experiment'
        self.experiment.mkdir()
        self.training = self.root / 'training'
        self.training.mkdir()
        self.revision = {'branch': 'codex/append-test', 'commit': 'a' * 40}
        buffer = BytesIO()
        Image.new('RGB', (2, 2), (40, 90, 140)).save(buffer, format='PNG')
        self.png = buffer.getvalue()
        self.original_images = []
        for name in ('source_training_audit.json', 'source_audit.log', 'prompt_manifest.csv', 'index.html'):
            (self.experiment / name).write_text('original ' + name)
        write_json(self.experiment / 'status.json', {'status': 'inference_verified',
            'inference_git': append.ORIGINAL_GIT, 'training_git': append.source.TRAINING_GIT,
            'completed_steps': [500, 750, 1000], 'expected_images': 180, 'sampling': append.prompt_forms.SAMPLING})
        (self.experiment / 'repro_outputs').mkdir()
        (self.experiment / 'repro_outputs/SUMMARY.md').write_text('original summary')
        for step in (500, 750, 1000):
            directory = self.experiment / f'step{step}'
            directory.mkdir()
            planned, records = append.prompt_forms.prompt_rows(step), []
            for row in planned:
                path = directory / row['image_path']
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(self.png)
                self.original_images.append(path)
                records.append({**row, 'status': 'ok', 'image_sha256': sha256(path)})
            write_json(directory / 'provenance.json', {'status': 'succeeded', 'git': append.ORIGINAL_GIT,
                'sampling': append.prompt_forms.SAMPLING, 'step': step, 'training_images': 72, 'generated_images': 60})
            for name, values in (('generation_manifest.jsonl', planned), ('generation_status.jsonl', records)):
                (directory / name).write_text(''.join(json.dumps(row) + '\n' for row in values))
            for name in ('training_references.json', 'index.html', 'training_contact_sheet.jpg',
                         'fullrank_contact_sheet.jpg', 'lora_r64_contact_sheet.jpg'):
                (directory / name).write_bytes(b'original artifact')
            for method in ('fullrank', 'lora_r64'):
                write_json(directory / 'checkpoints' / method / 'receipt.json', {'hashes': {}})
        self.experiment_patch = patch.object(append, 'EXPERIMENT', self.experiment)
        self.experiment_patch.start()
        self.addCleanup(self.experiment_patch.stop)

    def cleanup(self):
        if not self.root.resolve().is_relative_to(self.temporary_root):
            raise ValueError('cleanup must stay below the test temporary root')
        shutil.rmtree(self.root)

    def test_targets_are_450_unique_siblings_of_old_images_without_collisions(self):
        expected = {self.experiment / f'step{step}/images/{method}/{material}__{form}__seed{seed}.png'
            for step, method, material, form, seed in itertools.product((500, 750, 1000),
                ('fullrank', 'lora_r64'), ('wood', 'metal', 'matte', 'plastic', 'ceramic'),
                ('adjective', 'entirely', 'surface'), (44, 45, 46, 47, 48))}
        actual = append.check_new_targets()
        self.assertEqual(len(actual), 450)
        self.assertEqual(set(actual), expected)
        self.assertTrue(set(actual).isdisjoint(self.original_images))
        self.assertTrue(all(path.parent.is_dir() for path in actual))

    def test_occupied_seed44_is_rejected_without_changing_it(self):
        path = append.targets()[0]
        path.write_bytes(b'existing seed44 sentinel')
        before = sha256(path)
        with self.assertRaises(FileExistsError):
            append.check_new_targets()
        self.assertEqual(sha256(path), before)

    def test_original_locks_cover_all180_and_reject_changes_or_incomplete_results(self):
        extension = self.experiment / 'extensions/older-extension/sentinel.txt'
        extension.parent.mkdir(parents=True)
        extension.write_text('extension is outside the original lock set')
        locks = append.original_locks()
        png_locks = {Path(path): digest for path, digest in locks.items() if path.endswith('.png')}
        self.assertEqual(set(png_locks), set(self.original_images))
        self.assertEqual(len(png_locks), 180)
        self.assertTrue(all(digest == sha256(path) for path, digest in png_locks.items()))
        self.assertNotIn(str(extension), locks)
        path = self.original_images[0]
        path.write_bytes(b'modified old image')
        with self.assertRaises(ValueError):
            append.source.check_locks(locks)
        with self.assertRaisesRegex(ValueError, 'recorded hash'):
            append.original_locks()
        path.write_bytes(self.png)
        status_path = self.experiment / 'status.json'
        state = json.loads(status_path.read_text())
        changed = {**state, 'status': 'generating'}
        write_json(status_path, changed)
        with self.assertRaisesRegex(ValueError, 'complete original'):
            append.original_locks()
        write_json(status_path, state)
        provenance_path = self.experiment / 'step750/provenance.json'
        provenance = json.loads(provenance_path.read_text())
        write_json(provenance_path, {**provenance, 'status': 'dry_run'})
        with self.assertRaisesRegex(ValueError, 'provenance'):
            append.original_locks()

    def run_mocked(self, output, execute=False):
        selections = [{'method': method} for method in ('fullrank', 'lora_r64')]
        training_sentinel = self.training / 'checkpoint-sentinel.bin'
        training_sentinel.write_bytes(b'unchanged checkpoint')
        training_locks = {str(training_sentinel): sha256(training_sentinel)}

        def render(dataset, destination, step, index, execute_inference, device, **options):
            self.assertEqual(dataset, self.training / 'training_dataset')
            self.assertEqual(index, self.training / 'training_index.json')
            self.assertEqual(options['image_root'], self.experiment / f'step{step}')
            self.assertEqual((execute_inference, device), (execute, 'cuda:0'))
            self.assertEqual(options['reference_count'], 72)
            self.assertEqual(options['sampling_seeds'], (44, 45, 46, 47, 48))
            self.assertEqual(options['selection_loader'](index, dataset), selections)
            with self.assertRaisesRegex(ValueError, 'selection source'):
                options['selection_loader'](index, self.root / 'wrong_dataset')
            with self.assertRaisesRegex(ValueError, 'selection source'):
                options['selection_loader'](self.root / 'wrong_index', dataset)
            rows = options['probe_rows'](step)
            self.assertEqual((len(rows), {row['seed'] for row in rows}), (150, {44, 45, 46, 47, 48}))
            result = {'status': 'succeeded' if execute else 'dry_run', 'expected_images': 150,
                'sampling': {**append.prompt_forms.SAMPLING, 'seeds': [44, 45, 46, 47, 48]}, 'training_images': 72}
            write_json(destination / 'provenance.json', result)
            return result

        with patch.dict(os.environ, {'COLORPEEL_OUTPUT_ROOT': str(self.root)}), \
             patch.object(append, 'git_info', return_value=self.revision), \
             patch.object(append.source, 'SOURCE', self.training), \
             patch.object(append.source, 'source_revision', return_value=append.source.TRAINING_GIT), \
             patch.object(append.source, 'source_audit', return_value=(selections, training_locks)) as audit, \
             patch.object(append.renderer, 'run', side_effect=render) as renderer:
            result = append.run(output, 2, execute)
        return result, audit, renderer

    def test_dry_run_preserves_original_locks_and_records_actual_seeds_and_sources(self):
        before = append.original_locks()
        sampling = copy.deepcopy(append.prompt_forms.SAMPLING)
        output = self.experiment / 'extensions/new_seed_append'
        result, audit, renderer = self.run_mocked(output)
        self.assertEqual(result['status'], 'dry_run_verified')
        self.assertEqual(result['completed_steps'], [500, 750, 1000])
        self.assertEqual(result['seeds'], [44, 45, 46, 47, 48])
        self.assertEqual(result['sampling']['seeds'], [44, 45, 46, 47, 48])
        self.assertEqual((result['expected_added_images'], result['original_images'], result['expected_total_images']), (450, 180, 630))
        self.assertFalse(result['retraining'])
        self.assertTrue(result['original_artifacts_unchanged'])
        audit.assert_called_once_with(output)
        self.assertEqual(renderer.call_count, 3)
        self.assertEqual(append.original_locks(), before)
        self.assertEqual(append.prompt_forms.SAMPLING, sampling)
        self.assertEqual(json.loads((output / 'original_result_locks.json').read_text()), before)
        self.assertEqual(len(json.loads((output / 'new_image_plan.json').read_text())['images']), 450)
        self.assertTrue(all(not path.exists() for path in append.targets()))
        self.assertFalse((self.experiment / 'extensions/seed44_48.claim').exists())
        for bad_output in (output, self.root / 'outside', self.experiment / 'extensions/nested/new'):
            with self.subTest(output=bad_output), self.assertRaisesRegex(ValueError, 'fresh extension'), \
                 patch.dict(os.environ, {'COLORPEEL_OUTPUT_ROOT': str(self.root)}):
                append.run(bad_output, 0)

    def test_preexisting_execute_claim_blocks_renderer_and_preserves_originals(self):
        before = append.original_locks()
        claim = self.experiment / 'extensions/seed44_48.claim'
        claim.mkdir(parents=True)
        (claim / 'owner.json').write_text('existing owner sentinel')
        output = self.experiment / 'extensions/duplicate_execute'
        with patch.dict(os.environ, {'COLORPEEL_OUTPUT_ROOT': str(self.root)}), \
             patch.object(append, 'git_info', return_value=self.revision), \
             patch.object(append.source, 'source_audit', return_value=([], {})), \
             patch.object(append.renderer, 'run') as renderer, self.assertRaises(FileExistsError):
            append.run(output, 0, execute=True)
        renderer.assert_not_called()
        self.assertEqual((claim / 'owner.json').read_text(), 'existing owner sentinel')
        self.assertEqual(json.loads((output / 'status.json').read_text())['status'], 'failed')
        self.assertEqual(append.original_locks(), before)
        self.assertTrue(all(not path.exists() for path in append.targets()))


if __name__ == '__main__':
    unittest.main()
