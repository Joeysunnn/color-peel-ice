"""Verify immutable interrupted-prefix reuse and generation of only missing rows."""

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.mailbox_color_sd15_v1 import inference, parallel_job, prompts
from tests import test_mailbox_color_inference as helpers


class MailboxColorParallelJobTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix='mailbox-parallel-', dir=os.environ.get('TMP'))
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        environment = patch.dict(os.environ, {'COLORPEEL_OUTPUT_ROOT': str(self.root)})
        environment.start()
        self.addCleanup(environment.stop)
        self.method, self.step = 'fullrank', 400

    def fixture(self, count=3, fragment=False):
        self.source = self.root / 'source'
        dataset = self.source / 'training_dataset'
        dataset.mkdir(parents=True)
        image = dataset / 'actual.png'
        Image.new('RGB', (512, 512), 'orange').save(image)
        refs = [{'id': str(i), 'image': str(image), 'image_sha256': inference.sha256(image),
                 'caption': 'a photo of sphere shape in <C*> color'} for i in range(9)]
        manifest = dataset / 'training_assets_manifest.jsonl'
        manifest.write_text(''.join(json.dumps(row) + '\n' for row in refs))
        checkpoint = self.source / 'training' / self.method / 'run' / 'checkpoints'
        snapshot = checkpoint / 'checkpoint-400'
        snapshot.mkdir(parents=True)
        for filename in ('final.bin', 'pytorch_model.bin', 'pytorch_model_1.bin'):
            (checkpoint / filename if filename == 'final.bin' else snapshot / filename).write_text(filename)
        self.color = {'method': self.method, 'token': '<C*>', 'branch': 'color', 'status': 'verified', 'steps': 1000,
            'run': str(checkpoint.parent), 'checkpoint': str(checkpoint), 'weight': 'final.bin',
            'hashes': {str(checkpoint / 'final.bin'): inference.sha256(checkpoint / 'final.bin')}}
        subject_weight = self.source / 'subject_reference' / 'weight.bin'
        subject_weight.parent.mkdir()
        subject_weight.write_text('fixed Subject1000')
        self.subject = {'method': self.method, 'token': '<S*>', 'branch': 'subject', 'status': 'verified',
            'steps': 1000, 'training_steps': 1000, 'paired_embedding_finite': True,
            'checkpoint': str(subject_weight.parent), 'weight': 'weight.bin',
            'hashes': {str(subject_weight): inference.sha256(subject_weight)}}
        self.audit = {'source': str(self.source), 'source_code': str(self.root / 'old_checkout'),
            'training_git': {'branch': 'old', 'commit': 'old_commit'},
            'inference_git': {'branch': 'new', 'commit': 'new_commit'},
            'color_runs': [self.color], 'subject_runs': [self.subject],
            'source_locks': {str(manifest): inference.sha256(manifest)}}
        self.old = self.source / 'step400' / self.method
        self.old.mkdir(parents=True)
        self.exported = self.export(self.color, 400, self.old / 'checkpoints' / 'color')
        rows = prompts.prompt_rows(self.method, self.step)
        (self.old / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.ledger_rows = []
        for row in rows[:count]:
            target = self.old / row['image_path']
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(image, target)
            self.ledger_rows.append({**row, 'status': 'ok', 'image_sha256': inference.sha256(target)})
        self.ledger = self.old / 'generation_status.jsonl'
        self.ledger.write_bytes((''.join(json.dumps(row) + '\n' for row in self.ledger_rows)).encode())
        if fragment:
            with self.ledger.open('ab') as handle:
                handle.write(b'{"id":"unfinished')
            orphan = self.old / rows[count]['image_path']
            orphan.parent.mkdir(exist_ok=True)
            orphan.write_bytes(b'partial PNG, never reusable')
        prompt_source = prompts.source_record()
        prompt_source['path'] = str(Path(self.audit['source_code']) / prompts.SOURCE.relative_to(prompts.ROOT))
        self.state = {'status': 'succeeded' if count == 403 else 'failed', 'error': 'KeyboardInterrupt: ',
            'git': self.audit['training_git'], 'method': self.method, 'step': 400, 'subject_step': 1000,
            'dataset': str(dataset), 'training_images': 9, 'expected_images': 403, 'generated_images': count + int(fragment),
            'sampling': prompts.SAMPLING, 'subject_training': self.subject, 'prompt_source': prompt_source,
            'color_training': self.exported, 'source_locks': self.audit['source_locks']}
        inference.write(self.old / 'provenance.json', self.state)

    def export(self, color, step, output):
        output.mkdir(parents=True)
        weight = output / 'color.bin'
        weight.write_text('same paired Color')
        snapshot = parallel_job._snapshot(color, step)
        return {**color, 'checkpoint': str(output), 'steps': step, 'training_steps': 1000,
            'paired_embedding_finite': True, 'source_checkpoint': color['checkpoint'], 'source_hashes': color['hashes'],
            'source_snapshot_hashes': snapshot, 'hashes': {str(weight): inference.sha256(weight)},
            'source_adapter_sha256': snapshot[str(Path(color['checkpoint']) / f'checkpoint-{step}/pytorch_model.bin')],
            'source_text_encoder_sha256': snapshot[str(Path(color['checkpoint']) / f'checkpoint-{step}/pytorch_model_1.bin')]}

    def inspect(self):
        return parallel_job.inspect_source(self.source, self.audit, self.method, self.step)

    def run_job(self, output, execute=False):
        with patch.object(parallel_job, 'git_info', return_value=self.audit['inference_git']):
            return parallel_job.run(self.source, self.audit, output, self.method, self.step, 'cpu', execute)

    def source_hashes(self):
        return {str(path): inference.sha256(path) for path in self.source.rglob('*') if path.is_file()}

    def test_dryrun_copies_valid_prefix_and_records_unfinished_tail_without_source_changes(self):
        self.fixture(3, fragment=True)
        before = self.source_hashes()
        output = self.root / 'dryrun'
        state = self.run_job(output)
        self.assertEqual(state['reused_images'], 3)
        self.assertEqual(state['newly_generated_images'], 0)
        self.assertEqual(len(list((output / 'images').glob('*.png'))), 3)
        reuse = json.loads((output / 'reuse_manifest.json').read_text())
        self.assertGreater(reuse['ignored_tail']['ledger_fragment_bytes'], 0)
        self.assertEqual(len(reuse['ignored_tail']['unrecorded_images']), 1)
        self.assertEqual(self.source_hashes(), before)

    def test_reject_wrong_hash_prompt_nonprefix_and_noninterrupt_failure(self):
        self.fixture()
        initial = self.ledger.read_text()
        mutations = []
        wrong_hash = [dict(row) for row in self.ledger_rows]
        wrong_hash[0]['image_sha256'] = '0' * 64
        mutations.append(wrong_hash)
        wrong_prompt = [dict(row) for row in self.ledger_rows]
        wrong_prompt[0]['prompt'] += ' changed'
        mutations.append(wrong_prompt)
        mutations.extend((self.ledger_rows[1:], [self.ledger_rows[0], self.ledger_rows[0]], list(reversed(self.ledger_rows))))
        for rows in mutations:
            with self.subTest(rows=rows):
                self.ledger.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                with self.assertRaises(ValueError):
                    self.inspect()
        self.ledger.write_text(initial)
        self.state['error'] = 'RuntimeError: real inference failure'
        inference.write(self.old / 'provenance.json', self.state)
        with self.assertRaisesRegex(ValueError, 'explicitly interrupted'):
            self.inspect()

    def test_wrong_sampling_snapshot_or_manifest_cannot_be_reused(self):
        self.fixture()
        self.state['sampling'] = {**prompts.SAMPLING, 'guidance_scale': 3.5}
        inference.write(self.old / 'provenance.json', self.state)
        with self.assertRaises(ValueError):
            self.inspect()
        self.state['sampling'] = prompts.SAMPLING
        self.state['color_training']['source_adapter_sha256'] = '0' * 64
        inference.write(self.old / 'provenance.json', self.state)
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            self.inspect()
        manifest = self.old / 'generation_manifest.jsonl'
        manifest.write_text(manifest.read_text().replace('bowl on the table', 'bowl in a room'))
        with self.assertRaisesRegex(ValueError, 'manifest'):
            self.inspect()

    def test_missing_unit_plans_all403_and_existing_incomplete_unit_is_rejected(self):
        self.fixture()
        inspected = parallel_job.inspect_source(self.source, self.audit, self.method, 400)
        self.assertEqual(len(inspected['rows']), 3)
        (self.old / 'provenance.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.inspect()
        target = self.old.rename(self.root / 'preserved_old_unit')
        self.assertTrue(target.exists())
        inspected = self.inspect()
        self.assertEqual(inspected['rows'], [])

    def test_execute_generates_only_last_three_joint_rows_with_both_real_adapter_dispatch(self):
        base = self.root
        for method, installer in (('fullrank', 'dual_fullrank'), ('lora_r64', 'dual_lora')):
            with self.subTest(method=method):
                self.root, self.method = base / method, method
                self.root.mkdir()
                self.fixture(400)
                before = self.source_hashes()
                modules = helpers.MailboxColorInferenceTest.modules(self)
                with patch.dict(sys.modules, modules), \
                     patch.object(parallel_job.training, 'export_step', side_effect=self.export), \
                     patch.object(parallel_job.evaluate, 'read_json', side_effect=lambda path: helpers.MailboxColorInferenceTest.model_json(self, path)):
                    state = self.run_job(self.root / 'executed', True)
                self.assertEqual(state['status'], 'succeeded')
                self.assertEqual(state['reused_images'], 400)
                self.assertEqual(state['newly_generated_images'], 3)
                self.assertEqual(len(self.calls), 3)
                self.assertTrue(all(tokens == ('<S*>.bin', '<C*>.bin') for tokens, _, _ in self.calls))
                self.assertEqual(sum(row[0] == installer for row in self.installs), 1)
                self.assertEqual(self.source_hashes(), before)

    def test_complete_unit_copies403_without_loading_any_pipeline(self):
        self.fixture(403)
        with patch.object(parallel_job.training, 'export_step', side_effect=self.export), \
             patch.object(parallel_job.evaluate, 'load_pipeline') as load:
            state = self.run_job(self.root / 'copied_complete', True)
        load.assert_not_called()
        self.assertEqual(state['reused_images'], 403)
        self.assertEqual(state['newly_generated_images'], 0)


if __name__ == '__main__':
    unittest.main()
