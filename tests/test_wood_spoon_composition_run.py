"""Verify four inference-only jobs, bounded GPU ownership and failure reporting."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from experiments.wood_spoon_composition_v1 import inference, run as runner


class WoodSpoonCompositionRunTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='wood-composition-run-', dir=os.environ.get('TMP'))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / 'experiment'
        self.git = {'commit': 'cpu-test', 'branch': 'test'}
        self.preflights, self.calls, self.events = [], [], []
        self.subject_steps = (600,)
        self.source_subjects = []
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {'TMP': str(self.root),
            'TEMP': str(self.root), 'COLORPEEL_OUTPUT_ROOT': str(self.root)}))
        self.stack.enter_context(patch.object(runner, 'git_info', return_value=self.git))
        self.stack.enter_context(patch.object(runner.source, 'prepare', side_effect=self.prepare))
        self.stack.enter_context(patch.object(runner.inference, 'run', side_effect=self.preflight))

    def prepare(self, folder, *, texture_step=1000, include_material=True, subject_step=600, include_subject=True):
        self.include_subject = include_subject
        self.source_options = (texture_step, include_material)
        self.source_subjects.append(subject_step)
        folder.mkdir(parents=True)
        runner.write(folder / 'audit.json', {'inference_git': self.git})
        self.events.append('source_audit')

    def preflight(self, audit_path, output, method, branch, device='cuda:0', execute=False, attribute_step=1000, subject_step=600, literal_subject=False):
        self.assertEqual(literal_subject, not self.include_subject)
        self.assertEqual(attribute_step, self.source_options[0] if branch == 'texture' else 1000)
        self.assertTrue(Path(audit_path).is_file())
        self.assertFalse(execute)
        self.assertEqual(device, 'cuda:0')
        self.assertIn(subject_step, self.subject_steps)
        folder = branch if len(self.subject_steps) == 1 else f'step{subject_step}_{branch}'
        self.assertEqual(output, self.output / 'preflight' / folder / method)
        output.mkdir(parents=True)
        state = {'status': 'dry_run', 'expected_images': 31, 'generated_images': 0}
        runner.write(output / 'provenance.json', state)
        self.preflights.append((branch, method))
        self.events.append('preflight')
        return state

    def process_factory(self, fail=False, parallel=False):
        owner = self
        lock = threading.Lock()
        expected_jobs = (4 if self.source_options[1] else 2) * len(self.subject_steps)
        barrier = threading.Barrier(min(3, expected_jobs)) if parallel else None
        active = set()

        class Process:
            def __init__(self, command, **kwargs):
                self.command = command
                self.output = Path(command[command.index('--output') + 1])
                self.method = command[command.index('--method') + 1]
                self.branch = command[command.index('--branch') + 1]
                self.subject_step = int(command[command.index('--subject-step') + 1])
                self.gpu = kwargs['env']['CUDA_VISIBLE_DEVICES']
                owner.assertEqual(command[command.index('-m') + 1],
                    'experiments.wood_spoon_composition_v1.inference')
                owner.assertIn('--execute-inference', command)
                owner.assertNotIn('--execute-training', command)
                owner.assertEqual(command[command.index('--device') + 1], 'cuda:0')
                folder = self.branch if len(owner.subject_steps) == 1 else f'step{self.subject_step}_{self.branch}'
                owner.assertEqual(self.output, owner.output / folder / self.method)
                audit = owner.output / 'source_audit'
                if len(owner.subject_steps) > 1:
                    audit /= f'step{self.subject_step}'
                owner.assertEqual(Path(command[command.index('--audit') + 1]), audit / 'audit.json')
                owner.assertEqual(kwargs['env']['PYTHONDONTWRITEBYTECODE'], '1')
                state = json.loads((owner.output / 'status.json').read_text())
                owner.assertEqual(state['status'], 'inference')
                owner.assertEqual(len(owner.preflights), expected_jobs)
                with lock:
                    owner.assertNotIn(self.gpu, active)
                    active.add(self.gpu)
                    self.index = len(owner.calls)
                    self.pid = 10000 + self.index
                    owner.calls.append((self.branch, self.method, self.gpu))
                    owner.events.append('spawn')

            def wait(self):
                if barrier is not None and self.index < 3:
                    barrier.wait(timeout=10)
                if not fail:
                    self.output.mkdir(parents=True)
                    step = int(self.command[self.command.index('--attribute-step') + 1])
                    rows = [{**row, 'status': 'ok', 'image_sha256': 'fixture'}
                            for row in inference.prompt_rows(self.method, self.branch, step, subject_step=self.subject_step,
                                                             literal_subject='--literal-subject' in self.command)]
                    (self.output / 'generation_status.jsonl').write_text(
                        ''.join(json.dumps(row) + '\n' for row in rows))
                    runner.write(self.output / 'provenance.json',
                        {'status': 'succeeded', 'generated_images': 31})
                with lock:
                    active.remove(self.gpu)
                return 1 if fail else 0

        return Process

    def test_dry_run_audits_then_preflights_four_without_starting_processes(self):
        with patch.object(runner.subprocess, 'Popen') as process:
            result = runner.run(self.output, [0, 2, 3])
        self.assertEqual(result['status'], 'dry_run_verified')
        self.assertEqual(set(self.preflights), {(b, m) for b in runner.BRANCHES for m in runner.METHODS})
        self.assertEqual(self.events, ['source_audit'] + ['preflight'] * 4)
        process.assert_not_called()
        commands = json.loads((self.output / 'repro_outputs/COMMANDS.json').read_text())
        self.assertEqual(len(commands), 4)
        self.assertFalse(result['training'])
        self.assertFalse(result['monitoring'])

    def test_three_gpu_queues_execute_each_condition_once_and_align_all_seeds(self):
        self.source_options = (1000, True)
        with patch.object(runner.subprocess, 'Popen', side_effect=self.process_factory(parallel=True)):
            result = runner.run(self.output, [0, 2, 3], execute=True)
        expected = {(b, m) for b in runner.BRANCHES for m in runner.METHODS}
        self.assertEqual(len(self.calls), 4)
        self.assertEqual({(b, m) for b, m, gpu in self.calls}, expected)
        self.assertEqual({gpu for b, m, gpu in self.calls}, {'0', '2', '3'})
        self.assertEqual(result['status'], 'execution_verified')
        self.assertEqual(result['generated_images'], 124)
        self.assertEqual({(r['branch'], r['method']) for r in result['completed_jobs']}, expected)
        page = (self.output / 'index.html').read_text(encoding='utf8')
        self.assertEqual(page.count('<tr><th>seed'), 31)
        for seed in range(30, 61):
            fragment = page.split(f'<tr><th>seed{seed}</th>', 1)[1].split('</tr>', 1)[0]
            paths = [f'{b}/{m}/images/seed{seed}.png' for b in runner.BRANCHES for m in runner.METHODS]
            self.assertEqual([fragment.index(path) for path in paths], sorted(fragment.index(path) for path in paths))
        for p in self.output.glob('queue_gpu*.json'):
            self.assertTrue(all(row['status'] == 'succeeded' for row in json.loads(p.read_text())))

    def test_failed_child_without_provenance_is_recorded_and_not_retried(self):
        self.source_options = (1000, True)
        with patch.object(runner.subprocess, 'Popen', side_effect=self.process_factory(fail=True)):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                runner.run(self.output, [0], execute=True)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(json.loads((self.output / 'status.json').read_text())['status'], 'failed')
        queue = json.loads((self.output / 'queue_gpu0.json').read_text())
        self.assertEqual(queue[0]['status'], 'failed')
        self.assertEqual(queue[0]['returncode'], 1)

    def test_texture500_only_dispatches_two_jobs_and_keeps_methods_together(self):
        self.source_options = (500, False)
        with patch.object(runner.subprocess, 'Popen', side_effect=self.process_factory(parallel=True)):
            result = runner.run(self.output, [2, 3], execute=True, texture_step=500, texture_only=True)
        self.assertEqual(self.source_options, (500, False))
        self.assertEqual({(b, m) for b, m, gpu in self.calls}, {('texture', m) for m in runner.METHODS})
        self.assertEqual(result['generated_images'], 62)
        self.assertEqual(result['attribute_steps'], {'texture': 500})
        self.assertFalse((self.output / 'material').exists())
        page = (self.output / 'index.html').read_text(encoding='utf8')
        self.assertIn('S600 + T500', page)
        self.assertNotIn('S+M', page)
        self.assertEqual(page.count('<tr><th>seed'), 31)
        self.assertEqual(page.count('<td><img'), 62)

    def test_subject200_400_with_fixed_texture500_audit_all_then_dispatch_four(self):
        self.source_options = (500, False)
        self.subject_steps = (200, 400)
        with patch.object(runner.subprocess, 'Popen', side_effect=self.process_factory(parallel=True)):
            result = runner.run(self.output, [0, 2, 3], execute=True, texture_step=500,
                                texture_only=True, subject_steps=self.subject_steps)
        self.assertEqual(self.source_subjects, [200, 400])
        self.assertEqual(self.events[:6], ['source_audit'] * 2 + ['preflight'] * 4)
        self.assertEqual(result['generated_images'], 124)
        self.assertEqual({(r['subject_step'], r['method']) for r in result['completed_jobs']},
                         {(s, m) for s in (200, 400) for m in runner.METHODS})
        commands = json.loads((self.output / 'repro_outputs/COMMANDS.json').read_text())
        self.assertEqual(len({row['folder'] for row in commands}), 4)
        self.assertTrue(all(row['branch'] == 'texture' and
                            row['command'][row['command'].index('--attribute-step') + 1] == '500'
                            for row in commands))
        page = (self.output / 'index.html').read_text(encoding='utf8')
        self.assertIn('S200/S400 + T500', page)
        self.assertNotIn('S600', page)
        self.assertEqual(page.count('<td><img'), 124)
        for step in (200, 400):
            for method in runner.METHODS:
                self.assertIn(f'step{step}_texture/{method}/images/seed30.png', page)


    def test_subject200_400_600_texture750_dispatches_six_and_compares186_images(self):
        self.source_options = (750, False)
        self.subject_steps = (200, 400, 600)
        with patch.object(runner.subprocess, 'Popen', side_effect=self.process_factory(parallel=True)):
            result = runner.run(self.output, [0, 2, 3], execute=True, texture_step=750,
                                texture_only=True, subject_steps=self.subject_steps)
        self.assertEqual(self.source_subjects, [200, 400, 600])
        self.assertEqual(self.events[:9], ['source_audit'] * 3 + ['preflight'] * 6)
        self.assertEqual(result['generated_images'], 186)
        self.assertEqual({(r['subject_step'], r['method']) for r in result['completed_jobs']},
                         {(s, m) for s in self.subject_steps for m in runner.METHODS})
        commands = json.loads((self.output / 'repro_outputs/COMMANDS.json').read_text())
        self.assertEqual(len({row['folder'] for row in commands}), 6)
        self.assertTrue(all(row['branch'] == 'texture' and
                            row['command'][row['command'].index('--attribute-step') + 1] == '750'
                            for row in commands))
        page = (self.output / 'index.html').read_text(encoding='utf8')
        self.assertIn('S200/S400/S600 + T750', page)
        self.assertEqual(page.count('<td><img'), 186)
        for step in self.subject_steps:
            for method in runner.METHODS:
                self.assertIn(f'step{step}_texture/{method}/images/seed30.png', page)

    def test_literal_wood_spoon_only_audits_and_dispatches_two_texture500_jobs(self):
        self.source_options = (500, False)
        with patch.object(runner.subprocess, 'Popen', side_effect=self.process_factory(parallel=True)):
            result = runner.run(self.output, [2, 3], execute=True, texture_step=500, literal_subject=True)
        self.assertFalse(self.include_subject)
        self.assertTrue(result['literal_subject'])
        self.assertEqual(result['subject_steps'], [])
        self.assertEqual(result['generated_images'], 62)
        commands = json.loads((self.output / 'repro_outputs/COMMANDS.json').read_text())
        self.assertEqual(len(commands), 2)
        self.assertTrue(all('--literal-subject' in row['command'] and row['branch'] == 'texture' for row in commands))
        page = (self.output / 'index.html').read_text(encoding='utf8')
        self.assertIn('literal wood spoon', page)
        self.assertNotIn('S600', page)
        self.assertEqual(page.count('<td><img'), 62)
        self.assertIn('Only the Texture500', page)


if __name__ == '__main__':
    unittest.main()
