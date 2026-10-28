"""Check that continuation never schedules the completed180 fullrank again."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.subject_trellis_spoons_v1 import continue_training as continuation


class SpoonContinuationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='spoon-continue-', dir=os.environ.get('TMP'))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / 'source'
        self.output = self.source / 'metal_spoon/continuations/new'
        self.git = {'commit': 'new', 'branch': 'test'}
        self.fullrank = {'method': 'fullrank', 'steps': 1000}
        self.receipt = {'source_git': continuation.reuse.OLD_GIT, 'locks': {}}

    def test_remaining180_jobs_skip_fullrank_and_use_reuse_inference(self):
        jobs = continuation.remaining_jobs(self.root / 'data', self.root / 'output', 2)
        self.assertEqual([j['name'] for j in jobs], ['lora_r64', 'step500', 'step750', 'step1000'])
        self.assertEqual(jobs[0]['command'][jobs[0]['command'].index('--method') + 1], 'lora_r64')
        self.assertTrue(all('--reuse-fullrank' in j['command'] for j in jobs[1:]))

    def launch(self, command, **kwargs):
        folder = Path(command[command.index('--output') + 1])
        folder.mkdir(parents=True)
        if '--method' in command:
            self.assertEqual(command[command.index('--method') + 1], 'lora_r64')
            continuation.training._write(folder / 'status.json', {'status': 'training_verified', 'completed_steps': 1000})
            continuation.training._write(folder / 'training_index.json', {'git': self.git, 'runs': [{'method': 'lora_r64'}]})
        else:
            step = int(command[command.index('--step') + 1])
            continuation.training._write(folder / 'provenance.json', {'status': 'succeeded', 'generated_images': 20,
                'training_images': 180, 'step': step, 'sampling': continuation.inference.SAMPLING,
                'contact_sheets': ['all', 'fullrank', 'rank64']})
        return SimpleNamespace(returncode=0)

    def patches(self):
        return [patch.object(continuation, 'resolve_run_dir', side_effect=lambda path: Path(path).resolve()),
            patch.object(continuation.reuse, 'SOURCE_ROOT', self.source),
            patch.object(continuation, 'git_info', return_value=self.git),
            patch.object(continuation.reuse, 'completed_fullrank', return_value=(self.fullrank, self.receipt)),
            patch.object(continuation, '_locks')]

    def test_only_missing_jobs_run_and_original_bytes_stay_untouched(self):
        from contextlib import ExitStack
        original = self.source / 'metal_spoon/all180/training/fullrank/original.bin'
        original.parent.mkdir(parents=True)
        original.write_bytes(b'completed model must never be rewritten')
        with ExitStack() as stack:
            for item in self.patches():
                stack.enter_context(item)
            launch = stack.enter_context(patch.object(continuation.subprocess, 'run', side_effect=self.launch))
            stack.enter_context(patch.object(continuation.reuse, 'selection_loader', return_value=[]))
            subset = stack.enter_context(patch.object(continuation.pipeline, 'run', return_value={'status': 'execution_verified'}))
            result = continuation.run('metal_spoon', self.output, 0, True)
        self.assertEqual(launch.call_count, 4)
        self.assertTrue(subset.call_args.args[4])
        self.assertEqual(subset.call_args.args[1], self.output / 'blue_green72')
        self.assertEqual(result['status'], 'execution_verified')
        self.assertEqual(result['inference_steps'], {'all180': [500, 750, 1000], 'blue_green72': [500, 750, 1000]})
        self.assertEqual(original.read_bytes(), b'completed model must never be rewritten')
        index = json.loads((self.output / 'all180/training_index.json').read_text())
        self.assertEqual(index['source_gits'], {'fullrank': continuation.reuse.OLD_GIT, 'lora_r64': self.git})

    def test_failed_rank64_stops_before_inference_or72(self):
        from contextlib import ExitStack
        with ExitStack() as stack:
            for item in self.patches():
                stack.enter_context(item)
            launch = stack.enter_context(patch.object(continuation.subprocess, 'run', return_value=SimpleNamespace(returncode=1)))
            subset = stack.enter_context(patch.object(continuation.pipeline, 'run'))
            with self.assertRaisesRegex(RuntimeError, 'lora_r64 failed'):
                continuation.run('metal_spoon', self.output, 0, True)
        self.assertEqual(launch.call_count, 1)
        subset.assert_not_called()
        self.assertEqual(json.loads((self.output / 'status.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
