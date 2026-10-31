"""Only four missing cases run; six historical cases are copied unchanged."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.wood_spoon_composition_v1 import collect_t500 as runner


class CollectTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=os.environ.get('TMP'))
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.output = self.root / 'experiment'
        self.events = []
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for mock in (patch.dict(os.environ, {'COLORPEEL_OUTPUT_ROOT': str(self.root)}),
                     patch.object(runner, 'git_info', return_value={'commit': 'test', 'branch': 'test'}),
                     patch.object(runner.reuse_composition, 'copy_completed', side_effect=self.copy),
                     patch.object(runner.source180, 'prepare', side_effect=self.audit),
                     patch.object(runner.inference, 'run', side_effect=self.preflight)):
            self.stack.enter_context(mock)

    def ledger(self, folder, step, method):
        folder.mkdir(parents=True)
        rows = runner.inference.prompt_rows(method, 'texture', 500, step,
                                            subject_images=72 if step < 700 else 180)
        (folder / 'generation_status.jsonl').write_text(''.join(json.dumps({**r, 'status': 'ok'}) + '\n' for r in rows))
        runner.write(folder / 'provenance.json', {'status': 'succeeded', 'generated_images': 31})

    def copy(self, source, destination, step, method):
        self.assertTrue(str(source).startswith(str(runner.BASE)))
        self.assertIn(step, (200, 400, 600))
        self.ledger(destination, step, method)
        self.events.append('copy')

    def audit(self, folder, step):
        self.assertIn(step, (750, 1000))
        folder.mkdir(parents=True)
        runner.write(folder / 'audit.json', {})
        self.events.append('audit')

    def preflight(self, audit, folder, method, branch, **kwargs):
        self.assertTrue(audit.is_file())
        self.assertEqual(branch, 'texture')
        self.assertEqual(kwargs['subject_images'], 180)
        self.assertEqual(kwargs['attribute_step'], 500)
        self.events.append('preflight')

    def process(self, fail=False):
        owner = self
        class Process:
            def __init__(self, command, **kwargs):
                owner.assertEqual(owner.events[:12], ['copy'] * 6 + ['audit'] * 2 + ['preflight'] * 4)
                self.pid = 1000 + len(owner.events)
                owner.events.append('spawn')
                self.folder = Path(command[command.index('--output') + 1])
                self.step = int(command[command.index('--subject-step') + 1])
                self.method = command[command.index('--method') + 1]
                owner.assertEqual(command[command.index('--subject-images') + 1], '180')
                owner.assertIn(kwargs['env']['CUDA_VISIBLE_DEVICES'], ('0', '2', '3'))

            def wait(self):
                if not fail:
                    owner.ledger(self.folder, self.step, self.method)
                return 1 if fail else 0
        return Process

    def test_reuses186_and_generates124_in_one310_image_review(self):
        with patch.object(runner.subprocess, 'Popen', side_effect=self.process()):
            state = runner.run(self.output, [0, 2, 3], True)
        self.assertEqual(state['status'], 'execution_verified')
        self.assertEqual((state['copied_images'], state['generated_images'], state['total_images']), (186, 124, 310))
        self.assertEqual(self.events.count('spawn'), 4)
        page = (self.output / 'index.html').read_text(encoding='utf8')
        self.assertEqual(page.count('<td><img'), 310)
        self.assertEqual(page.count('<tr><th>seed'), 31)

    def test_failed_new_job_is_recorded_without_retry_or_touching_copied_outputs(self):
        with patch.object(runner.subprocess, 'Popen', side_effect=self.process(fail=True)):
            with self.assertRaisesRegex(RuntimeError, 'no automatic retry'):
                runner.run(self.output, [0], True)
        self.assertEqual(self.events.count('spawn'), 1)
        self.assertEqual(json.loads((self.output / 'status.json').read_text())['status'], 'failed')
        self.assertEqual(len(list(self.output.glob('s72_step*/*/generation_status.jsonl'))), 6)


if __name__ == '__main__':
    unittest.main()
