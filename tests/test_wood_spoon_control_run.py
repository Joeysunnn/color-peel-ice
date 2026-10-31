"""Only four missing controls run; reused outputs remain byte copies."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.wood_spoon_composition_v1 import control_run as runner


COPIED = {'mug_without_t', 'mug_with_t', 'wood_spoon_with_t', 'subject_with_t'}


class ControlRunTest(unittest.TestCase):
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
                     patch.object(runner.source, 'prepare', side_effect=self.audit),
                     patch.object(runner.inference, 'references', return_value=({}, {})),
                     patch.object(runner.control_reuse, 'copy_existing', side_effect=self.copy),
                     patch.object(runner.inference, 'run', side_effect=self.preflight)):
            self.stack.enter_context(mock)

    def audit(self, folder, **kwargs):
        self.assertEqual(kwargs, {'texture_step': 500, 'include_material': False})
        folder.mkdir(parents=True)
        runner.write(folder / 'audit.json', {})
        self.events.append(('audit', None))
        return {}

    def save_image_record(self, folder, case):
        folder.mkdir(parents=True)
        row = {**runner.inference.prompt_row(case), 'status': 'ok'}
        (folder / 'generation_status.jsonl').write_text(json.dumps(row) + '\n')
        runner.write(folder / 'provenance.json', {'status': 'succeeded', 'generated_images': 1})

    def copy(self, folder, case, audit):
        if case not in COPIED:
            return None
        self.save_image_record(folder, case)
        self.events.append(('copy', case))
        return {'copied': True}

    def preflight(self, audit, folder, case):
        self.assertNotIn(case, COPIED)
        self.assertTrue(audit.is_file())
        self.events.append(('preflight', case))

    def process(self, fail=False):
        owner = self
        class Process:
            def __init__(self, command, **kwargs):
                self.case = command[command.index('--case') + 1]
                self.folder = Path(command[command.index('--output') + 1])
                self.pid = 1000 + len(owner.events)
                owner.assertNotIn(self.case, COPIED)
                owner.assertEqual(sum(k == 'copy' for k, v in owner.events), 4)
                owner.assertEqual(sum(k == 'preflight' for k, v in owner.events), 4)
                owner.assertIn(kwargs['env']['CUDA_VISIBLE_DEVICES'], ('2', '3'))
                owner.events.append(('spawn', self.case))

            def wait(self):
                if not fail:
                    owner.save_image_record(self.folder, self.case)
                return 1 if fail else 0
        return Process

    def test_exact_eight_case_review_copies_four_and_only_generates_missing_four(self):
        with patch.object(runner.subprocess, 'Popen', side_effect=self.process()):
            state = runner.run(self.output, [2, 3], True)
        self.assertEqual(state['status'], 'execution_verified')
        self.assertEqual((state['copied_images'], state['generated_images'], state['total_images']), (4, 4, 8))
        self.assertEqual({v for k, v in self.events if k == 'spawn'}, set(runner.inference.CASES) - COPIED)
        page = (self.output / 'index.html').read_text(encoding='utf8')
        self.assertEqual(page.count('<img'), 8)
        self.assertEqual(page.count('<tr><th>'), 5)
        self.assertIn('Without T', page)
        self.assertIn('With T500', page)

    def test_preflight_copies_existing_images_without_launching_any_pipeline(self):
        with patch.object(runner.subprocess, 'Popen') as process:
            state = runner.run(self.output, [2])
        self.assertEqual(state['status'], 'dry_run_verified')
        process.assert_not_called()
        self.assertEqual(len(list(self.output.glob('cases/*/generation_status.jsonl'))), 4)

    def test_failed_child_is_recorded_without_retry_or_replacing_copied_cases(self):
        with patch.object(runner.subprocess, 'Popen', side_effect=self.process(fail=True)):
            with self.assertRaisesRegex(RuntimeError, 'no automatic retry'):
                runner.run(self.output, [2], True)
        self.assertEqual(sum(k == 'spawn' for k, v in self.events), 1)
        self.assertEqual(json.loads((self.output / 'status.json').read_text())['status'], 'failed')
        self.assertEqual(len(list(self.output.glob('cases/*/generation_status.jsonl'))), 4)


if __name__ == '__main__':
    unittest.main()
