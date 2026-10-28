"""Exercise600-step staging metadata and the full coordinator acceptance path."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.subject_trellis_dataset_v1 import train_pipeline as shared
from experiments.subject_trellis_spoons600_v1 import stage_training as staging, train_pipeline as runner


class Spoon600PipelineTest(unittest.TestCase):
    def test_staging_preserves_bytes_and_keeps_old1000_metadata(self):
        from tests.test_trellis_spoon_stage import TrellisSpoonStageTest
        fixture = TrellisSpoonStageTest()
        fixture.setUp()
        try:
            spec = {**fixture.spec, 'request': staging.REQUEST, 'max_train_steps': 600,
                    'checkpoint_steps': [200, 400, 600]}
            before = fixture.source_hashes()
            with patch.object(staging, 'SOURCES', {'metal_spoon': spec}):
                audit = staging.stage(fixture.candidates, fixture.root / 'new600')
                self.assertEqual(staging.verify_dataset(fixture.root / 'new600'), audit)
            self.assertEqual(audit['sampling']['updates'], 600)
            self.assertEqual((audit['sampling']['complete_cycles'], audit['sampling']['extra_prefix_images']), (8, 24))
            self.assertEqual(audit['rows'], 72)
            approval = staging.shared.read(fixture.root / 'new600/approval.json')
            self.assertEqual(approval['checkpoint_steps'], [200, 400, 600])
            self.assertEqual(approval['request'], staging.REQUEST)
            old = staging.previous.stage(fixture.candidates, fixture.root / 'old1000', True)
            self.assertEqual(old['sampling']['updates'], 1000)
            self.assertEqual(staging.shared.read(fixture.root / 'old1000/approval.json')['checkpoint_steps'], [500, 750, 1000])
            self.assertEqual(fixture.source_hashes(), before)
        finally:
            fixture.doCleanups()

    def test600_queue_accepts_both_audits_and_all_three_inference_steps(self):
        with tempfile.TemporaryDirectory(prefix='spoon600-pipeline-', dir=os.environ.get('TMP')) as temporary:
            root = Path(temporary).resolve()
            output = root / 'experiment'
            git = {'commit': 'cpu', 'branch': 'test'}

            def stage(candidates, dataset, subset):
                self.assertTrue(subset)
                dataset.mkdir(parents=True)
                refs = [{'color': 'green', 'view': 'center', 'material': 'M0_R50', 'light': 'studio'}] * 72
                (dataset / 'training_assets_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in refs))
                audit = {'rows': 72, 'subject_id': 'metal_spoon', 'source_mode': 'synthetic_native_geometry',
                         'selected_colors': ['green', 'blue']}
                runner.training._write(dataset / 'dataset_audit.json', audit)
                return audit

            def launch(command, **kwargs):
                folder = Path(command[command.index('--output') + 1])
                folder.mkdir(parents=True)
                if '--method' in command:
                    method = command[command.index('--method') + 1]
                    runner.training._write(folder / 'status.json', {'status': 'training_verified', 'completed_steps': 600})
                    runner.training._write(folder / 'training_index.json', {'git': git, 'runs': [
                        {'status': 'verified', 'method': method, 'branch': 'subject', 'token': '<S*>', 'steps': 600}]})
                else:
                    step = int(command[command.index('--step') + 1])
                    self.assertIn(step, (200, 400, 600))
                    runner.training._write(folder / 'provenance.json', {'status': 'succeeded', 'generated_images': 36,
                        'step': step, 'sampling': runner.inference.SAMPLING, 'training_images': 72,
                        'contact_sheets': ['training', 'original', 'rank64']})
                return SimpleNamespace(returncode=0)

            with ExitStack() as stack:
                stack.enter_context(patch.object(shared, 'git_info', return_value=git))
                stack.enter_context(patch.object(shared, 'resolve_run_dir', side_effect=lambda path: Path(path).resolve()))
                stack.enter_context(patch.object(runner.stage_training, 'stage', side_effect=stage))
                stack.enter_context(patch.object(runner.training, 'build_configuration_bundle', side_effect=lambda data, plan: plan.mkdir()))
                stack.enter_context(patch.object(runner.training, 'verify_dataset', return_value={}))
                process = stack.enter_context(patch.object(shared.subprocess, 'run', side_effect=launch))
                result = runner.run('metal_spoon', output, 2, True)
            self.assertEqual(result['status'], 'execution_verified')
            self.assertEqual(result['completed_steps'], {'fullrank': 600, 'lora_r64': 600})
            self.assertEqual(result['inference_steps'], [200, 400, 600])
            self.assertEqual(process.call_count, 5)
            exposures = json.loads((output / 'train_outputs/EXPOSURES.json').read_text())
            self.assertEqual((exposures['updates_per_method'], exposures['complete_cycles'], exposures['prefix_images']), (600, 8, 24))
            self.assertIn('(600 per method)', (output / 'train_outputs/SUMMARY.md').read_text())


if __name__ == '__main__':
    unittest.main()
