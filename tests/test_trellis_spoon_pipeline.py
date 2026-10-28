"""CPU checks for isolated spoon queues and dependency-selected shared planning."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.subject_trellis_dataset_v1 import train_pipeline as shared
from experiments.subject_trellis_spoons_v1 import train_pipeline as spoon


class SpoonPipelineTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='spoon-pipeline-', dir=os.environ.get('TMP'))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def test_commands_target_spoon_modules_and_paired_steps(self):
        jobs = spoon.commands(self.root / 'data', self.root / 'run', 2)
        self.assertEqual([job['name'] for job in jobs], ['fullrank', 'lora_r64', 'step500', 'step750', 'step1000'])
        for job in jobs:
            self.assertIn('experiments.subject_trellis_spoons_v1.', job['command'][3])
            self.assertNotIn('subject_trellis_dataset_v1', job['command'][3])
        self.assertEqual(jobs[-1]['environment'], {'CUDA_VISIBLE_DEVICES': '2'})

    def test_subject_queue_groups_both_doses_and_stops_on_failure(self):
        output = self.root / 'metal'
        with patch.object(spoon, 'resolve_run_dir', side_effect=lambda value: Path(value).resolve()), \
             patch.object(spoon, 'git_info', return_value={'commit': 'cpu'}), \
             patch.object(shared, 'run', return_value={'status': 'dry_run_verified'}) as run:
            state = spoon.run('metal_spoon', output, 0)
        self.assertEqual(state['completed_experiments'], ['all180', 'blue_green72'])
        self.assertEqual([call.args[1] for call in run.call_args_list], [output / 'all180', output / 'blue_green72'])
        self.assertEqual([call.args[4] for call in run.call_args_list], [False, True])
        self.assertTrue(all(call.kwargs['subject_noun'] == 'spoon' for call in run.call_args_list))
        failed = self.root / 'wood'
        with patch.object(spoon, 'resolve_run_dir', side_effect=lambda value: Path(value).resolve()), \
             patch.object(spoon, 'git_info', return_value={'commit': 'cpu'}), \
             patch.object(shared, 'run', side_effect=RuntimeError('declared failure')) as run, \
             self.assertRaisesRegex(RuntimeError, 'declared failure'):
            spoon.run('wood_spoon', failed, 2, True)
        self.assertEqual(run.call_count, 1)
        self.assertFalse((failed / 'blue_green72').exists())

    def test_shared_planner_uses_injected_subject_modules(self):
        def stage(candidates, dataset, subset):
            dataset.mkdir(parents=True)
            count = 72 if subset else 180
            references = [{'color': 'blue', 'view': 'center', 'light': 'studio', 'material': 'M0_R50'}] * count
            (dataset / 'training_assets_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in references))
            return {'rows': count, 'selected_colors': ['blue', 'green'], 'source_mode': 'synthetic_native_geometry', 'subject_id': 'wood_spoon'}

        module = SimpleNamespace(ROOT=self.root / 'code', METHODS=('fullrank', 'lora_r64'),
            _write=shared.training._write,
            build_configuration_bundle=lambda data, out: out.mkdir())
        infer = SimpleNamespace(STEPS=(500, 750, 1000), expected_images=lambda data: 36)
        with patch.object(shared, 'resolve_run_dir', side_effect=lambda value: Path(value).resolve()), \
             patch.object(shared, 'git_info', return_value={'commit': 'cpu'}):
            state = shared.run(self.root / 'source/dataset', self.root / 'comparison', 2,
                blue_green_only=True, stage_module=SimpleNamespace(stage=stage), training_module=module,
                inference_module=infer, command_builder=lambda *args: [], subject_noun='spoon')
        self.assertEqual((state['status'], state['subject_id'], state['training_images']), ('dry_run_verified', 'wood_spoon', 72))
        self.assertEqual(state['expected_images_total'], 108)
        text = (self.root / 'comparison/train_outputs/SCIENTIFIC_CHANGELOG.md').read_text()
        self.assertIn('spoon initializer', text)
        self.assertNotIn('mailbox initializer', text)


if __name__ == '__main__':
    unittest.main()
