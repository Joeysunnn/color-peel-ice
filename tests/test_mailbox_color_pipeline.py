"""Check preflight integration and finite training-to-inference queue boundaries."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.mailbox_color_sd15_v1 import pipeline
from tests import test_mailbox_color_data as fixtures


class MailboxColorPipelineTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.MailboxColorDataTest(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.outputs.mkdir()
        self.output = self.fixture.outputs / 'experiment'
        self.git = {'branch': 'codex/color_cpu_test', 'commit': 'f' * 40}

    def test_preflight_stages_real_nine_and_all_eight_inference_matrices_without_training(self):
        def subject(output):
            output.mkdir()
            pipeline.write(output / 'subject_index.json', {'status': 'verified'})
        with patch.object(pipeline, 'git_info', return_value=self.git), \
             patch.object(pipeline.inference, 'git_info', return_value=self.git), \
             patch.object(pipeline.inference, 'prepare_subject', side_effect=subject), \
             patch.object(pipeline.training, 'run', return_value={'status': 'dry_run_verified'}) as trainer, \
             patch.object(pipeline.subprocess, 'Popen', side_effect=AssertionError('no workers during dry-run')), \
             patch.object(pipeline.shutil, 'disk_usage', return_value=SimpleNamespace(free=40 * 1024**3)):
            state = pipeline.run(self.output, [0, 2])
        self.assertEqual(state['status'], 'preflight_verified')
        self.assertEqual(state['expected_images'], 3224)
        self.assertEqual(trainer.call_count, 2)
        self.assertTrue(all(len(call.args) == 4 for call in trainer.call_args_list))
        for method in pipeline.training.METHODS:
            config = pipeline.read_config(self.output / 'training_plan' / 'configs' / (method + '.json'))
            self.assertEqual(config['args']['modifier_token'], '<C*>')
            self.assertEqual(config['args']['checkpoint_steps'], [400, 600, 800, 1000])
            for step in pipeline.training.STEPS:
                folder = self.output / 'preflight' / f'step{step}' / method
                provenance = pipeline.read_json(folder / 'provenance.json')
                self.assertEqual(provenance['status'], 'dry_run')
                self.assertEqual(provenance['expected_images'], 403)
                self.assertEqual(len((folder / 'generation_manifest.jsonl').read_text().splitlines()), 403)
                self.assertTrue((folder / 'index.html').is_file())
        self.assertEqual(pipeline.data.verify_dataset(self.output / 'training_dataset')['rows'], 9)

    def run_worker(self, failure=False):
        self.output.mkdir()
        pipeline.write(self.output / 'plan.json', {'git': self.git, 'gpus': [0, 2]})
        calls = []
        def inference(*args, **kwargs):
            calls.append((args, kwargs))
            if failure and args[3] == 600:
                raise RuntimeError('inference failed')
        with patch.object(pipeline, 'git_info', return_value=self.git), \
             patch.object(pipeline.training, 'run', return_value={'completed_steps': 1000, 'status': 'training_verified'}) as trainer, \
             patch.object(pipeline.inference, 'run', side_effect=inference), \
             patch.dict(pipeline.os.environ):
            if failure:
                with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                    pipeline.worker(self.output, 'lora_r64', 2)
                state = pipeline.read_json(self.output / 'lora_r64.status.json')
            else:
                state = pipeline.worker(self.output, 'lora_r64', 2)
        self.assertEqual(trainer.call_count, 1)
        self.assertEqual(trainer.call_args.args[-1], True)
        for args, kwargs in calls:
            self.assertEqual(args[2], 'lora_r64')
            self.assertEqual(kwargs['device'], 'cuda:0')
            self.assertTrue(kwargs['execute_inference'])
            self.assertEqual(kwargs['subject_index'], self.output / 'subject_reference' / 'subject_index.json')
            self.assertEqual(kwargs['training_index'], self.output / 'training' / 'lora_r64' / 'training_index.json')
        return state, [args[3] for args, _ in calls]

    def test_worker_runs_one_training_then_all_four_paired_tests(self):
        state, steps = self.run_worker()
        self.assertEqual(steps, [400, 600, 800, 1000])
        self.assertEqual(state['status'], 'execution_verified')
        self.assertEqual(state['completed_inference_steps'], steps)
        self.assertFalse(state['monitoring'])

    def test_worker_stops_failed_inference_without_retry_or_later_tests(self):
        state, steps = self.run_worker(True)
        self.assertEqual(steps, [400, 600])
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['completed_inference_steps'], [400])


if __name__ == '__main__':
    unittest.main()
