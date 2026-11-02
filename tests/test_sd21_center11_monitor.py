"""Exercise recovery decisions and file-level completion without GPU execution."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.sd21_mailbox_center11_v1 import monitor as m


def observations(training=False, inference=False):
    return {name: {'training': {'complete': training, 'finite': True},
                   'inference': {'complete': inference}} for name in m.METHODS}


class MonitorTest(unittest.TestCase):
    def test_live_process_prevents_recovery_even_after_two_dead_checks(self):
        self.assertEqual(m.next_action(observations(), [123], 3, []), ('watching', None))
        self.assertEqual(m.next_action(observations(), [], 1, []), ('confirming_exit', None))
        self.assertEqual(m.next_action(observations(), [], 2, []), ('recover', ('fullrank', 'training')))

    def test_successful_stages_reused_and_limits_enforced(self):
        status = observations(training=True)
        self.assertEqual(m.next_action(status, [], 2, []), ('recover', ('fullrank', 'inference')))
        status['fullrank']['inference']['complete'] = True
        status['lora_r64']['training']['complete'] = False
        self.assertEqual(m.next_action(status, [], 2, []), ('recover', ('lora_r64', 'training')))
        attempts = [{'method': 'lora_r64', 'phase': 'training'}] * 2
        self.assertEqual(m.next_action(status, [], 2, attempts)[0], 'needs_attention')
        self.assertEqual(m.next_action(observations(True, True), [], 0, []), ('completed', None))
        status['fullrank']['training']['finite'] = False
        self.assertEqual(m.next_action(status, [123], 0, [])[0], 'needs_attention')

    def test_process_identity_uses_scope_and_real_entrypoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / 'experiment_parent'
            commands = {
                '101': ['python', '-m', 'experiments.sd21_mailbox_center11_v1.run', '--output', str(parent / 'experiment')],
                '102': ['python', '-m', 'experiments.sd21_mailbox_center11_v1.inference', '--output', str(parent / 'recovery/inference')],
                '103': ['python', '/code/src/train/train_colorpeel.py', '--output_dir', str(parent / 'experiment/training/checkpoints')],
                '104': ['python', '-m', 'experiments.sdxl_mailbox_v1.run', '--output', '/another/experiment'],
                '105': ['python', '-m', 'experiments.sd21_mailbox_center11_v1.monitor', '--parent', str(parent)],
                '106': ['python', '-m', 'experiments.sd21_mailbox_center11_v1.run', '--output', str(parent) + '-other/experiment'],
            }
            for pid, args in commands.items():
                (root / pid).mkdir()
                (root / pid / 'cmdline').write_bytes('\0'.join(args).encode() + b'\0')
            self.assertEqual(set(m.active_processes(parent, root)), {101, 102, 103})

    def test_gpu_zero_only_and_no_launch_when_occupied(self):
        stats = '0, GPU-zero, 6\n1, GPU-one, 30000\n2, GPU-two, 30000\n3, GPU-three, 30000\n'
        for processes, expected in [('GPU-one, 9\n', True), ('GPU-zero, 10\n', False)]:
            with patch.object(m.subprocess, 'check_output', side_effect=[stats, processes]):
                self.assertIs(m.gpu0_idle(), expected)
        with patch.object(m, 'verify_checkout'), patch.object(m, 'active_processes', return_value=[10]), \
                patch.object(m.subprocess, 'Popen') as popen:
            self.assertIsNone(m.launch_recovery(Path('/unused'), {}, 'fullrank', 'training', {}))
            popen.assert_not_called()

    def test_training_requires_all_states_and_continuous_finite_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / 'checkpoints'
            checkpoint.mkdir()
            metrics = [{'step': s, 'total_loss': 0.1, 'losses_finite': {'all': True}} for s in range(1, 1501)]
            path = checkpoint / 'training_metrics.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in metrics))
            m.write(root / 'manifest.json', {'status': 'succeeded', 'returncode': 0,
                    'git': {'commit': m.COMMIT, 'branch': m.BRANCH}})
            for step in m.STEPS:
                folder = checkpoint / f'checkpoint-{step}'
                folder.mkdir()
                for name in m.STATE_FILES:
                    (folder / name).write_bytes(b'state')
            self.assertTrue(m.training_info(root)['complete'])
            (checkpoint / 'checkpoint-500/optimizer.bin').write_bytes(b'')
            self.assertFalse(m.training_info(root)['complete'])
            metrics[-1]['losses_finite']['all'] = False
            path.write_text(''.join(json.dumps(row) + '\n' for row in metrics))
            self.assertFalse(m.training_info(root)['finite'])

    def test_completion_needs_297_hashed_images_and_matching_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, training = Path(temporary) / 'inference', Path(temporary) / 'training'
            (output / 'ckpts').mkdir(parents=True)
            m.write(output / 'status.json', {'status': 'succeeded', 'generated_images': 297, 'completed_steps': list(m.STEPS)})
            m.write(output / 'ckpts/index.json', {'training': {'run': str(training)},
                    'steps': {str(s): {'paired_embedding_finite': True} for s in m.STEPS}})
            for step in m.STEPS:
                folder = output / f'step{step}'
                (folder / 'images').mkdir(parents=True)
                planned, actual = [], []
                for i in range(99):
                    path = folder / 'images' / f'{i}.png'
                    path.write_bytes(f'fixture {step} {i}'.encode())
                    row = {'id': f'{step}_{i}', 'image_path': f'images/{i}.png'}
                    planned.append(row)
                    actual.append({**row, 'status': 'ok', 'image_sha256': m.digest(path)})
                for name, rows in [('generation_manifest.jsonl', planned), ('generation_status.jsonl', actual)]:
                    (folder / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))
            info = m.inference_info(output, training)
            self.assertTrue(info['complete'])
            self.assertEqual(info['images'], 297)
            with self.assertRaisesRegex(ValueError, 'selected training'):
                m.inference_info(output, Path(temporary) / 'wrong_training')
            (output / 'step500/images/0.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                m.inference_info(output, training)

    def test_recovery_is_fresh_original_config_and_gpu0(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            (parent / 'monitor').mkdir()
            job = {'run': '/old/20261102-172731__study__variant__87d6e18__42', 'config_path': '/frozen/config.json'}
            record = {'attempts': [], 'locations': {'fullrank': {'training': job['run'], 'inference': '/old/inference'}}}
            with patch.object(m, 'verify_checkout'), patch.object(m, 'active_processes', return_value=[]), \
                    patch.object(m, 'gpu0_idle', return_value=True), patch.object(m.subprocess, 'run') as preflight, \
                    patch.object(m.subprocess, 'Popen') as popen:
                popen.return_value.pid = 99
                m.launch_recovery(parent, job, 'fullrank', 'training', record)
                call = popen.call_args
                self.assertEqual(call.kwargs['cwd'], m.CHECKOUT)
                self.assertEqual(call.kwargs['env']['CUDA_VISIBLE_DEVICES'], '0')
                self.assertEqual(preflight.call_args.kwargs['cwd'], m.CHECKOUT)
                self.assertIn('contract.verify_dataset', preflight.call_args.args[0][3])
                self.assertIn('/frozen/config.json', call.args[0])
                self.assertNotIn('--resume', call.args[0])
                self.assertTrue(Path(record['locations']['fullrank']['training']).is_relative_to(parent / 'recovery'))
                self.assertEqual(len(record['attempts']), 1)
                self.assertEqual(m.read(parent / 'monitor/state.json')['attempts'][0]['pid'], 99)

    def test_code_error_evidence_prevents_blind_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'failure.log'
            path.write_text('Traceback\nValueError: dataset changed\n')
            self.assertEqual(m.fatal_evidence([path])['log'], str(path))
            path.write_text('RuntimeError: CUDA out of memory\n')
            self.assertIsNone(m.fatal_evidence([path]))

    def test_crash_before_directory_creation_keeps_intent_and_advances_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            (parent / 'monitor').mkdir()
            job = {'run': '/old/20261102-172731__study__variant__87d6e18__42', 'config_path': '/frozen/config.json'}
            record = {'attempts': [], 'locations': {'fullrank': {'training': job['run'], 'inference': '/old/inference'}}}
            with patch.object(m, 'verify_checkout'), patch.object(m, 'active_processes', return_value=[]), \
                    patch.object(m, 'gpu0_idle', return_value=True), patch.object(m.subprocess, 'run'), \
                    patch.object(m.subprocess, 'Popen') as popen:
                with patch.object(Path, 'mkdir', side_effect=OSError('simulated crash before mkdir')):
                    with self.assertRaisesRegex(OSError, 'simulated crash'):
                        m.launch_recovery(parent, job, 'fullrank', 'training', record)
                popen.assert_not_called()
                persisted = m.read(parent / 'monitor/state.json')
                self.assertEqual(persisted['attempts'][0]['attempt'], 1)
                self.assertFalse((parent / 'recovery/fullrank/training-1').exists())
                popen.return_value.pid = 101
                m.launch_recovery(parent, job, 'fullrank', 'training', persisted)
                self.assertEqual(persisted['attempts'][-1]['attempt'], 2)
                self.assertTrue((parent / 'recovery/fullrank/training-2').is_dir())


if __name__ == '__main__':
    unittest.main()
