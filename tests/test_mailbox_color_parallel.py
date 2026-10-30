"""Validate exact-process handoff and exclusive three-GPU dispatch."""

import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.mailbox_color_sd15_v1 import parallel


class ColorParallelTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='color-parallel-', dir=os.environ.get('TMP'))
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.source = self.root / 'experiment'
        self.source.mkdir()
        self.output = self.root / 'continuations' / 'parallel'
        self.git = {'branch': 'codex/test', 'commit': 'f' * 40}
        self.addCleanup(patch.stopall)
        patch.object(parallel, 'SOURCE', self.source).start()
        patch.object(parallel, 'git_info', return_value=self.git).start()
        patch.dict(os.environ, {'COLORPEEL_OUTPUT_ROOT': str(self.root)}).start()

    def handoff_fixture(self):
        state = {'worker_pids': {'fullrank': 11, 'lora_r64': 12}, 'gpus': [2, 3]}
        parallel.write(self.source / 'status.json', state)
        parallel.write(self.root / 'LAUNCH_RECEIPT.json', {'coordinator_pid': 10})
        identities = {}
        for method, pid, gpu in (('fullrank', 11, 2), ('lora_r64', 12, 3)):
            folder = self.source / 'training' / method
            folder.mkdir(parents=True)
            parallel.write(folder / 'status.json', {'status': 'training_verified', 'completed_steps': 1000})
            identities[pid] = {'pid': pid, 'uid': 1006, 'starttime': '123', 'ignores_sigint': False,
                'args': [parallel.sys.executable, '-u', '-B', '-m', 'experiments.mailbox_color_sd15_v1.pipeline',
                    '--output', str(self.source), '--worker-method', method, '--gpu', str(gpu)]}
        self.output.mkdir(parents=True)
        return identities

    def test_handoff_signals_only_validated_inference_workers_once(self):
        identities = self.handoff_fixture()
        def killed(pid, sig):
            identities.pop(pid)
        with patch.object(parallel, 'process_info', side_effect=lambda pid: identities.get(pid)), \
             patch.object(parallel.os, 'getuid', return_value=1006, create=True), \
             patch.object(parallel.os, 'kill', side_effect=killed) as kill:
            state = parallel.stop_original(self.output)
        self.assertEqual([call.args for call in kill.call_args_list], [(11, signal.SIGINT), (12, signal.SIGINT)])
        self.assertEqual(state['status'], 'original_workers_and_coordinator_exited')
        self.assertFalse(state['training_restarted'])

    def test_unfinished_training_wrong_identity_and_ignored_signal_reject_before_any_signal(self):
        identities = self.handoff_fixture()
        for invalid in ('training', 'identity', 'ignored', 'uid'):
            with self.subTest(invalid=invalid):
                original = dict(identities[12])
                train = self.source / 'training' / 'lora_r64' / 'status.json'
                if invalid == 'training':
                    parallel.write(train, {'status': 'training', 'completed_steps': 999})
                elif invalid == 'identity':
                    identities[12] = {**original, 'args': ['another', 'process']}
                elif invalid == 'ignored':
                    identities[12] = {**original, 'ignores_sigint': True}
                else:
                    identities[12] = {**original, 'uid': 999}
                with patch.object(parallel, 'process_info', side_effect=lambda pid: identities.get(pid)), \
                     patch.object(parallel.os, 'getuid', return_value=1006, create=True), \
                     patch.object(parallel.os, 'kill') as kill:
                    with self.assertRaises(ValueError):
                        parallel.stop_original(self.output)
                    kill.assert_not_called()
                identities[12] = original
                parallel.write(train, {'status': 'training_verified', 'completed_steps': 1000})

    def test_three_queues_run_all_eight_jobs_exactly_once_without_training(self):
        audit = {'source_locks': {'locked': 'hash'}, 'inference_git': self.git}
        commands = []
        def popen(command, **kwargs):
            commands.append((command, kwargs['env']['CUDA_VISIBLE_DEVICES']))
            return SimpleNamespace(pid=123, wait=lambda: 0)
        with patch.object(parallel, 'source_audit', return_value=audit), \
             patch.object(parallel, 'stop_original', return_value={'status': 'exited'}), \
             patch.object(parallel.inference, 'check_locks'), \
             patch.object(parallel.parallel_job, 'inspect_source', return_value={'rows': [1] * 10, 'source_locks': {}}), \
             patch.object(parallel.subprocess, 'Popen', side_effect=popen):
            state = parallel.run(self.output, [0, 2, 3], True, True)
        self.assertEqual(state['status'], 'execution_verified')
        self.assertEqual(state['reused_images'], 80)
        self.assertEqual(state['pending_images'], 3144)
        self.assertFalse(state['training_restarted'])
        jobs = [(cmd[cmd.index('--job-method') + 1], int(cmd[cmd.index('--job-step') + 1])) for cmd, _ in commands]
        self.assertEqual(len(jobs), 8)
        self.assertEqual(set(jobs), {(m, s) for m in parallel.training.METHODS for s in parallel.training.STEPS})
        self.assertTrue(all(gpu in ('0', '2', '3') for _, gpu in commands))

    def test_dry_run_cannot_signal_or_create_worker(self):
        with patch.object(parallel, 'stop_original') as stop, patch.object(parallel.subprocess, 'Popen') as child:
            with self.assertRaises(ValueError):
                parallel.run(self.output, [0, 2, 3], False, True)
            stop.assert_not_called()
            child.assert_not_called()


if __name__ == '__main__':
    unittest.main()
