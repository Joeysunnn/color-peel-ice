"""Check the deployed queue includes training, paired inference and the full seed range."""

from pathlib import Path
from unittest.mock import patch

from experiments.sd21_wood_branches_v1 import run


def test_six_jobs_train_then_infer_without_compositions():
    def config(branch, method, smoke=False):
        return {'run': {'variant': branch + '_' + method},
                'args': {'max_train_steps': 2 if smoke else 1000}}

    with patch.object(run, 'build_config', side_effect=config):
        jobs = run.make_jobs(Path('/new/experiment'), {'commit': 'abcdef1234'}, (0, 2, 3))
    assert len(jobs) == 6
    assert [(job['branch'], job['method']) for job in jobs] == [
        (branch, method) for branch in run.BRANCHES for method in run.METHODS]
    assert [job['gpu'] for job in jobs] == [0, 2, 3, 0, 2, 3]
    for job in jobs:
        command = job['inference_command']
        start = command.index('--seeds') + 1
        assert command[start:start + 31] == list(map(str, range(30, 61)))
        assert '--execute' in command
        assert command[command.index('--run') + 1] == job['run']
        assert job['training_command'][-1] == job['run']
        assert job['config']['args']['max_train_steps'] == 1000


def test_literal_queue_receives_same_seeds(tmp_path):
    calls = []
    def capture(command, **kwargs):
        calls.append(command)

    with patch.object(run.subprocess, 'run', side_effect=capture):
        state = run.run_queue(2, [], tmp_path, {}, literal=True)
    assert state['status'] == 'succeeded'
    assert len(calls) == 1
    command = calls[0]
    assert command[command.index('--branch') + 1] == 'literal'
    start = command.index('--seeds') + 1
    assert command[start:start + 31] == list(map(str, range(30, 61)))
