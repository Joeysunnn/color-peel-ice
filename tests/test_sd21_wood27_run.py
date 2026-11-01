"""Check that the27-image launch queues train both methods and automatically infer."""

from pathlib import Path
from unittest.mock import patch

from experiments.sd21_wood_subject27_v1 import run
from scripts.launch.colorpeel_run import validate_run_id


def test_two_methods_have_exact_training_config_and_automatic_inference():
    revision = {'commit': 'abcdef1234567890', 'branch': 'codex/test'}
    with patch.object(run.contract.previous.subject_training, 'read_json',
                      return_value={'subject_id': 'wood_spoon', 'rows': 72}):
        jobs = run.jobs(Path('/new/experiment'), revision, (2, 3))
    assert [(job['method'], job['gpu']) for job in jobs] == [('fullrank', 2), ('lora_r64', 3)]
    for job in jobs:
        config = job['config']
        assert config['run']['study'] == run.contract.STUDY
        assert config['args']['max_train_steps'] == 1000
        assert config['args']['checkpoint_steps'] == [400, 600, 800, 1000]
        assert config['args']['concepts_list'] == str(run.contract.DATASET / 'concepts.json')
        assert config['args']['initializer_token'] == 'spoon'
        validate_run_id(Path(job['run']), config, revision['commit'])
        assert job['training_command'][-1] == job['run']
        command = job['inference_command']
        assert 'experiments.sd21_wood_subject27_v1.inference' in command
        assert command[command.index('--run') + 1] == job['run']
        assert command[command.index('--method') + 1] == job['method']
        assert '--execute' in command


def test_queue_uses_injected27_verifier_and_reporter(tmp_path):
    calls = []
    config = {'args': {'max_train_steps': 1000, 'checkpoint_steps': [400, 600, 800, 1000]}}
    group = tmp_path / 'fullrank'
    group.mkdir()
    job = {'branch': 'subject', 'method': 'fullrank', 'run': str(group / 'run'),
           'group': str(group), 'config': config, 'training_command': ['train'],
           'inference_command': ['infer']}
    def reporter(folder, state, selected):
        (folder / 'train_outputs').mkdir(exist_ok=True)
        calls.append(state['status'])
    with patch.object(run.shared, 'git_info', return_value={'commit': 'same'}), \
         patch.object(run.shared, 'verify_dataset', side_effect=AssertionError('old72 verifier called')), \
         patch.object(run.shared.subprocess, 'run') as execute, \
         patch.object(run.contract, 'verify_dataset', return_value={'rows': 27}) as verify:
        result = run.shared.run_queue(2, [job], tmp_path, {'commit': 'same'},
            dataset_verifier=run.contract.verify_dataset, training_reporter=reporter)
    assert result['status'] == 'succeeded'
    assert execute.call_count == 2
    assert verify.call_count == 2
    assert calls == ['training', 'inference', 'succeeded']
