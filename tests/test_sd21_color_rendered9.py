import pytest

from experiments.sd21_color_gray9_v1 import contract as old, run as runner
from experiments.sd21_color_rendered9_v1 import contract, inference
from experiments.mailbox_color_sd15_v1 import prompts
from experiments.sd15_token_local_kv_v1.snapshots import checkpoint_plan


@pytest.mark.parametrize('method', contract.METHODS)
def test_only_dose_and_save_arguments_change_from_previous_sd21(method, monkeypatch, tmp_path):
    monkeypatch.setenv(contract.DATASET_ENV, str(tmp_path / 'dataset'))
    before, after = old.build_config('color', method), contract.build_config('color', method)
    assert {k for k in before['args'] if before['args'][k] != after['args'][k]} == {'max_train_steps', 'checkpoint_steps'}
    assert after['args']['max_train_steps'] == 800
    assert checkpoint_plan(after['args']) == [400, 600, 800]
    assert contract.authorization()['transfer_seeds'] == list(range(30, 35))
    assert 'rendered_background9' in str(contract.SOURCE_ROOT)
    assert old.STEPS == (400, 600, 800, 1000)


def test_exact_300_color_transfer_images():
    expected = [r['prompt'] for r in prompts.conditions() if r['arm'] == 'color_only']
    all_rows = []
    for method in contract.METHODS:
        for step in contract.STEPS:
            rows = inference.prompt_rows(method, step)
            assert len(rows) == 50
            assert list(dict.fromkeys(r['prompt'] for r in rows)) == expected
            assert {r['seed'] for r in rows} == set(range(30, 35))
            assert all(r['cfg'] == 6 and r['color_adapter_loaded'] for r in rows)
            all_rows.extend(rows)
    assert len({r['id'] for r in all_rows}) == 300
    with pytest.raises(ValueError):
        inference.prompt_rows('fullrank', 1000)


def test_queue_uses_new_inference_and_separate_smoke(monkeypatch, tmp_path):
    monkeypatch.setenv(contract.DATASET_ENV, str(tmp_path / 'dataset'))
    for smoke in (False, True):
        jobs = runner.jobs(tmp_path / str(smoke), {'commit': 'abcdef0'}, (2, 3), smoke,
                           contract=contract, inference=inference)
        assert [j['gpu'] for j in jobs] == [2, 3]
        for job in jobs:
            assert inference.__name__ in job['inference_command']
            assert job['config']['args']['max_train_steps'] == (2 if smoke else 800)
            assert job['config']['args']['checkpoint_steps'] == ([2] if smoke else [400, 600, 800])
            assert 'resume_from_checkpoint' not in job['config']['args']
            assert job['config']['data_lineage']['source'] == str(contract.SOURCE_ROOT)
