import pytest

from experiments.sd21_color_rendered9_200_v1 import contract as previous
from experiments.sd21_color_rendered9_150_v1 import contract, inference
from experiments.sd21_color_gray9_v1 import run
from experiments.mailbox_color_sd15_v1 import prompts
from experiments.sd15_token_local_kv_v1.snapshots import checkpoint_plan


@pytest.mark.parametrize('method', contract.METHODS)
def test_fresh150_preserves_other_arguments_and_saves_both_pairs(method, monkeypatch, tmp_path):
    monkeypatch.setenv(contract.DATASET_ENV, str(tmp_path / 'dataset'))
    old, new = previous.build_config('color', method), contract.build_config('color', method)
    assert {k for k in old['args'] if old['args'][k] != new['args'][k]} == {'max_train_steps', 'checkpoint_steps'}
    assert new['args']['max_train_steps'] == 150
    assert checkpoint_plan(new['args']) == [100, 150]
    assert 'resume_from_checkpoint' not in new['args']
    assert contract.SOURCE_INDEX_SHA == previous.SOURCE_INDEX_SHA
    assert contract.source_snapshot is previous.source_snapshot
    assert previous.STEPS == (200,)


def test_exact640_matrix_and_queue(monkeypatch, tmp_path):
    monkeypatch.setenv(contract.DATASET_ENV, str(tmp_path / 'dataset'))
    expected = [r['prompt'] for r in prompts.conditions() if r['arm'] == 'color_only']
    rows = []
    for method in contract.METHODS:
        for step in (100, 150):
            group = inference.prompt_rows(method, step)
            assert len(group) == 160
            assert {r['seed'] for r in group} == set(range(30, 46))
            assert list(dict.fromkeys(r['prompt'] for r in group)) == expected
            assert all(r['cfg'] == 6 and r['color_adapter_loaded'] for r in group)
            rows.extend(group)
    assert len({r['id'] for r in rows}) == 640
    with pytest.raises(ValueError):
        inference.prompt_rows('fullrank', 200)
    jobs = run.jobs(tmp_path, {'commit': 'abcdef0'}, (2, 3), contract=contract, inference=inference)
    assert [j['gpu'] for j in jobs] == [2, 3]
    assert all(inference.__name__ in j['inference_command'] and j['config']['args']['checkpoint_steps'] == [100, 150] for j in jobs)
