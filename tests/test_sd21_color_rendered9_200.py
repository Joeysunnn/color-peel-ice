import pytest

from experiments.sd21_color_rendered9_v1 import contract as previous
from experiments.sd21_color_rendered9_200_v1 import contract, inference
from experiments.sd21_color_gray9_v1 import run
from experiments.mailbox_color_sd15_v1 import prompts


@pytest.mark.parametrize('method', contract.METHODS)
def test_fresh200_preserves_all_other_training_arguments(method, monkeypatch, tmp_path):
    monkeypatch.setenv(contract.DATASET_ENV, str(tmp_path / 'dataset'))
    old, new = previous.build_config('color', method), contract.build_config('color', method)
    assert {k for k in old['args'] if old['args'][k] != new['args'][k]} == {'max_train_steps', 'checkpoint_steps'}
    assert new['args']['max_train_steps'] == 200
    assert new['args']['checkpoint_steps'] == [200]
    assert 'resume_from_checkpoint' not in new['args']
    assert contract.SOURCE_INDEX_SHA == previous.SOURCE_INDEX_SHA
    assert contract.source_snapshot is previous.source_snapshot
    assert previous.STEPS == (400, 600, 800)


def test_exact320_matrix_and_new_queue(monkeypatch, tmp_path):
    monkeypatch.setenv(contract.DATASET_ENV, str(tmp_path / 'dataset'))
    expected = [r['prompt'] for r in prompts.conditions() if r['arm'] == 'color_only']
    all_rows = []
    for method in contract.METHODS:
        rows = inference.prompt_rows(method, 200)
        assert len(rows) == 160
        assert {r['seed'] for r in rows} == set(range(30, 46))
        assert list(dict.fromkeys(r['prompt'] for r in rows)) == expected
        assert all(r['cfg'] == 6 and r['color_adapter_loaded'] for r in rows)
        all_rows.extend(rows)
    assert len({r['id'] for r in all_rows}) == 320
    with pytest.raises(ValueError):
        inference.prompt_rows('fullrank', 800)
    jobs = run.jobs(tmp_path, {'commit': 'abcdef0'}, (0, 3), contract=contract, inference=inference)
    assert [j['gpu'] for j in jobs] == [0, 3]
    assert all(inference.__name__ in j['inference_command'] and j['config']['args']['checkpoint_steps'] == [200] for j in jobs)
