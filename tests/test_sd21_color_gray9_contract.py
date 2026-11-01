from pathlib import Path

import pytest

from experiments.mailbox_color_sd15_v1.training import build_config as old_config
from experiments.sd21_color_gray9_v1 import contract


@pytest.mark.parametrize('method', contract.METHODS)
def test_color_recipe_only_changes_base_and_paths(method, monkeypatch):
    dataset = Path('external-test-gray9').resolve()
    monkeypatch.setenv(contract.DATASET_ENV, str(dataset))
    old = old_config(method, dataset)
    new = contract.build_config('color', method)
    assert {k for k in old['args'] if old['args'][k] != new['args'][k]} == {'pretrained_model_name_or_path'}
    assert new['args']['max_train_steps'] == 1000
    assert new['args']['checkpoint_steps'] == [400, 600, 800, 1000]
    assert new['args']['modifier_token'] == '<C*>'
    assert new['args']['initializer_token'] == 'orange'
    assert new['args']['adam_weight_decay'] == 0
    assert new['args']['hflip'] is True
    assert new['run']['seed'] == 42
    assert 'instance_mask_dir' not in new['args']


def test_smoke_and_authorization_are_explicit(monkeypatch):
    monkeypatch.setenv(contract.DATASET_ENV, str(Path('external-test-gray9').resolve()))
    config = contract.build_config('color', 'fullrank', smoke=True)
    assert config['args']['max_train_steps'] == 2
    assert config['args']['checkpoint_steps'] == [2]
    assert contract.authorization()['transfer_seeds'] == list(range(30, 41))
    assert contract.authorization()['object_mask_loss'] is False
    with pytest.raises(ValueError):
        contract.build_config('subject', 'fullrank')


def test_missing_dataset_context_fails(monkeypatch):
    monkeypatch.delenv(contract.DATASET_ENV, raising=False)
    with pytest.raises(ValueError, match=contract.DATASET_ENV):
        contract.dataset_path()
