"""Check S400/T600 routing independently of GPU runtime and learned weights."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from experiments.sd21_wood_branches_v1 import composition


def test_both_methods_same_prompt_steps_and_seeds():
    for method in ('fullrank', 'lora_r64'):
        rows = composition.prompt_rows(method)
        assert len(rows) == len({row['id'] for row in rows}) == 31
        assert [row['seed'] for row in rows] == list(range(30, 61))
        assert {row['subject_step'] for row in rows} == {400}
        assert {row['attribute_step'] for row in rows} == {600}
        assert {row['prompt'] for row in rows} == {'a photo of <S*> spoon with <T*> texture'}
    with pytest.raises(ValueError):
        composition.prompt_rows('mixed')


@pytest.mark.parametrize('method', ['fullrank', 'lora_r64'])
def test_loads_both_matching_adapters_then_both_embeddings(method):
    subject = {'method': method, 'checkpoint': '/s400', 'weight': 'weights.bin', 'token': '<S*>'}
    texture = {'method': method, 'checkpoint': '/t600', 'weight': 'weights.bin', 'token': '<T*>'}
    pipe = Mock()
    events = []
    pipe.load_textual_inversion.side_effect = lambda path, **kw: events.append(('token', kw['weight_name']))
    pipe.unet.load_attn_procs.side_effect = lambda path, **kw: events.append(('primary', path))
    dual = Mock(side_effect=lambda *args, **kw: events.append(('dual', kw)))
    load = Mock(side_effect=lambda path, **kw: 'S-state' if path.as_posix().startswith('/s400') else 'T-state')
    modules = {
        'torch': SimpleNamespace(load=load),
        'src.methods.colorpeel_ice.dual_token_local_kv': SimpleNamespace(install_dual_token_local_kv=dual),
        'experiments.lora_kv_material_v1.attention': SimpleNamespace(install_dual_token_local_lora_kv=dual),
    }
    with patch.dict(sys.modules, modules), patch.object(composition.inference, 'load_pipeline', return_value=pipe) as base:
        result = composition.load_pipeline(subject, texture, 'cuda:0')
    assert result is pipe
    base.assert_called_once_with([], 'cuda:0')
    assert events[-2:] == [('token', '<S*>.bin'), ('token', '<T*>.bin')]
    if method == 'fullrank':
        assert events[0] == ('primary', '/s400')
        dual.assert_called_once_with(pipe.unet, 'T-state', primary_label='subject')
    else:
        dual.assert_called_once_with(pipe.unet, 'S-state', 'T-state', rank=64, alpha=64)


def test_rejects_unfinished_or_different_training_revision_before_pairs():
    for state in ({'status': 'running', 'git': composition.SOURCE_GIT, 'smoke': False},
                  {'status': 'succeeded', 'git': {'commit': 'wrong'}, 'smoke': False},
                  {'status': 'succeeded', 'git': composition.SOURCE_GIT, 'smoke': True}):
        with patch.object(composition, 'verify_model', return_value={}), \
             patch.object(composition, 'read_json', return_value=state), \
             pytest.raises(ValueError):
            composition.prepare()
