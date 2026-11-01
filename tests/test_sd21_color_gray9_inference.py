from experiments.sd21_color_gray9_v1 import inference, run
from experiments.mailbox_color_sd15_v1 import prompts


def test_color_matrix_and_cfg_do_not_inherit_subject_settings():
    expected = [r['prompt'] for r in prompts.conditions() if r['arm'] == 'color_only']
    for method in ('fullrank', 'lora_r64'):
        total = []
        for step in (400, 600, 800, 1000):
            rows = inference.prompt_rows(method, step)
            assert len(rows) == 110
            assert list(dict.fromkeys(r['prompt'] for r in rows)) == expected
            assert {r['seed'] for r in rows} == set(range(30, 41))
            assert all(r['cfg'] == 6 and r['color_adapter_loaded'] and '<S*>' not in r['prompt'] for r in rows)
            total.extend(rows)
        assert len({r['id'] for r in total}) == 440


def test_two_step_check_is_separate_from_formal(monkeypatch, tmp_path):
    monkeypatch.setenv('COLORPEEL_SD21_COLOR_DATASET', str(tmp_path / 'dataset'))
    formal = run.jobs(tmp_path / 'formal', {'commit': 'abcdef0'}, (2, 3))
    smoke = run.jobs(tmp_path / 'smoke', {'commit': 'abcdef0'}, (2, 3), smoke=True)
    assert [j['gpu'] for j in formal] == [2, 3]
    assert all(j['config']['args']['max_train_steps'] == 1000 for j in formal)
    assert all(j['config']['args']['checkpoint_steps'] == [400, 600, 800, 1000] for j in formal)
    assert all(j['config']['args']['max_train_steps'] == 2 and '--smoke' in j['inference_command'] for j in smoke)
    assert all('resume_from_checkpoint' not in j['config']['args'] for j in formal)
    assert len(inference.prompt_rows('lora_r64', 2, smoke=True)) == 1
