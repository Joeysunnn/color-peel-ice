"""Check four27-image Subject/Texture combinations, including dry-run review paths."""

import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import pytest

from experiments.sd21_wood_branches_v1 import composition as shared
from experiments.sd21_wood_subject27_v1 import composition


def test_four_jobs_use_two_subject_steps_and_matching_methods():
    jobs = composition.jobs(Path('/new'), (2, 3))
    assert [(job['subject_step'], job['method'], job['gpu']) for job in jobs] == [
        (400, 'fullrank', 2), (400, 'lora_r64', 3), (600, 'fullrank', 2), (600, 'lora_r64', 3)]
    rows = [row for job in jobs for row in shared.prompt_rows(
        job['method'], subject_step=job['subject_step'], subject_images=27)]
    assert len(rows) == len({row['id'] for row in rows}) == 124
    assert {row['subject_images'] for row in rows} == {27}
    assert {row['attribute_step'] for row in rows} == {600}
    for job in jobs:
        command = job['command']
        assert command[command.index('--subject-step') + 1] == str(job['subject_step'])
        assert command[command.index('--method') + 1] == job['method']
        assert job['output'].endswith(str(Path(f'S{job["subject_step"]}') / job['method']))


def source_audit(tmp_path, method='fullrank', step=600):
    image = tmp_path / 'actual.png'
    Image.new('RGB', (512, 512), 'green').save(image)
    digest = shared.sha256(image)
    refs = {branch: [{'image': str(image), 'image_sha256': digest,
                     'caption': f'a photo of {token} spoon', 'id': f'{branch}_{i}'} for i in range(count)]
            for branch, count, token in (('subject', 27, '<S*>'), ('texture', 16, '<T*>'))}
    revision = {'branch': 'codex/new', 'commit': 'new'}
    audit = {'inference_git': revision, 'training_gits': {'subject': 'old675', 'texture': 'old190'},
             'sources': {'subject': {method: {'method': method, 'step': step}},
                         'texture': {method: {'method': method, 'step': 600}}},
             'references': refs, 'source_locks': {str(image): digest}}
    path = tmp_path / 'audit.json'
    shared.write(path, audit)
    return path, revision


@pytest.mark.parametrize('method', ['fullrank', 'lora_r64'])
def test_dryrun_preserves27_and16_references_and_s600_metadata(tmp_path, method):
    audit, revision = source_audit(tmp_path, method)
    output = tmp_path / 'new' / 'S600' / method
    with patch.object(shared, 'git_info', return_value=revision), \
         patch.object(shared, 'resolve_run_dir', side_effect=lambda path: path.resolve()):
        state = shared.generate(audit, output, method, subject_step=600, subject_images=27)
    assert state['status'] == 'dry_run_verified'
    assert (state['subject_step'], state['subject_images'], state['expected_images']) == (600, 27, 31)
    refs = json.loads((output / 'training_references.json').read_text())['references']
    assert {key: len(rows) for key, rows in refs.items()} == {'subject': 27, 'texture': 16}
    rows = [json.loads(line) for line in (output / 'generation_manifest.jsonl').read_text().splitlines()]
    assert [row['seed'] for row in rows] == list(range(30, 61))
    assert all(row['subject_step'] == 600 and row['subject_images'] == 27 for row in rows)
    assert '../../index.html' in (output / 'index.html').read_text(encoding='utf8')


def test_wrong_subject_step_rejected_before_creating_output(tmp_path):
    audit, revision = source_audit(tmp_path, step=600)
    output = tmp_path / 'wrong'
    with patch.object(shared, 'git_info', return_value=revision), \
         patch.object(shared, 'resolve_run_dir', side_effect=lambda path: path.resolve()), \
         pytest.raises(ValueError, match='requested steps'):
        shared.generate(audit, output, 'fullrank', subject_step=400, subject_images=27)
    assert not output.exists()
