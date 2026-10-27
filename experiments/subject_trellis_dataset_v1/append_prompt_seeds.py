"""Append seeds44-48 inside the completed literal-prompt experiment, exclusively."""

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import sys

from experiments.subject_marble_color30_v1 import inference as renderer
from experiments.subject_trellis_dataset_v1 import prompt_forms, run_prompt_forms as source, training
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

EXPERIMENT = Path('/var/tmp/colorpeel-r12user5/runs/trellis_blue_green72_prompt_forms_v1/'
                  '20261028-001330__literal180__9a6d618__42')
ORIGINAL_GIT = {'branch': 'codex/trellis72-material-prompt-forms-v1',
                'commit': '9a6d618e8992d896d58f1e2dc3df08b2120d0456'}
SEEDS = (44, 45, 46, 47, 48)


def original_locks():
    """Freeze explicit original artifacts, never recursively include extensions."""
    state = read_json(EXPERIMENT / 'status.json')
    if (state.get('status') != 'inference_verified' or state.get('inference_git') != ORIGINAL_GIT
            or state.get('training_git') != source.TRAINING_GIT
            or state.get('completed_steps') != list(prompt_forms.STEPS)
            or state.get('expected_images') != 180 or state.get('sampling') != prompt_forms.SAMPLING):
        raise ValueError('require the complete original42/43 literal-prompt experiment')
    locks = {str(EXPERIMENT / name): sha256(EXPERIMENT / name) for name in
             ('status.json', 'source_training_audit.json', 'source_audit.log', 'prompt_manifest.csv', 'index.html')}
    locks.update({str(path): sha256(path) for path in (EXPERIMENT / 'repro_outputs').iterdir() if path.is_file()})
    for step in prompt_forms.STEPS:
        directory = EXPERIMENT / f'step{step}'
        provenance = read_json(directory / 'provenance.json')
        expected = prompt_forms.prompt_rows(step)
        manifest = [json.loads(line) for line in (directory / 'generation_manifest.jsonl').read_text().splitlines()]
        records = [json.loads(line) for line in (directory / 'generation_status.jsonl').read_text().splitlines()]
        if (provenance.get('status') != 'succeeded' or provenance.get('git') != ORIGINAL_GIT
                or provenance.get('sampling') != prompt_forms.SAMPLING or provenance.get('step') != step
                or provenance.get('training_images') != 72 or provenance.get('generated_images') != 60
                or manifest != expected or len(records) != 60 or len({row['id'] for row in records}) != 60):
            raise ValueError('original step matrix or provenance differs')
        by_id = {row['id']: row for row in expected}
        for row in records:
            if row['id'] not in by_id or any(row.get(key) != value for key, value in by_id[row['id']].items()):
                raise ValueError('original recorded prompt or seed differs')
            path = directory / row['image_path']
            if (not path.resolve().is_relative_to(directory) or path.is_symlink()
                    or row.get('status') not in ('ok', 'safety_filtered') or sha256(path) != row['image_sha256']):
                raise ValueError('original image differs from its recorded hash')
            locks[str(path)] = row['image_sha256']
        for name in ('provenance.json', 'training_references.json', 'generation_manifest.jsonl',
                     'generation_status.jsonl', 'index.html', 'training_contact_sheet.jpg',
                     'fullrank_contact_sheet.jpg', 'lora_r64_contact_sheet.jpg'):
            locks[str(directory / name)] = sha256(directory / name)
        for method in prompt_forms.METHODS:
            path = directory / 'checkpoints' / method / 'receipt.json'
            receipt = read_json(path)
            source.check_locks(receipt['hashes'])
            locks.update(receipt['hashes'])
            locks[str(path)] = sha256(path)
    source.check_locks(locks)
    return locks


def targets():
    return [EXPERIMENT / f'step{step}' / row['image_path'] for step in prompt_forms.STEPS
            for row in prompt_forms.prompt_rows(step, SEEDS)]


def check_new_targets():
    paths = targets()
    if len(paths) != 450 or len(set(paths)) != 450:
        raise ValueError('require exactly450 unique new image paths')
    for path in paths:
        if not path.resolve().is_relative_to(EXPERIMENT) or path.exists() or path.is_symlink():
            raise FileExistsError('new-seed image path is occupied or escapes the experiment')
    return paths


def report(output, state):
    folder = output / 'repro_outputs'
    folder.mkdir(exist_ok=True)
    source.write(output / 'status.json', state)
    source.write(folder / 'status.json', state)
    notes = {
        'SUMMARY.md': f"Status: {state['status']}; completed new-seed steps: {state['completed_steps']}. Expected450 added images; original180 retained. No retraining.\n",
        'COMMANDS.md': shlex.join([sys.executable, '-B', '-m', 'experiments.subject_trellis_dataset_v1.append_prompt_seeds',
            '--output', str(output), '--gpu', str(state['gpu'])] + (['--execute-inference'] if state['execute'] else [])) + '\n',
        'LOG.md': 'New provenance/manifests/sheets live in this extension. PNG image_root is the original step directory. Old root status and receipts remain unchanged.\n',
        'SCIENTIFIC_CHANGELOG.md': 'Only inference seeds change from42/43 to44/45/46/47/48. '
            'Same15 literal prompts, methods, paired500/750/1000 snapshots,PNDM100,CFG3.5,512px,fp16,safety checker. No training.\n',
        'COMPARABILITY_REPORT.md': 'Original180 PNGs and original reports/checkpoints are hash-locked. '
            'New PNGs are exclusively created alongside them; new sheets show the five added seed rows. '
            'Full matrix expected630 images across seven seeds. Changing seed samples generation variability, not a new training run.\n',
        'PATCHES.md': 'Explicit seed-list and image-root hooks keep default behavior. Source changed locally, committed/pushed, then cloned/pulled on server. '
            'Existing target files are rejected and new PNGs use exclusive creation, never overwrite.\n'}
    for name, text in notes.items():
        (folder / name).write_text(text, encoding='utf8')


def run(output, gpu, execute=False):
    output = resolve_run_dir(Path(output))
    if output.parent != EXPERIMENT / 'extensions' or output.exists() or gpu < 0:
        raise ValueError('require a fresh extension directly inside the original experiment/extensions')
    revision = git_info()
    old_locks = original_locks()
    new_paths = check_new_targets()
    output.mkdir(parents=True)
    state = {'status': 'auditing', 'inference_git': revision, 'training_git': source.TRAINING_GIT,
        'original_inference_git': ORIGINAL_GIT, 'experiment': str(EXPERIMENT), 'extension': str(output),
        'seeds': list(SEEDS), 'gpu': gpu, 'execute': execute, 'retraining': False,
        'expected_added_images': 450, 'original_images': 180, 'expected_total_images': 630,
        'completed_steps': [], 'sampling': {**prompt_forms.SAMPLING, 'seeds': list(SEEDS)}}
    report(output, state)
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', DIFFUSERS_OFFLINE='1')
        selections, training_locks = source.source_audit(output)
        locks = {**old_locks, **training_locks}
        source.write(output / 'original_result_locks.json', old_locks)
        source.write(output / 'new_image_plan.json', {'seeds': list(SEEDS), 'images': [str(path) for path in new_paths]})
        state['original_artifact_locks'] = len(old_locks)
        if execute:
            claim = EXPERIMENT / 'extensions/seed44_48.claim'
            claim.mkdir(exist_ok=False)
            source.write(claim / 'owner.json', {'git': revision, 'extension': str(output), 'seeds': list(SEEDS)})

        def load_selections(index, dataset):
            if Path(index).resolve() != source.SOURCE / 'training_index.json' or Path(dataset).resolve() != source.SOURCE / 'training_dataset':
                raise ValueError('selection source differs from the audited training run')
            source.source_revision()
            source.check_locks(locks)
            return selections

        def rows(step):
            return prompt_forms.prompt_rows(step, SEEDS)

        planned = [row for step in prompt_forms.STEPS for row in rows(step)]
        with (output / 'prompt_manifest.csv').open('x', newline='', encoding='utf8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(planned[0]))
            writer.writeheader()
            writer.writerows(planned)
        (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Seeds44–48;450 appended images</h1>'
            + ''.join(f'<p><a href="step{step}/index.html">Checkpoint {step}</a></p>' for step in prompt_forms.STEPS), encoding='utf8')
        if execute:
            relative = output.relative_to(EXPERIMENT).as_posix()
            with (EXPERIMENT / 'index_seed44_48.html').open('x', encoding='utf8') as handle:
                handle.write('<!doctype html><meta charset="utf-8"><h1>Literal prompts: seeds42–48</h1>'
                    '<p>Original180 images plus450 added images expected. Each PNG remains in the original step/images folder.</p>'
                    '<p><a href="index.html">Original seeds42/43</a></p>'
                    f'<p><a href="{relative}/index.html">Seeds44–48 and status</a></p>')
        for step in prompt_forms.STEPS:
            if git_info() != revision:
                raise ValueError('append checkout changed')
            source.check_locks(locks)
            state.update(status='generating' if execute else 'dry_run', active_step=step)
            report(output, state)
            result = renderer.run(source.SOURCE / 'training_dataset', output / f'step{step}', step,
                source.SOURCE / 'training_index.json', execute, 'cuda:0', training_module=training,
                reference_count=72, experiment='trellis72_literal_prompt_seeds44_48',
                reference_label='TRELLIS.2 blue/green72 seeds44–48', probe_rows=rows, probe_labels=prompt_forms.LABELS,
                selection_loader=load_selections, sampling_seeds=SEEDS, image_root=EXPERIMENT / f'step{step}',
                baseline_note='Same literal prompt experiment and weights; append only seeds44–48 alongside immutable42/43 images.')
            if (result['status'] != ('succeeded' if execute else 'dry_run') or result['expected_images'] != 150
                    or result['sampling'] != state['sampling'] or result.get('training_images') != 72
                    or (execute and (result['generated_images'] != 150 or len(result['contact_sheets']) != 3))):
                raise ValueError('incomplete added-seed matrix')
            state['completed_steps'].append(step)
            report(output, state)
        source.source_revision()
        source.check_locks(locks)
        if git_info() != revision:
            raise ValueError('append checkout changed')
        if execute and any(not path.is_file() for path in new_paths):
            raise ValueError('missing new image in original image folder')
        state.update(status='append_verified' if execute else 'dry_run_verified', active_step=None,
                     original_artifacts_unchanged=True)
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report(output, state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--execute-inference', action='store_true')
    args = parser.parse_args()
    print(run(args.output, args.gpu, args.execute_inference)['status'])
