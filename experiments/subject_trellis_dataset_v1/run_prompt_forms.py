"""Test literal material prompt forms using the completed blue/green72 snapshots."""

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.subject_marble_color30_v1 import inference as renderer
from experiments.subject_trellis_dataset_v1 import prompt_forms, training
from experiments.sd15_token_local_kv_v1 import snapshots
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

SOURCE = Path('/var/tmp/colorpeel-r12user5/runs/subject_trellis_blue_green72_training_v1/'
              '20261027-204842__both_methods1000_transfer__b3b850d__42')
SOURCE_CHECKOUT = Path('/var/tmp/colorpeel-r12user5/checkouts/trellis72-blue-green-transfer-v1')
TRAINING_GIT = {'branch': 'codex/trellis72-blue-green-transfer-v1',
                'commit': 'b3b850d9047f793b7d23e3a61d43d19e28972557'}
AUDIT_SCRIPT = '''
import contextlib,json,sys
from pathlib import Path
with contextlib.redirect_stdout(sys.stderr):
    from experiments.subject_marble_color30_v1 import inference
    from experiments.subject_trellis_dataset_v1 import training
    from scripts.launch.colorpeel_run import git_info
    root=Path.cwd().resolve()
    if any(not Path(module.__file__).resolve().is_relative_to(root) for module in (inference,training)):
        raise ValueError('audit imports must come from the frozen training checkout')
    source=Path(sys.argv[1])
    dataset=source/'training_dataset'
    training.verify_dataset(dataset)
    runs=inference._selections(source/'training_index.json',dataset,training_module=training)
    revision=git_info()
print(json.dumps({'git':revision,'runs':runs},allow_nan=False))
'''


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def source_revision():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(SOURCE_CHECKOUT), *args], text=True).strip()
    if git('status', '--porcelain'):
        raise ValueError('frozen training checkout is dirty')
    revision = {'branch': git('rev-parse', '--abbrev-ref', 'HEAD'), 'commit': git('rev-parse', 'HEAD')}
    if revision != TRAINING_GIT:
        raise ValueError('frozen training checkout revision changed')
    return revision


def check_locks(locks):
    if any(sha256(path) != digest for path, digest in locks.items()):
        raise ValueError('completed training input or checkpoint changed')


def recorded_snapshot_locks(selection):
    checkpoint = Path(selection['checkpoint']).resolve()
    locks = {}
    for step in prompt_forms.STEPS:
        receipt_path = SOURCE / f'step{step}/checkpoints' / selection['method'] / 'receipt.json'
        receipt = read_json(receipt_path)
        pair = {str(checkpoint / f'checkpoint-{step}' / name)
                for name in ('pytorch_model.bin', 'pytorch_model_1.bin')}
        recorded = receipt.get('source_snapshot_hashes', {})
        if (receipt.get('method') != selection['method'] or receipt.get('step') != step
                or receipt.get('steps') != step or receipt.get('training_steps') != 1000
                or receipt.get('source_checkpoint') != str(checkpoint) or set(recorded) != pair
                or receipt.get('source_hashes') != selection['hashes']
                or receipt.get('paired_embedding_finite') is not True
                or receipt.get('source_adapter_sha256') != recorded[str(checkpoint / f'checkpoint-{step}/pytorch_model.bin')]
                or receipt.get('source_text_encoder_sha256') != recorded[str(checkpoint / f'checkpoint-{step}/pytorch_model_1.bin')]):
            raise ValueError('historical paired snapshot receipt differs')
        check_locks(recorded)
        locks.update(recorded)
        locks[str(receipt_path)] = sha256(receipt_path)
    return locks


def source_audit(output):
    source_revision()
    state = read_json(SOURCE / 'status.json')
    index = read_json(SOURCE / 'training_index.json')
    if (state.get('status') != 'execution_verified' or state.get('git') != TRAINING_GIT
            or state.get('completed_steps') != {'fullrank': 1000, 'lora_r64': 1000}
            or state.get('training_images') != 72 or state.get('inference_steps') != [500, 750, 1000]
            or index.get('git') != TRAINING_GIT):
        raise ValueError('require the completed specified blue/green72 experiment')
    paths = [SOURCE / name for name in ('status.json', 'training_index.json')]
    paths.extend(SOURCE / 'training_dataset' / name for name in
                 ('concepts.json', 'training_assets_manifest.jsonl', 'dataset_audit.json', 'source_locks.json', 'approval.json'))
    locks = {str(path): sha256(path) for path in paths}
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
    for key in ('PYTHONPATH', 'PYTHONHOME', 'GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_COMMON_DIR'):
        env.pop(key, None)
    result = subprocess.run([sys.executable, '-B', '-c', AUDIT_SCRIPT, str(SOURCE)],
        cwd=SOURCE_CHECKOUT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output / 'source_audit.log').write_text(result.stderr, encoding='utf8')
    if result.returncode:
        raise RuntimeError('strict original-revision training audit failed; see source_audit.log')
    audit = json.loads(result.stdout)
    if audit.get('git') != TRAINING_GIT or audit.get('runs') != index.get('runs'):
        raise ValueError('original-revision audit differs from the frozen training index')
    for step in prompt_forms.STEPS:
        path = SOURCE / f'step{step}/provenance.json'
        provenance = read_json(path)
        if (provenance.get('status') != 'succeeded' or provenance.get('git') != TRAINING_GIT
                or provenance.get('step') != step or provenance.get('generated_images') != 36
                or provenance.get('training') != audit['runs']):
            raise ValueError('historical inference audit differs')
        locks[str(path)] = sha256(path)
    for selection in audit['runs']:
        checkpoint = Path(selection['checkpoint']).resolve()
        if not checkpoint.is_relative_to(SOURCE / 'training' / selection['method']):
            raise ValueError('training checkpoint path escapes its declared method')
        locks.update(selection['hashes'])
        locks.update(recorded_snapshot_locks(selection))
        for step in prompt_forms.STEPS:
            for name in snapshots.STATE_FILES:
                path = checkpoint / f'checkpoint-{step}' / name
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError('missing or empty paired snapshot state')
                digest = sha256(path)
                if str(path) in locks and locks[str(path)] != digest:
                    raise ValueError('recorded snapshot changed during audit')
                locks[str(path)] = digest
    source_revision()
    check_locks(locks)
    write(output / 'source_training_audit.json', {**audit, 'source_locks': locks})
    return audit['runs'], locks


def report(output, state):
    folder = output / 'repro_outputs'
    folder.mkdir(exist_ok=True)
    write(output / 'status.json', state)
    write(folder / 'status.json', state)
    (folder / 'SUMMARY.md').write_text(
        f"Status: {state['status']}; verified steps: {state['completed_steps']}; expected180 images. No retraining.\n", encoding='utf8')
    (folder / 'COMMANDS.md').write_text(shlex.join([sys.executable, '-B', '-m',
        'experiments.subject_trellis_dataset_v1.run_prompt_forms', '--output', str(output),
        '--gpu', str(state['gpu'])] + (['--execute-inference'] if state['execute'] else [])) + '\n', encoding='utf8')
    notes = {
        'LOG.md': 'See source_audit.log, source_training_audit.json and each step provenance/status/manifests.\n',
        'SCIENTIFIC_CHANGELOG.md': 'Only prompt phrasing changes:15 literal conditions, no photo prefix or mailbox noun. '
            'Matte is retained literally even in made entirely of matte. Existing weights/data/samplers unchanged. '
            'Both methods, seeds42/43, checkpoints500/750/1000. PNDM100,CFG3.5,512px,fp16,safety checker.\n',
        'COMPARABILITY_REPORT.md': 'Source training is audited strictly in its original clean b3b850d checkout. '
            'New inference revision is recorded separately; no source index or manifest is relabeled. '
            'Each adapter uses its same-step Subject embedding. Compared with previous prompts, both surface '
            'phrasing and removal of photo/mailbox wording change; this is not a one-word-only material ablation.\n',
        'PATCHES.md': 'New prompt builder, audited inference runner, and explicit selection_loader/baseline-note '
            'hooks. Default historical audit/sampling paths remain unchanged. Code deployed through Git only.\n'}
    for name, text in notes.items():
        (folder / name).write_text(text, encoding='utf8')


def run(output, gpu, execute=False):
    output = resolve_run_dir(Path(output))
    if (gpu < 0 or output.exists() or output.is_relative_to(training.ROOT)
            or output.is_relative_to(SOURCE) or output.is_relative_to(SOURCE_CHECKOUT)):
        raise ValueError('require a new external inference directory and free nonnegative GPU')
    revision = git_info()
    output.mkdir(parents=True)
    state = {'status': 'source_audit', 'inference_git': revision, 'training_git': TRAINING_GIT,
        'source': str(SOURCE), 'source_checkout': str(SOURCE_CHECKOUT), 'gpu': gpu,
        'execute': execute, 'retraining': False, 'expected_images': 180, 'completed_steps': [],
        'sampling': prompt_forms.SAMPLING}
    report(output, state)
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', DIFFUSERS_OFFLINE='1')
        selections, locks = source_audit(output)
        state['source_locks'] = locks

        def load_selections(index, dataset):
            if Path(index).resolve() != SOURCE / 'training_index.json' or Path(dataset).resolve() != SOURCE / 'training_dataset':
                raise ValueError('source selection paths differ from the audited experiment')
            source_revision()
            check_locks(locks)
            return selections

        all_rows = [row for step in prompt_forms.STEPS for row in prompt_forms.prompt_rows(step)]
        with (output / 'prompt_manifest.csv').open('x', newline='', encoding='utf8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        for step in prompt_forms.STEPS:
            if git_info() != revision:
                raise ValueError('inference checkout changed')
            source_revision()
            check_locks(locks)
            state.update(status='generating' if execute else 'dry_run', active_step=step)
            report(output, state)
            result = renderer.run(SOURCE / 'training_dataset', output / f'step{step}', step,
                SOURCE / 'training_index.json', execute, 'cuda:0', training_module=training,
                reference_count=72, experiment='trellis_blue_green72_material_prompt_forms_v1',
                reference_label='TRELLIS.2 blue/green72 prompt forms', probe_rows=prompt_forms.prompt_rows,
                probe_labels=prompt_forms.LABELS, selection_loader=load_selections,
                baseline_note='Three literal material templates without photo prefix or mailbox noun; frozen blue/green72 Subject weights and unchanged sampling.')
            if (result['status'] != ('succeeded' if execute else 'dry_run')
                    or result['expected_images'] != 60 or result['training_images'] != 72
                    or (execute and (result['generated_images'] != 60 or len(result['contact_sheets']) != 3))):
                raise ValueError('incomplete prompt-form matrix')
            state['completed_steps'].append(step)
            report(output, state)
        source_revision()
        check_locks(locks)
        if git_info() != revision:
            raise ValueError('inference checkout changed')
        state.update(status='inference_verified' if execute else 'dry_run_verified', active_step=None)
        (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Material prompt forms</title>'
            '<h1>Three literal prompt forms; five materials</h1>'
            + ''.join(f'<p><a href="step{step}/index.html">Checkpoint {step}</a></p>' for step in prompt_forms.STEPS), encoding='utf8')
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
