"""Audit the preserved wood180 Subject750/1000 pairs with the fixed Texture500."""

import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.wood_spoon_composition_v1 import source
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

SUBJECT_ROOT = Path('/var/tmp/colorpeel-r12user5/runs/subject_trellis_spoons_training_v1/'
    '20261028-094541__four_comparisons__d2e6239__42/wood_spoon')
CONTINUATION_ROOT = SUBJECT_ROOT / 'continuations/20261028-102906__remaining_jobs__c0ad7bd__42/all180'
SUBJECT_CHECKOUT = Path('/var/tmp/colorpeel-r12user5/checkouts/trellis-spoons-continue-v1')
OLD_GIT = {'commit': 'd2e6239c12f54465b33759558129bf8813ad7c16',
    'branch': 'codex/trellis-spoons-training-v1'}
CONTINUATION_GIT = {'commit': 'c0ad7bdd6d72b43677444151866695b6c02494e3',
    'branch': 'codex/trellis-spoons-continue-v1'}


def _subject(output, step, locks):
    program = '''import json,pathlib,sys
from experiments.subject_trellis_spoons_v1 import training,reuse
from scripts.launch.colorpeel_run import git_info
root,continuation,checkout=map(pathlib.Path,sys.argv[1:4])
assert pathlib.Path(training.__file__).resolve().is_relative_to(checkout)
assert pathlib.Path(reuse.__file__).resolve().is_relative_to(checkout)
dataset=root/'all180/training_dataset'
audit=training.verify_dataset(dataset)
assert audit['subject_id']=='wood_spoon' and audit['rows']==180
parents=reuse.selection_loader(continuation/'training_index.json',dataset)
print('WOOD180_COMPOSITION_AUDIT='+json.dumps({'parents':parents,'git':git_info()}))
'''
    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
    environment.pop('PYTHONPATH', None)
    process = subprocess.run([sys.executable, '-B', '-c', program, str(SUBJECT_ROOT),
        str(CONTINUATION_ROOT), str(SUBJECT_CHECKOUT)], cwd=SUBJECT_CHECKOUT, env=environment,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (output / 'subject180_training_audit.log').write_text(process.stdout, encoding='utf8')
    markers = [line.split('=', 1)[1] for line in process.stdout.splitlines()
               if line.startswith('WOOD180_COMPOSITION_AUDIT=')]
    if process.returncode or len(markers) != 1:
        raise ValueError('Frozen180 reuse audit failed; see subject180_training_audit.log')
    audited = json.loads(markers[0])
    if audited.get('git') != CONTINUATION_GIT:
        raise ValueError('Frozen continuation checkout revision changed')
    index = source._read(locks, CONTINUATION_ROOT / 'training_index.json')
    if (index.get('git') != CONTINUATION_GIT or index.get('runs') != audited['parents']
            or index.get('source_gits') != {'fullrank': OLD_GIT, 'lora_r64': CONTINUATION_GIT}):
        raise ValueError('Wood180 index differs from the strict mixed-revision source audit')
    for path, digest in index.get('reuse_receipt', {}).get('locks', {}).items():
        source._lock(locks, path, digest)
    parents = source._methods(audited['parents'], 'subject', '<S*>', 1000)
    provenance = source._read(locks, CONTINUATION_ROOT / f'step{step}/provenance.json')
    if (provenance.get('status') != 'succeeded' or provenance.get('step') != step
            or provenance.get('git') != CONTINUATION_GIT or provenance.get('training') != audited['parents']
            or provenance.get('training_images') != 180 or provenance.get('generated_images') != 20):
        raise ValueError('Expected the completed original wood180 paired-step inference')
    dataset = SUBJECT_ROOT / 'all180/training_dataset'
    for name in ('dataset_audit.json', 'source_locks.json', 'approval.json', 'concepts.json'):
        source._read(locks, dataset / name)
    for path, digest in source._read(locks, dataset / 'source_locks.json').items():
        source._lock(locks, path, digest)
    references = source._references(locks, dataset / 'training_assets_manifest.jsonl', 180)
    selections = {}
    for method, parent in parents.items():
        expected = ((SUBJECT_ROOT / 'all180/training/fullrank/runs') if method == 'fullrank'
                    else CONTINUATION_ROOT / 'training/lora_r64/runs')
        if Path(parent['run']).parent != expected:
            raise ValueError('Wood180 method does not belong to its original training directory')
        directory = CONTINUATION_ROOT / f'step{step}/checkpoints' / method
        derived = source._read(locks, directory / 'receipt.json')
        selections[method] = source.verify_pair(parent, derived, step, directory, locks)
    return selections, references


def prepare(output, step):
    if type(step) is not int or step not in (750, 1000):
        raise ValueError('Select the existing wood180 Subject750 or Subject1000 snapshot')
    output = resolve_run_dir(Path(output)).resolve()
    if (output.exists() or output.is_relative_to(source.ROOT)
            or any(output.is_relative_to(path) or path.is_relative_to(output)
                   for path in (SUBJECT_ROOT, source.SUBJECT_ROOT, source.NATURAL_ROOT, source.MATERIAL_ROOT))):
        raise ValueError('Use a fresh external audit directory outside frozen experiments')
    inference_git = git_info()
    output.mkdir(parents=True)
    # Reuse the existing strict T500 audit without changing its established guards.
    # Its auxiliary S72 audit remains explicitly recorded, never used for this inference.
    result = source.prepare(output / 'base72_texture500', texture_step=500, include_material=False)
    locks = result['source_locks']
    source._lock(locks, output / 'base72_texture500/audit.json')
    selections, references = _subject(output, step, locks)
    result['sources']['subject'] = selections
    result['references']['subject'] = references
    result['training_gits']['subject'] = {'fullrank': OLD_GIT, 'lora_r64': CONTINUATION_GIT}
    result['auxiliary_source_audit'] = {'path': str(output / 'base72_texture500/audit.json'),
        'note': 'Auxiliary S72 was audited while reusing the strict Texture500 loader; only S180 selections above are used.'}
    source.verify_locks(locks)
    if result['inference_git'] != inference_git or git_info() != inference_git:
        raise ValueError('Inference checkout changed during the wood180 source audit')
    (output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    return result
