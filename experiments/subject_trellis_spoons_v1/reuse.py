"""Read-only reuse of the two completed fullrank runs at their recorded revision."""

from pathlib import Path

from experiments.sd15_token_local_kv_v1.snapshots import STATE_FILES
from experiments.subject_multicolor_multimaterial_b.inference import _locks
from experiments.subject_trellis_spoons_v1 import training
from experiments.texture_material_disentangled_v1 import train as shared
from scripts.launch.colorpeel_run import git_info, read_config

SOURCE_ROOT = Path('/var/tmp/colorpeel-r12user5/runs/subject_trellis_spoons_training_v1/'
    '20261028-094541__four_comparisons__d2e6239__42')
OLD_GIT = {'commit': 'd2e6239c12f54465b33759558129bf8813ad7c16',
    'branch': 'codex/trellis-spoons-training-v1'}


def completed_fullrank(dataset, source_dir):
    dataset, source_dir = Path(dataset).resolve(), Path(source_dir).resolve()
    audit = training.verify_dataset(dataset)
    subject = audit['subject_id']
    if (subject not in ('metal_spoon', 'wood_spoon') or audit['rows'] != 180
            or dataset != (SOURCE_ROOT / subject / 'all180/training_dataset').resolve()
            or source_dir != (SOURCE_ROOT / subject / 'all180/training/fullrank').resolve()):
        raise ValueError('reuse only the completed fullrank of the same frozen spoon180 experiment')
    status_path, index_path = source_dir / 'status.json', source_dir / 'training_index.json'
    status, index = training.read_json(status_path), training.read_json(index_path)
    selections = index.get('runs', [])
    if (status.get('status') != 'training_verified' or status.get('completed_steps') != 1000
            or status.get('method') != 'fullrank' or status.get('git') != OLD_GIT
            or index.get('git') != OLD_GIT or len(selections) != 1):
        raise ValueError('require an audited completed fullrank at the fixed original revision')
    selection = selections[0]
    run = Path(selection['run']).resolve()
    if (run.parent != source_dir / 'runs' or Path(selection['checkpoint']).resolve() != run / 'checkpoints'
            or any(selection.get(key) != value for key, value in {
                'status': 'verified', 'method': 'fullrank', 'branch': 'subject',
                'concept_id': subject, 'token': '<S*>', 'steps': 1000}.items())):
        raise ValueError('completed fullrank path, identity or method differs')
    expected = training.build_config('fullrank', dataset)
    if read_config(run / 'config.yaml') != expected:
        raise ValueError('completed fullrank config differs from the spoon recipe')
    _locks(selection.get('hashes', {}))
    recipe = {'method': 'fullrank', 'branch': 'subject', 'concept_id': subject, 'token': '<S*>'}
    if shared.audit_run(recipe, run, expected, expected_git=OLD_GIT) != selection:
        raise ValueError('completed fullrank index differs from its actual audit')
    snapshots = [run / 'checkpoints' / f'checkpoint-{step}' / name
                 for step in (500, 750, 1000) for name in STATE_FILES]
    if any(not path.is_file() or path.is_symlink() or not path.stat().st_size for path in snapshots):
        raise ValueError('completed fullrank has a missing or empty paired snapshot')
    locks = {**selection['hashes'], **{str(path): shared.sha256(path)
             for path in (status_path, index_path, *snapshots)}}
    _locks(locks)
    return selection, {'subject_id': subject, 'dataset': str(dataset), 'source_dir': str(source_dir),
        'source_git': OLD_GIT.copy(), 'locks': locks, 'retrained': False}


def selection_loader(index_path, dataset):
    """Audit each method against its real source revision, without changing old files."""
    index, current = training.read_json(index_path), git_info()
    runs = index.get('runs', [])
    if (index.get('git') != current or len(runs) != 2
            or {run.get('method') for run in runs} != set(training.METHODS)
            or index.get('source_gits') != {'fullrank': OLD_GIT, 'lora_r64': current}):
        raise ValueError('require two explicit methods and their actual source revisions')
    receipt = index.get('reuse_receipt', {})
    _locks(receipt.get('locks', {}))
    fullrank, expected_receipt = completed_fullrank(dataset, receipt.get('source_dir', ''))
    if receipt != expected_receipt:
        raise ValueError('fullrank reuse receipt differs from its immutable source')
    result = []
    for selection in runs:
        if selection['method'] == 'fullrank':
            if selection != fullrank:
                raise ValueError('combined fullrank selection differs from the completed source')
            result.append(fullrank)
            continue
        expected = training.build_config('lora_r64', dataset)
        recipe = {'method': 'lora_r64', 'branch': 'subject',
            'concept_id': expected['subject_id'], 'token': '<S*>'}
        _locks(selection.get('hashes', {}))
        if shared.audit_run(recipe, Path(selection['run']), expected) != selection:
            raise ValueError('rank64 selection differs from its current-revision audit')
        result.append(selection)
    return result
