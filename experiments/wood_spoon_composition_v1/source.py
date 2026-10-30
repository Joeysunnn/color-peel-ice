"""Read-only, selected-wood audits for paired S600, Material1000 and Texture1000."""

import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.natural_material_initializer_v1.source import build_config as initializer_config
from experiments.natural_texture_material_v1 import training_config as natural_config
from experiments.natural_texture_material_v1.training_source import verify_dataset
from experiments.sd15_token_local_kv_v1 import snapshots
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.texture_material_disentangled_v1 import step_snapshots, train as shared
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
SUBJECT_ROOT = Path('/var/tmp/colorpeel-r12user5/runs/subject_trellis_spoons600_v1/'
    '20261028-140359__blue_green72_both600__bb84ce1__42/wood_spoon')
SUBJECT_CHECKOUT = Path('/var/tmp/colorpeel-r12user5/checkouts/trellis-spoons72-600-v1')
NATURAL_ROOT = Path('/var/tmp/colorpeel-r12user5/runs/natural_texture_material_v1/'
    '20261018-190309__natural_tm_training__9077221__42')
MATERIAL_ROOT = Path('/var/tmp/colorpeel-r12user5/runs/natural_material_initializer_v1/'
    '20261019-095320__wood_material_init_material__9178429__42')
SUBJECT_GIT = {'commit': 'bb84ce132cb0db56d2966e7e73f5cc7a992a9120',
    'branch': 'codex/trellis-spoons72-600-v1'}
NATURAL_GIT = {'commit': '907722118502fecc1529000571c464b374224049', 'branch': 'codex/texture-material-v1'}
MATERIAL_GIT = {'commit': '91784296098cb81b6ef2b272d7c5fcf2fd3fddb5', 'branch': 'codex/texture-material-v1'}
METHODS = ('fullrank', 'lora_r64')


def verify_locks(locks):
    if not locks or any(not Path(path).is_absolute() or sha256(path) != digest
                        for path, digest in locks.items()):
        raise ValueError('A frozen composition source changed')


def _lock(locks, path, expected=None):
    path = Path(path).resolve()
    digest = sha256(path)
    if ((expected is not None and expected != digest)
            or (str(path) in locks and locks[str(path)] != digest)):
        raise ValueError(f'Source hash changed: {path}')
    locks[str(path)] = digest
    return digest


def _read(locks, path):
    _lock(locks, path)
    return read_json(path)


def _methods(rows, branch, token, steps):
    selected = [row for row in rows if row.get('branch') == branch
        and row.get('concept_id') == ('wood_spoon' if branch == 'subject' else 'wood_spoon_' + branch)]
    if (len(selected) != 2 or {row.get('method') for row in selected} != set(METHODS)
            or any(row.get('token') != token or row.get('steps') != steps
                   or row.get('status') != 'verified' for row in selected)):
        raise ValueError(f'Expected exactly two verified wood {branch} selections at step {steps}')
    return {row['method']: row for row in selected}


def _wood_material_plan(plan):
    source_id = plan.get('source_id')
    # The pinned wood-only9178429 plan predates the top-level source_id field.
    if 'source_id' not in plan and plan.get('git') == MATERIAL_GIT:
        source_id = 'wood_spoon'
    recipes = [row for row in natural_config.recipes() if row['concept_id'] == 'wood_spoon_material']
    return source_id == 'wood_spoon' and plan.get('recipes') == recipes


def verify_pair(parent, derived, step, directory, locks):
    """Verify receipts/hashes and actual paired tensors, without exporting any files."""
    import torch

    directory = Path(directory).resolve()
    token, method = parent['token'], parent['method']
    if (type(step) is not int or step != parent['steps']
            or any(derived.get(key) != parent.get(key) for key in
                   ('run', 'method', 'branch', 'concept_id', 'source_id', 'token', 'weight', 'status'))
            or derived.get('steps') != step or derived.get('step') != step
            or derived.get('training_steps') != parent['steps']
            or derived.get('source_checkpoint') != parent['checkpoint']
            or derived.get('checkpoint') != str(directory) or derived.get('path') != str(directory)
            or derived.get('source_hashes') != parent['hashes']
            or derived.get('paired_embedding_finite') is not True):
        raise ValueError('Derived receipt does not match the selected source/step')
    if _read(locks, directory / 'receipt.json') != derived:
        raise ValueError('Derived receipt differs from its source index')
    for path, digest in parent['hashes'].items():
        _lock(locks, path, digest)
    snapshot = Path(parent['checkpoint']) / f'checkpoint-{step}'
    snapshot_hashes = {str((snapshot / name).resolve()): _lock(locks, snapshot / name)
                       for name in ('pytorch_model.bin', 'pytorch_model_1.bin')}
    if (derived.get('source_snapshot_hashes') != snapshot_hashes
            or derived.get('source_adapter_sha256') != snapshot_hashes[str((snapshot / 'pytorch_model.bin').resolve())]
            or derived.get('source_text_encoder_sha256') != snapshot_hashes[str((snapshot / 'pytorch_model_1.bin').resolve())]):
        raise ValueError('Snapshot source hashes differ')
    required = {**parent['hashes'], **snapshot_hashes}
    required.update({str(directory / name): _lock(locks, directory / name)
                     for name in (parent['weight'], token + '.bin', 'adaptation_config.json')})
    if (derived.get('hashes') != required
            or derived.get('derived_adapter_sha256') != required[str(directory / parent['weight'])]
            or derived.get('derived_token_sha256') != required[str(directory / (token + '.bin'))]
            or _read(locks, directory / 'adaptation_config.json') !=
               _read(locks, Path(parent['checkpoint']) / 'adaptation_config.json')):
        raise ValueError('Derived payload hashes or adaptation configuration differ')
    if parent['branch'] == 'subject':
        token_id, _ = snapshots._embedding_audit(Path(parent['checkpoint']) / 'embedding_update_audit.json', step, token)
        adapter = torch.load(snapshot / 'pytorch_model.bin', map_location='cpu')
        table = torch.load(snapshot / 'pytorch_model_1.bin', map_location='cpu')[snapshots.EMBEDDING_KEY]
        if (not isinstance(table, torch.Tensor) or table.ndim != 2 or table.shape[1] != 768
                or not 0 <= token_id < table.shape[0]):
            raise ValueError('Subject snapshot embedding table differs')
        embedding = table[token_id].clone().contiguous()
        del table
    else:
        adapter, embedding, _, record = step_snapshots._paired(torch, parent, step)
        if record['source_snapshot_hashes'] != snapshot_hashes or derived.get('final_snapshot_matches_export') is not True:
            raise ValueError('Natural final snapshot provenance differs')
    shapes, empty = snapshots.adapter_schema(method)
    if not bool(torch.isfinite(embedding).all()):
        raise ValueError('Paired snapshot embedding is nonfinite')
    snapshots._check_state(torch, adapter, shapes, {}, 'source snapshot')
    exported = torch.load(directory / parent['weight'], map_location='cpu')
    snapshots._check_state(torch, exported, shapes, empty, 'derived')
    token_state = torch.load(directory / (token + '.bin'), map_location='cpu')
    if (not isinstance(token_state, dict) or set(token_state) != {token}
            or not torch.equal(embedding, token_state[token])
            or any(not torch.equal(adapter[key], exported[key]) for key in shapes)):
        raise ValueError('Derived adapter/token tensors differ from the paired snapshot')
    verify_locks(locks)
    return derived


def _references(locks, ledger, count):
    _lock(locks, ledger)
    rows = [json.loads(line) for line in Path(ledger).read_text(encoding='utf8').splitlines()]
    if len(rows) != count or len({row['id'] for row in rows}) != count:
        raise ValueError('Actual training reference count differs')
    for row in rows:
        for field in ('image', 'mask'):
            _lock(locks, row[field], row[field + '_sha256'])
    return [{**row, 'caption': row.get('caption', row.get('prompt'))} for row in rows]


def _subject(output, locks):
    # Execute the original strict Subject audit in its unchanged training checkout.
    program = '''import json,pathlib,sys
from experiments.subject_trellis_spoons600_v1 import training
from experiments.subject_marble_color30_v1 import inference
from scripts.launch.colorpeel_run import git_info
root,checkout=map(pathlib.Path,sys.argv[1:3])
assert pathlib.Path(training.__file__).resolve().is_relative_to(checkout)
assert pathlib.Path(inference.__file__).resolve().is_relative_to(checkout)
dataset=root/'training_dataset'
audit=training.verify_dataset(dataset)
assert audit['subject_id']=='wood_spoon' and audit['rows']==72
parents=inference._selections(root/'training_index.json',dataset,training_module=training)
print('WOOD_COMPOSITION_AUDIT='+json.dumps({'parents':parents,'git':git_info()}))
'''
    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
    environment.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-B', '-c', program, str(SUBJECT_ROOT), str(SUBJECT_CHECKOUT)],
        cwd=SUBJECT_CHECKOUT, env=environment, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (output / 'subject_training_audit.log').write_text(result.stdout, encoding='utf8')
    if result.returncode:
        raise ValueError('Frozen Subject training audit failed; see subject_training_audit.log')
    markers = [line.split('=', 1)[1] for line in result.stdout.splitlines() if line.startswith('WOOD_COMPOSITION_AUDIT=')]
    if len(markers) != 1:
        raise ValueError('Frozen Subject audit result missing')
    audit = json.loads(markers[0])
    if audit['git'] != SUBJECT_GIT:
        raise ValueError('Frozen Subject checkout revision changed')
    index = _read(locks, SUBJECT_ROOT / 'training_index.json')
    if index.get('git') != SUBJECT_GIT or index.get('runs') != audit['parents']:
        raise ValueError('Subject index differs from original strict audit')
    parents = _methods(audit['parents'], 'subject', '<S*>', 600)
    provenance = _read(locks, SUBJECT_ROOT / 'step600/provenance.json')
    if (provenance.get('status') != 'succeeded' or provenance.get('step') != 600
            or provenance.get('git') != SUBJECT_GIT or provenance.get('training') != audit['parents']
            or provenance.get('training_images') != 72 or provenance.get('generated_images') != 36):
        raise ValueError('Expected the completed original Subject600 inference')
    dataset = SUBJECT_ROOT / 'training_dataset'
    for name in ('dataset_audit.json', 'source_locks.json', 'approval.json', 'concepts.json'):
        _read(locks, dataset / name)
    for path, digest in _read(locks, dataset / 'source_locks.json').items():
        _lock(locks, path, digest)
    references = _references(locks, dataset / 'training_assets_manifest.jsonl', 72)
    sources = {}
    for method, parent in parents.items():
        directory = SUBJECT_ROOT / 'step600/checkpoints' / method
        derived = _read(locks, directory / 'receipt.json')
        sources[method] = verify_pair(parent, derived, 600, directory, locks)
    return sources, references


def prepare(output):
    output = resolve_run_dir(Path(output)).resolve()
    if output.exists() or output.is_relative_to(ROOT) or any(
            output.is_relative_to(path) or path.is_relative_to(output)
            for path in (SUBJECT_ROOT, NATURAL_ROOT, MATERIAL_ROOT)):
        raise ValueError('Use a fresh external audit directory separate from frozen sources')
    inference_git = git_info()
    output.mkdir(parents=True)
    locks = {}
    subject, subject_refs = _subject(output, locks)
    state = _read(locks, NATURAL_ROOT / 'status.json')
    if (state.get('status') != 'execution_verified' or state.get('git') != NATURAL_GIT
            or state.get('max_steps_per_run') != 1000):
        raise ValueError('Natural training completion/revision differs')
    dataset = Path(state['source_dataset'])
    protocol, _, dataset_locks = verify_dataset(dataset)
    if _read(locks, NATURAL_ROOT / 'protocol.json') != protocol:
        raise ValueError('Natural training protocol differs from the accepted dataset')
    for path, digest in dataset_locks.items():
        _lock(locks, path, digest)
    natural_index = _read(locks, NATURAL_ROOT / 'training/training_index.json')
    texture_index = _read(locks, NATURAL_ROOT / 'step1000/training_index.json')
    material_state = _read(locks, MATERIAL_ROOT / 'status.json')
    plan = _read(locks, MATERIAL_ROOT / 'plan.json')
    material_index = _read(locks, MATERIAL_ROOT / 'training/training_index.json')
    material_exports = _read(locks, MATERIAL_ROOT / 'exports/step1000/index.json')
    if (natural_index.get('git') != NATURAL_GIT or texture_index.get('git') != NATURAL_GIT
            or texture_index.get('training_steps') != 1000 or texture_index.get('source_training_steps') != 1000
            or material_state.get('status') != 'execution_verified' or material_state.get('git') != MATERIAL_GIT
            or plan.get('git') != MATERIAL_GIT or not _wood_material_plan(plan)
            or plan.get('initializer', {}).get('word') != 'material'
            or plan.get('source_training') != str(NATURAL_ROOT)
            or plan.get('protocol') != protocol
            or material_index.get('git') != MATERIAL_GIT or material_exports.get('git') != MATERIAL_GIT
            or material_exports.get('step') != 1000):
        raise ValueError('Natural/initializer source indices, initializer or revisions differ')
    # The historical ablation lock index also contains unrelated sources and steps.
    # Bind that index, but validate only selected weights below and dataset files above.
    _read(locks, MATERIAL_ROOT / 'source_locks.json')
    sources, references = {'subject': subject}, {'subject': subject_refs}
    for branch, parent_index, derived_index in (('material', material_index, material_exports),
                                               ('texture', natural_index, texture_index)):
        token = '<M*>' if branch == 'material' else '<T*>'
        parents = _methods(parent_index['runs'], branch, token, 1000)
        derived = _methods(derived_index['runs'], branch, token, 1000)
        references[branch] = _references(locks, dataset / 'staging' / ('wood_spoon_' + branch) /
            'training_assets_manifest.jsonl', 8 if branch == 'material' else 16)
        sources[branch] = {}
        for method, parent in parents.items():
            recipe = next(row for row in natural_config.recipes()
                          if row['concept_id'] == 'wood_spoon_' + branch and row['method'] == method)
            expected = natural_config.build_config(recipe, dataset / 'staging', protocol)
            if branch == 'material':
                expected = initializer_config(expected, method, 'wood_spoon', 'material')
            run = Path(parent['run'])
            expected_root = MATERIAL_ROOT if branch == 'material' else NATURAL_ROOT
            if run.parent != expected_root / 'training':
                raise ValueError('Selected training run belongs to another experiment')
            actual = shared.audit_run(recipe, run, expected,
                expected_git=MATERIAL_GIT if branch == 'material' else NATURAL_GIT)
            if actual != parent:
                raise ValueError('Selected natural training index differs from actual training audit')
            directory = (MATERIAL_ROOT / 'exports/step1000/checkpoints' / method if branch == 'material'
                         else NATURAL_ROOT / 'step1000/checkpoints' / (method + '__wood_spoon_texture'))
            sources[branch][method] = verify_pair(parent, derived[method], 1000, directory, locks)
    verify_locks(locks)
    if git_info() != inference_git:
        raise ValueError('Inference checkout changed during the source audit')
    result = {'sources': sources, 'source_locks': locks,
        'training_gits': {'subject': SUBJECT_GIT, 'material': MATERIAL_GIT, 'texture': NATURAL_GIT},
        'references': references, 'inference_git': inference_git}
    (output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    return result
