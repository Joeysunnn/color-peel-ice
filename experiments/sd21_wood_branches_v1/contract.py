"""Frozen wood S72/M8/T16 data and inherited recipes for the SD2.1-base comparison."""

import copy
import json
from pathlib import Path

from experiments.natural_material_initializer_v1.source import build_config as material_config
from experiments.natural_texture_material_v1 import dataset_protocol as natural_schema
from experiments.natural_texture_material_v1 import training_config as natural_training
from experiments.natural_texture_material_v1.training_source import verify_dataset as verify_natural
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.subject_trellis_spoons_v1 import training as subject_training
from experiments.subject_trellis_spoons600_v1.stage_training import verify_dataset as verify_subject

STUDY = 'sd21_wood_branches_v1'
STEPS = (400, 600, 800, 1000)
METHODS = ('fullrank', 'lora_r64')
BRANCHES = ('subject', 'material', 'texture')
MODEL_DIR = Path('/var/tmp/colorpeel-r12user5/assets/sd21_base_4e63672_common_checker')
DEFAULT_OUTPUT_ROOT = Path('/home/r12user5/Documents/Jiawei/colorpeel-runs/sd21_wood_branches_v1')
SUBJECT_DATASET = Path('/var/tmp/colorpeel-r12user5/runs/subject_trellis_spoons600_v1/'
    '20261028-140359__blue_green72_both600__bb84ce1__42/wood_spoon/training_dataset')
NATURAL_DATASET = Path('/var/tmp/colorpeel-r12user5/runs/natural_texture_material_v1/'
    '20261018-175054__counterfactual_dataset__c503bcd__42')
NATURAL_PROTOCOL_SHA256 = '2f0dbf61f0b969e17dc2c9beb4b65b4b0a4814ca2a62e38b6ac062c94d456127'
NATURAL_PROTOCOL_CANONICAL_SHA256 = '9746286e465a978f72b24ded443dc666d590d41e61f53b390d38a66481487259'
TOKENS = {'subject': '<S*>', 'material': '<M*>', 'texture': '<T*>'}
INITIALIZERS = {'subject': 'spoon', 'material': 'material', 'texture': 'texture'}
HASHES = {
    'subject': {'concepts.json': '3377416196b5d75af9f91cc86098046a11e9add47cfb6b5ee7fdf0962818fc9c',
        'training_assets_manifest.jsonl': '1c14f188ad3d38532678672b060e4d6c45b8c296ab1505df109d092ef33b8d08',
        'dataset_audit.json': '9c1f1afa0e9aca90b3c929946854c505f5143930c171c63b3d73deb66c246285'},
    'material': {'concepts.json': 'b0531402221e82b399a4eb14b974960ca3cb3dc8389108eb70a2295924a4b402',
        'training_assets_manifest.jsonl': '6a94bfa1537213e5530beeb4fdbc5e7958ee035edd5078cc040c6395f421735b',
        'dataset_receipt.json': '1dd2c91637996856e3c6958962ac47b9e53f01616e5b07fa7fc1ffdc5e4897a3'},
    'texture': {'concepts.json': '3798d81214b21253ac17083d39734f23fdd49b016127a7ea848c1fc7d2558e20',
        'training_assets_manifest.jsonl': 'cd189af5ebcf9373ebaa10bd110e45bc599c63a0dd188fc3d6f68f010e111ee4',
        'dataset_receipt.json': 'c9b121b4b597d61946c123206b79c3c4149507f2d329fc29b2ba1bcb9a59e650'}}
SOURCE_NOTE = ('Dataset protocols retain their historical SD1.5 model metadata as immutable data lineage only. '
    'New training starts from the separately recorded SD2.1-base512 model, without old adapters or resume. '
    'Subject images come from the600-step experiment; its historical dose/approval is not rewritten.')


def dataset_sources():
    result = {}
    for branch, count in (('subject', 72), ('material', 8), ('texture', 16)):
        dataset = SUBJECT_DATASET if branch == 'subject' else NATURAL_DATASET / 'staging' / ('wood_spoon_' + branch)
        result[branch] = {'branch': branch, 'dataset': str(dataset), 'rows': count,
            'concepts': str(dataset / 'concepts.json'), 'manifest': str(dataset / 'training_assets_manifest.jsonl'),
            'concepts_sha256': HASHES[branch]['concepts.json'],
            'manifest_sha256': HASHES[branch]['training_assets_manifest.jsonl'],
            'hashes': {str(dataset / name): digest for name, digest in HASHES[branch].items()},
            'token': TOKENS[branch], 'initializer': INITIALIZERS[branch]}
    return result


def _natural_protocol():
    path = NATURAL_DATASET / 'plan/protocol.json'
    if sha256(path) != NATURAL_PROTOCOL_SHA256:
        raise ValueError('Frozen natural dataset protocol bytes changed')
    protocol = read_json(path)
    if natural_schema.protocol_sha256(protocol) != NATURAL_PROTOCOL_CANONICAL_SHA256:
        raise ValueError('Frozen natural dataset protocol content changed')
    return protocol


def build_config(branch, method, model_dir=MODEL_DIR, smoke=False):
    if branch not in BRANCHES or method not in METHODS or type(smoke) is not bool:
        raise ValueError('Select a declared wood branch/method and explicit smoke flag')
    spec = dataset_sources()[branch]
    if branch == 'subject':
        config = subject_training.build_config(method, Path(spec['dataset']))
    else:
        recipe = next(row for row in natural_training.recipes()
                      if row['concept_id'] == 'wood_spoon_' + branch and row['method'] == method)
        config = natural_training.build_config(recipe, NATURAL_DATASET / 'staging', _natural_protocol())
        if branch == 'material':
            config = material_config(config, method, 'wood_spoon', 'material')
    config = copy.deepcopy(config)
    steps = 2 if smoke else 1000
    config['status'] = 'sd21_wood_branch_prepared'
    config['run'].update(study=STUDY, variant=f'wood_spoon_{branch}_{method}_{"smoke2" if smoke else "steps1000"}')
    config['args'].update(pretrained_model_name_or_path=str(model_dir), max_train_steps=steps,
                          checkpoint_steps=[2] if smoke else list(STEPS))
    config['data_lineage'] = {'branch': branch, 'source': spec, 'note': SOURCE_NOTE,
                            'data_reused_read_only': True, 'old_weights_reused': False}
    return config


def verify_dataset(branch):
    if branch not in BRANCHES:
        raise ValueError('Select subject, material or texture')
    spec = dataset_sources()[branch]
    dataset, locks = Path(spec['dataset']), dict(spec['hashes'])
    for path, digest in locks.items():
        if sha256(path) != digest:
            raise ValueError(f'Frozen {branch} dataset metadata bytes changed: {path}')
    if branch == 'subject':
        audit = verify_subject(dataset)
        if (audit.get('subject_id') != 'wood_spoon' or audit.get('rows') != 72
                or set(audit.get('selected_colors', [])) != {'blue', 'green'}):
            raise ValueError('Require the frozen wood-only blue/green72 source')
        for name in ('approval.json', 'source_locks.json'):
            path = dataset / name
            locks[str(path)] = sha256(path)
        locks.update(read_json(dataset / 'source_locks.json'))
    else:
        _natural_protocol()
        _, audit, natural_locks = verify_natural(NATURAL_DATASET)
        locks.update(natural_locks)
    if any(locks.get(path) != digest for path, digest in spec['hashes'].items()):
        raise ValueError('Upstream data locks disagree with the frozen selected dataset')
    rows = [json.loads(line) for line in Path(spec['manifest']).read_text(encoding='utf8').splitlines()]
    if len(rows) != spec['rows'] or len({row['id'] for row in rows}) != spec['rows']:
        raise ValueError('Frozen training rows/count/order differ')
    for row in rows:
        caption = row.get('caption', row.get('prompt'))
        if not isinstance(caption, str) or caption.count(TOKENS[branch]) != 1:
            raise ValueError('Frozen training caption token differs')
        for field in ('image', 'mask'):
            path = row[field]
            if not Path(path).is_absolute() or sha256(path) != row[field + '_sha256']:
                raise ValueError('Frozen training RGB or mask bytes changed')
            locks[path] = row[field + '_sha256']
    if any(sha256(path) != digest for path, digest in locks.items()):
        raise ValueError('Frozen source changed while validating the dataset')
    return {**spec, 'references': [{**row, 'caption': row.get('caption', row.get('prompt'))} for row in rows],
            'source_locks': locks, 'source_metadata': audit, 'source_note': SOURCE_NOTE}
