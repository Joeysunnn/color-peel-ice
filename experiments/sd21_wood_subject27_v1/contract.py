"""Train the approved frozen27 images with the unchanged SD2.1 Subject recipe."""

import copy
from pathlib import Path

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd21_wood_branches_v1 import contract as previous
from experiments.subject_trellis_wood27_v1 import prepare

STUDY = 'sd21_wood_subject27_v1'
BRANCHES = ('subject',)
METHODS, TOKENS, STEPS, MODEL_DIR = previous.METHODS, previous.TOKENS, previous.STEPS, previous.MODEL_DIR
INITIALIZERS = previous.INITIALIZERS
DEFAULT_OUTPUT_ROOT = Path('/var/tmp/colorpeel-r12user5/runs') / STUDY
SOURCE_ROOT = Path('/var/tmp/colorpeel-r12user5/runs/subject_trellis_wood27_v1/'
    '20261101-113822__subject_trellis_wood27_v1__dataset27__08756e7__42')
DATASET = SOURCE_ROOT / 'dataset'
AUDIT_SHA256 = '787c75f0709bc0a418e0e654077388231e1d3cdb2fa47674cb771f286b9c8da0'
CONCEPTS_SHA256 = 'e2d74fedf6d9d95e4041684fe2593771c4946fc33d835b68f766fbb5742455d1'
MANIFEST_SHA256 = '4f53aa3cd6b98096ca1737890e99a354b2f213c1e37f3db1c5bb0948f60ea58f'
SOURCE_NOTE = ('User approved these27 images for this new training run. Original dataset training_eligible=False '
    'and historical dataset-only status remain unchanged; the new run records training authorization separately. '
    'Reuse24 studio endpoint renders plus original/LCh-blue/LCh-green photos read-only, in their frozen order. '
    'Keep the previous SD2.1 Subject method, initialization, losses and all other science arguments.')


def dataset_sources():
    return {'subject': {'branch': 'subject', 'dataset': str(DATASET), 'rows': 27,
        'concepts': str(DATASET / 'concepts.json'), 'manifest': str(DATASET / 'training_assets_manifest.jsonl'),
        'concepts_sha256': CONCEPTS_SHA256, 'manifest_sha256': MANIFEST_SHA256,
        'hashes': {str(SOURCE_ROOT / 'dataset_audit.json'): AUDIT_SHA256,
            str(DATASET / 'concepts.json'): CONCEPTS_SHA256,
            str(DATASET / 'training_assets_manifest.jsonl'): MANIFEST_SHA256},
        'token': '<S*>', 'initializer': 'spoon'}}


def build_config(branch, method, model_dir=MODEL_DIR, smoke=False):
    if branch != 'subject':
        raise ValueError('The27-image experiment trains only the Subject branch')
    config = copy.deepcopy(previous.build_config('subject', method, model_dir=model_dir, smoke=smoke))
    config['args']['concepts_list'] = str(DATASET / 'concepts.json')
    config['data_manifest'] = str(DATASET / 'training_assets_manifest.jsonl')
    config['status'] = 'sd21_wood_subject27_prepared'
    config['run'].update(study=STUDY,
        variant=f'wood_spoon_subject27_{method}_{"smoke2" if smoke else "steps1000"}')
    config['data_lineage'] = {'branch': branch, 'source': dataset_sources()['subject'],
        'note': SOURCE_NOTE, 'data_reused_read_only': True, 'old_weights_reused': False}
    return config


def verify_dataset(branch):
    if branch != 'subject':
        raise ValueError('The27-image experiment accepts only the Subject dataset')
    spec = dataset_sources()['subject']
    locks = dict(spec['hashes'])
    for path, digest in locks.items():
        if sha256(path) != digest:
            raise ValueError('Frozen27 dataset metadata bytes changed')
    audit = read_json(SOURCE_ROOT / 'dataset_audit.json')
    if (audit.get('status') != 'dataset27_constructed_pending_user_review'
            or audit.get('rows') != 27 or audit.get('synthetic_rows') != 24 or audit.get('photo_rows') != 3
            or audit.get('synthetic_copies_byte_identical') is not True
            or audit.get('training_started') is not False or audit.get('monitoring') is not False
            or len(audit.get('files', {})) != 69):
        raise ValueError('Expected the frozen27 dataset-only receipt and69 output files')
    for relative, digest in audit['files'].items():
        relative = Path(relative)
        path = (SOURCE_ROOT / relative).resolve()
        if (relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(SOURCE_ROOT.resolve())
                or sha256(path) != digest):
            raise ValueError('Frozen27 dataset output path or bytes changed')
        locks[str(path)] = digest
    upstream = read_json(SOURCE_ROOT / 'source_locks.json')
    if not upstream or upstream != audit.get('source_locks'):
        raise ValueError('Original27 upstream source locks changed')
    for path, digest in upstream.items():
        if not Path(path).is_absolute() or (path in locks and locks[path] != digest) or sha256(path) != digest:
            raise ValueError('Original27 upstream source bytes changed')
        locks[path] = digest
    if any(locks.get(path) != digest for path, digest in spec['hashes'].items()):
        raise ValueError('Dataset-only receipt disagrees with the frozen27 source hashes')
    rows = prepare.verify_output(SOURCE_ROOT)
    if any(sha256(path) != digest for path, digest in locks.items()):
        raise ValueError('Frozen27 sources changed during verification')
    return {**spec, 'references': rows, 'source_locks': locks,
            'source_metadata': audit, 'source_note': SOURCE_NOTE}
