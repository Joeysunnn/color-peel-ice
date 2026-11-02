"""The eleven reviewed center-view Mailbox inputs and unchanged SD2.1 recipe."""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd21_wood_branches_v1.contract import MODEL_DIR
from experiments.subject_multicolor_multimaterial_b import training as baseline
from experiments.subject_trellis_wood27_v1.prepare import stage_row

ROOT = Path(__file__).resolve().parents[2]
STUDY = 'sd21_mailbox_center11_v1'
METHODS, BRANCHES, STEPS = ('fullrank', 'lora_r64'), ('subject',), (500, 1000, 1500)
TOKENS, INITIALIZERS, CONCEPT_IDS = {'subject': '<S*>'}, {'subject': 'mailbox'}, {'subject': 'mailbox'}
DATASET_ENV = 'COLORPEEL_MAILBOX_CENTER11_DATASET'
SOURCE_ROOT = Path('/var/tmp/colorpeel-r12user5/runs/sd21_mailbox_subject27_v1/'
                   '20261101-221114__both600__a513371__42/experiment/dataset')
APPROVED_PATH = Path(__file__).with_name('approved11.json')


def dataset_path():
    return Path(os.environ[DATASET_ENV]).resolve()


def check_locks(locks):
    if not locks or any(sha256(path) != digest for path, digest in locks.items()):
        raise ValueError('reviewed input or staged dataset changed')


def build_config(branch, method, model_dir=MODEL_DIR, smoke=False):
    if branch != 'subject' or method not in METHODS or smoke:
        raise ValueError('require center11 Subject fullrank/rank64 formal1500 recipe')
    dataset = dataset_path()
    config = baseline.build_config(method, dataset)
    config.update(status='sd21_mailbox_center11_authorized', subject_id='mailbox')
    config['run'].update(study=STUDY, variant=f'mailbox_center11_{method}_steps1500')
    config['args'].update(pretrained_model_name_or_path=str(model_dir), max_train_steps=1500,
                          checkpoint_steps=list(STEPS))
    config['environment'][DATASET_ENV] = str(dataset)
    config['data_lineage'] = {'dataset': str(dataset), 'source': str(SOURCE_ROOT),
        'rows': 11, 'synthetic_rows': 8, 'photo_rows': 3, 'old_weights_reused': False,
        'approved_inputs_sha256': sha256(APPROVED_PATH), 'source_metadata_unchanged': True}
    return config


def concepts(rows):
    return [{'instance_prompt': [row['caption']], 'instance_data_dir': str(Path(row['image']).parent),
             'instance_mask_dir': str(Path(row['mask']).parent)} for row in rows]


def stage_dataset(dataset):
    dataset = Path(dataset).resolve()
    if dataset.exists() or dataset.name != 'dataset' or dataset.is_relative_to(ROOT) or dataset.is_relative_to(SOURCE_ROOT):
        raise ValueError('require fresh external experiment/dataset')
    source_manifest = SOURCE_ROOT / 'training_assets_manifest.jsonl'
    source = [json.loads(line) for line in source_manifest.read_text(encoding='utf8').splitlines()]
    approved = read_json(APPROVED_PATH)
    locks = {str(source_manifest): sha256(source_manifest), str(APPROVED_PATH): sha256(APPROVED_PATH)}
    staged = []
    for expected in approved:
        matches = [row for row in source if row['id'] == expected['id']]
        if len(matches) != 1 or any(matches[0].get(key) != value for key, value in expected.items()):
            raise ValueError('source differs from the reviewed eleven images/captions')
        row = matches[0]
        for field in ('image', 'mask'):
            path = Path(row[field]).resolve()
            if not path.is_relative_to(SOURCE_ROOT / 'assets'):
                raise ValueError('source image/mask outside the frozen parent dataset')
            locks[str(path)] = row[field + '_sha256']
        check_locks(locks)
        staged.append(stage_row(row, dataset.parent, image_source=row['image'], mask_source=row['mask']))
    (dataset / 'training_assets_manifest.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in staged), encoding='utf8')
    baseline._write(dataset / 'concepts.json', concepts(staged))
    baseline._write(dataset / 'source_locks.json', locks)
    baseline._write(dataset / 'dataset_audit.json', {'rows': 11, 'synthetic_rows': 8, 'photo_rows': 3,
        'subject_id': 'mailbox', 'source_locks': locks,
        'files': {str(p): sha256(p) for p in dataset.rglob('*') if p.is_file()},
        'source_metadata_unchanged': True, 'copies_byte_identical': True})
    return verify_dataset('subject')


def verify_dataset(branch):
    if branch != 'subject':
        raise ValueError('only Subject is authorized')
    dataset = dataset_path()
    audit = read_json(dataset / 'dataset_audit.json')
    if (audit['rows'], audit['synthetic_rows'], audit['photo_rows'], audit['subject_id']) != (11, 8, 3, 'mailbox'):
        raise ValueError('invalid center11 dataset counts')
    locks = {**audit['source_locks'], **audit['files'],
             str(dataset / 'dataset_audit.json'): sha256(dataset / 'dataset_audit.json')}
    check_locks(locks)
    rows = [json.loads(line) for line in (dataset / 'training_assets_manifest.jsonl').read_text(encoding='utf8').splitlines()]
    approved = read_json(APPROVED_PATH)
    if len(rows) != 11 or len(approved) != 11:
        raise ValueError('require exactly eleven approved rows')
    for row, expected in zip(rows, approved):
        if any(row.get(key) != value for key, value in expected.items()) or row['caption'].count('<S*>') != 1:
            raise ValueError('reviewed order, caption or hash differs')
        for field, mode in (('image', 'RGB'), ('mask', 'L')):
            path = Path(row[field])
            if (not path.is_relative_to(dataset / 'assets') or sha256(path) != row[field + '_sha256']
                    or set(path.parent.iterdir()) != {path}):
                raise ValueError('require the unchanged one-image/one-mask concept')
            with Image.open(path) as image:
                if image.mode != mode or image.size != (512, 512):
                    raise ValueError('require original512 RGB and L mask')
                if field == 'mask' and set(np.unique(np.asarray(image))) != {0, 255}:
                    raise ValueError('require nonempty binary object mask')
    if read_json(dataset / 'concepts.json') != concepts(rows):
        raise ValueError('one-image one-caption pairing differs')
    return {'branch': branch, 'dataset': str(dataset), 'rows': 11, 'references': rows,
        'concepts': str(dataset / 'concepts.json'), 'manifest': str(dataset / 'training_assets_manifest.jsonl'),
        'source_locks': locks, 'token': '<S*>', 'initializer': 'mailbox'}
