"""Frozen gray9 Color data and inherited Color recipes on SD2.1-base512."""

import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from experiments.mailbox_color_sd15_v1.training import build_config as color_config
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd21_wood_branches_v1.contract import MODEL_DIR

ROOT = Path(__file__).resolve().parents[2]
STUDY = 'sd21_color_gray9_v1'
BRANCHES, METHODS = ('color',), ('fullrank', 'lora_r64')
STEPS = (400, 600, 800, 1000)
TOKENS, INITIALIZERS = {'color': '<C*>'}, {'color': 'orange'}
CONCEPT_IDS = {'color': 'mailbox_original_orange_gray9'}
DATASET_ENV = 'COLORPEEL_SD21_COLOR_DATASET'
DEFAULT_OUTPUT_ROOT = Path('/home/r12user5/Documents/Jiawei/colorpeel-runs') / STUDY
SOURCE_ROOT = Path('/var/tmp/colorpeel-r12user5/runs/color_geometry_lightness_v1/'
                   '20261101-163802__orange9_gray128__9e81dcd__42')
SOURCE_INDEX_SHA = '53f156e5c7a8256d047712d6be285c592e0e06f9295f6f217f8553968473ec28'
SOURCE_NOTE = ('The user explicitly authorized gray9 Color training on SD2.1. Historical preview '
               'training_eligible=false remains unchanged. New authorization belongs to this run. '
               'Reuse RGB/captions/order without object-mask loss, rerendering or old learned weights.')


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf8')


def dataset_path(dataset=None):
    value = dataset if dataset is not None else os.environ.get(DATASET_ENV)
    if not value:
        raise ValueError(f'{DATASET_ENV} must identify this experiment staged dataset')
    return Path(value).resolve()


def authorization():
    return {'status': 'user_authorized_training', 'dataset': 'uniform_gray9',
            'branch': 'color', 'methods': list(METHODS), 'max_train_steps': 1000,
            'checkpoint_steps': list(STEPS), 'training_seed': 42,
            'transfer_seeds': list(range(30, 41)), 'object_mask_loss': False}


def source_snapshot():
    index = SOURCE_ROOT / 'files.json'
    if sha256(index) != SOURCE_INDEX_SHA:
        raise ValueError('Frozen gray9 source index changed')
    locks = {str(index.resolve()): SOURCE_INDEX_SHA}
    for name, digest in read_json(index).items():
        path = (SOURCE_ROOT / name).resolve()
        if not path.is_relative_to(SOURCE_ROOT.resolve()) or sha256(path) != digest:
            raise ValueError('Frozen gray9 source bytes or path changed')
        locks[str(path)] = digest
    rows = read_json(SOURCE_ROOT / 'manifest.json')
    expected = [(s, v) for s in ('cube', 'sphere', 'cylinder') for v in (0, 8, 16)]
    if len(rows) != 9 or [(r['shape'], r['view']) for r in rows] != expected:
        raise ValueError('Frozen gray9 source count/order changed')
    for row in rows:
        if row['caption'] != f"a photo of {row['shape']} shape in <C*> color":
            raise ValueError('Frozen Color caption changed')
        image_path = (SOURCE_ROOT / row['image']).resolve()
        mask_path = (SOURCE_ROOT / row['mask']).resolve()
        if (locks.get(str(image_path)) != row['image_sha256']
                or locks.get(str(mask_path)) != row['mask_sha256']):
            raise ValueError('Frozen gray9 manifest disagrees with file index')
        with Image.open(image_path) as image:
            if image.mode != 'RGB' or image.size != (512, 512):
                raise ValueError('Require actual512px RGB training images')
            rgb = np.asarray(image)
        mask = np.asarray(Image.open(mask_path))
        if mask.shape != (512, 512) or set(np.unique(mask)) != {0, 255} or not np.all(rgb[mask == 0] == 128):
            raise ValueError('Frozen gray background or source mask differs')
    return rows, locks


def stage_dataset(output):
    output = dataset_path(output)
    if output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(SOURCE_ROOT) or SOURCE_ROOT.is_relative_to(output):
        raise ValueError('Stage only into a fresh external dataset directory')
    sources, locks = source_snapshot()
    output.mkdir(parents=True)
    rows, concepts = [], []
    for source in sources:
        path = output / 'images' / source['id'] / 'image.png'
        path.parent.mkdir(parents=True)
        shutil.copyfile(SOURCE_ROOT / source['image'], path)
        rows.append({'id': source['id'], 'caption': source['caption'], 'shape': source['shape'],
                     'view': source['view'], 'image': str(path), 'image_sha256': source['image_sha256'],
                     'source_image': str(SOURCE_ROOT / source['image']), 'uses_object_mask': False})
        concepts.append({'instance_prompt': [source['caption']], 'instance_data_dir': str(path.parent)})
    _write(output / 'concepts.json', concepts)
    (output / 'training_assets_manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf8')
    _write(output / 'authorization.json', authorization())
    _write(output / 'source_locks.json', locks)
    names = ('concepts.json', 'training_assets_manifest.jsonl', 'authorization.json', 'source_locks.json')
    _write(output / 'dataset_audit.json', {'rows': 9, 'source': str(SOURCE_ROOT), 'source_index_sha256': SOURCE_INDEX_SHA,
        'files': {name: sha256(output / name) for name in names}, 'uses_masks': False,
        'source_note': SOURCE_NOTE, 'sampling': {'shuffle': False, 'complete_cycles': 111, 'extra_prefix_images': 1}})
    return verify_dataset('color', dataset=output)


def verify_dataset(branch, *, dataset=None):
    if branch != 'color':
        raise ValueError('Only Color is authorized in gray9')
    dataset = dataset_path(dataset)
    sources, locks = source_snapshot()
    audit = read_json(dataset / 'dataset_audit.json')
    if audit['rows'] != 9 or audit['source_index_sha256'] != SOURCE_INDEX_SHA or audit['uses_masks'] is not False:
        raise ValueError('Staged gray9 audit differs')
    if read_json(dataset / 'authorization.json') != authorization() or read_json(dataset / 'source_locks.json') != locks:
        raise ValueError('Staged gray9 authorization/source locks differ')
    for name, digest in audit['files'].items():
        path = (dataset / name).resolve()
        if not path.is_relative_to(dataset) or sha256(path) != digest:
            raise ValueError('Staged gray9 metadata changed')
        locks[str(path)] = digest
    locks[str(dataset / 'dataset_audit.json')] = sha256(dataset / 'dataset_audit.json')
    rows = [json.loads(line) for line in (dataset / 'training_assets_manifest.jsonl').read_text(encoding='utf8').splitlines()]
    if len(rows) != 9:
        raise ValueError('Staged gray9 row count differs')
    concepts = []
    for row, source in zip(rows, sources):
        path = dataset / 'images' / source['id'] / 'image.png'
        if (row['id'] != source['id'] or row['caption'] != source['caption'] or row['image'] != str(path)
                or row['image_sha256'] != source['image_sha256'] or sha256(path) != source['image_sha256']
                or row['uses_object_mask'] is not False or sorted(p.name for p in path.parent.iterdir()) != ['image.png']):
            raise ValueError('Staged gray9 RGB, caption, source order or image-only folder differs')
        locks[str(path)] = source['image_sha256']
        concepts.append({'instance_prompt': [source['caption']], 'instance_data_dir': str(path.parent)})
    if read_json(dataset / 'concepts.json') != concepts:
        raise ValueError('Gray9 concepts/image/caption alignment differs')
    return {'branch': branch, 'dataset': str(dataset), 'rows': 9, 'token': '<C*>', 'initializer': 'orange',
            'references': rows, 'source_locks': locks, 'source_metadata': audit, 'source_note': SOURCE_NOTE,
            'concepts': str(dataset / 'concepts.json'), 'manifest': str(dataset / 'training_assets_manifest.jsonl')}


def build_config(branch, method, model_dir=MODEL_DIR, smoke=False, *, dataset=None):
    if branch != 'color' or method not in METHODS or type(smoke) is not bool:
        raise ValueError('Select declared Color method and explicit smoke flag')
    dataset = dataset_path(dataset)
    config = color_config(method, dataset)
    config['status'] = 'authorized_sd21_color_gray9'
    config['run'].update(study=STUDY, variant=f'gray9_color_{method}_{"smoke2" if smoke else "steps1000"}')
    config['args'].update(pretrained_model_name_or_path=str(model_dir), max_train_steps=2 if smoke else 1000,
                          checkpoint_steps=[2] if smoke else list(STEPS))
    config['environment'][DATASET_ENV] = str(dataset)
    config['data_lineage'] = {'branch': branch, 'source': str(SOURCE_ROOT), 'dataset': str(dataset),
                            'source_note': SOURCE_NOTE, 'old_weights_reused': False}
    return config
