"""Frozen Mailbox sources and the inherited SD2.1 Subject training recipe."""

import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd21_wood_branches_v1.contract import MODEL_DIR
from experiments.subject_multicolor_multimaterial_b import training as baseline
from experiments.subject_trellis_wood27_v1.prepare import select24, stage_row, verify_output

ROOT = Path(__file__).resolve().parents[2]
STUDY = 'sd21_mailbox_subject27_v1'
METHODS, BRANCHES, STEPS = ('fullrank', 'lora_r64'), ('subject',), (200, 400, 600)
TOKENS, INITIALIZERS, CONCEPT_IDS = {'subject': '<S*>'}, {'subject': 'mailbox'}, {'subject': 'mailbox'}
DATASET_ENV = 'COLORPEEL_MAILBOX27_DATASET'
SOURCE_ROOT = Path('/var/tmp/colorpeel-r12user5/runs/mailbox_source_framing72_v1/'
    '20261101-214327__threeviews72__d501491__42/result')
PHOTO_ROOT = Path('/home/r12user5/Documents/Jiawei/colorpeel-runs/'
    '20260916-163006__natural_image_subject_color_pilot__subject_recolor_mailbox_training_assets__bce4364__42/category')
ORIGINAL = ROOT / 'experiments/natural_material_supermat_v1/assets/source_mailbox.jpg'
SOURCE_HASHES = {
    str(SOURCE_ROOT / 'dataset_status.json'): '7c2b0ecad1dbeaec8a30254ac9901b7f65feba34f31016b804750d02f3d33260',
    str(SOURCE_ROOT / 'dataset/manifest.json'): '0c63047abad5039429ab5f73d79aa399b302536ad408fa37e782b5e780510ceb',
    str(ORIGINAL): 'ba9c6624d9998cdfcd16308b0cf8fd8f3778909d8655efe2ba9cebba370483d0',
    str(PHOTO_ROOT / 'blue/images/image.png'): '5475c02c3713ce6353bd262f584f16105d24c23285ec1312eb427fb87edd4dc5',
    str(PHOTO_ROOT / 'green/images/image.png'): '15b93e4e49502870daa3b71f524220943ccfaf2114f810c54da827e0a210aed0',
    str(PHOTO_ROOT / 'blue/masks/image.png'): 'b4d119d110ec356fca9daa9003c8505ba04f7ee694d0d8785e9c5281f63a8cf7',
    str(PHOTO_ROOT / 'green/masks/image.png'): 'b4d119d110ec356fca9daa9003c8505ba04f7ee694d0d8785e9c5281f63a8cf7',
}


def dataset_path():
    return Path(os.environ[DATASET_ENV]).resolve()


def check_locks(locks):
    if not locks or any(sha256(path) != digest for path, digest in locks.items()):
        raise ValueError('frozen input or staged dataset changed')


def build_config(branch, method, model_dir=MODEL_DIR, smoke=False):
    if branch != 'subject' or method not in METHODS or smoke:
        raise ValueError('require Mailbox subject fullrank/rank64 formal600 recipe')
    dataset = dataset_path()
    config = baseline.build_config(method, dataset)
    config.update(status='sd21_mailbox_subject27_authorized', subject_id='mailbox')
    config['run'].update(study=STUDY, variant=f'mailbox27_{method}_steps600')
    config['args'].update(pretrained_model_name_or_path=str(model_dir), max_train_steps=600,
                          checkpoint_steps=list(STEPS))
    config['environment'][DATASET_ENV] = str(dataset)
    config['data_lineage'] = {'dataset': str(dataset), 'synthetic_source': str(SOURCE_ROOT),
        'rows': 27, 'synthetic_rows': 24, 'photo_rows': 3, 'old_weights_reused': False,
        'source_metadata_unchanged': True}
    return config


def stage_dataset(dataset):
    dataset = Path(dataset).resolve()
    if dataset.exists() or dataset.is_relative_to(SOURCE_ROOT) or dataset.is_relative_to(ROOT):
        raise ValueError('require fresh external staged dataset')
    root = dataset.parent
    if dataset.name != 'dataset':
        raise ValueError('dataset must be experiment/dataset')
    check_locks(SOURCE_HASHES)
    source_status = read_json(SOURCE_ROOT / 'dataset_status.json')
    manifest = read_json(SOURCE_ROOT / 'dataset/manifest.json')
    if (source_status['status'] != 'completed_pending_review' or source_status['images_completed'] != 72
            or source_status['native_inputs_unchanged'] is not True or len(manifest['records']) != 72):
        raise ValueError('require completed source-framed72')
    selected = select24(manifest['records'])
    locks = dict(SOURCE_HASHES)
    staged = []
    for row in selected:
        image, mask = [SOURCE_ROOT / 'dataset' / row[key] for key in ('image', 'mask')]
        locks.update({str(image): row['image_sha256'], str(mask): row['mask_sha256']})
        check_locks({str(image): row['image_sha256'], str(mask): row['mask_sha256']})
        staged.append(stage_row({**row, 'subject_id': 'mailbox'}, root, image_source=image, mask_source=mask))
    photo_mask = PHOTO_ROOT / 'blue/masks/image.png'
    source_folder = root / 'sources'
    source_folder.mkdir()
    shutil.copyfile(ORIGINAL, source_folder / 'original_mailbox.jpg')
    original_rgb = np.asarray(Image.open(ORIGINAL).convert('RGB'))
    mask_rgb = np.asarray(Image.open(photo_mask).convert('L'))
    if original_rgb.shape != (512, 512, 3) or mask_rgb.shape != (512, 512):
        raise ValueError('photo and paired masks must keep original512 canvas')
    for color in ('orange', 'blue', 'green'):
        original = color == 'orange'
        row = {'id': 'photo_original' if original else f'photo_lch_{color}',
            'subject_id': 'mailbox', 'color': color, 'view': 'original_photo', 'light': 'original_photo',
            'controls': None, 'kind': 'original_photograph' if original else 'historical_lch_recolor',
            'caption': f'a photo of <S*> mailbox in {color} color made of glossy painted metal',
            'mask_sha256': SOURCE_HASHES[str(photo_mask)]}
        if original:
            item = stage_row(row, root, image=original_rgb, mask_source=photo_mask)
            if not np.array_equal(np.asarray(Image.open(item['image'])), original_rgb):
                raise ValueError('JPEG decoded pixels changed during PNG conversion')
            item['parent_image'] = str(ORIGINAL)
        else:
            image = PHOTO_ROOT / color / 'images/image.png'
            row['image_sha256'] = SOURCE_HASHES[str(image)]
            item = stage_row(row, root, image_source=image, mask_source=photo_mask)
        staged.append(item)
    ledger = dataset / 'training_assets_manifest.jsonl'
    ledger.write_text(''.join(json.dumps(row) + '\n' for row in staged), encoding='utf8')
    baseline._write(dataset / 'concepts.json', [{'instance_prompt': [r['caption']],
        'instance_data_dir': str(Path(r['image']).parent), 'instance_mask_dir': str(Path(r['mask']).parent)} for r in staged])
    baseline._write(dataset / 'source_locks.json', locks)
    verify_output(root)
    check_locks(locks)
    files = {str(p): sha256(p) for p in dataset.rglob('*') if p.is_file()}
    baseline._write(dataset / 'dataset_audit.json', {'rows': 27, 'synthetic_rows': 24, 'photo_rows': 3,
        'subject_id': 'mailbox', 'source_locks': locks, 'files': files,
        'original_rgb_preserved': True, 'synthetic_and_recolor_copies_byte_identical': True,
        'photo_mask_note': 'Use historical paired photo mask b4d119d... for all3 photos; distinct from TRELLIS source mask550c1e13....',
        'source_flags_unchanged': True})
    return verify_dataset('subject')


def verify_dataset(branch):
    if branch != 'subject':
        raise ValueError('only Subject is authorized')
    dataset = dataset_path()
    audit = read_json(dataset / 'dataset_audit.json')
    if (audit['rows'], audit['synthetic_rows'], audit['photo_rows'], audit['subject_id']) != (27, 24, 3, 'mailbox'):
        raise ValueError('invalid staged dataset counts')
    locks = {**audit['source_locks'], **audit['files'],
             str(dataset / 'dataset_audit.json'): sha256(dataset / 'dataset_audit.json')}
    if any(audit['source_locks'].get(path) != digest for path, digest in SOURCE_HASHES.items()):
        raise ValueError('source locks differ from frozen protocol')
    check_locks(locks)
    references = verify_output(dataset.parent)
    if any(row['subject_id'] != 'mailbox' for row in references):
        raise ValueError('wrong subject')
    return {'branch': 'subject', 'dataset': str(dataset), 'rows': 27, 'references': references,
        'concepts': str(dataset / 'concepts.json'), 'manifest': str(dataset / 'training_assets_manifest.jsonl'),
        'source_locks': locks, 'token': '<S*>', 'initializer': 'mailbox'}
