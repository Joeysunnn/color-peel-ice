"""Byte-copy the authorized 180 candidates into new Subject training staging."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from experiments.subject_trellis_dataset_v1.build import PROTOCOL, ROOT, rows, sha256, write

CANDIDATES = Path('/var/tmp/colorpeel-r12user5/runs/subject_trellis_dataset_v1/'
                  '20261027-192223__material180__0d7fce4__42/dataset')
MANIFEST_SHA = '49825606688d019563d16f148392cf34a7931c1ccbf48d866bbb7850e2df6bfc'
CONCEPTS_SHA = 'd77f11b75ae4d3969ff9f943dcd9096db9ba9367db325fd081ec01af7abc3f6c'
REQUEST = ('现在用这180张数据训练subject token，沿用之前original和rank64两组设置，各1000步，'
           '保存500/750/1000步并做相同的材质迁移测试。严格本地修改提交推送、服务器拉取运行，'
           '旧结果不要动，启动后不用监控。')
FILES = ('concepts.json', 'training_assets_manifest.jsonl', 'source_locks.json', 'approval.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def check_locks(locks):
    if not locks or any(not Path(path).is_file() or sha256(path) != digest for path, digest in locks.items()):
        raise ValueError('immutable candidate, provenance or training source changed')


def pair_arrays(image, mask):
    for path, mode in ((image, 'RGB'), (mask, 'L')):
        if path.is_symlink() or set(path.parent.iterdir()) != {path}:
            raise ValueError('require one image and one matching mask per directory')
        with Image.open(path) as handle:
            if handle.mode != mode or handle.size != (512, 512):
                raise ValueError('training image and mask must be aligned512 RGB/L')
    with Image.open(image) as handle, Image.open(mask) as region:
        rgb, binary = np.array(handle), np.array(region)
    if set(np.unique(binary)) != {0, 255} or not np.all(rgb[binary == 0] == 127):
        raise ValueError('require a nonempty binary mask and exact neutral exterior')


def source_snapshot(candidates):
    candidates = Path(candidates).resolve()
    if candidates != CANDIDATES.resolve():
        raise ValueError('only the authorized complete material180 run is accepted')
    root = candidates.parent
    locks = {str(candidates / 'manifest.json'): MANIFEST_SHA,
             str(candidates / 'concepts.json'): CONCEPTS_SHA,
             str(root / 'protocol.json'): sha256(PROTOCOL)}
    check_locks(locks)
    config, manifest = read(root / 'protocol.json'), read(candidates / 'manifest.json')
    records = manifest['records']
    expected = {row['id']: row for row in rows(config)}
    verification, status = read(root / 'verification.json'), read(root / 'repro_outputs/status.json')
    if (manifest['schema'] != config['schema'] or len(records) != 180 or len(expected) != 180
            or len({row['id'] for row in records}) != 180 or {row['id'] for row in records} != set(expected)
            or manifest.get('training_started') is not False or manifest.get('training_eligible_images') != 0
            or verification.get('status') != 'technical_checks_passed_scientific_review_pending'
            or [verification.get(key) for key in ('images', 'pair_groups', 'single_factor_pairs')] != [180, 30, 210]
            or verification.get('manifest_sha256') != MANIFEST_SHA or verification.get('concepts_sha256') != CONCEPTS_SHA
            or status.get('status') != 'dataset180_constructed_pending_user_review'
            or status.get('images_completed') != 180 or status.get('training_started') is not False
            or status.get('verification') != verification):
        raise ValueError('require the complete technically verified untrained source dataset')
    locks.update(manifest['source_hashes'])
    if (locks.get(config['asset']) != config['asset_sha256']
            or locks.get(config['camera_source']) != config['camera_source_sha256']):
        raise ValueError('native asset or source-camera provenance differs')
    for relative in ('verification.json', 'repro_outputs/status.json', 'runtime.json', 'cameras.json', 'dataset/pairs.json'):
        locks[str(root / relative)] = sha256(root / relative)
    concepts = []
    for row in records:
        if any(row.get(key) != value for key, value in expected[row['id']].items()):
            raise ValueError('source caption, controls or review flags differ from protocol')
        paths = {key: (candidates / row[key]).resolve() for key in ('image', 'mask')}
        for key, path in paths.items():
            if Path(row[key]).is_absolute() or not path.is_relative_to(candidates):
                raise ValueError('candidate path escapes its source dataset')
            locks[str(path)] = row[key + '_sha256']
        pair_arrays(paths['image'], paths['mask'])
        concepts.append({'instance_prompt': [row['caption']], 'instance_data_dir': str(paths['image'].parent),
                         'instance_mask_dir': str(paths['mask'].parent)})
    if read(candidates / 'concepts.json') != concepts:
        raise ValueError('source one-image one-caption pairing differs')
    check_locks(locks)
    return records, locks, list(config['colors'])


def training_row(source, dataset):
    row = {**source, 'training_eligible': True, 'review_status': 'user_authorized_training',
           'caption_status': 'user-approved parameter-derived supervision; physical accuracy unverified'}
    for key, folder in (('image', 'images'), ('mask', 'masks')):
        row[key] = str(dataset / 'assets' / source['id'] / folder / 'image.png')
        row['source_' + key] = str((CANDIDATES / source[key]).resolve())
        row['source_' + key + '_sha256'] = source[key + '_sha256']
    return row


def concepts_for(records):
    return [{'instance_prompt': [row['caption']], 'instance_data_dir': str(Path(row['image']).parent),
             'instance_mask_dir': str(Path(row['mask']).parent)} for row in records]


def audit_for(dataset, colors):
    return {'schema': 'trellis_subject180_training/v1', 'status': 'verified', 'rows': 180,
        'training_eligible_images': 180, 'candidates': str(CANDIDATES.resolve()),
        'source_mode': 'synthetic_native_geometry', 'selected_colors': colors,
        'files': {name: sha256(dataset / name) for name in FILES},
        'sampling': {'shuffle': False, 'updates': 1000, 'complete_cycles': 5, 'extra_prefix_images': 100,
                     'exposures': 'first100 rows six times; remaining80 rows five times; source order unchanged'},
        'limitations': 'Synthetic single-geometry supervision; training authorization is not proof of material accuracy or disentanglement.'}


def stage(candidates, output):
    candidates, output = Path(candidates).resolve(), Path(output).resolve()
    if (output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(candidates.parent)
            or candidates.is_relative_to(output)):
        raise ValueError('stage only into a new external directory, preserving all source results')
    sources, locks, colors = source_snapshot(candidates)
    output.mkdir(parents=True)
    records = [training_row(source, output) for source in sources]
    for row in records:
        for key in ('image', 'mask'):
            path = Path(row[key])
            path.parent.mkdir(parents=True)
            shutil.copyfile(row['source_' + key], path)
    write(output / 'concepts.json', concepts_for(records))
    (output / 'training_assets_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf8')
    write(output / 'source_locks.json', locks)
    write(output / 'approval.json', {'request': REQUEST, 'scope': 'all180 images, original/fullrank and rank64,1000 updates each',
        'selected_colors': colors, 'checkpoint_steps': [500, 750, 1000], 'monitoring': False,
        'not_claimed': 'No per-image physical-material or learned-disentanglement validation is implied.'})
    write(output / 'dataset_audit.json', audit_for(output, colors))
    return verify_dataset(output)


def verify_dataset(dataset):
    dataset = Path(dataset).resolve()
    sources, locks, colors = source_snapshot(CANDIDATES)
    records = [json.loads(line) for line in (dataset / 'training_assets_manifest.jsonl').read_text(encoding='utf8').splitlines()]
    expected = [training_row(row, dataset) for row in sources]
    if (records != expected or read(dataset / 'source_locks.json') != locks
            or read(dataset / 'concepts.json') != concepts_for(expected)
            or read(dataset / 'approval.json').get('request') != REQUEST):
        raise ValueError('staged source order, image/mask pairing, caption or approval changed')
    for row in records:
        if row['caption'].count('<S*>') != 1:
            raise ValueError('each Subject caption must contain exactly one token')
        for key in ('image', 'mask'):
            path = Path(row[key])
            if not path.is_relative_to(dataset) or sha256(path) != row[key + '_sha256']:
                raise ValueError('training copy differs from its immutable source hash')
        pair_arrays(Path(row['image']), Path(row['mask']))
    audit = audit_for(dataset, colors)
    if read(dataset / 'dataset_audit.json') != audit:
        raise ValueError('training dataset audit changed')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(stage(args.candidates, args.output)))
