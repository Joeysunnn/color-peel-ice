"""Verify rendered candidates and record matched, single-factor M/R pairs."""

import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.subject_trellis_dataset_v1.build import PROTOCOL, array_hash, rows, sha256

SPLIT_POLICY = ('No random image split: all samples share one generated geometry and matched '
                'rendering conditions. These candidates do not provide independent-source generalization evidence.')


def checked_file(root, relative, digest=None):
    root, relative = Path(root).resolve(), Path(relative)
    path = root / relative
    if relative.is_absolute() or not path.resolve().is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise ValueError('missing artifact or path outside dataset')
    if digest is not None and sha256(path) != digest:
        raise ValueError('artifact hash mismatch')
    return path


def read_image(path, mode):
    with Image.open(path) as image:
        if image.mode != mode or image.size != (512, 512):
            raise ValueError('require aligned 512px RGB/L/RGBA artifacts')
        return np.array(image)


def verify_record(output, row, config, geometry):
    image = checked_file(output / 'dataset', row['image'], row['image_sha256'])
    mask_file = checked_file(output / 'dataset', row['mask'], row['mask_sha256'])
    if any(set(path.parent.iterdir()) != {path} for path in (image, mask_file)):
        raise ValueError('image and mask directories must each contain one file')
    rgb, mask = read_image(image, 'RGB'), read_image(mask_file, 'L')
    if set(np.unique(mask)) != {0, 255}:
        raise ValueError('require a nonempty binary mask with background')
    inside = mask == 255
    signatures = {'mask': array_hash(inside), 'normal': array_hash(geometry['normal'])}
    if not np.array_equal(inside, geometry['mask']) or row['geometry_buffer_sha256'] != signatures:
        raise ValueError('image mask or geometry buffer differs within the view')
    raw = read_image(checked_file(output, row['raw_image'], row['raw_sha256']), 'RGB')
    if not np.array_equal(rgb[inside], raw[inside]) or not np.all(rgb[~inside] == 127) or not np.all(raw[~inside] == 0):
        raise ValueError('training RGB must retain raw foreground and neutral exterior')
    cutout = read_image(checked_file(output, row['foreground_rgba']), 'RGBA')
    if (not np.array_equal(cutout[..., 3], mask) or not np.array_equal(cutout[inside, :3], raw[inside])
            or not np.all(cutout[~inside, :3] == 0)):
        raise ValueError('RGBA pairing or transparent hidden RGB differs')
    map_file = checked_file(output, row['maps'], row['maps_sha256'])
    targets = {'base_color': np.asarray(config['colors'][row['color']])[:, None],
               'metallic': row['controls']['metallic'], 'roughness': row['controls']['roughness'], 'alpha': 1.0}
    with np.load(map_file, allow_pickle=False) as maps:
        if set(maps.files) != set(targets):
            raise ValueError('unexpected material maps')
        for name, target in targets.items():
            values = maps[name]
            shape = (3, 512, 512) if name == 'base_color' else (512, 512)
            if values.dtype != np.float32 or values.shape != shape or not np.isfinite(values).all():
                raise ValueError('nonfinite or misaligned material map')
            foreground = values[:, inside] if name == 'base_color' else values[inside]
            exterior = values[:, ~inside] if name == 'base_color' else values[~inside]
            if not np.allclose(foreground, target, atol=1e-6, rtol=0) or not np.allclose(exterior, 0, atol=1e-6, rtol=0):
                raise ValueError('material map differs from declared intervention')
    return rgb, inside


def one_factor_pairs(group):
    if len(group) != 6:
        raise ValueError('each matched group requires six M/R states')
    anchor = group[0]
    shared = ('view', 'color', 'light', 'pair_group', 'geometry_buffer_sha256')
    if 'subject_id' in anchor:
        shared += ('subject_id',)
    if any(any(row[key] != anchor[key] for key in shared) or any(
            row['controls'][key] != anchor['controls'][key] for key in ('base_color', 'alpha')) for row in group):
        raise ValueError('pair group changes a fixed control')
    by_control = {(row['controls']['metallic'], row['controls']['roughness']): row for row in group}
    metallic = sorted({key[0] for key in by_control})
    roughness = sorted({key[1] for key in by_control})
    if len(metallic) != 2 or len(roughness) != 3 or set(by_control) != set(itertools.product(metallic, roughness)):
        raise ValueError('require unique complete two-by-three M/R grid')
    edges = [('metallic', (metallic[0], r), (metallic[1], r)) for r in roughness]
    edges += [('roughness', (m, a), (m, b)) for m in metallic for a, b in zip(roughness, roughness[1:])]
    result = []
    for axis, left, right in edges:
        first, second = by_control[left], by_control[right]
        fixed = 'roughness' if axis == 'metallic' else 'metallic'
        result.append({'axis': axis, 'from': first['id'], 'to': second['id'],
            'from_value': first['controls'][axis], 'to_value': second['controls'][axis],
            'matched_controls': {**{key: anchor[key] for key in shared},
                fixed: first['controls'][fixed], 'base_color': anchor['controls']['base_color'], 'alpha': 1.0}})
    return result


def verify(output, protocol_path=PROTOCOL):
    output = Path(output).resolve()
    if (output / 'dataset/pairs.json').exists() or (output / 'verification.json').exists():
        raise FileExistsError('verification outputs already exist; do not overwrite receipts')
    protocol = checked_file(output, 'protocol.json', sha256(protocol_path))
    config = json.loads(protocol.read_text())
    manifest_path = checked_file(output, 'dataset/manifest.json')
    manifest = json.loads(manifest_path.read_text())
    records = manifest['records']
    expected = {row['id']: row for row in rows(config)}
    if (len(expected) != 180 or len(records) != 180 or len({row['id'] for row in records}) != 180
            or {row['id'] for row in records} != set(expected) or manifest['schema'] != config['schema']
            or manifest.get('training_started') is not False or manifest.get('training_eligible_images') != 0):
        raise ValueError('require complete unique full-factorial 180 untrained candidates')
    geometry, images, masks, groups, concepts = {}, {}, {}, {}, []
    for view in config['views_yaw_offset_radians']:
        mask = read_image(checked_file(output, f'geometry/{view}_mask.png'), 'L')
        normal = np.load(checked_file(output, f'geometry/{view}_normal.npy'), allow_pickle=False)
        if (set(np.unique(mask)) != {0, 255} or normal.dtype != np.float32
                or normal.shape != (3, 512, 512) or not np.isfinite(normal).all()):
            raise ValueError('invalid frozen geometry buffers')
        geometry[view] = {'mask': mask == 255, 'normal': normal}
    for row in records:
        if any(row.get(key) != value for key, value in expected[row['id']].items()):
            raise ValueError('manifest control, caption or candidate status differs from protocol')
        images[row['id']], masks[row['id']] = verify_record(output, row, config, geometry[row['view']])
        groups.setdefault(row['pair_group'], []).append(row)
        concepts.append({'instance_prompt': [row['caption']],
            'instance_data_dir': str((output / 'dataset' / row['image']).parent),
            'instance_mask_dir': str((output / 'dataset' / row['mask']).parent)})
    concept_path = checked_file(output, 'dataset/concepts.json')
    if json.loads(concept_path.read_text()) != concepts or any(row['caption'].count('<S*>') != 1 for row in records):
        raise ValueError('one-image one-caption concepts differ')
    pairs, blocks = [], []
    for name, group in groups.items():
        edges = one_factor_pairs(group)
        blocks.append({'id': name, 'image_ids': [row['id'] for row in group]})
        for edge in edges:
            first, second = edge['from'], edge['to']
            difference = np.abs(images[first].astype(np.int16) - images[second].astype(np.int16))[masks[first]]
            if not np.any(difference):
                raise ValueError('single-factor edge produces byte-identical foreground RGB')
            pairs.append({**edge, 'foreground_rgb_mae_0_255': float(difference.mean())})
    if len(groups) != 30 or len(pairs) != 210:
        raise ValueError('require thirty matched groups and 210 single-factor edges')
    statistics = {}
    for axis in ('metallic', 'roughness'):
        values = [edge['foreground_rgb_mae_0_255'] for edge in pairs if edge['axis'] == axis]
        statistics[axis] = {'count': len(values), 'min': min(values), 'median': float(np.median(values)), 'max': max(values)}
    summary = {'status': 'technical_checks_passed_scientific_review_pending', 'images': 180,
        'pair_groups': 30, 'single_factor_pairs': 210, 'foreground_rgb_mae_0_255': statistics,
        'manifest_sha256': sha256(manifest_path), 'concepts_sha256': sha256(concept_path),
        'training_started': False, 'training_eligible_images': 0, 'split_policy': SPLIT_POLICY,
        'technical_checks': ['complete_unique_factorial', 'artifact_hashes', 'RGB_mask_raw_cutout_pairing',
            'fixed_geometry_per_view', 'uniform_BC_M_R_alpha', 'one_image_one_caption', 'nonidentical_single_factor_edges'],
        'scientific_threshold': 'None: RGB differences are descriptive, not material correctness or disentanglement scores.'}
    with (output / 'dataset/pairs.json').open('x', encoding='utf8') as handle:
        json.dump({'groups': blocks, 'pairs': pairs, 'split_policy': SPLIT_POLICY}, handle, indent=2, allow_nan=False)
    with (output / 'verification.json').open('x', encoding='utf8') as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    return summary
