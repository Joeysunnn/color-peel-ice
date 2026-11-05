"""Freeze previous SAM-pink RGB/crop and replace only its alpha with saved BiRefNet."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.subject_trellis_counterfactual_v1.run import check_source_receipt, fixed_crop_condition
from experiments.wood_spoon_input_cause_v1.prepare import official_crop_box
from scripts.launch.colorpeel_run import git_info
from src.methods.colorpeel_ice.single_image_subject import check_locks, fresh, sha256, write_json


def controlled_pair(hard, soft):
    if hard.shape != (512, 512, 4) or soft.shape != hard.shape:
        raise ValueError('require aligned RGBA512 inputs')
    support = hard[..., 3] > 0
    if not support.any() or not set(np.unique(hard[..., 3])) <= {0, 255}:
        raise ValueError('reference alpha must be nonempty binary SAM')
    if (soft[support, 3] == 0).any() or (hard[~support, :3] != 0).any():
        raise ValueError('soft alpha must cover old SAM; old outside RGB must be zero')
    changed = hard.copy()
    changed[..., 3] = soft[..., 3]
    crop = [int(value) for value in official_crop_box(hard[..., 3])]
    if crop != official_crop_box(hard[..., 3]):
        raise ValueError('this diagnostic requires integer reference crop coordinates')
    return {'hard_replay': hard.copy(), 'soft_only': changed}, crop


def prepare(hard_receipt, soft_receipt, output):
    parents = [Path(path).resolve() for path in (hard_receipt, soft_receipt)]
    checked = [check_source_receipt(path) for path in parents]
    records = [json.loads(path.read_text()) for path in parents]
    if any(records[0][key] != records[1][key] for key in ('source_sha256', 'mask_sha256')):
        raise ValueError('controls must share original photograph and semantic SAM')
    locks = {path: digest for row in checked for path, digest in row['sha256'].items()}
    output = Path(output).resolve()
    if any(output.is_relative_to(Path(path).parent) for path in locks):
        raise ValueError('output must be outside source directories')
    arrays = [np.array(Image.open(record['cutout_path'])) for record in records]
    variants, crop = controlled_pair(*arrays)
    output = fresh(output)
    audit = {'git': git_info(), 'fixed_crop_xyxy': crop, 'seed': 42,
             'parents': [str(path) for path in parents], 'source_locks': locks, 'variants': {},
             'production_input_changed': False, 'training_started': False,
             'quality_accepted_by_user': False}
    for name, rgba in variants.items():
        folder = output / name
        folder.mkdir()
        cutout, alpha, expected = [folder / file for file in
                                   ('cutout512.png', 'alpha512.png', 'expected_condition.png')]
        Image.fromarray(rgba).save(cutout)
        Image.fromarray(rgba[..., 3]).save(alpha)
        condition = fixed_crop_condition(Image.fromarray(rgba), crop)
        condition.save(expected)
        restricted = rgba.copy()
        restricted[arrays[0][..., 3] == 0, 3] = 0
        if condition.tobytes() != fixed_crop_condition(Image.fromarray(restricted), crop).tobytes():
            raise ValueError('expanded alpha support unexpectedly changed fixed black RGB conditioning')
        generated = {str(path): sha256(path) for path in (cutout, alpha, expected)}
        receipt = {**records[0], 'cutout_path': str(cutout), 'cutout_sha256': generated[str(cutout)],
            'conditioning_alpha_path': str(alpha), 'conditioning_alpha_sha256': generated[str(alpha)],
            'soft_alpha': name == 'soft_only', 'alpha_matches_mask': name == 'hard_replay',
            'source_locks': {**locks, **generated},
            'diagnostic': {'variant': name, 'git': audit['git'], 'fixed_crop_xyxy': crop,
                'rgb_policy': 'Every RGB byte copied from previous hard-SAM pink cutout; no recoloring.',
                'alpha_parent': str(parents[name == 'soft_only']),
                'no_inference_effect_from_expanded_alpha_support_at_fixed_black_rgb_and_crop': True}}
        write_json(folder / 'source_receipt.json', receipt)
        check_source_receipt(folder / 'source_receipt.json')
        audit['variants'][name] = {'receipt': str(folder / 'source_receipt.json'),
            'rgb_exactly_equal_reference': bool(np.array_equal(rgba[..., :3], arrays[0][..., :3])),
            'alpha_changed_pixels': int((rgba[..., 3] != arrays[0][..., 3]).sum()),
            'automatic_crop_not_used': official_crop_box(rgba[..., 3]), 'files_sha256': generated}
    check_locks(locks)
    write_json(output / 'input_audit.json', audit)
    return audit
