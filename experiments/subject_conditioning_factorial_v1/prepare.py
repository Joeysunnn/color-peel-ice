"""Freeze uncolored RGB512 and independently vary recolor support and alpha."""

from itertools import product
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.subject_trellis_counterfactual_v1.run import check_source_receipt
from experiments.wood_spoon_input_cause_v1.prepare import official_crop_box
from scripts.launch.colorpeel_run import git_info
from src.methods.colorpeel_ice.natural_subject_recolor_border_repair import derive_repaired_alpha
from src.methods.colorpeel_ice.natural_subject_recolor_pilot import recolor
from src.methods.colorpeel_ice.single_image_subject import check_locks, fresh, sha256, write_json


def build_variants(rgb, sam, soft, source_rgb=False):
    if (rgb.dtype != np.uint8 or rgb.shape != (512, 512, 3)
            or any(array.dtype != np.uint8 or array.shape != (512, 512) for array in (sam, soft))
            or not set(np.unique(sam)) <= {0, 255} or not sam.any()
            or not ((soft > 0) & (soft < 255)).any()):
        raise ValueError('require RGB512, binary SAM and genuinely soft L512 alpha')
    if not source_rgb and (((sam > 0) & (soft == 0)).any() or (rgb[soft == 0] != 0).any()):
        raise ValueError('frozen matte RGB must cover all SAM pixels with transparent RGB zero')
    supports = {'sam': sam > 0, 'biref': soft > 0}
    colors, metrics = {}, {}
    for scope, support in supports.items():
        colors['original', scope] = rgb.copy()
        feather, parameters = derive_repaired_alpha(support)
        editable = support & np.any(rgb != 255, axis=-1)
        if editable.any():
            colored, measured, _ = recolor(rgb, editable.astype(np.uint8) * 255,
                np.where(editable, feather, 0).astype(np.uint16), 0.)
        else:
            colored, measured = rgb.copy(), {'status': 'unchanged_all_white'}
        if not np.array_equal(colored[~support], rgb[~support]):
            raise ValueError('pink changed RGB outside its selected recolor support')
        colors['pink', scope] = colored
        metrics[scope] = {'blend': parameters, 'recolor': measured}
    variants = []
    for color, scope, alpha in product(('original', 'pink'), ('sam', 'biref'), ('sam', 'biref')):
        pre_alpha = colors[color, scope]
        geometry = sam if alpha == 'sam' else soft
        rgba = np.dstack([np.where(geometry[..., None] > 0, pre_alpha, 0), geometry])
        variants.append({'name': '__'.join((color, scope, alpha)), 'color': color,
            'recolor_support': scope, 'geometry_alpha': alpha, 'pre_alpha_rgb': pre_alpha.copy(),
            'auto_crop_xyxy': official_crop_box(geometry),
            'rgba': rgba, 'recolor_metrics': metrics[scope] if color == 'pink' else {'status': 'unrecolored'}})
    return variants


def expected_auto_condition(rgba):
    """Mirror pinned official RGBA512 preprocessing, including fractional crop boxes."""
    crop = official_crop_box(rgba[..., 3])
    pixels = np.asarray(Image.fromarray(rgba).crop(crop)).astype(np.float32) / 255
    return Image.fromarray((pixels[..., :3] * pixels[..., 3:4] * 255).astype(np.uint8))


def prepare(matte_receipt, output, source_rgb=False):
    matte_receipt, output = Path(matte_receipt).resolve(), Path(output).resolve()
    checked = check_source_receipt(matte_receipt)
    parent = json.loads(matte_receipt.read_text())
    if ('conditioning_alpha_path' not in parent or 'conditioning_recolor' in parent
            or parent['conditioning_alpha']['method'] != 'pinned_BiRefNet_full_native_RGBA_then_joint_LANCZOS/v1'):
        raise ValueError('require the audited uncolored SAM-associated BiRefNet matte receipt')
    locks = checked['sha256']
    if any(output.is_relative_to(Path(path).parent) for path in locks):
        raise ValueError('new preparation must be separate from all input folders')
    with Image.open(parent['cutout_path']) as image, Image.open(parent['mask_path']) as mask:
        rgba, sam = np.array(image), np.array(mask)
    rgb = rgba[..., :3]
    if source_rgb:
        rgb_source = Path(parent['image512_path'])
        digest = parent['image512_sha256']
        if (not rgb_source.is_absolute() or locks.get(str(rgb_source)) != digest
                or sha256(rgb_source) != digest):
            raise ValueError('source RGB512 requires its unchanged parent source lock')
        with Image.open(rgb_source) as image:
            if image.mode != 'RGB' or image.size != (512, 512):
                raise ValueError('source RGB512 must be RGB mode on the unchanged 512 canvas')
            rgb = np.array(image)
    revision = git_info()
    variants = build_variants(rgb, sam, rgba[..., 3], source_rgb=source_rgb)
    output = fresh(output)
    rgb_path = output / 'frozen_base_rgb512.png'
    Image.fromarray(rgb).save(rgb_path)
    locks = {**locks, str(rgb_path): sha256(rgb_path)}
    manifest = {'git': revision, 'source_locks': locks, 'matte_receipt': str(matte_receipt),
        'seed': 42, 'crop_policy': 'official automatic crop from each geometry alpha > 204',
        'variants': [], 'expected_meshes': 8, 'source_rgb': source_rgb,
        'scope_semantics': 'Recolor only. Original and all pixels outside pink support retain frozen base RGB.',
        'rgb_preprocessing': ('Frozen original image512 RGB from independent RGB LANCZOS and full-canvas letterbox; '
            'identical for every factor cell, no alpha-dependent RGB resizing or new resizing.' if source_rgb else
            'Frozen uncolored BiRefNet joint-RGBA-resize RGB512, identical for every factor cell; no new resizing.'),
        'alpha_serialization': 'Pre-alpha RGB is saved separately; final RGBA zeros RGB only where selected geometry alpha is zero.',
        'original_scope_pairs_expected_identical': True,
        'training_started': False, 'production_defaults_changed': False, 'quality_accepted_by_user': False}
    if source_rgb:
        manifest.update(frozen_rgb_source=str(rgb_source), frozen_rgb_source_sha256=digest)
    for variant in variants:
        folder = output / variant['name']
        folder.mkdir()
        files = {'pre_alpha_rgb': folder / 'pre_alpha_rgb512.png', 'cutout': folder / 'cutout512.png',
                 'conditioning_alpha': folder / 'alpha512.png', 'expected_condition': folder / 'expected_condition.png'}
        Image.fromarray(variant['pre_alpha_rgb']).save(files['pre_alpha_rgb'])
        Image.fromarray(variant['rgba']).save(files['cutout'])
        Image.fromarray(variant['rgba'][..., 3]).save(files['conditioning_alpha'])
        expected_auto_condition(variant['rgba']).save(files['expected_condition'])
        generated_locks = {str(path): sha256(path) for path in files.values()}
        diagnostic = {key: value for key, value in variant.items() if key not in ('rgba', 'pre_alpha_rgb')}
        diagnostic.update(pre_alpha_rgb_path=str(files['pre_alpha_rgb']),
                          pre_alpha_rgb_sha256=generated_locks[str(files['pre_alpha_rgb'])],
                          scope_semantics=manifest['scope_semantics'], rgb_preprocessing=manifest['rgb_preprocessing'])
        receipt = {**parent, 'source_locks': {**locks, **generated_locks},
            'alpha_matches_mask': variant['geometry_alpha'] == 'sam',
            'soft_alpha': variant['geometry_alpha'] == 'biref', 'diagnostic': diagnostic,
            'conditioning_alpha': {'method': 'factorial_frozen_alpha_selection/v1',
                'selection': variant['geometry_alpha'], 'parent_receipt': str(matte_receipt),
                'parent_conditioning_alpha': parent['conditioning_alpha']}}
        for name in ('cutout', 'conditioning_alpha'):
            receipt[name + '_path'], receipt[name + '_sha256'] = str(files[name]), generated_locks[str(files[name])]
        receipt_path = folder / 'source_receipt.json'
        write_json(receipt_path, receipt)
        check_source_receipt(receipt_path)
        manifest['variants'].append({**diagnostic, 'source_receipt': str(receipt_path),
            'source_receipt_sha256': sha256(receipt_path), 'expected_condition': str(files['expected_condition']),
            'expected_condition_sha256': generated_locks[str(files['expected_condition'])],
            'files_sha256': generated_locks})
    check_locks(locks)
    if git_info() != revision:
        raise ValueError('checkout changed during preparation')
    write_json(output / 'manifest.json', manifest)
    return manifest
