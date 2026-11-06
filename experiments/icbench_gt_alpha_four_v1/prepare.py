"""Prepare fixed original/pink RGB with GT/SAM binary or cropped BiRefNet soft alpha."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.subject_trellis_counterfactual_v1.run import check_source_receipt
from scripts.launch.colorpeel_run import git_info
from src.methods.colorpeel_ice import subject_alpha
from src.methods.colorpeel_ice.natural_subject_recolor_border_repair import derive_repaired_alpha
from src.methods.colorpeel_ice.natural_subject_recolor_pilot import recolor
from src.methods.colorpeel_ice.single_image_subject import check_locks, fresh, sha256, write_json


def padded_gt_crop(mask):
    ys, xs = np.where(mask > 0)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    padding = math.ceil(0.1 * max(x1 - x0, y1 - y0))
    return [max(0, x0 - padding), max(0, y0 - padding),
            min(512, x1 + padding), min(512, y1 + padding)], padding


def pink_rgb(rgb, mask, mask_kind='GT'):
    feather, parameters = derive_repaired_alpha(mask > 0)
    editable = (mask > 0) & np.any(rgb != 255, axis=-1)
    if editable.any():
        colored, metrics, _ = recolor(rgb, editable.astype(np.uint8) * 255,
            np.where(editable, feather, 0).astype(np.uint16), 0.)
    else:
        colored, metrics = rgb.copy(), {'status': 'unchanged_all_foreground_exact_white'}
    if not np.array_equal(colored[mask == 0], rgb[mask == 0]):
        raise ValueError(f'pink RGB changed outside {mask_kind}')
    return colored, {'hue_degrees': 0., 'rgb_edit_support': f'{mask_kind} excluding exact-white pixels',
        'soft_blend_parameters': parameters, 'recolor_metrics': metrics,
        'exact_white_pixels_preserved': int(((mask > 0) & ~editable).sum())}


def prepare(source_receipt, output, soft=False, segmenter=None, mask_kind='GT'):
    if mask_kind not in ('GT', 'SAM'):
        raise ValueError('mask_kind must be GT or SAM')
    label = mask_kind.lower()
    source_receipt, output = Path(source_receipt).resolve(), Path(output).resolve()
    checked = check_source_receipt(source_receipt)
    parent = json.loads(source_receipt.read_text())
    if 'conditioning_alpha_path' in parent or 'conditioning_recolor' in parent:
        raise ValueError(f'require the original semantic {mask_kind} receipt, not a derived condition')
    locks = dict(checked['sha256'])
    rgb_path = Path(parent['image512_path'])
    if (not rgb_path.is_absolute() or locks.get(str(rgb_path)) != parent['image512_sha256']
            or sha256(rgb_path) != parent['image512_sha256']):
        raise ValueError('require the locked original image512 RGB')
    if any(output.is_relative_to(path) for path in
           (subject_alpha.ASSETS, *(Path(path).parent for path in locks))):
        raise ValueError('output must be outside all source and model folders')
    with Image.open(rgb_path) as image, Image.open(parent['mask_path']) as opened_mask:
        if (image.mode != 'RGB' or image.size != (512, 512)
                or opened_mask.mode != 'L' or opened_mask.size != image.size):
            raise ValueError(f'require aligned RGB512 and {mask_kind} L512')
        rgb, mask = np.array(image), np.array(opened_mask)
    if not set(np.unique(mask)) <= {0, 255} or not mask.any():
        raise ValueError(f'require a nonempty binary {mask_kind} mask')
    revision = git_info()
    output = fresh(output)
    alpha_kind = 'soft' if soft else 'hard'
    manifest = {'schema': f'icbench_{label}_alpha_four_prepare/v1', 'git': revision,
        'status': 'started', 'source_receipt': str(source_receipt),
        'source_receipt_sha256': locks[str(source_receipt)], 'source_locks': locks,
        'mask_kind': mask_kind, 'geometry_alpha': alpha_kind, 'variants': [], 'training_started': False,
        'quality_accepted_by_user': False,
        'rgb_policy': f'Fixed image512 RGB; pink recolors {mask_kind} only. No alpha-dependent RGB resize.',
        'rgba_serialization': 'Fixed pre-alpha RGB saved separately; zero RGB only where selected alpha is zero.',
        'crop_policy': 'TRELLIS official automatic crop from each geometry alpha > 204'}

    def save(image, filename):
        path = output / filename
        image.save(path)
        locks[str(path)] = sha256(path)
        return path

    try:
        colors = {'original': rgb}
        colors['pink'], recolor_metadata = pink_rgb(rgb, mask, mask_kind)
        alpha = mask.copy()
        alpha_metadata = {'method': f'unchanged_{mask_kind}_binary/v1', 'semantic_mask_modified': False}
        if soft:
            crop, padding = padded_gt_crop(mask)
            cropped = Image.fromarray(rgb).crop(crop)
            save(cropped, 'biref_input_crop.png')
            metadata = {'backend': 'injected_for_test'}
            owned = segmenter is None
            if owned:
                segmenter, metadata, model_locks = subject_alpha.load_segmenter()
                locks.update(model_locks)
            alpha_metadata = {'method': f'{mask_kind}_bbox_padding10pct_BiRefNet_then_{mask_kind}_clip/v1',
                'segmentation': metadata, 'crop_xyxy_exclusive': crop, 'padding_pixels': padding,
                'padding_rule': f'ceil(0.1 * max({mask_kind} half-open bbox width,height)) per side; clip to512',
                f'{label}_clipped': True, 'semantic_mask_modified': False,
                f'soft_values_preserved_inside_{label}': True, 'opacity_normalized': False,
                'hole_filling': False, 'output_alpha_resized': False,
                'model_preprocessing': 'Existing pinned BiRefNet wrapper; model-internal resizing unchanged.',
                'component_policy': f'{mask_kind} clipping is equivalent to keeping intersecting alpha>0 components then intersecting {mask_kind}; no component relabeling.'}
            manifest['conditioning_alpha'] = alpha_metadata
            try:
                segmented = segmenter(cropped.copy())
            finally:
                if owned:
                    segmenter.cpu()
            save(segmented, 'biref_raw_rgba_crop.png')
            if segmented.mode == 'RGBA':
                save(segmented.getchannel('A'), 'biref_raw_alpha_crop.png')
            if segmented.mode != 'RGBA' or segmented.size != cropped.size:
                raise ValueError('BiRefNet must preserve the cropped RGBA canvas')
            rgba = np.array(segmented)
            if not np.array_equal(rgba[..., :3], np.asarray(cropped)):
                raise ValueError('BiRefNet changed source crop RGB')
            full_alpha = np.zeros_like(mask)
            x0, y0, x1, y1 = crop
            full_alpha[y0:y1, x0:x1] = rgba[..., 3]
            save(Image.fromarray(full_alpha), 'biref_raw_alpha512.png')
            alpha = np.where(mask > 0, full_alpha, 0).astype(np.uint8)
            save(Image.fromarray(alpha), f'{label}_clipped_alpha512.png')
            if not alpha.any():
                raise ValueError(f'{mask_kind}-clipped BiRef alpha is empty')
            if not (alpha > 204).any():
                raise ValueError(f'{mask_kind}-clipped BiRef alpha has no values > 204 for official crop')
            if not ((alpha > 0) & (alpha < 255)).any():
                raise ValueError(f'{mask_kind}-clipped BiRef alpha has no soft values')
        manifest['conditioning_alpha'] = alpha_metadata
        for color, pixels in colors.items():
            name = f'{color}__{alpha_kind}'
            folder = output / name
            folder.mkdir()
            pre_rgb = save(Image.fromarray(pixels), f'{name}/pre_alpha_rgb512.png')
            alpha_path = save(Image.fromarray(alpha), f'{name}/alpha512.png')
            cutout = save(Image.fromarray(np.dstack([
                np.where(alpha[..., None] > 0, pixels, 0), alpha])), f'{name}/cutout512.png')
            receipt = {**parent, 'source_locks': dict(locks), 'mask_kind': mask_kind,
                'semantic_receipt_path': str(source_receipt),
                'semantic_receipt_sha256': locks[str(source_receipt)],
                'cutout_path': str(cutout), 'cutout_sha256': locks[str(cutout)],
                'conditioning_alpha_path': str(alpha_path),
                'conditioning_alpha_sha256': locks[str(alpha_path)],
                'conditioning_alpha': alpha_metadata, 'alpha_matches_mask': bool(np.array_equal(alpha, mask)),
                'alpha_matches_conditioning_alpha': True, 'transparent_rgb_zero': True,
                'diagnostic': {'name': name, 'color': color, 'geometry_alpha': alpha_kind,
                    'pre_alpha_rgb_path': str(pre_rgb), 'pre_alpha_rgb_sha256': locks[str(pre_rgb)],
                    'rgb_policy': manifest['rgb_policy']}}
            if color == 'pink':
                receipt['conditioning_recolor'] = recolor_metadata
            receipt_path = folder / 'source_receipt.json'
            write_json(receipt_path, receipt)
            check_source_receipt(receipt_path)
            manifest['variants'].append({'name': name, 'source_receipt': str(receipt_path),
                'source_receipt_sha256': sha256(receipt_path), **receipt['diagnostic']})
        check_locks(locks)
        if git_info() != revision:
            raise ValueError('checkout changed during preparation')
        manifest['status'] = 'succeeded'
        write_json(output / 'manifest.json', manifest)
        write_json(output / 'status.json', manifest)
        return manifest
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'status.json', manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-receipt', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--soft', action='store_true')
    parser.add_argument('--mask-kind', choices=('GT', 'SAM'), default='GT')
    args = parser.parse_args()
    prepare(args.source_receipt, args.output, soft=args.soft, mask_kind=args.mask_kind)
