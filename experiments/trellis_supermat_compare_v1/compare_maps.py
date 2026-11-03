"""Compare image-plane A/R/M estimates; no calibrated surface correspondence is assumed."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


SIZE = (512, 512)
MIN_RETENTION = 0.5
MIN_PIXELS = 50
EROSION = 2
MASK_THRESHOLD = 0.9
ALPHA_THRESHOLD = 0.9
LIMITATIONS = [
    'Common ROI is approximate image-plane overlap, not verified 3D surface correspondence.',
    'Inspect the ROI overlay to confirm both methods still sample the same visible lid surface.',
    'TRELLIS camera is estimated; silhouette IoU alone does not prove surface alignment.',
    'SuperMat albedo and TRELLIS native base color are model estimates, not physical ground truth.',
    'M/R medians summarize the shared ROI; no training or visual acceptance is implied.',
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def read_image(path, mode):
    with Image.open(path) as image:
        if image.size != SIZE:
            raise ValueError(f'Require unresized512px input: {path}')
        if mode == 'RGBA' and image.mode != mode:
            raise ValueError('Source must retain its RGBA foreground mask')
        return np.asarray(image.convert(mode)).copy()


def read_scalar_png(path):
    with Image.open(path) as image:
        if image.size != SIZE or image.mode not in ('L', 'RGB', 'RGBA'):
            raise ValueError('SuperMat scalar maps must be unresized8bit grayscale PNGs')
        values = np.asarray(image)
        if values.ndim == 3:
            if not np.array_equal(values[..., 0], values[..., 1]) or not np.array_equal(values[..., 0], values[..., 2]):
                raise ValueError('SuperMat scalar PNG contains unequal RGB channels')
            values = values[..., 0]
        return values.astype(np.float64) / 255


def erode(mask, radius=EROSION):
    """Square erosion with image-exterior pixels explicitly false."""
    height, width = mask.shape
    padded = np.pad(mask, radius, constant_values=False)
    result = np.ones_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result &= padded[dy:dy + height, dx:dx + width]
    return result


def load_trellis(path):
    # Mandatory raw float maps: never reconstruct statistics from display PNGs.
    with np.load(path, allow_pickle=False) as archive:
        if not {'base_color', 'roughness', 'metallic', 'mask', 'alpha'}.issubset(archive.files):
            raise ValueError('TRELLIS attributes.npz lacks required A/R/M/mask/alpha arrays')
        if any(not np.issubdtype(archive[name].dtype, np.floating)
               for name in ('base_color', 'roughness', 'metallic')):
            raise ValueError('TRELLIS A/R/M must be raw floating-point attributes')
        arrays = {name: np.array(archive[name], dtype=np.float64) for name in
                  ('base_color', 'roughness', 'metallic', 'mask', 'alpha')}
    if arrays['base_color'].shape == (3, *SIZE):
        arrays['base_color'] = arrays['base_color'].transpose(1, 2, 0)
    if (arrays['base_color'].shape != (*SIZE, 3)
            or any(arrays[name].shape != SIZE for name in ('roughness', 'metallic', 'mask', 'alpha'))
            or any(not np.isfinite(value).all() for value in arrays.values())):
        raise ValueError('TRELLIS raw attributes require finite512px HWC/CHW RGB and HW scalar arrays')
    if np.any((arrays['mask'] < 0) | (arrays['mask'] > 1)):
        raise ValueError('TRELLIS mask must be in [0,1]')
    return arrays


def distribution(array, roi):
    raw = np.asarray(array, dtype=np.float64)[roi]
    clipped = np.clip(raw, 0, 1)
    quantiles = (0.05, 0.25, 0.75, 0.95)
    return {'pixel_count': int(raw.size), 'raw_min': float(raw.min()), 'raw_max': float(raw.max()),
            'raw_median': float(np.median(raw)), 'raw_quantiles': dict(zip(
                ('q05', 'q25', 'q75', 'q95'), map(float, np.quantile(raw, quantiles)))),
            'median': float(np.median(clipped)), 'min': float(clipped.min()), 'max': float(clipped.max()),
            'quantiles': dict(zip(('q05', 'q25', 'q75', 'q95'), map(float, np.quantile(clipped, quantiles)))),
            'clipping_fraction': float(np.mean((raw < 0) | (raw > 1)))}


def display(array, mask=None):
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if mask is not None:
        array = np.where(mask[..., None], array, 127 / 255)
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))


def overlay(rgb, roi, common=None):
    array = np.asarray(rgb, dtype=np.float64).copy()
    outline = roi & ~erode(roi, 1)
    array[outline] = [1, 0, 1]
    if common is not None:
        array[common] = 0.55 * array[common] + 0.45 * np.array([0, 1, 0])
        array[common & ~erode(common, 1)] = [0, 1, 0]
    return display(array)


def compare(supermat, trellis_review, source, roi, output):
    supermat, trellis_review, source, roi, output = [Path(p).resolve() for p in
                                                  (supermat, trellis_review, source, roi, output)]
    protected = (supermat, trellis_review, source.parent, roi.parent, Path(__file__).resolve().parents[2])
    if output.exists() or any(output == p or output in p.parents for p in protected):
        raise ValueError('Require a fresh output separate from input directories')
    if any(output.is_relative_to(p) for p in (supermat, trellis_review, Path(__file__).resolve().parents[2])):
        raise ValueError('Output must be outside source map and repository directories')
    paths = {'source': source, 'reviewed_roi': roi, 'trellis_raw': trellis_review / 'attributes.npz',
             'trellis_review': trellis_review / 'review.json', 'trellis_mask': trellis_review / 'fitted_mask.png',
             **{'supermat_' + name: supermat / (name + '.png') for name in ('albedo', 'roughness', 'metallic')}}
    actual_condition = trellis_review.parent / 'actual_condition.png'
    if actual_condition.is_file():
        paths['trellis_actual_condition'] = actual_condition
    locks = {str(path): sha256(path) for path in paths.values()}
    rgba = read_image(source, 'RGBA') / 255.0
    foreground = rgba[..., 3] > 0.5
    reviewed_roi = read_image(roi, 'L') >= 128
    if not foreground.any() or not reviewed_roi.any():
        raise ValueError('Source foreground and reviewed ROI must be nonempty')
    sm = {'albedo': read_image(paths['supermat_albedo'], 'RGB') / 255.0,
          **{name: read_scalar_png(paths['supermat_' + name]) for name in ('roughness', 'metallic')}}
    trellis = load_trellis(paths['trellis_raw'])
    fitted = read_image(paths['trellis_mask'], 'L') >= 128
    if not np.array_equal(fitted, trellis['mask'] > 0.5):
        raise ValueError('TRELLIS fitted_mask.png differs from raw mask at0.5')
    review = json.loads(paths['trellis_review'].read_text(encoding='utf8'))
    camera_iou = float((foreground & fitted).sum() / max(int((foreground | fitted).sum()), 1))
    recorded_iou = review['metrics']['original512_mask_iou']
    if not np.isfinite(recorded_iou) or abs(recorded_iou - camera_iou) > 1e-8:
        raise ValueError('Camera review IoU differs from current source/raw mask')
    intersection = reviewed_roi & foreground & (trellis['mask'] >= MASK_THRESHOLD) & (trellis['alpha'] >= ALPHA_THRESHOLD)
    common = erode(intersection)
    count = int(common.sum())
    retained = float(count / reviewed_roi.sum())
    eligible = retained >= MIN_RETENTION and count >= MIN_PIXELS
    output.mkdir(parents=True)
    Image.fromarray(common.astype(np.uint8) * 255).save(output / 'common_roi.png')
    coverage = {'reviewed_roi_pixels': int(reviewed_roi.sum()), 'intersection_before_erosion_pixels': int(intersection.sum()),
                'common_roi_pixels': count, 'retention_fraction': retained,
                'minimum_retention_fraction': MIN_RETENTION, 'minimum_pixels': MIN_PIXELS,
                'erosion_pixels': EROSION, 'erosion_kernel': 'square5x5; exterior false',
                'trellis_raw_mask_threshold': MASK_THRESHOLD, 'supermat_foreground': 'source RGBA alpha >0.5',
                'trellis_raw_alpha_threshold': ALPHA_THRESHOLD,
                'trellis_alpha_on_reviewed_roi': distribution(trellis['alpha'], reviewed_roi),
                'trellis_opaque_pixels_in_reviewed_roi': int((reviewed_roi & (trellis['alpha'] >= ALPHA_THRESHOLD)).sum()),
                'eligible': bool(eligible), 'roi_expanded': False}
    report = {'schema': 'trellis_supermat_maps_comparison/v1', 'status': 'ready_for_review' if eligible else 'failed',
              'coverage': coverage, 'camera_iou': camera_iou, 'camera': review['camera'],
              'source_locks': locks, 'common_roi': str(output / 'common_roi.png'),
              'common_roi_sha256': sha256(output / 'common_roi.png'), 'limitations': LIMITATIONS,
              'albedo_display': 'Native [0,1] maps clipped only; no extra gamma, no photometric Albedo accuracy claim.',
              'material_outputs': {}}
    foreground_rgb = rgba[..., :3] * rgba[..., 3:] + (127 / 255) * (1 - rgba[..., 3:])
    panels = [('source_rgb', 'Source RGB channels / original canvas', display(rgba[..., :3])),
              ('source_foreground', 'Source foreground / original canvas', display(foreground_rgb)),
              ('reviewed_roi_overlay', 'Reviewed ROI boundary (magenta)', overlay(foreground_rgb, reviewed_roi)),
              ('common_roi_overlay', 'Common ROI (green), reviewed ROI (magenta)', overlay(foreground_rgb, reviewed_roi, common)),
              ('supermat_mask', 'SuperMat foreground = source alpha', display(foreground.astype(float))),
              ('supermat_albedo', 'SuperMat Albedo / original view', display(sm['albedo'], foreground)),
              ('supermat_roughness', 'SuperMat Roughness / original view', display(sm['roughness'], foreground)),
              ('supermat_metallic', 'SuperMat Metallic / original view', display(sm['metallic'], foreground)),
              ('trellis_roi_overlay', 'TRELLIS ROI overlay / estimated view', overlay(np.asarray(display(trellis['base_color'], fitted)) / 255.0, reviewed_roi, common)),
              ('trellis_base_color', 'TRELLIS Base Color / estimated view', display(trellis['base_color'], fitted)),
              ('trellis_roughness', 'TRELLIS Roughness / estimated view', display(trellis['roughness'], fitted)),
              ('trellis_metallic', 'TRELLIS Metallic / estimated view', display(trellis['metallic'], fitted))]
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 18)
    except OSError:
        font = ImageFont.load_default()
    sheet = Image.new('RGB', (2048, 1770), 'white')
    draw = ImageDraw.Draw(sheet)
    (output / 'panels').mkdir()
    for index, (name, title, image) in enumerate(panels):
        image.save(output / 'panels' / (name + '.png'))
        x, y = (index % 4) * 512, (index // 4) * 550
        draw.text((x + 8, y + 8), title, fill='black', font=font)
        sheet.paste(image, (x, y + 32))
    draw.multiline_text((10, 1658), f'Status: {report["status"]}; common ROI={count} pixels; retention={retained:.3f}; camera IoU={camera_iou:.3f}\n'
        'Approximate image-plane correspondence only. Inspect both ROI overlays for the same lid surface.\n'
        'Statistics: SuperMat8bit PNG /255; TRELLIS raw float NPZ. Display PNGs do not supply TRELLIS scalar targets.', fill='black', font=font, spacing=6)
    sheet.save(output / 'comparison.png')
    conditioning = Image.new('RGB', (1024, 550), 'white')
    condition_draw = ImageDraw.Draw(conditioning)
    condition_draw.text((8, 8), 'SuperMat input foreground / original512 canvas', fill='black', font=font)
    conditioning.paste(display(foreground_rgb), (0, 32))
    condition_draw.text((520, 8), 'TRELLIS actual conditioning / display resized', fill='black', font=font)
    condition_info = {'supermat_display': 'Source alpha foreground composited on gray; original512 canvas',
                      'trellis_actual_condition': str(actual_condition) if actual_condition.is_file() else None}
    if actual_condition.is_file():
        with Image.open(actual_condition) as opened:
            condition_info['trellis_original_size'] = list(opened.size)
            condition_image = opened.convert('RGBA')
        background = Image.new('RGBA', condition_image.size, (127, 127, 127, 255))
        background.alpha_composite(condition_image)
        condition_image = ImageOps.contain(background.convert('RGB'), SIZE, Image.Resampling.LANCZOS)
        conditioning.paste(condition_image, (512 + (512 - condition_image.width) // 2, 32 + (512 - condition_image.height) // 2))
    else:
        condition_draw.text((530, 90), 'Actual conditioning image not exported', fill='black', font=font)
    conditioning.save(output / 'conditioning_comparison.png')
    report['conditioning'] = condition_info
    display(trellis['alpha']).save(output / 'trellis_alpha.png')
    if any(sha256(path) != digest for path, digest in locks.items()):
        report.update(status='failed', error='Input changed during comparison; no scalar material outputs published.')
        eligible = False
    if eligible:
        for name, maps in (('supermat', sm), ('trellis', trellis)):
            stats = {channel: distribution(maps[channel], common) for channel in ('roughness', 'metallic')}
            record = {'schema': 'scalar_material_from_common_roi/v1', 'status': 'ready_for_review', 'method': name,
                'roughness': stats['roughness']['median'], 'metallic': stats['metallic']['median'], 'distributions': stats,
                'aggregation': 'Median after clipping [0,1]; raw distribution and clipping fraction retained',
                'numeric_source': 'uint8 PNG /255; quantization step1/255' if name == 'supermat' else 'raw float attributes.npz; no PNG quantization',
                'coverage': coverage, 'camera_iou': camera_iou, 'source_locks': locks,
                'common_roi': report['common_roi'], 'common_roi_sha256': report['common_roi_sha256'], 'limitations': LIMITATIONS}
            folder = output / name
            folder.mkdir()
            write_json(folder / 'material.json', record)
            report['material_outputs'][name] = str(folder / 'material.json')
    elif 'error' not in report:
        report['error'] = 'Insufficient shared ROI: need >=50% of reviewed ROI and >=50pixels after erosion; ROI was not expanded.'
    write_json(output / 'report.json', report)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>TRELLIS / SuperMat A/R/M</title>'
        f'<h1>A/R/M comparison: {report["status"]}</h1><p>Approximate image-plane overlap; inspect overlays for the same surface. '
        'No training or automatic visual acceptance.</p><p><a href="report.json">Statistics and provenance</a> | '
        '<a href="common_roi.png">Shared ROI</a> | <a href="trellis_alpha.png">TRELLIS alpha</a></p>'
        '<a href="comparison.png"><img style="max-width:100%" src="comparison.png"></a>'
        '<h2>Conditioning inputs</h2><a href="conditioning_comparison.png"><img style="max-width:100%" src="conditioning_comparison.png"></a>', encoding='utf8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('supermat', 'trellis-review', 'source', 'roi', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.supermat, args.trellis_review, args.source, args.roi, args.output)
    print(json.dumps({'status': result['status'], 'coverage': result['coverage'], 'camera_iou': result['camera_iou']}))
    raise SystemExit(0 if result['status'] == 'ready_for_review' else 1)
