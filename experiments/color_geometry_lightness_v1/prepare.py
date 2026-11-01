"""Nine fixed-chromaticity previews from verified neutral Blender geometry."""

import argparse
import html
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from experiments.mailbox_color_sd15_v1.data import source_snapshot, check_locks
from experiments.sd15_token_local_kv_v1.contract import sha256
from scripts.methods.colorpeel_ice import run_d1_gamut_only_lch_injection as neutral
from scripts.methods.colorpeel_ice.render_d1_color_calibration_preflight import _mask_interior
from src.methods.colorpeel_ice import neutral_lch_injection as color

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/home/r12user5/Documents/Jiawei/colorpeel-runs/natural_image_subject_color_pilot/gamut_aware_lch_injection_f1af34e_20260912_0080')
AB = np.array([63.21805661727065, 54.85769274644318])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf8')


def remap(image, mask):
    if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
            or mask.shape != image.shape[:2] or not set(np.unique(mask)) <= {0, 255}):
        raise ValueError('require RGB uint8 and aligned binary source mask')
    inside = mask == 255
    indices, geometry = _mask_interior(inside.ravel().tolist(), mask.shape[1], mask.shape[0])
    source_lab = color.linear_rgb_to_lab(color._srgb_to_linear(image.reshape(-1, 3)))
    interior_L = source_lab[indices, 0]
    q05, center, q95 = np.quantile(interior_L, [.05, .5, .95])
    scale = (q95 - q05) / 2
    if scale < .5:
        raise ValueError('source has insufficient lightness contrast; do not amplify noise')
    # Strictly increasing, smooth and bounded; never clip RGB or reduce chroma.
    mapped_L = 50 + 8 * np.tanh((source_lab[inside.ravel(), 0] - center) / scale)
    lab = np.column_stack((mapped_L, np.broadcast_to(AB, (len(mapped_L), 2))))
    linear = color.lab_to_linear_rgb(lab)
    if not np.isfinite(linear).all() or (linear < 0).any() or (linear > 1).any():
        raise ValueError('fixed target is outside sRGB; no silent clipping or chroma reduction')
    output = image.copy()
    output[inside] = color._linear_to_srgb_uint8(linear)
    decoded = color.linear_rgb_to_lab(color._srgb_to_linear(output.reshape(-1, 3)))
    interior = decoded[indices]
    error = np.linalg.norm(interior[:, 1:] - AB, axis=1)
    dominant = np.median(interior, axis=0)
    source_order = np.argsort(interior_L)
    reversals = np.diff(interior[source_order, 0]) < -.5
    statistics = {
        'source_L_q05_median_q95': [float(q05), float(center), float(q95)],
        'mapping_scale': float(scale), 'interior_geometry': geometry,
        'output_L_q05_median_q95': np.quantile(interior[:, 0], [.05, .5, .95]).tolist(),
        'output_L_q95_minus_q05': float(np.quantile(interior[:, 0], .95) - np.quantile(interior[:, 0], .05)),
        'dominant_lab': dominant.tolist(), 'e_ch': float(np.linalg.norm(dominant[1:] - AB)),
        'pixel_ch_error_p50_p95_max': np.quantile(error, [.5, .95, 1]).tolist(),
        'decoded_L_order_reversals_over_point5': int(reversals.sum()),
        'rgb_clipping_pixels': 0, 'chroma_reduction': 0,
        'outside_mask_changed_pixels': int(np.any(output[~inside] != image[~inside], axis=1).sum()),
        'source_background_preserved': True,
    }
    statistics['numeric_pass'] = bool(statistics['e_ch'] <= 3 and error.max() <= 1
                                     and statistics['output_L_q95_minus_q05'] >= 6
                                     and not reversals.any()
                                     and statistics['outside_mask_changed_pixels'] == 0)
    return output, statistics


def review(output, rows):
    sheet = Image.new('RGB', (960, 9 * 360), 'white')
    draw = ImageDraw.Draw(sheet)
    parts = ['<!doctype html><meta charset="utf-8"><title>Color geometry lightness9</title>',
             '<style>body{font:16px sans-serif;background:#eee;margin:24px}table{border-collapse:collapse}',
             'td{vertical-align:top;padding:8px}img{max-width:100%}th{padding:12px}code{word-break:break-all}</style>',
             '<h1>Fixed a/b + bounded geometry lightness: nine candidates</h1>',
             '<p>Preview only; not training-approved. Target Lab a=63.2181, b=54.8577; L=42..58.</p>',
             '<p>Historical neutral Blender renders reused byte-for-byte. Binary masks retained without invented alpha/feathering.',
             ' Background/shadow unchanged from neutral; historical Emission has different floor spill.',
             ' This is controlled image synthesis, not physical PBR or a strict shading-only ablation.</p>',
             '<p><a href="audit.json">Numerical audit</a> | <a href="manifest.json">Sources and captions</a></p>',
             '<table><tr><th>Historical Emission</th><th>Frozen neutral</th><th>New fixed a/b</th></tr>']
    for i, row in enumerate(rows):
        case = output / row['id']
        parts.append('<tr>')
        for j, name in enumerate(('emission.png', 'neutral.png', 'image.png')):
            img = Image.open(case / name).convert('RGB')
            sheet.paste(img.resize((320, 320), Image.Resampling.LANCZOS), (j * 320, i * 360 + 30))
            draw.text((j * 320 + 5, i * 360 + 5), f"{row['id']} | {name}", fill='black')
            url = f"{row['id']}/{name}"
            parts.append(f'<td><a href="{url}"><img width="512" src="{url}"></a></td>')
        parts.append('</tr><tr><td colspan="3">' + html.escape(row['caption'])
                     + f" | E_ch={row['metrics']['e_ch']:.3f}; L contrast={row['metrics']['output_L_q95_minus_q05']:.2f}"
                     + f' | <a href="{row["id"]}/edge.png">Nearest-neighbor edge close-up</a></td></tr>')
        mask = np.asarray(Image.open(case / 'mask.png')) > 0
        ys, xs = np.nonzero(mask)
        box = (max(0, int(xs.min()) - 8), max(0, int(ys.min()) - 8),
               min(512, int(xs.max()) + 9), min(512, int(ys.max()) + 9))
        crop = Image.open(case / 'image.png').crop(box)
        crop.resize((crop.width * 4, crop.height * 4), Image.Resampling.NEAREST).save(case / 'edge.png')
    sheet.save(output / 'comparison.png')
    (output / 'index.html').write_text('\n'.join(parts) + '</table>', encoding='utf8')


def run(output):
    output = output.resolve()
    if output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(SOURCE):
        raise ValueError('require a fresh external output directory')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('source checkout must be clean')
    neutral_rows, _ = neutral._source_records(SOURCE)
    emission_rows, locks, target, old_roots = source_snapshot()
    if any(output.is_relative_to(r) or r.is_relative_to(output) for r in old_roots):
        raise ValueError('output overlaps historical inputs')
    if not np.array_equal([target['a'], target['b']], AB):
        raise ValueError('target differs from current Color branch')
    for name in (neutral.SOURCE_MANIFEST_NAME, neutral.SOURCE_CONTRACT_NAME):
        locks[str(SOURCE / name)] = sha256(SOURCE / name)
    for row in neutral_rows.values():
        for kind in ('image', 'mask', 'metadata'):
            path = (SOURCE / row[kind + '_relative_path']).resolve()
            if not path.is_relative_to(SOURCE):
                raise ValueError('neutral source path escapes root')
            locks[str(path)] = row[kind + '_sha256']
    check_locks(locks)
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', {
        'git': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'target': target, 'source_neutral': str(SOURCE), 'count': 9,
        'mapping': 'L_out=50+8*tanh((L_src-median)/(0.5*(q95-q05))); quantiles from eroded interior',
        'L_bounds': [42, 58], 'boundary': 'frozen binary mask, no feather, no morphology on output',
        'training_authorized': False, 'visual_review': 'pending', 'new_blender_render': False,
        'limitations': 'Preserves spatial shading order, not absolute illumination. Baseline floor spill differs.'})
    rows = []
    for old in emission_rows:
        identifier = f"{old['shape']}_v{old['view']:02d}"
        source = neutral_rows[f"neutral__{old['shape']}__v{old['view']}"]
        case = output / identifier
        case.mkdir()
        for kind, filename in (('image', 'neutral.png'), ('mask', 'mask.png'), ('metadata', 'neutral_metadata.json')):
            shutil.copyfile(SOURCE / source[kind + '_relative_path'], case / filename)
        shutil.copyfile(old['source_image'], case / 'emission.png')
        image = np.asarray(Image.open(case / 'neutral.png').convert('RGB'))
        mask = np.asarray(Image.open(case / 'mask.png').convert('L'))
        if image.shape != (512, 512, 3):
            raise ValueError('expected frozen 512px source')
        result, metrics = remap(image, mask)
        Image.fromarray(result).save(case / 'image.png')
        if not np.array_equal(np.asarray(Image.open(case / 'image.png')), result):
            raise ValueError('PNG pixel roundtrip changed')
        row = {'id': identifier, 'caption': old['caption'], 'shape': old['shape'], 'view': old['view'],
               'neutral_record': source, 'emission_record': old, 'metrics': metrics,
               'image': f'{identifier}/image.png', 'mask': f'{identifier}/mask.png',
               'image_sha256': sha256(case / 'image.png'), 'training_eligible': False}
        write_json(case / 'metrics.json', metrics)
        rows.append(row)
    check_locks(locks)
    write_json(output / 'source_locks.json', locks)
    write_json(output / 'manifest.json', rows)
    audit = {'count': len(rows), 'numeric_pass': all(r['metrics']['numeric_pass'] for r in rows),
             'max_pixel_ch_error': max(r['metrics']['pixel_ch_error_p50_p95_max'][2] for r in rows),
             'min_L_contrast': min(r['metrics']['output_L_q95_minus_q05'] for r in rows),
             'training_started': False, 'visual_review': 'pending', 'old_sources_hash_verified': True}
    write_json(output / 'audit.json', audit)
    review(output, rows)
    write_json(output / 'files.json', {str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file()})
    print(json.dumps(audit))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
