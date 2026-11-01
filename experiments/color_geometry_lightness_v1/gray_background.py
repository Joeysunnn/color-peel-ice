"""Replace only the background of the accepted nine previews with sRGB128."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.color_geometry_lightness_v1.prepare import ROOT, write_json

SOURCE = Path('/var/tmp/colorpeel-r12user5/runs/color_geometry_lightness_v1/20261101-162859__orange9__7f6865a__42')
SOURCE_INDEX_SHA = 'e8e30ebc32d0e26110b2f3866c9d8ce88838f616276af02fc23bef4f4963ff43'


def verify_source():
    if sha256(SOURCE / 'files.json') != SOURCE_INDEX_SHA:
        raise ValueError('accepted preview source index changed')
    for name, digest in read_json(SOURCE / 'files.json').items():
        path = (SOURCE / name).resolve()
        if not path.is_relative_to(SOURCE) or sha256(path) != digest:
            raise ValueError('accepted preview source file changed')


def run(output):
    output = output.resolve()
    if output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(SOURCE) or SOURCE.is_relative_to(output):
        raise ValueError('require a new external output, disjoint from source')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('source checkout must be clean')
    verify_source()
    rows = read_json(SOURCE / 'manifest.json')
    if len(rows) != 9 or {(r['shape'], r['view']) for r in rows} != {
            (s, v) for s in ('cube', 'sphere', 'cylinder') for v in (0, 8, 16)}:
        raise ValueError('require exactly the accepted nine shape/view pairs')
    output.mkdir(parents=True)
    sheet = Image.new('RGB', (1152, 1242), 'white')
    draw = ImageDraw.Draw(sheet)
    parts = ['<!doctype html><meta charset="utf-8"><title>Color geometry: uniform gray9</title>',
             '<style>body{font:16px sans-serif;margin:24px;background:#eee}img{max-width:100%}',
             'td{padding:8px}th{padding:12px}</style><h1>Color geometry — uniform neutral gray</h1>',
             '<p>9 images, sRGB background (128,128,128). Foreground RGB and binary masks unchanged.',
             ' Cast shadows outside the object are removed. No feathering or retraining.</p>',
             '<p><a href="audit.json">Pixel audit</a> | <a href="contact_sheet.png">Nine-image overview</a></p>',
             '<table><tr><th>Accepted original</th><th>Uniform gray background</th></tr>']
    manifest = []
    for i, row in enumerate(rows):
        case = output / row['id']
        case.mkdir()
        source_image = SOURCE / row['image']
        source_mask = SOURCE / row['mask']
        before = np.asarray(Image.open(source_image).convert('RGB'))
        mask = np.asarray(Image.open(source_mask).convert('L'))
        if before.shape != (512, 512, 3) or mask.shape != (512, 512) or set(np.unique(mask)) != {0, 255}:
            raise ValueError('expected 512px RGB and binary mask')
        inside = mask == 255
        after = before.copy()
        after[~inside] = 128
        Image.fromarray(after).save(case / 'image.png')
        shutil.copyfile(source_image, case / 'before.png')
        shutil.copyfile(source_mask, case / 'mask.png')
        saved = np.asarray(Image.open(case / 'image.png'))
        changed_inside = int(np.any(saved[inside] != before[inside], axis=1).sum())
        background_ok = bool(np.all(saved[~inside] == 128))
        if changed_inside or not background_ok or sha256(case / 'mask.png') != sha256(source_mask):
            raise ValueError('foreground/mask preservation or gray-background check failed')
        manifest.append({'id': row['id'], 'shape': row['shape'], 'view': row['view'], 'caption': row['caption'],
                         'image': f"{row['id']}/image.png", 'mask': f"{row['id']}/mask.png",
                         'image_sha256': sha256(case / 'image.png'), 'mask_sha256': sha256(case / 'mask.png'),
                         'source_image': str(source_image), 'source_image_sha256': row['image_sha256'],
                         'foreground_changed_pixels': changed_inside, 'background_rgb': [128, 128, 128],
                         'background_uniform': background_ok, 'source_foreground_metrics': row['metrics'],
                         'training_eligible': False})
        x, y = (i % 3) * 384, (i // 3) * 414
        draw.text((x + 8, y + 8), row['id'], fill='black')
        sheet.paste(Image.fromarray(saved).resize((384, 384), Image.Resampling.LANCZOS), (x, y + 30))
        parts.append(f'<tr><td><a href="{row["id"]}/before.png"><img width="512" src="{row["id"]}/before.png"></a></td>'
                     f'<td><a href="{row["id"]}/image.png"><img width="512" src="{row["id"]}/image.png"></a>'
                     f'<p>{row["id"]} | foreground changed: 0 pixels</p></td></tr>')
    verify_source()
    write_json(output / 'manifest.json', manifest)
    audit = {'count': 9, 'foreground_changed_pixels': 0, 'all_backgrounds_exact_srgb128': True,
             'all_masks_byte_identical': True, 'old_source_hashes_verified_before_after': True,
             'source_geometry_user_accepted': True, 'background_visual_review': 'pending',
             'training_started': False}
    write_json(output / 'audit.json', audit)
    write_json(output / 'protocol.json', {'git': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                                        'source': str(SOURCE), 'source_index_sha256': SOURCE_INDEX_SHA,
                                        'background_rgb_srgb_uint8': [128, 128, 128],
                                        'method': 'mask==255 preserves source RGB; mask==0 replaced with gray128',
                                        'scope': 'background replacement only, no rendering or training',
                                        'limitations': 'Binary mask edges retained; cast shadows removed; foreground shading unchanged.'})
    sheet.save(output / 'contact_sheet.png')
    (output / 'index.html').write_text('\n'.join(parts) + '</table>', encoding='utf8')
    write_json(output / 'files.json', {str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file()})
    print(json.dumps(audit))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
