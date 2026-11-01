"""Apply the unchanged foreground remap and review freshly rendered backgrounds."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from experiments.color_geometry_lightness_v1.prepare import remap, write_json, ROOT
from experiments.color_geometry_lightness_v1.render_background import SOURCE
from experiments.mailbox_color_sd15_v1.data import source_snapshot, check_locks
from experiments.sd15_token_local_kv_v1.contract import sha256


def run(output, inputs):
    output = output.resolve()
    if output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(SOURCE):
        raise ValueError('require fresh external review directory')
    old_rows, locks, target, _ = source_snapshot()
    captions = {(r['shape'], r['view']): r['caption'] for r in old_rows}
    cases = []
    for root in inputs:
        index = json.loads((root / 'files.json').read_text())
        for relative, digest in index.items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()) or sha256(path) != digest:
                raise ValueError('render output hash mismatch')
        locks.update(json.loads((root / 'source_locks.json').read_text()))
        cases.extend(sorted(root.glob('neutral__*/metadata.json')))
    check_locks(locks)
    if len(cases) not in (3, 9):
        raise ValueError('expected sphere3 or complete9')
    output.mkdir(parents=True)
    rows = []
    for metadata in cases:
        record = json.loads(metadata.read_text())
        shape, view = record['request']['shape'], record['request']['view_index']
        identifier = f'{shape}_v{view:02d}'
        dest = output / identifier
        dest.mkdir()
        source = metadata.parent
        old = SOURCE / 'neutral' / 'renders' / source.name
        mask = np.asarray(Image.open(source / 'mask.png').convert('L'))
        old_mask = np.asarray(Image.open(old / 'mask.png').convert('L'))
        if not np.array_equal(mask, old_mask):
            raise ValueError(f'{identifier}: foreground mask changed')
        neutral = np.asarray(Image.open(source / 'image.png').convert('RGB'))
        mapped, metrics = remap(neutral, mask)
        if not metrics['numeric_pass']:
            raise ValueError(f'{identifier}: mapping failed numerical checks')
        for src, name in ((source / 'image.png', 'neutral.png'), (source / 'mask.png', 'mask.png'),
                          (metadata, 'metadata.json'), (old / 'image.png', 'old_neutral.png')):
            shutil.copyfile(src, dest / name)
        Image.fromarray(mapped).save(dest / 'image.png')
        row = {'id': identifier, 'shape': shape, 'view': view, 'caption': captions[shape, view],
               'image': f'{identifier}/image.png', 'mask': f'{identifier}/mask.png',
               'image_sha256': sha256(dest / 'image.png'), 'source_render': str(source),
               'metrics': metrics, 'old_mask_identical': True,
               'coverage': record['coverage'], 'training_eligible': False}
        rows.append(row)
    rows.sort(key=lambda r: (['sphere', 'cube', 'cylinder'].index(r['shape']), r['view']))
    if len({(r['shape'], r['view']) for r in rows}) != len(rows):
        raise ValueError('duplicate cases')
    sheet = Image.new('RGB', (1024, 366 * (len(rows) // 3)), 'white')
    draw = ImageDraw.Draw(sheet)
    page = ['<!doctype html><meta charset="utf-8"><title>Rendered gray studio</title>',
            '<style>body{font:16px sans-serif;margin:24px;background:#eee}img{width:32%}</style>',
            '<h1>Rendered continuous gray background</h1>',
            '<p>Background and cast shadows are original render pixels. Foreground uses the unchanged fixed a/b, bounded-L mapping.',
            ' Camera, lights, material and object masks match the frozen source. No training.</p>',
            '<p>Each row: old neutral / new neutral / new mapped RGB.</p>']
    for i, row in enumerate(rows):
        case = output / row['id']
        img = Image.open(case / 'image.png')
        sheet.paste(img.resize((340, 340), Image.Resampling.LANCZOS), ((i % 3) * 342, (i // 3) * 366 + 24))
        draw.text(((i % 3) * 342 + 5, (i // 3) * 366 + 5), row['id'], fill='black')
        page.append(f'<h2>{row["id"]}</h2>')
        for name in ('old_neutral.png', 'neutral.png', 'image.png'):
            link = f'{row["id"]}/{name}'
            page.append(f'<a href="{link}"><img src="{link}"></a>')
    sheet.save(output / 'contact_sheet.png')
    (output / 'index.html').write_text('\n'.join(page), encoding='utf8')
    check_locks(locks)
    write_json(output / 'manifest.json', rows)
    write_json(output / 'source_locks.json', locks)
    write_json(output / 'audit.json', {'count': len(rows), 'target': target,
               'numeric_pass': True, 'old_masks_identical': True, 'background_changed_pixels_by_mapping': 0,
               'camera_light_material_state_equal': True, 'training_started': False,
               'visual_review': 'pending', 'render_inputs': [str(p) for p in inputs]})
    write_json(output / 'files.json', {str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file()})
    print(json.dumps({'count': len(rows), 'numeric_pass': True, 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    run(args.output, args.inputs)
