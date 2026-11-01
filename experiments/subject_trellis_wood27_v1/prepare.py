"""Prepare24 unchanged studio endpoint renders plus three original-photo anchors."""

import argparse
from collections import Counter
import html
from itertools import product
import json
from pathlib import Path
import shutil
import textwrap
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd21_wood_branches_v1 import contract as source_contract
from src.methods.colorpeel_ice.natural_subject_recolor_pilot import recolor
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
STUDY = 'subject_trellis_wood27_v1'
SOURCE_INPUT = Path('/var/tmp/colorpeel-r12user5/runs/subject_trellis_spoons_v1/'
                    '20261028-013740__assets__2943bf6__42/inputs/wood_spoon')
ORIGINAL = ROOT / 'experiments/natural_material_threeway_v1/assets/wood_spoon_source.png'
ORIGINAL_SHA = 'b17fb667be977d2ed879f3c216011e6dde8324951a6f331136a7c52ab2bc4f9d'
MASK_SHA = '9647fbc0ca9ad0c8155ca17ad1e8987573c5adc061af6dbd5062000b0e230383'
HUES = {'blue': 240.0, 'green': 120.0}
PALETTE = ROOT / 'experiments/natural_image_subject_color_pilot/configs/d1_subject_recolor_mailbox_pilot_protocol_v1.json'


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def select24(rows):
    chosen = [row for row in rows if row['color'] in HUES and row['light'] == 'studio'
              and row['controls']['metallic'] in (0, 1) and row['controls']['roughness'] in (.15, .85)]
    expected = set(product(('left', 'center', 'right'), HUES, (0, 1), (.15, .85)))
    actual = [(row['view'], row['color'], row['controls']['metallic'], row['controls']['roughness']) for row in chosen]
    if len(chosen) != 24 or set(actual) != expected or len({row['id'] for row in chosen}) != 24:
        raise ValueError('require every color/material endpoint/view once under studio, exactly24 images')
    return chosen


def recolor_photo(rgb, soft_alpha, hue):
    # Pure white is exactly neutral. Matrix roundoff makes its computed L slightly
    # greater than100; leave its RGB untouched instead of clipping Lab or RGB.
    editable = (soft_alpha > 0) & np.any(rgb != 255, axis=-1)
    edit_mask = editable.astype(np.uint8) * 255
    alpha16 = soft_alpha.astype(np.uint16) * 257
    alpha16[~editable] = 0
    result, metrics, maps = recolor(rgb, edit_mask, alpha16, hue)
    if not np.array_equal(result[~editable], rgb[~editable]):
        raise ValueError('photo background or exact white pixels changed')
    metrics.update(pure_white_support_pixels_preserved=int(((soft_alpha > 0) & ~editable).sum()),
                   source_alpha_support_pixels=int((soft_alpha > 0).sum()),
                   hue_degrees=hue, editing_mask='source_alpha>0 and RGB is not exact white')
    return result, metrics, maps, edit_mask


def stage_row(row, output, image_source=None, mask_source=None, image=None, mask=None):
    result = dict(row)
    for name, source, pixels in (('image', image_source, image), ('mask', mask_source, mask)):
        path = output / 'dataset/assets' / row['id'] / ('images' if name == 'image' else 'masks') / 'image.png'
        path.parent.mkdir(parents=True)
        if source is not None:
            shutil.copyfile(source, path)
            if sha256(path) != row[name + '_sha256']:
                raise ValueError('selected render copy differs from its immutable source')
            result['parent_' + name] = str(source)
        else:
            Image.fromarray(pixels).save(path)
        result[name] = str(path)
        result[name + '_sha256'] = sha256(path)
        result['review_image'] = path.relative_to(output).as_posix() if name == 'image' else result['review_image']
    result.update(training_eligible=False, review_status='dataset_constructed_pending_user_review')
    if result['caption'].count('<S*>') != 1:
        raise ValueError('require exactly one Subject token per caption')
    return result


def review(output, rows):
    # The manifest retains source order; show the three photographic anchors first.
    displayed = rows[24:] + rows[:24]
    canvas = Image.new('RGB', (4 * 256, 7 * 344 + 42), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), 'Wood spoon27: 24 TRELLIS studio endpoints + original / LCh blue / LCh green. NO TRAINING.', fill='black')
    cards = []
    for position, row in enumerate(displayed):
        slot = position if position < 3 else position + 1
        x, y = slot % 4 * 256, slot // 4 * 344 + 42
        with Image.open(row['image']) as image:
            canvas.paste(image.convert('RGB').resize((248, 248)), (x + 4, y))
        draw.text((x + 5, y + 252), row['id'], fill='black')
        for line, caption in enumerate(textwrap.wrap(row['caption'], 35)):
            draw.text((x + 5, y + 270 + line * 14), caption, fill='black')
        cards.append(f'<figure><a href="{row["review_image"]}"><img loading="lazy" src="{row["review_image"]}"></a>'
                     f'<figcaption>{html.escape(row["id"])}<br>{html.escape(row["caption"])}</figcaption></figure>')
    canvas.save(output / 'contact_sheet27.png')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>wood spoon27 dataset</title>'
        '<style>body{font:16px sans-serif}section{display:flex;flex-wrap:wrap;gap:12px}figure{width:256px;margin:0}img{width:256px}figcaption{overflow-wrap:anywhere}</style>'
        '<h1>Wood spoon:27 actual dataset RGB images and captions</h1>'
        '<p>24 byte-identical TRELLIS studio endpoint renders, one resized original photo, two hue-only LCh recolors. No training.</p>'
        '<p>Photographs retain their original background and illumination; they are not additional studio renders. LCh palette angles differ from renderer RGB constants.</p>'
        '<a href="contact_sheet27.png">Contact sheet</a><section>' + ''.join(cards) + '</section>', encoding='utf8')


def verify_output(output):
    output = Path(output).resolve()
    rows = [json.loads(line) for line in (output / 'dataset/training_assets_manifest.jsonl').read_text(encoding='utf8').splitlines()]
    if len(rows) != 27 or len({row['id'] for row in rows}) != 27:
        raise ValueError('dataset must contain exactly27 unique rows')
    select24(rows[:24])
    if [row['id'] for row in rows[24:]] != ['photo_original', 'photo_lch_blue', 'photo_lch_green']:
        raise ValueError('photo anchors or manifest order differs')
    concepts = []
    for row in rows:
        for field, mode in (('image', 'RGB'), ('mask', 'L')):
            path = Path(row[field])
            if not path.is_relative_to(output / 'dataset/assets') or sha256(path) != row[field + '_sha256']:
                raise ValueError('dataset path or bytes differ from manifest')
            if set(path.parent.iterdir()) != {path}:
                raise ValueError('require one image and one mask per caption concept')
            with Image.open(path) as image:
                if image.mode != mode or image.size != (512, 512):
                    raise ValueError('require aligned512px RGB and L mask')
                if field == 'mask' and set(np.unique(np.array(image))) != {0, 255}:
                    raise ValueError('training object mask must be nonempty binary0/255')
        if row['caption'].count('<S*>') != 1 or row['training_eligible'] is not False:
            raise ValueError('caption token or dataset-only policy differs')
        concepts.append({'instance_prompt': [row['caption']], 'instance_data_dir': str(Path(row['image']).parent),
                         'instance_mask_dir': str(Path(row['mask']).parent)})
    if read_json(output / 'dataset/concepts.json') != concepts:
        raise ValueError('one-image one-caption pairing differs')
    return rows


def run(output):
    output = resolve_run_dir(Path(output))
    if output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(SOURCE_INPUT.parent):
        raise ValueError('use a fresh external dataset folder separate from source assets')
    revision = git_info()
    frozen = source_contract.verify_dataset('subject')
    selected = select24(frozen['references'])
    source_receipt = read_json(SOURCE_INPUT / 'source_receipt.json')
    if (sha256(ORIGINAL) != ORIGINAL_SHA or source_receipt['source_sha256'] != ORIGINAL_SHA
            or sha256(SOURCE_INPUT / 'mask512.png') != MASK_SHA
            or source_receipt['mask_sha256'] != MASK_SHA
            or source_receipt['transform']['original_size'] != [2560, 2560]
            or source_receipt['transform']['cropped'] is not False):
        raise ValueError('original photo, frozen segmentation or original-canvas transform differs')
    palette = {row['name']: row['hue_degrees'] for row in read_json(PALETTE)['auxiliary_palette']}
    if any(palette[name] != hue for name, hue in HUES.items()):
        raise ValueError('inherited Subject LCh palette changed')
    locks = {**frozen['source_locks'], str(ORIGINAL): ORIGINAL_SHA,
             str(SOURCE_INPUT / 'mask512.png'): MASK_SHA, str(PALETTE): sha256(PALETTE),
             str(SOURCE_INPUT / 'source_receipt.json'): sha256(SOURCE_INPUT / 'source_receipt.json')}
    output.mkdir(parents=True)
    (output / 'sources').mkdir()
    for source, name in ((ORIGINAL, 'original_full_resolution.png'), (SOURCE_INPUT / 'mask512.png', 'source_soft_alpha512.png'),
                         (SOURCE_INPUT / 'source_receipt.json', 'source_receipt.json')):
        shutil.copyfile(source, output / 'sources' / name)
    protocol = {'study': STUDY, 'schema': 'wood_spoon_subject27_dataset/v1', 'git': revision,
        'composition': {'synthetic': 24, 'original_photo': 1, 'lch_recolors': 2, 'total': 27},
        'selection': {'colors': ['blue', 'green'], 'metallic': [0, 1], 'roughness': [.15, .85],
                      'views': ['left', 'center', 'right'], 'lights': ['studio']},
        'synthetic_source': frozen['dataset'], 'synthetic_policy': 'byte-copy existing RGB/mask/caption; retain source order',
        'photo_policy': 'full original RGB canvas Lanczos2560to512; preserve original background and illumination',
        'training_mask': 'existing soft alpha>=128; no new segmentation or morphology',
        'lch': {'source': str(PALETTE), 'space': 'CIELAB D65', 'hues_degrees': HUES,
                'preserve_L': True, 'preserve_source_C_except_gamut': True, 'gamut_margin': .995,
                'binary_search_iterations': 32, 'alpha8_to_uint16_multiplier': 257,
                'pure_white_policy': 'exactRGB255 pixels unchanged as neutral; exclude only from editing mask',
                'training_mask_unchanged_by_editing_mask': True},
        'training_started': False, 'training_authorized': False, 'monitoring': False}
    write(output / 'protocol.json', protocol)
    rows = [stage_row(row, output, image_source=row['image'], mask_source=row['mask']) for row in selected]
    with Image.open(ORIGINAL) as source:
        rgb = np.array(source.convert('RGB').resize((512, 512), Image.Resampling.LANCZOS))
    with Image.open(SOURCE_INPUT / 'mask512.png') as source:
        alpha = np.array(source.convert('L'))
    object_mask = np.where(alpha >= 128, 255, 0).astype(np.uint8)
    Image.fromarray(rgb).save(output / 'sources/original_rgb512.png')
    base = {'subject_id': 'wood_spoon', 'view': 'original_photo', 'light': 'original_photo', 'controls': None,
            'caption': 'a photo of <S*> spoon with a wood surface', 'color': 'original', 'kind': 'original_photograph'}
    rows.append(stage_row({**base, 'id': 'photo_original'}, output, image=rgb, mask=object_mask))
    measurements = {}
    for color, hue in HUES.items():
        colored, metrics, maps, edit_mask = recolor_photo(rgb, alpha, hue)
        Image.fromarray(edit_mask).save(output / 'sources' / f'lch_{color}_editing_mask.png')
        np.savez_compressed(output / 'sources' / f'lch_{color}_maps.npz', **maps)
        rows.append(stage_row({**base, 'id': f'photo_lch_{color}', 'kind': 'lch_recolor', 'color': color,
                               'caption': f'a photo of <S*> spoon in {color} color with a wood surface',
                               'lch_hue_degrees': hue, 'parent_photo': 'photo_original'},
                              output, image=colored, mask=object_mask))
        measurements[color] = metrics
    ledger = output / 'dataset/training_assets_manifest.jsonl'
    ledger.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
    concepts = [{'instance_prompt': [row['caption']], 'instance_data_dir': str(Path(row['image']).parent),
                 'instance_mask_dir': str(Path(row['mask']).parent)} for row in rows]
    write(output / 'dataset/concepts.json', concepts)
    write(output / 'lch_measurements.json', measurements)
    write(output / 'source_locks.json', locks)
    verified = verify_output(output)
    if any(sha256(path) != digest for path, digest in locks.items()) or git_info() != revision:
        raise ValueError('source files or execution revision changed while building the dataset')
    review(output, verified)
    files = {path.relative_to(output).as_posix(): sha256(path) for path in sorted(output.rglob('*')) if path.is_file()}
    state = {'status': 'dataset27_constructed_pending_user_review', 'git': revision, 'rows': 27,
             'synthetic_rows': 24, 'photo_rows': 3, 'colors': dict(Counter(row['color'] for row in rows)),
             'synthetic_copies_byte_identical': True, 'training_started': False, 'monitoring': False,
             'lch_measurements': measurements, 'files': files, 'source_locks': locks}
    write(output / 'dataset_audit.json', state)
    archive = output.with_name(output.name + '.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                bundle.write(path, path.relative_to(output).as_posix())
    print(json.dumps({'status': state['status'], 'output': str(output), 'rows': 27,
                      'archive': str(archive), 'archive_sha256': sha256(archive)}), flush=True)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.output)
