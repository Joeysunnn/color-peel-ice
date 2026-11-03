"""Reuse audited same-photo RGB/masks, with explicit new caption provenance."""

import html
import json
from pathlib import Path
import shutil

from PIL import Image

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sdxl_mailbox_v1.data import source_rows, _check_locks
from experiments.sd21_mailbox_center11_v1.contract import APPROVED_PATH
from experiments.subject_multicolor_multimaterial_b.training import _write
from . import contract


def caption(row, branch, category):
    text = row.get('prompt', row.get('caption'))
    if branch == 'subject':
        if text.count('mailbox') != 1:
            raise ValueError('expected the reviewed Subject category exactly once')
        text = text.replace('mailbox', category)
    elif branch == 'material':
        if text.count('made of <M*>') != 1:
            raise ValueError('unexpected historical Material grammar')
        text = text.replace('made of <M*>', 'with a <M*> surface')
    return text


def validate_caption(tokenizer, text, branch):
    if not isinstance(text, str) or text.count(contract.TOKENS[branch]) != 1:
        raise ValueError('caption must contain exactly one branch modifier')
    if any(token in text for other, token in contract.TOKENS.items() if other != branch):
        raise ValueError('caption leaks another branch modifier')
    ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    if len(ids) > 77 or ids.count(tokenizer.convert_tokens_to_ids(contract.TOKENS[branch])) != 1:
        raise ValueError('caption truncates or does not encode exactly one modifier')
    return ids


def concepts(rows):
    result = []
    for row in rows:
        item = {'instance_prompt': [row['caption']], 'instance_data_dir': str(Path(row['image']).parent)}
        if row['mask']:
            item['instance_mask_dir'] = str(Path(row['mask']).parent)
        result.append(item)
    return result


def prepare(output, manifest, tokenizer):
    output = Path(output).resolve()
    if manifest['source']['image_sha256'] != contract.SOURCE_SHA:
        raise ValueError('this image has no audited derived dataset; Mailbox data cannot be attached to it')
    if output.exists() or output.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('require a fresh external dataset directory')
    inputs, locks = source_rows()
    approved = read_json(APPROVED_PATH)
    by_id = {row['id']: row for row in inputs['subject']}
    selected = []
    for expected in approved:
        row = by_id[expected['id']]
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError('Subject sources differ from the approved center11')
        selected.append(row)
    inputs['subject'] = selected
    locks[str(APPROVED_PATH.resolve())] = sha256(APPROVED_PATH)
    if tokenizer.add_tokens(list(contract.TOKENS.values())) != 4:
        raise ValueError('require the unmodified base tokenizer for caption preflight')
    output.mkdir(parents=True)
    files, sections = {}, []
    for branch in contract.BRANCHES:
        rows = []
        for index, source in enumerate(inputs[branch]):
            row = {'id': source['id'], 'source_caption': source.get('prompt', source.get('caption')),
                   'caption': caption(source, branch, manifest['subject_category'])}
            row['caption_token_ids'] = validate_caption(tokenizer, row['caption'], branch)
            for field in ('image', 'mask'):
                path = source.get(field)
                row[field] = row[field + '_sha256'] = None
                if path:
                    dest = output / branch / f'{index:03d}' / field / 'image.png'
                    dest.parent.mkdir(parents=True)
                    shutil.copyfile(path, dest)
                    row.update({field: str(dest), field + '_sha256': source[field + '_sha256'],
                                'source_' + field: path})
                    files[str(dest)] = source[field + '_sha256']
            rows.append(row)
        folder = output / branch
        (folder / 'training_assets_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
        _write(folder / 'concepts.json', concepts(rows))
        for name in ('training_assets_manifest.jsonl', 'concepts.json'):
            files[str(folder / name)] = sha256(folder / name)
        cards = ''.join('<figure><img src="' + html.escape(Path(row['image']).relative_to(output).as_posix())
                        + '"><figcaption>' + html.escape(row['caption']) + '</figcaption></figure>' for row in rows)
        sections.append(f'<h2>{branch}: {len(rows)}</h2><section>{cards}</section>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Qwen initializer training inputs</title>'
        '<style>body{font:16px sans-serif;margin:24px}section{display:flex;flex-wrap:wrap;gap:16px}'
        'figure{width:256px;margin:0}img{width:256px}figcaption{overflow-wrap:anywhere}</style>'
        '<h1>Actual 44 training images and captions, SD2.1 512px</h1>' + ''.join(sections), encoding='utf8')
    _check_locks({**locks, **files})
    _write(output / 'receipt.json', {'source_locks': locks, 'files': files, 'counts': contract.COUNTS,
        'source_image_sha256': contract.SOURCE_SHA, 'category': manifest['subject_category'],
        'copies_byte_identical': True, 'old_weights_reused': False})
    return {branch: verify_dataset(output, branch) for branch in contract.BRANCHES}


def verify_dataset(output, branch):
    from src.train.instance_mask_utils import pair_instance_images_and_masks
    output = Path(output).resolve()
    receipt = read_json(output / 'receipt.json')
    if receipt['counts'] != contract.COUNTS or receipt['source_image_sha256'] != contract.SOURCE_SHA:
        raise ValueError('dataset source or counts differ')
    locks = {**receipt['source_locks'], **receipt['files'], str(output / 'receipt.json'): sha256(output / 'receipt.json')}
    _check_locks(locks)
    folder = output / branch
    rows = [json.loads(line) for line in (folder / 'training_assets_manifest.jsonl').read_text(encoding='utf8').splitlines()]
    if len(rows) != contract.COUNTS[branch] or read_json(folder / 'concepts.json') != concepts(rows):
        raise ValueError('one-image/caption pairing differs')
    for row in rows:
        if row['mask']:
            pair_instance_images_and_masks(Path(row['image']).parent, Path(row['mask']).parent)
        for field, mode in (('image', 'RGB'), ('mask', 'L')):
            if not row[field]:
                if branch != 'color' or field != 'mask':
                    raise ValueError('required RGB or foreground mask missing')
                continue
            path = Path(row[field])
            if not path.is_relative_to(folder) or set(path.parent.iterdir()) != {path}:
                raise ValueError('one-image concept directory changed')
            with Image.open(path) as image:
                if image.mode != mode or image.size != (512, 512):
                    raise ValueError('unchanged 512px RGB/L inputs required')
    return {'branch': branch, 'dataset': str(folder), 'rows': len(rows), 'references': rows,
            'source_locks': locks, 'concepts': str(folder / 'concepts.json'),
            'manifest': str(folder / 'training_assets_manifest.jsonl'), 'token': contract.TOKENS[branch]}
