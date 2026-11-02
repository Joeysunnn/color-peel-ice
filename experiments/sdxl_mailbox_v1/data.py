"""Copy audited historical RGB/masks; resize only when reading training inputs."""

import hashlib
import itertools
import json
from pathlib import Path
import shutil

from PIL import Image, ImageChops, ImageDraw

from .contract import BRANCHES, LATENT_SIZE, RESOLUTION, TRAINING

ROOT = Path(__file__).resolve().parents[2]
SUBJECT = Path('/var/tmp/colorpeel-r12user5/runs/sd21_mailbox_subject27_v1/'
               '20261101-221114__both600__a513371__42/experiment/dataset')
COLOR = Path('/var/tmp/colorpeel-r12user5/runs/color_geometry_lightness_v1/'
             '20261101-175834__rendered_background9__e44d9cb__42/result9')
COLOR_INDEX_SHA = 'c712520a3b92c2b1c7f5990f4e595e0db692529a948758cf63af60a4dcdbc700'
NATURAL = Path('/var/tmp/colorpeel-r12user5/runs/natural_texture_material_v1/'
               '20261018-175054__counterfactual_dataset__c503bcd__42')
NATURAL_PROTOCOL_SHA = '2f0dbf61f0b969e17dc2c9beb4b65b4b0a4814ca2a62e38b6ac062c94d456127'
NATURAL_CANONICAL_SHA = '9746286e465a978f72b24ded443dc666d590d41e61f53b390d38a66481487259'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf8')


def _check_locks(locks):
    if not locks or any(not Path(path).is_absolute() or sha256(path) != digest
                        for path, digest in locks.items()):
        raise ValueError('frozen source or staged file differs from its hash')


def _ledger(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line]


def source_rows():
    """Validate stored provenance without changing historical eligibility flags."""
    audit_path = SUBJECT / 'dataset_audit.json'
    audit = read_json(audit_path)
    if (audit['rows'], audit['synthetic_rows'], audit['photo_rows'], audit['subject_id']) != (27, 24, 3, 'mailbox'):
        raise ValueError('require the frozen Mailbox24+3 data')
    locks = {**audit['source_locks'], **audit['files'], str(audit_path): sha256(audit_path)}
    _check_locks(locks)
    subject = _ledger(SUBJECT / 'training_assets_manifest.jsonl')
    if [row['id'] for row in subject[24:]] != ['photo_original', 'photo_lch_blue', 'photo_lch_green']:
        raise ValueError('photo order differs')
    result = {'subject': subject}

    index = COLOR / 'files.json'
    if sha256(index) != COLOR_INDEX_SHA:
        raise ValueError('rendered-background Color9 index changed')
    locks[str(index)] = COLOR_INDEX_SHA
    for name, digest in read_json(index).items():
        path = (COLOR / name).resolve()
        if not path.is_relative_to(COLOR.resolve()):
            raise ValueError('Color source path escapes its root')
        locks[str(path)] = digest
    expected = [(shape, view) for shape in ('cube', 'sphere', 'cylinder') for view in (0, 8, 16)]
    color = read_json(COLOR / 'manifest.json')
    if len(color) != 9 or {(row['shape'], row['view']) for row in color} != set(expected):
        raise ValueError('Color geometry/view coverage differs')
    color.sort(key=lambda row: expected.index((row['shape'], row['view'])))
    for row in color:
        if row['caption'] != f"a photo of {row['shape']} shape in <C*> color":
            raise ValueError('Color caption changed')
        row['image'] = str((COLOR / row['image']).resolve())
        if locks.get(row['image']) != row['image_sha256']:
            raise ValueError('Color row differs from its pinned index')
        row['mask'] = None
        row['mask_sha256'] = None
    result['color'] = color

    protocol_path = NATURAL / 'plan/protocol.json'
    if sha256(protocol_path) != NATURAL_PROTOCOL_SHA:
        raise ValueError('natural dataset protocol changed')
    protocol = read_json(protocol_path)
    locks.update(protocol['source_locks'])
    locks[str(protocol_path)] = NATURAL_PROTOCOL_SHA
    staging_audit_path = NATURAL / 'staging/audit.json'
    staging_audit = read_json(staging_audit_path)
    locks[str(staging_audit_path)] = sha256(staging_audit_path)
    if staging_audit['protocol_sha256'] != NATURAL_CANONICAL_SHA:
        raise ValueError('natural staging protocol differs')
    rendered = read_json(NATURAL / 'renders/manifest.json')
    if rendered['status'] != 'complete' or rendered['protocol_sha256'] != NATURAL_CANONICAL_SHA:
        raise ValueError('natural renders do not use the pinned protocol')
    render_rows = {row['id']: row for row in rendered['records']}
    for branch in ('material', 'texture'):
        folder = NATURAL / 'staging' / ('mailbox_' + branch)
        receipt_path = folder / 'dataset_receipt.json'
        receipt = read_json(receipt_path)
        locks[str(receipt_path)] = staging_audit['datasets']['mailbox_' + branch]['dataset_receipt_sha256']
        locks[str(folder / 'concepts.json')] = receipt['concepts_sha256']
        locks[str(folder / 'training_assets_manifest.jsonl')] = receipt['training_assets_manifest_sha256']
        locks[str(NATURAL / 'renders/manifest.json')] = receipt['source_render_manifest_sha256']
        if (receipt['status'], receipt['source_id'], receipt['branch'], receipt['image_count'],
                receipt['protocol_sha256']) != ('verified', 'mailbox', branch, TRAINING[branch]['images'], NATURAL_CANONICAL_SHA):
            raise ValueError('Mailbox natural dataset receipt differs')
        items = _ledger(folder / 'training_assets_manifest.jsonl')
        nuisances = ('smooth_dielectric', 'rough_metal') if branch == 'texture' else ('mailbox_material',)
        expected_conditions = list(itertools.product(('sphere', 'cube'), ('red', 'blue'),
                                                     ('soft_front', 'side_directional'), nuisances))
        if [(row['shape'], row['color'], row['light'], row['material_id']) for row in items] != expected_conditions:
            raise ValueError('Mailbox natural factorial order differs')
        for row in items:
            original = render_rows[row['id']]
            if row['source_id'] != 'mailbox' or row['concept_id'] != 'mailbox_' + branch:
                raise ValueError('require Mailbox natural inputs')
            source_image = (NATURAL / 'renders' / original['image']).resolve()
            source_mask = (NATURAL / 'renders' / original['mask']).resolve()
            if (row['source_image'] != str(source_image) or row['source_mask'] != str(source_mask)
                    or row['image_sha256'] != original['image_sha256']
                    or row['source_mask_sha256'] != original['mask_sha256']):
                raise ValueError('staged natural RGB/mask lineage differs')
            locks[str(source_image)] = original['image_sha256']
            locks[str(source_mask)] = original['mask_sha256']
            light = protocol['dataset']['light_phrases'][row['light']]
            prompt = f"a photo of a {row['color']} {row['shape']}"
            if branch == 'texture':
                material = protocol['materials'][row['material_id']]['literal']
                prompt += f' with <T*> texture made of {material} under {light}'
            else:
                prompt += f' made of <M*> under {light}'
            if row['prompt'] != prompt:
                raise ValueError('natural caption differs from pinned protocol')
            with Image.open(source_mask) as raw, Image.open(row['mask']) as mask:
                binary = raw.convert('L').point(lambda value: 255 if value >= 128 else 0)
                if ImageChops.difference(binary, mask.convert('L')).getbbox() is not None:
                    raise ValueError('natural staged mask differs from renderer threshold')
        result[branch] = items
    for branch, items in result.items():
        _validate_rows(items, branch)
        for row in items:
            locks[row['image']] = row['image_sha256']
            if row.get('mask'):
                locks[row['mask']] = row['mask_sha256']
    _check_locks(locks)
    return result, locks


def _validate_rows(items, branch):
    spec = TRAINING[branch]
    if len(items) != spec['images'] or len({row['id'] for row in items}) != len(items):
        raise ValueError('training count or unique IDs differ')
    for row in items:
        prompt = row.get('prompt', row.get('caption'))
        if not isinstance(prompt, str) or prompt.count(spec['token']) != 1:
            raise ValueError('training caption must contain its branch token once')
        for name in ('image', 'mask'):
            if name == 'mask' and branch == 'color':
                if row.get(name) is not None:
                    raise ValueError('Color uses the whole valid image, not an object mask')
                continue
            path = Path(row[name])
            if not path.is_absolute() or sha256(path) != row[name + '_sha256']:
                raise ValueError('training image/mask hash differs')
            with Image.open(path) as opened:
                if opened.size != (512, 512) or opened.mode != ('RGB' if name == 'image' else 'L'):
                    raise ValueError('require unchanged512px RGB and binary L mask')
                if name == 'mask':
                    values = {value for value, count in enumerate(opened.histogram()) if count}
                    if not values <= {0, 255} or 255 not in values:
                        raise ValueError('object mask must contain binary foreground')


def prepare_data(output):
    """Create one fresh dataset root, returning manifest paths for all branches."""
    output = Path(output).resolve()
    sources = (ROOT, SUBJECT, COLOR, NATURAL)
    if output.exists() or any(output.is_relative_to(path.resolve()) or path.resolve().is_relative_to(output)
                              for path in sources):
        raise ValueError('use a new external directory separate from source data and code')
    inputs, locks = source_rows()
    output.mkdir(parents=True)
    specs, files = {}, {}
    for branch in BRANCHES:
        copied = []
        for index, row in enumerate(inputs[branch]):
            directory = output / branch / f'{index:03d}'
            directory.mkdir(parents=True)
            new = {'id': row['id'], 'prompt': row.get('prompt', row.get('caption'))}
            for name in ('image', 'mask'):
                source = row.get(name)
                if source is None:
                    new[name] = new[name + '_sha256'] = None
                    continue
                target = directory / (name + '.png')
                shutil.copyfile(source, target)
                new[name], new[name + '_sha256'] = str(target), row[name + '_sha256']
                new['source_' + name] = source
                files[str(target)] = row[name + '_sha256']
            copied.append(new)
        _validate_rows(copied, branch)
        manifest = output / branch / 'manifest.json'
        _write(manifest, copied)
        files[str(manifest)] = sha256(manifest)
        specs[branch] = {'manifest': str(manifest), 'rows': len(copied),
                         'token': TRAINING[branch]['token'], 'initializer': TRAINING[branch]['initializer']}
    _check_locks(locks)
    _write(output / 'receipt.json', {'schema': 'sdxl_mailbox_data/v1', 'branches': specs,
           'source_locks': locks, 'files': files, 'copied_source_bytes_unchanged': True,
           'training_preprocessing': {'resolution': RESOLUTION, 'image_resize': 'LANCZOS',
               'mask_resize': 'NEAREST', 'latent_mask_size': LATENT_SIZE, 'valid_latent_border': 1,
               'shuffle': False, 'color_object_mask': False}})
    verify(output)
    return specs


def verify(output):
    receipt = read_json(Path(output) / 'receipt.json')
    if receipt.get('schema') != 'sdxl_mailbox_data/v1' or set(receipt['branches']) != set(BRANCHES):
        raise ValueError('invalid staged dataset receipt')
    _check_locks(receipt['source_locks'])
    _check_locks(receipt['files'])
    for branch, spec in receipt['branches'].items():
        _validate_rows(read_json(spec['manifest']), branch)
    return receipt


def load_rows(manifest):
    manifest = Path(manifest)
    branch = manifest.parent.name
    receipt = read_json(manifest.parent.parent / 'receipt.json')
    if receipt['files'].get(str(manifest.resolve())) != sha256(manifest):
        raise ValueError('staged manifest changed')
    items = read_json(manifest)
    _validate_rows(items, branch)
    return items


def rows(output, branch):
    return load_rows(Path(output) / branch / 'manifest.json')


def load_training_image(row):
    """Return RGB1024 and L128 mask; caller converts them to model tensors."""
    with Image.open(row['image']) as opened:
        image = opened.convert('RGB').resize((RESOLUTION, RESOLUTION), Image.Resampling.LANCZOS)
    valid = Image.new('L', (LATENT_SIZE, LATENT_SIZE), 0)
    ImageDraw.Draw(valid).rectangle((1, 1, LATENT_SIZE - 2, LATENT_SIZE - 2), fill=255)
    if row.get('mask'):
        with Image.open(row['mask']) as opened:
            mask = opened.convert('L').resize((RESOLUTION, RESOLUTION), Image.Resampling.NEAREST)
            mask = mask.resize((LATENT_SIZE, LATENT_SIZE), Image.Resampling.NEAREST)
        valid = ImageChops.multiply(valid, mask)
    if valid.getbbox() is None:
        raise ValueError('object has no valid latent foreground')
    return image, valid
