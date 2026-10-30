"""Copy the original Mailbox's nine verified Emission renders without new rendering."""

import json
import os
from pathlib import Path
import shutil

from PIL import Image

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from scripts.launch.colorpeel_run import expand_value, read_config, resolve_run_dir
from scripts.methods.colorpeel_ice import stage_d1_emission_color_transfer as emission
from src.methods.colorpeel_ice import emission_color_branch_pilot as pilot

ROOT = Path(__file__).resolve().parents[2]
LEGACY_CONFIG = ROOT / 'experiments/color_material_composition_v1/configs/orange_token_local_color_short100.yaml'
ORIGINAL_CONFIG = ROOT / 'experiments/natural_material_supermat_v1/configs/subject_eroded_material_v1.json'
ORIGINAL_IMAGE = ROOT / 'experiments/natural_material_supermat_v1/assets/source_mailbox.jpg'
HISTORICAL_ROOT = '/home/r12user5/Documents/Jiawei/colorpeel-runs'
FILES = ('concepts.json', 'training_assets_manifest.jsonl', 'source_locks.json', 'approval.json')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def check_locks(locks):
    if not locks or any(not Path(path).is_absolute() or not Path(path).is_file()
                        or sha256(Path(path)) != digest for path, digest in locks.items()):
        raise ValueError('immutable Emission or original Mailbox source changed')


def source_snapshot():
    legacy = read_config(LEGACY_CONFIG)['color_material_source']
    environment = {'COLORPEEL_RUN_ROOT': os.environ.get('COLORPEEL_RUN_ROOT', HISTORICAL_ROOT)}
    source = Path(expand_value(legacy['source_root'], environment)).resolve()
    staging = Path(expand_value(legacy['staging_root'], environment)).resolve()
    protocol_path = (ROOT / legacy['protocol']).resolve()
    locks = {str(protocol_path): legacy['protocol_sha256'],
             str(staging / 'staging_manifest.json'): legacy['staging_manifest_sha256'],
             str(staging / 'concepts.json'): legacy['concepts_sha256']}
    check_locks(locks)
    protocol = emission.protocol(protocol_path)
    original = read_json(ORIGINAL_CONFIG)
    if (original['source_sample_id'] != '010_D1GT_81_130'
            or protocol['target']['stable_id'] != 'D1GT:81/130.png'
            or protocol['training']['initializer_token'] != 'orange'
            or sha256(ORIGINAL_IMAGE) != original['source_image_sha256']):
        raise ValueError('require the frozen original orange Mailbox, not a recolored input')
    pilot_protocol = pilot.load_protocol()
    target = next(row for row in pilot_protocol['targets'] if row['stable_id'] == protocol['target']['stable_id'])
    if any(protocol['target'][key] != target[key] for key in ('a', 'b')):
        raise ValueError('Mailbox target differs from the approved natural-image color')
    selected = emission.verified_source(source, protocol)
    manifest = read_json(staging / 'staging_manifest.json')
    records = manifest.get('records', [])
    if (manifest.get('record_count') != 9 or len(records) != 9
            or manifest.get('source_analysis_sha256') != protocol['source_pilot']['analysis_sha256']
            or manifest.get('protocol_sha256') != legacy['protocol_sha256']
            or [row.get('request_id') for row in records] != list(selected)):
        raise ValueError('historical orange9 source order or staging provenance differs')
    concepts, rows = [], []
    for record in records:
        request_id = record['request_id']
        row = selected[request_id]
        staged = staging / request_id / 'img.png'
        if (any(record.get(key) != row[key] for key in ('shape', 'view_index', 'prompt'))
                or record.get('source_image_sha256') != row['image_sha256']
                or sha256(staged) != row['image_sha256']):
            raise ValueError('historical orange9 pixels or captions changed')
        concepts.append({'instance_prompt': [row['prompt']], 'instance_data_dir': str(staged.parent)})
        rows.append({'id': request_id, 'caption': row['prompt'], 'shape': row['shape'],
                     'view': row['view_index'], 'source_image': str(row['image'].resolve()),
                     'source_image_sha256': row['image_sha256']})
        locks[str(staged)] = row['image_sha256']
    if read_json(staging / 'concepts.json') != concepts:
        raise ValueError('historical orange9 concept/image/caption alignment changed')
    source_manifest = read_json(source / emission.MANIFEST_NAME)
    for row in source_manifest['records']:
        for field in ('image', 'metadata'):
            path = (source / row[field + '_relative_path']).resolve()
            if not path.is_relative_to(source):
                raise ValueError('source render artifact escapes the frozen source')
            locks[str(path)] = row[field + '_sha256']
    locks.update({str(source / emission.ANALYSIS_NAME): protocol['source_pilot']['analysis_sha256'],
                  str(source / emission.MANIFEST_NAME): sha256(source / emission.MANIFEST_NAME),
                  str(LEGACY_CONFIG.resolve()): sha256(LEGACY_CONFIG),
                  str(ORIGINAL_CONFIG.resolve()): sha256(ORIGINAL_CONFIG),
                  str(ORIGINAL_IMAGE.resolve()): original['source_image_sha256'],
                  str(ROOT / pilot.PROTOCOL_RELPATH): sha256(ROOT / pilot.PROTOCOL_RELPATH)})
    check_locks(locks)
    return rows, locks, protocol['target'], (source, staging)


def approval():
    return {'status': 'user_authorized_training', 'source': 'original_mailbox_emission9',
            'methods': ['fullrank', 'lora_r64'], 'max_train_steps': 1000,
            'checkpoint_steps': [400, 600, 800, 1000], 'inference_seeds': list(range(30, 61)),
            'scope': 'SD1.5 Color training, existing transfer prompts and paired Subject+Color adapters; preserve all historical results.'}


def training_rows(dataset, sources):
    return [{**row, 'image': str(dataset / 'images' / row['id'] / 'image.png'),
             'image_sha256': row['source_image_sha256']} for row in sources]


def concepts_for(rows):
    return [{'instance_prompt': [row['caption']], 'instance_data_dir': str(Path(row['image']).parent)}
            for row in rows]


def audit_for(dataset, rows, target):
    return {'schema': 'mailbox_color_sd15_emission9/v1', 'status': 'verified', 'rows': 9,
            'subject_id': 'mailbox', 'source_mode': 'verified_original_mailbox_emission',
            'target': target, 'initializer_token': 'orange', 'uses_masks': False,
            'source_order': [row['id'] for row in rows],
            'files': {name: sha256(dataset / name) for name in FILES},
            'sampling': {'shuffle': False, 'updates': 1000, 'complete_cycles': 111,
                         'extra_prefix_images': 1},
            'limitations': 'Nine emission-rendered basic shapes preserve the extracted target color, not original Mailbox geometry or material; authorization is not evidence of disentanglement.'}


def stage_dataset(output):
    output = resolve_run_dir(Path(output))
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError('stage only into a new directory outside the repository')
    sources, locks, target, roots = source_snapshot()
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots):
        raise ValueError('new staging must preserve all historical sources')
    output.mkdir(parents=True)
    rows = training_rows(output, sources)
    for row in rows:
        image = Path(row['image'])
        image.parent.mkdir(parents=True)
        shutil.copyfile(row['source_image'], image)
    write_json(output / 'concepts.json', concepts_for(rows))
    (output / 'training_assets_manifest.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf8')
    write_json(output / 'source_locks.json', locks)
    write_json(output / 'approval.json', approval())
    write_json(output / 'dataset_audit.json', audit_for(output, rows, target))
    return verify_dataset(output)


def verify_dataset(dataset):
    dataset = resolve_run_dir(Path(dataset))
    if dataset.is_relative_to(ROOT):
        raise ValueError('training data must remain outside the repository')
    sources, locks, target, roots = source_snapshot()
    if any(dataset.is_relative_to(root) or root.is_relative_to(dataset) for root in roots):
        raise ValueError('training data overlaps historical sources')
    if read_json(dataset / 'source_locks.json') != locks:
        raise ValueError('immutable Emission source locks changed')
    rows = [json.loads(line) for line in (dataset / 'training_assets_manifest.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    expected = training_rows(dataset, sources)
    if rows != expected or read_json(dataset / 'concepts.json') != concepts_for(expected):
        raise ValueError('orange9 training pixels, source order or captions changed')
    for row in rows:
        path = Path(row['image'])
        if path.is_symlink() or set(path.parent.iterdir()) != {path} or sha256(path) != row['image_sha256']:
            raise ValueError('require byte-identical copies and one actual training image per concept')
        with Image.open(path) as image:
            if image.mode != 'RGB' or image.size != (512, 512):
                raise ValueError('require nine actual512px RGB training images')
    if read_json(dataset / 'approval.json') != approval():
        raise ValueError('training approval or1000-step checkpoint plan changed')
    audit = audit_for(dataset, rows, target)
    if read_json(dataset / 'dataset_audit.json') != audit:
        raise ValueError('training dataset audit differs')
    check_locks(locks)
    return audit
