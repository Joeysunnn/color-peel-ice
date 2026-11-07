"""Stage and audit independent SD2.1 branches for one discovered object."""

import json
from pathlib import Path
import re
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd21_wood_branches_v1 import snapshots
from experiments.sd21_wood_branches_v1.contract import MODEL_DIR
from experiments.subject_multicolor_multimaterial_b.training import _write
from scripts.launch.colorpeel_run import git_info, resolve_run_dir
from src.train.checkpoint_utils import validate_checkpoint_plan

ROOT = Path(__file__).resolve().parents[3]
BRANCHES = ('subject', 'color', 'material', 'texture')
TOKENS = dict(zip(BRANCHES, ('<S*>', '<C*>', '<M*>', '<T*>')))
COUNTS = dict(zip(BRANCHES, (11, 9, 8, 16)))
DEFAULT_STEPS = dict(zip(BRANCHES, (1000, 150, 200, 400)))
STUDY = 'single_image_four_branch_v1'


def training_plan(branches=None, steps=None):
    selected = list(BRANCHES if branches is None else branches)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(BRANCHES):
        raise ValueError('select distinct supported branches')
    overrides = {} if steps is None else dict(steps)
    if set(overrides) - set(BRANCHES) or any(type(value) is not int or value < 1 for value in overrides.values()):
        raise ValueError('training steps must be positive integers for supported branches')
    return {branch: overrides.get(branch, DEFAULT_STEPS[branch]) for branch in selected}


def checkpoint_plan(plan, checkpoint_steps=None):
    plan = training_plan(list(plan), plan)
    overrides = {} if checkpoint_steps is None else dict(checkpoint_steps)
    if set(overrides) - set(plan):
        raise ValueError('checkpoint steps require selected training branches')
    result = {}
    for branch, maximum in plan.items():
        values = overrides.get(branch, [maximum])
        if not isinstance(values, (list, tuple)):
            raise ValueError('checkpoint steps must be an ordered sequence')
        values = list(values)
        validate_checkpoint_plan(values, maximum, maximum)
        if values[-1] != maximum:
            raise ValueError('checkpoint steps must include the final training step')
        result[branch] = values
    return result


def source_locks(record):
    if re.fullmatch(r'object_[0-9]+', record.get('id', '')) is None:
        raise ValueError('require a discovered object identifier')
    locks = {str(Path(record[key]).resolve()): record[key + '_sha256']
             for key in ('source_image', 'object_mask', 'vlm_input')}
    snapshots.check_locks(locks)
    return locks


def _validate_image(path, field):
    with Image.open(path) as image:
        if image.mode != ('RGB' if field == 'image' else 'L') or image.size != (512, 512):
            raise ValueError('training requires paired 512px RGB/L images')
        if field == 'mask':
            pixels = np.asarray(image)
            if not pixels.any() or not set(np.unique(pixels).tolist()) <= {0, 255}:
                raise ValueError('training mask must be nonempty binary 0/255')


def _caption_ids(tokenizer, text, branch):
    token = TOKENS[branch]
    if not isinstance(text, str) or text.count(token) != 1 or any(
            other in text for name, other in TOKENS.items() if name != branch):
        raise ValueError('caption must contain exactly its own modifier once')
    ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    if len(ids) > 77 or ids.count(tokenizer.convert_tokens_to_ids(token)) != 1:
        raise ValueError('caption truncates or fails modifier tokenization')
    return ids


def concepts(rows):
    return [{'instance_prompt': [row['caption']],
             'instance_data_dir': str(Path(row['image']).parent),
             **({'instance_mask_dir': str(Path(row['mask']).parent)} if row['mask'] else {})}
            for row in rows]


def stage_datasets(output, record, datasets, branches, tokenizer):
    """Copy locked newly derived rows; no historical Mailbox resolver is consulted."""
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError('dataset staging must use a fresh external directory')
    locks = source_locks(record)
    files, counts = {}, {}
    if set(branches) - set(datasets):
        raise ValueError('selected branch lacks newly derived data')
    if tokenizer.add_tokens([TOKENS[branch] for branch in branches]) != len(branches):
        raise ValueError('caption preflight requires a fresh base tokenizer')
    output.mkdir(parents=True)
    for branch in branches:
        data = datasets[branch]
        if (data['source_image_sha256'] != record['source_image_sha256']
                or data['source_mask_sha256'] != record['object_mask_sha256']):
            raise ValueError('derived data belongs to another photograph or object mask')
        snapshots.check_locks(data['source_locks'])
        if any(key in locks and locks[key] != value for key, value in data['source_locks'].items()):
            raise ValueError('derived source locks disagree')
        locks.update(data['source_locks'])
        sources = data['rows']
        if len(sources) != COUNTS[branch] or len({row['id'] for row in sources}) != len(sources):
            raise ValueError(f'{branch} requires exactly {COUNTS[branch]} unique derived rows')
        rows = []
        for index, source in enumerate(sources):
            row = {'id': source['id'], 'caption': source['caption'],
                   'caption_token_ids': _caption_ids(tokenizer, source['caption'], branch)}
            for field in ('image', 'mask'):
                value = source.get(field)
                row[field] = row[field + '_sha256'] = None
                if value is None:
                    if field != 'mask' or branch != 'color':
                        raise ValueError('required image or object mask missing')
                    continue
                path = Path(value).resolve()
                digest = sha256(path)
                if data['source_locks'].get(str(path)) != digest:
                    raise ValueError('derived image/mask must be covered by source locks')
                _validate_image(path, field)
                destination = output / branch / f'{index:03d}' / field / 'image.png'
                destination.parent.mkdir(parents=True)
                shutil.copyfile(path, destination)
                row.update({field: str(destination), field + '_sha256': digest, 'source_' + field: str(path)})
                files[str(destination)] = digest
            rows.append(row)
        folder = output / branch
        (folder / 'training_assets_manifest.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
        _write(folder / 'concepts.json', concepts(rows))
        for name in ('training_assets_manifest.jsonl', 'concepts.json'):
            files[str(folder / name)] = sha256(folder / name)
        counts[branch] = len(rows)
    receipt = {'schema': 'colorpeel_single_object_dataset/v1', 'object': record,
               'counts': counts, 'source_locks': locks, 'files': files,
               'old_weights_reused': False, 'copies_byte_identical': True}
    snapshots.check_locks({**locks, **files})
    _write(output / 'receipt.json', receipt)
    return receipt


def verify_dataset(output, branch):
    output = Path(output).resolve()
    receipt = read_json(output / 'receipt.json')
    if branch not in receipt['counts'] or receipt['counts'][branch] != COUNTS[branch]:
        raise ValueError('selected branch count differs')
    locks = {**receipt['source_locks'], **receipt['files'],
             str(output / 'receipt.json'): sha256(output / 'receipt.json')}
    snapshots.check_locks(locks)
    folder = output / branch
    rows = [json.loads(line) for line in (folder / 'training_assets_manifest.jsonl').read_text().splitlines()]
    if len(rows) != COUNTS[branch] or read_json(folder / 'concepts.json') != concepts(rows):
        raise ValueError('one-image caption pairing changed')
    for row in rows:
        for field in ('image', 'mask'):
            if row[field] is None:
                if field != 'mask' or branch != 'color':
                    raise ValueError('missing required image or mask')
                continue
            path = Path(row[field])
            if not path.is_relative_to(folder) or set(path.parent.iterdir()) != {path}:
                raise ValueError('one-image staging directory changed')
            _validate_image(path, field)
    return {'branch': branch, 'dataset': str(folder), 'rows': len(rows), 'references': rows,
            'source_locks': locks, 'concepts': str(folder / 'concepts.json'),
            'manifest': str(folder / 'training_assets_manifest.jsonl'), 'token': TOKENS[branch]}


def audit_material_rows(data):
    """Bind staged Material pixels and scalar targets to the actual render metadata."""
    target = data['material_target']
    snapshots.check_locks(data['source_locks'])
    for row in data['rows']:
        metadata_path = Path(row['metadata']).resolve()
        if (data['source_locks'].get(str(metadata_path)) != row['metadata_sha256']
                or sha256(metadata_path) != row['metadata_sha256']):
            raise ValueError('Material render metadata is not hash-locked')
        metadata = read_json(metadata_path)
        if any(row.get(key) != target[key] or metadata.get(key) != target[key]
               for key in ('roughness', 'metallic')):
            raise ValueError('actual Material render rows differ from the numeric target')
        for field in ('image', 'mask'):
            source = Path(row['source_' + field]).resolve()
            if (Path(metadata[field]).resolve() != source
                    or data['source_locks'].get(str(source)) != metadata[field + '_sha256']):
                raise ValueError('Material metadata describes different render pixels')
            with Image.open(source) as original, Image.open(row[field]) as staged:
                if field == 'image':
                    first, second = np.asarray(original.convert('RGB')), np.asarray(staged.convert('RGB'))
                else:
                    first, second = np.asarray(original.convert('L')) >= 128, np.asarray(staged.convert('L')) >= 128
                if not np.array_equal(first, second):
                    raise ValueError('Material pixels changed after rendering')
    return target


def prepare(output, object_record, initializer_manifest, datasets, branches=None, steps=None,
            checkpoint_steps=None):
    from transformers import CLIPTokenizer
    from src.train.initializer_token_utils import load_initializer_manifest
    from experiments.sd21_wood_branches_v1.prepare_model import verify_model
    from experiments.sd21_qwen_initializers_v1.material_target import compare_material_target

    output = resolve_run_dir(Path(output))
    initializer_manifest = Path(initializer_manifest).resolve()
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError('require a fresh external prepared experiment')
    plan = training_plan(branches, steps)
    checkpoints = checkpoint_plan(plan, checkpoint_steps)
    revision = git_info()
    locks = source_locks(object_record)
    model = verify_model(MODEL_DIR)
    tokenizer = CLIPTokenizer.from_pretrained(str(MODEL_DIR), subfolder='tokenizer', local_files_only=True)
    initializers = load_initializer_manifest(initializer_manifest, tokenizer, MODEL_DIR)
    if set(plan) - set(initializers.get('selected_branches', BRANCHES)):
        raise ValueError('training branch was not verified by the selected initializer manifest')
    if (Path(initializers['source']['image']).resolve() != Path(object_record['vlm_input']).resolve()
            or initializers['source']['image_sha256'] != object_record['vlm_input_sha256']):
        raise ValueError('initializer was not derived from this isolated object crop')
    output.mkdir(parents=True)
    material_target = None
    if 'material' in plan:
        target = audit_material_rows(datasets['material'])
        material_target = compare_material_target(initializers['response']['material'], **target)
        material_target['numeric_target'] = target
        _write(output / 'material_target_consistency.json', material_target)
        if material_target['training_blocked']:
            raise ValueError('high-confidence VLM/Material target conflict')
        locks[str(output / 'material_target_consistency.json')] = sha256(output / 'material_target_consistency.json')
    staged = stage_datasets(output / 'dataset', object_record, datasets, list(plan), tokenizer)
    locks.update({**staged['source_locks'], **staged['files'], **initializers['source_locks'],
                  **snapshots.model_locks(MODEL_DIR, model),
                  str(initializer_manifest): sha256(initializer_manifest),
                  str(output / 'dataset/receipt.json'): sha256(output / 'dataset/receipt.json')})
    audit = {'schema': 'colorpeel_single_object_training/v1', 'new_training_git': revision,
             'object': object_record, 'branches': list(plan), 'steps': plan, 'checkpoint_steps': checkpoints,
             'initializer_manifest': str(initializer_manifest), 'initializers': initializers,
             'source_locks': locks, 'base_model': model['model'],
             'material_target_consistency': material_target, 'fresh_training': True,
             'checkpoint_policy': 'paired snapshots at declared steps; final step required'}
    _write(output / 'source_audit.json', audit)
    _write(output / 'prepared.json', {'status': 'prepared', 'git': revision,
            'source_audit_sha256': sha256(output / 'source_audit.json')})
    verify_prepared(output)
    return output / 'source_audit.json'


def verify_prepared(experiment):
    experiment = Path(experiment).resolve()
    prepared, audit = read_json(experiment / 'prepared.json'), read_json(experiment / 'source_audit.json')
    if (prepared['source_audit_sha256'] != sha256(experiment / 'source_audit.json')
            or prepared['git'] != audit['new_training_git'] or git_info() != prepared['git']):
        raise ValueError('prepared recipe changed or prepare/train Git differs')
    snapshots.check_locks(audit['source_locks'])
    return audit


def build_config(experiment, branch, method='lora_r64', model_dir=MODEL_DIR, smoke=False, gpu=0):
    from experiments.subject_multicolor_multimaterial_b.training import build_config as subject_config
    from experiments.mailbox_color_sd15_v1.training import build_config as color_config
    from experiments.texture_material_disentangled_v1.train import build_config as natural_config

    experiment = Path(experiment).resolve()
    audit = read_json(experiment / 'source_audit.json')
    if (branch not in audit['branches'] or method != 'lora_r64' or Path(model_dir) != MODEL_DIR
            or type(smoke) is not bool or type(gpu) is not int or gpu < 0):
        raise ValueError('require a selected SD2.1 LoRA64 branch and nonnegative GPU')
    dataset = experiment / 'dataset' / branch
    if branch == 'subject':
        config = subject_config(method, dataset)
    elif branch == 'color':
        config = color_config(method, dataset)
    else:
        # Reuse optimizer/augmentation recipes only; remove all old data identity.
        template = {'method': method, 'branch': branch, 'token': TOKENS[branch],
                    'concept_id': 'polished_metal' if branch == 'material' else 'vertical_stripes'}
        config = natural_config(template, experiment / 'dataset', 'pilot')
        config.pop('texture_material')
    maximum = 2 if smoke else audit['steps'][branch]
    checkpoints = [2] if smoke else checkpoint_plan(audit['steps'], audit.get('checkpoint_steps'))[branch]
    config['status'] = 'single_image_object_training'
    config['run'] = {'study': STUDY, 'variant': audit['object']['id'] + '_' + branch +
                     ('_smoke2' if smoke else f'_steps{maximum}'), 'seed': 42}
    config['environment']['CUDA_VISIBLE_DEVICES'] = str(gpu)
    config['data_manifest'] = str(dataset / 'training_assets_manifest.jsonl')
    config['args'].update(pretrained_model_name_or_path=str(model_dir),
        concepts_list=str(dataset / 'concepts.json'), modifier_token=TOKENS[branch],
        initializer_token=audit['initializers']['branches'][branch]['initializer'],
        initializer_manifest=audit['initializer_manifest'], max_train_steps=maximum,
        checkpoint_steps=checkpoints, checkpointing_steps=maximum, gradient_checkpointing=True,
        concept_isolation_weight=0.0, adam_weight_decay=0.0, freeze_model='crossattn_kv')
    config['data_lineage'] = {'object_id': audit['object']['id'],
        'source_image_sha256': audit['object']['source_image_sha256'],
        'source_mask_sha256': audit['object']['object_mask_sha256'],
        'dataset': str(dataset), 'rows': COUNTS[branch], 'old_weights_reused': False,
        'initializer_manifest': audit['initializer_manifest']}
    return config


def recipe(experiment, branch, gpu=0):
    audit = read_json(Path(experiment) / 'source_audit.json')
    return SimpleNamespace(BRANCHES=(branch,), METHODS=('lora_r64',),
        STEPS=tuple(checkpoint_plan(audit['steps'], audit.get('checkpoint_steps'))[branch]),
        MODEL_DIR=MODEL_DIR, TOKENS=TOKENS,
        CONCEPT_IDS={branch: audit['object']['id'] + '_' + branch},
        verify_dataset=lambda selected: verify_dataset(Path(experiment) / 'dataset', selected),
        build_config=lambda selected, method, model_dir=MODEL_DIR, smoke=False:
            build_config(experiment, selected, method, model_dir, smoke, gpu))


def export(experiment, branch, train_run, output, smoke=False, gpu=0):
    audit = verify_prepared(experiment)
    checkpoint = Path(train_run).resolve() / 'checkpoints'
    initial_path = checkpoint / 'initializer_audit.json'
    initial, adaptation = read_json(initial_path), read_json(checkpoint / 'adaptation_config.json')
    selected = audit['initializers']['branches'][branch]
    manifest = Path(audit['initializer_manifest'])
    expected = {'initializer_manifest': str(manifest), 'initializer_manifest_sha256': sha256(manifest),
                'initializer': selected['initializer'], 'initializer_token_id': selected['initializer_token_id']}
    if (any(initial.get(key) != value or adaptation.get(key) != value for key, value in expected.items())
            or initial.get('modifier_token') != TOKENS[branch]
            or initial.get('initialized_equal_base') is not True or initial.get('initializer_unchanged') is not True
            or initial['base_initializer_sha256'] != initial['modifier_after_copy_sha256']):
        raise ValueError('actual initializer copy differs from the object manifest')
    result = snapshots.audit_export(train_run, branch, 'lora_r64', Path(output), smoke,
                                    experiment_contract=recipe(experiment, branch, gpu))
    provenance = {**expected, 'initializer_audit': str(initial_path),
                  'initializer_audit_sha256': sha256(initial_path)}
    result['source_locks'].update({str(initial_path): sha256(initial_path), str(manifest): sha256(manifest)})
    for step, item in result['steps'].items():
        item['initializer_provenance'] = provenance
        item['hashes'].update({str(initial_path): sha256(initial_path), str(manifest): sha256(manifest)})
        _write(Path(output) / f'step{step}/receipt.json', item)
    _write(Path(output) / 'index.json', result)
    return result
