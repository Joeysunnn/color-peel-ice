"""Copy verified completed S72/T500 groups without altering historical provenance."""

import json
from pathlib import Path
import shutil

from PIL import Image

from experiments.wood_spoon_composition_v1 import inference as I
from scripts.launch.colorpeel_run import read_config

BASE = Path('/var/tmp/colorpeel-r12user5/runs/wood_spoon_composition_v1')
SOURCES = {step: BASE / '20261030-221534__S200_S400_T500__6ce851f__30' / 'experiment'
           / f'step{step}_texture' for step in (200, 400)}
SOURCES[600] = BASE / '20261030-203708__S600_T500__03583f4__30/experiment/texture'


def no_symlinks(path):
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError('symlinks are forbidden in composition reuse')


def locks_checked(locks):
    if not locks:
        raise ValueError('missing historical source locks')
    for name, digest in locks.items():
        path = Path(name)
        no_symlinks(path)
        if not path.is_absolute() or not path.is_file() or I.sha256(path) != digest:
            raise ValueError('historical source lock changed')


def tree_hashes(directory):
    no_symlinks(directory)
    hashes = {}
    for path in directory.rglob('*'):
        no_symlinks(path)
        if path.is_file():
            hashes[path.relative_to(directory).as_posix()] = I.sha256(path)
        elif not path.is_dir():
            raise ValueError('unexpected source filesystem entry')
    return hashes


def training_checked(state, audit, method, step):
    for name, field, token, selected_step, total in (
            ('subject', 'subject_training', '<S*>', step, 600),
            ('texture', 'attribute_training', '<T*>', 500, 1000)):
        item = state.get(field, {})
        weight = 'pytorch_token_local_kv_weights.bin' if method == 'fullrank' else 'pytorch_lora_kv_weights.bin'
        expected = {'method': method, 'branch': name, 'token': token, 'status': 'verified',
            'steps': selected_step, 'step': selected_step, 'training_steps': total, 'weight': weight,
            'concept_id': 'wood_spoon' if name == 'subject' else 'wood_spoon_texture',
            'paired_embedding_finite': True}
        if (any(item.get(key) != value for key, value in expected.items())
                or item != audit.get('sources', {}).get(name, {}).get(method)):
            raise ValueError('historical selected training pair differs')
        hashes = item.get('hashes', {})
        if not hashes or any(state['source_locks'].get(path) != digest for path, digest in hashes.items()):
            raise ValueError('adapter/embedding pair is not source locked')
        run, checkpoint = Path(item['run']), Path(item['checkpoint'])
        required = [run / 'config.yaml', checkpoint / weight, checkpoint / (token + '.bin'),
                    checkpoint / 'adaptation_config.json']
        if any(not path.is_absolute() or str(path) not in hashes for path in required):
            raise ValueError('missing adapter, embedding or metadata lock')
        config = read_config(run / 'config.yaml')['args']
        if (config.get('modifier_token') != token or config.get('max_train_steps') != total
                or config.get('initializer_token') != ('spoon' if name == 'subject' else 'texture')):
            raise ValueError('training configuration differs')
        adaptation = I.read_json(checkpoint / 'adaptation_config.json')
        mode = 'token_local_kv' if method == 'fullrank' else 'lora_subject_kv'
        if (adaptation.get('modifier_tokens') != [token] or adaptation.get('adaptation_mode') != mode
                or (method == 'lora_r64' and any(adaptation.get(k) != v for k, v in
                    {'rank': 64, 'alpha': 64, 'mode': 'token_local_kv'}.items()))):
            raise ValueError('historical token local adapter metadata differs')


def copy_completed(source, destination, step, method):
    if type(step) is not int or step not in SOURCES or method not in I.METHODS:
        raise ValueError('only frozen Subject200/400/600 Texture500 groups may be reused')
    source, destination = Path(source).absolute(), Path(destination).absolute()
    no_symlinks(source)
    no_symlinks(destination)
    source, destination = source.resolve(), destination.resolve()
    if source != (SOURCES[step] / method).resolve():
        raise ValueError('source is not the fixed completed group')
    if (destination.exists() or destination.is_relative_to(I.ROOT)
            or destination.is_relative_to(source) or source.is_relative_to(destination)):
        raise ValueError('require a new external destination separate from source')
    before = tree_hashes(source)
    if 'reuse_receipt.json' in before:
        raise ValueError('source must be an original completed group')
    state = I.read_json(source / 'provenance.json')
    expected = {'status': 'succeeded', 'generated_images': 31, 'expected_images': 31,
        'method': method, 'branch': 'texture', 'subject_step': step, 'attribute_step': 500,
        'sampling': I.SAMPLING, 'prompt': I.PROMPTS['texture'],
        'training_images': {'subject': 72, 'texture': 16},
        'internal_adapter_slots': {'subject': '<S*>', 'material': '<T*>'},
        'attribute_semantics': 'texture'}
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError('completed historical generation contract differs')
    locks_checked(state.get('source_locks', {}))
    locks_checked(state.get('reference_copy_hashes', {}))
    audit_path = Path(state['audit'])
    if str(audit_path) not in state['source_locks']:
        raise ValueError('historical audit is not locked')
    audit = I.read_json(audit_path)
    training_checked(state, audit, method, step)
    refs = I.read_json(source / 'training_references.json').get('references', {})
    copies = {}
    for name, count in (('subject', 72), ('texture', 16)):
        rows = refs.get(name, [])
        original = audit.get('references', {}).get(name, [])
        if len(rows) != count or len(original) != count:
            raise ValueError('historical training reference count differs')
        for row, old in zip(rows, original):
            token = '<S*>' if name == 'subject' else '<T*>'
            if ({key: value for key, value in row.items() if key != 'review_image'} != old
                    or not isinstance(row.get('caption'), str) or row['caption'].count(token) != 1):
                raise ValueError('historical training reference caption or pairing differs')
            path = source / row['review_image']
            if not path.resolve().is_relative_to(source):
                raise ValueError('reference copy escapes group')
            digest = state['source_locks'].get(row['image'])
            if not digest or I.sha256(path) != digest or row.get('image_sha256', digest) != digest:
                raise ValueError('historical reference image differs')
            copies[str(path)] = digest
    if copies != state['reference_copy_hashes']:
        raise ValueError('historical reference copy locks differ')
    planned = [{key: value for key, value in row.items() if key != 'subject_images'}
               for row in I.prompt_rows(method, 'texture', 500, step)]
    manifest = [json.loads(line) for line in (source / 'generation_manifest.jsonl').read_text().splitlines()]
    ledger = [json.loads(line) for line in (source / 'generation_status.jsonl').read_text().splitlines()]
    if manifest != planned or len(ledger) != 31:
        raise ValueError('historical manifest or seed coverage differs')
    if {name for name in before if name.startswith('images/')} != {row['image_path'] for row in planned}:
        raise ValueError('historical generated image directory differs from the31-image manifest')
    for row, plan in zip(ledger, planned):
        if (any(row.get(key) != value for key, value in plan.items())
                or row.get('status') not in ('ok', 'safety_filtered')
                or row.get('subject_adapter_loaded') is not True
                or row.get('attribute_adapter_loaded') is not True or row.get('attribute_token') != '<T*>'):
            raise ValueError('historical generation ledger differs')
        image = source / row['image_path']
        if I.sha256(image) != row.get('image_sha256'):
            raise ValueError('historical generated image hash differs')
        with Image.open(image) as opened:
            if opened.format != 'PNG' or opened.mode != 'RGB' or opened.size != (512, 512):
                raise ValueError('historical generated image must be RGB512 PNG')
    if state.get('safety_filtered') != sum(row['status'] == 'safety_filtered' for row in ledger):
        raise ValueError('historical safety counts differ')
    shutil.copytree(source, destination)
    if tree_hashes(source) != before or tree_hashes(destination) != before:
        raise ValueError('source or copy bytes changed during reuse')
    receipt = {'status': 'verified_byte_copy', 'original_directory': str(source),
        'destination_directory': str(destination), 'original_inference_git': state.get('inference_git'),
        'subject_training_images': 72, 'subject_step': step, 'attribute_step': 500,
        'method': method, 'generated_images': 31, 'copied_files_sha256': before,
        'historical_provenance_unchanged': True, 'regenerated_images': 0}
    I.write(destination / 'reuse_receipt.json', receipt)
    return receipt
