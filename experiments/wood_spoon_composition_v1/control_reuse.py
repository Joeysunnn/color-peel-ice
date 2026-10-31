"""Reuse four fixed seed42 controls; missing evidence never falls back to sampling."""

import json
from pathlib import Path
import shutil

from PIL import Image

from experiments.wood_spoon_composition_v1 import inference as I
from experiments.wood_spoon_composition_v1.reuse_composition import locks_checked, no_symlinks

RUNS = Path('/var/tmp/colorpeel-r12user5/runs')
ORDINARY = RUNS / 'natural_texture_material_v1/20261018-213417__ordinary_transfer__9d70d7a__42/step500'
SOURCES = {'mug_without_t': ORDINARY, 'mug_with_t': ORDINARY,
    'wood_spoon_with_t': RUNS / 'wood_spoon_composition_v1/20261031-175743__literal_wood_spoon_T500__b7dedcb__30/experiment/texture/lora_r64',
    'subject_with_t': RUNS / 'wood_spoon_composition_v1/20261030-203708__S600_T500__03583f4__30/experiment/texture/lora_r64'}


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]


def ordinary_proof(state, audit):
    selected = audit['sources']['texture']['lora_r64']
    matches = [item for item in state.get('training', [])
               if item.get('method') == 'lora_r64' and item.get('concept_id') == 'wood_spoon_texture']
    if matches != [selected] or state.get('training_steps') != 500 or state.get('probe_suite') != 'ordinary_object_transfer':
        raise ValueError('ordinary transfer Texture500 source differs')
    old_locks = state.get('source_locks', {})
    hashes = selected.get('hashes', {})
    if not hashes or any(old_locks.get(path) != digest or audit['source_locks'].get(path) != digest
                         for path, digest in hashes.items()):
        raise ValueError('ordinary transfer selected source hashes differ')
    verified = dict(hashes)
    root = Path(state['source_run'])
    for path in (root / 'status.json', root / 'protocol.json', root / 'training/training_index.json',
                 root / 'step500/training_index.json'):
        if str(path) not in old_locks:
            raise ValueError('ordinary transfer training index is not source locked')
        verified[str(path)] = old_locks[str(path)]
    locks_checked(verified)
    if selected not in I.read_json(root / 'step500/training_index.json').get('runs', []):
        raise ValueError('ordinary transfer checkpoint differs from its training index')
    sampling = state.get('sampling', {})
    expected = {'scheduler': 'PNDM', 'num_inference_steps': 100, 'guidance_scale': 3.5,
                'resolution': 512, 'dtype': 'float16', 'safety_checker': True, 'seeds': [42, 43]}
    if any(sampling.get(key) != value for key, value in expected.items()):
        raise ValueError('ordinary transfer sampling differs')
    return verified


def copy_existing(output, case_id, audit):
    from experiments.wood_spoon_composition_v1.control_inference import CASES, prompt_row
    if case_id not in CASES:
        raise ValueError('unknown control case')
    if case_id not in SOURCES:
        return None
    source, output = SOURCES[case_id], Path(output).absolute()
    no_symlinks(source)
    no_symlinks(output)
    source, output = source.resolve(), output.resolve()
    if (output.exists() or output.is_relative_to(I.ROOT) or output.is_relative_to(source)
            or source.is_relative_to(output)):
        raise ValueError('require a fresh external control directory separate from historical results')
    evidence = [source / name for name in ('provenance.json', 'generation_manifest.jsonl', 'generation_status.jsonl')]
    for path in evidence:
        no_symlinks(path)
    before = {str(path): I.sha256(path) for path in evidence}
    state = I.read_json(evidence[0])
    manifest, ledger = lines(evidence[1]), lines(evidence[2])
    if state.get('status') != 'succeeded':
        raise ValueError('historical control is not complete')
    plan = prompt_row(case_id)
    is_mug = case_id.startswith('mug_')
    if is_mug:
        summary = source / 'summary.json'
        no_symlinks(summary)
        before[str(summary)] = I.sha256(summary)
        completed = I.read_json(summary)
        if (completed.get('status') != 'succeeded' or completed.get('generated_images') != 288
                or completed.get('missing_images') != 0 or len(manifest) != 288 or len(ledger) != 288):
            raise ValueError('ordinary transfer suite is incomplete')
        verified = ordinary_proof(state, audit)
        condition = 'learned_texture' if CASES[case_id]['use_texture'] else 'base'
        matches = [row for row in manifest if row.get('concept_id') == 'wood_spoon_texture'
            and row.get('method') == 'lora_r64' and row.get('seed') == 42
            and row.get('condition') == condition and row.get('context') == 'mug']
    else:
        use_subject = CASES[case_id]['use_subject']
        if (state.get('generated_images') != 31 or state.get('expected_images') != 31
                or len(manifest) != 31 or len(ledger) != 31 or state.get('method') != 'lora_r64'
                or state.get('branch') != 'texture' or state.get('attribute_step') != 500
                or state.get('sampling') != I.SAMPLING or state.get('prompt') != plan['prompt']
                or state.get('attribute_training') != audit['sources']['texture']['lora_r64']
                or state.get('subject_training') != (audit['sources']['subject']['lora_r64'] if use_subject else None)
                or state.get('subject_step') != (600 if use_subject else None)
                or state.get('training_images') != ({'subject': 72, 'texture': 16} if use_subject else {'texture': 16})):
            raise ValueError('historical composition control source or sampling differs')
        verified = state.get('source_locks', {})
        locks_checked(verified)
        locks_checked(state.get('reference_copy_hashes', {}))
        for name in (('subject', 'texture') if use_subject else ('texture',)):
            hashes = audit['sources'][name]['lora_r64'].get('hashes', {})
            if not hashes or any(verified.get(path) != digest or audit['source_locks'].get(path) != digest
                                 for path, digest in hashes.items()):
                raise ValueError('historical composition selected adapter/embedding is not locked')
        matches = [row for row in manifest if row.get('seed') == 42]
    if len(matches) != 1 or matches[0].get('prompt') != plan['prompt']:
        raise ValueError('historical control prompt is missing or ambiguous')
    original_plan = matches[0]
    ids = [row.get('id') for row in ledger]
    if len(set(ids)) != len(ids) or {row.get('id') for row in manifest} != set(ids):
        raise ValueError('historical control ledger coverage differs')
    rows = [row for row in ledger if row.get('id') == original_plan['id']]
    if len(rows) != 1:
        raise ValueError('historical control ledger is ambiguous')
    row = rows[0]
    expected_flags = ({'texture_adapter_loaded': CASES[case_id]['use_texture'], 'material_adapter_loaded': False}
        if is_mug else {'subject_adapter_loaded': CASES[case_id]['use_subject'],
                       'attribute_adapter_loaded': True, 'attribute_token': '<T*>'})
    if (any(row.get(key) != value for key, value in original_plan.items())
            or row.get('status') not in ('ok', 'safety_filtered')
            or any(row.get(key) != value for key, value in expected_flags.items())):
        raise ValueError('historical control ledger prompt or loaded adapter flags differ')
    image = source / row['image_path']
    no_symlinks(image)
    if not image.resolve().is_relative_to(source) or I.sha256(image) != row.get('image_sha256'):
        raise ValueError('historical control PNG hash differs')
    before[str(image)] = row['image_sha256']
    with Image.open(image) as opened:
        if opened.format != 'PNG' or opened.mode != 'RGB' or opened.size != (512, 512):
            raise ValueError('historical control PNG must be RGB512')
    output.mkdir(parents=True)
    target = output / plan['image_path']
    target.parent.mkdir()
    shutil.copyfile(image, target)
    archived = output / 'reuse_sources'
    archived.mkdir()
    for path in evidence:
        shutil.copyfile(path, archived / path.name)
        if I.sha256(archived / path.name) != before[str(path)]:
            raise ValueError('archived historical record bytes differ')
    I.write(archived / 'selected_original_manifest.json', original_plan)
    I.write(archived / 'selected_original_status.json', row)
    if I.sha256(target) != row['image_sha256'] or any(I.sha256(path) != digest for path, digest in before.items()):
        raise ValueError('historical source or copied control changed')
    normalized = {**plan, 'status': row['status'], 'image_sha256': row['image_sha256'], 'copied': True}
    for name, record in (('generation_manifest.jsonl', plan), ('generation_status.jsonl', normalized)):
        (output / name).write_text(json.dumps(record) + '\n', encoding='utf8')
    receipt = {'status': 'succeeded', 'generated_images': 1, 'copied': True, 'case_id': case_id,
        'original_directory': str(source), 'original_image_path': str(image),
        'original_git': state.get('inference_git', state.get('git')), 'original_records_sha256': before,
        'verified_source_locks': verified, 'image_sha256': row['image_sha256'],
        'regenerated_images': 0, 'prompt': plan['prompt'], 'seed': 42}
    I.write(output / 'reuse_receipt.json', receipt)
    I.write(output / 'provenance.json', receipt)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><p>'
        + plan['prompt'].replace('<', '&lt;').replace('>', '&gt;')
        + '</p><img src="images/seed42.png"><p><a href="reuse_sources/provenance.json">Historical source provenance</a></p>', encoding='utf8')
    return receipt
