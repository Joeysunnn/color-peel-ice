"""Continue one fixed inference matrix in a new directory from a verified immutable prefix."""

import json
from pathlib import Path
import shutil

from PIL import Image

from experiments.mailbox_color_sd15_v1 import inference, prompts, training
from experiments.sd15_token_local_kv_v1 import evaluate
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from scripts.launch.colorpeel_run import git_info, resolve_run_dir


def _selection(rows, method, token):
    matches = [row for row in rows if row.get('method') == method]
    if (len(matches) != 1 or matches[0].get('token') != token
            or matches[0].get('branch') != ('color' if token == '<C*>' else 'subject')
            or matches[0].get('status') != 'verified' or matches[0].get('steps') != 1000):
        raise ValueError('require one audited1000-step selection for each declared token and method')
    inference.check_locks(matches[0]['hashes'])
    return matches[0]


def _snapshot(selection, step):
    checkpoint = Path(selection['checkpoint'])
    paths = [checkpoint / f'checkpoint-{step}' / name for name in ('pytorch_model.bin', 'pytorch_model_1.bin')]
    return {str(path): sha256(path) for path in paths}


def _image(path, digest):
    if path.is_symlink() or sha256(path) != digest:
        raise ValueError('reused PNG hash differs')
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if image.mode != 'RGB' or image.size != (512, 512):
            raise ValueError('reused PNG must be512px RGB')


def _prefix(source, audit, method, step, rows, color, subject, snapshot):
    old = source / f'step{step}' / method
    if not old.exists():
        return [], {}, None, {'ledger_fragment_bytes': 0, 'unrecorded_images': []}
    manifest, provenance = old / 'generation_manifest.jsonl', old / 'provenance.json'
    actual_rows = [json.loads(line) for line in manifest.read_text(encoding='utf8').splitlines()]
    if actual_rows != rows:
        raise ValueError('original inference manifest differs from the exact requested matrix')
    state = read_json(provenance)
    if (state.get('status') not in ('succeeded', 'failed') or state.get('git') != audit['training_git']
            or state.get('method') != method or state.get('step') != step or state.get('subject_step') != 1000
            or state.get('dataset') != str(source / 'training_dataset') or state.get('training_images') != 9
            or state.get('expected_images') != 403 or state.get('sampling') != prompts.SAMPLING
            or state.get('subject_training') != subject):
        raise ValueError('original inference provenance or fixed Subject differs')
    if state['status'] == 'failed' and not state.get('error', '').startswith('KeyboardInterrupt:'):
        raise ValueError('only an explicitly interrupted original failure can be continued')
    prompt_source = state.get('prompt_source', {})
    expected_source = prompts.source_record()
    if (prompt_source.get('path') != str(Path(audit['source_code']) / prompts.SOURCE.relative_to(prompts.ROOT))
            or any(prompt_source.get(key) != expected_source[key] for key in ('sha256', 'historical_prompt_count', 'joint_prompts'))):
        raise ValueError('original prompt protocol differs')
    exported = state.get('color_training', {})
    if (exported.get('status') != 'verified' or exported.get('token') != '<C*>' or exported.get('branch') != 'color' or exported.get('method') != method
            or exported.get('steps') != step or exported.get('training_steps') != 1000
            or exported.get('paired_embedding_finite') is not True
            or exported.get('source_checkpoint') != color['checkpoint'] or exported.get('source_hashes') != color['hashes']
            or exported.get('source_snapshot_hashes') != snapshot
            or exported.get('source_adapter_sha256') != snapshot[str(Path(color['checkpoint']) / f'checkpoint-{step}/pytorch_model.bin')]
            or exported.get('source_text_encoder_sha256') != snapshot[str(Path(color['checkpoint']) / f'checkpoint-{step}/pytorch_model_1.bin')]):
        raise ValueError('original Color paired snapshot differs from the audited source')
    inference.check_locks(exported['hashes'])
    inference.check_locks(state['source_locks'])
    locks = {str(manifest): sha256(manifest), str(provenance): sha256(provenance)}
    ledger = old / 'generation_status.jsonl'
    raw = ledger.read_bytes() if ledger.exists() else b''
    lines = raw.splitlines(keepends=True)
    incomplete_tail = bool(lines and not lines[-1].endswith(b'\n'))
    fragment_bytes = 0
    if incomplete_tail:
        fragment_bytes = len(lines.pop())
    completed = []
    for index, line in enumerate(lines):
        row = json.loads(line)
        if (index >= len(rows) or {key: row.get(key) for key in rows[index]} != rows[index]
                or row.get('status') not in ('ok', 'safety_filtered') or not isinstance(row.get('image_sha256'), str)):
            raise ValueError('original ledger must be a unique ordered prefix of the requested matrix')
        image = old / rows[index]['image_path']
        _image(image, row['image_sha256'])
        locks[str(image)] = row['image_sha256']
        completed.append(row)
    count = len(completed)
    if (incomplete_tail and count == len(rows)) or state.get('generated_images') not in (count, min(count + 1, len(rows))):
        raise ValueError('original interrupted counts or trailing ledger fragment differ')
    if state['status'] == 'succeeded' and (count != len(rows) or incomplete_tail):
        raise ValueError('successful original inference must have the complete ledger')
    expected_images = {old / row['image_path'] for row in rows[:count]}
    allowed_images = expected_images | ({old / rows[count]['image_path']} if count < len(rows) else set())
    actual_images = set((old / 'images').rglob('*')) if (old / 'images').exists() else set()
    actual_images = {path for path in actual_images if path.is_file()}
    if not expected_images <= actual_images or actual_images - allowed_images:
        raise ValueError('original images contain a hole or more than one unrecorded trailing image')
    if state['generated_images'] > count and not (incomplete_tail or actual_images - expected_images):
        raise ValueError('original image count claims an unrecorded tail without any tail evidence')
    if ledger.exists():
        locks[str(ledger)] = sha256(ledger)
    ignored = {'ledger_fragment_bytes': fragment_bytes,
               'unrecorded_images': [str(path) for path in sorted(actual_images - expected_images)]}
    return completed, locks, exported, ignored


def inspect_source(source, audit, method, step):
    """Validate a source unit without creating output or loading a GPU model."""
    source = Path(source).resolve()
    if Path(audit['source']).resolve() != source:
        raise ValueError('source differs from its original-revision audit')
    rows = prompts.prompt_rows(method, step)
    inference.check_locks(audit['source_locks'])
    color = _selection(audit['color_runs'], method, '<C*>')
    subject = _selection(audit['subject_runs'], method, '<S*>')
    if subject.get('training_steps') != 1000 or subject.get('paired_embedding_finite') is not True:
        raise ValueError('Subject must be the prepared paired1000-step export')
    snapshot = _snapshot(color, step)
    prefix, locks, exported, ignored = _prefix(source, audit, method, step, rows, color, subject, snapshot)
    return {'rows': prefix, 'source_locks': locks, 'color': color, 'subject': subject,
            'source_snapshot_hashes': snapshot, 'original_color_export': exported, 'ignored_tail': ignored}


def run(source, audit, output, method, step, device='cuda:0', execute=False):
    source, output = Path(source).resolve(), resolve_run_dir(Path(output))
    if (output.exists() or output.is_relative_to(training.ROOT) or output.is_relative_to(source)
            or source.is_relative_to(output) or Path(audit['source']).resolve() != source
            or audit['inference_git'] != git_info()):
        raise ValueError('require a fresh external output, immutable source and matching inference revision')
    rows = prompts.prompt_rows(method, step)
    inspected = inspect_source(source, audit, method, step)
    color, subject = inspected['color'], inspected['subject']
    snapshot, prefix = inspected['source_snapshot_hashes'], inspected['rows']
    source_locks, old_export = inspected['source_locks'], inspected['original_color_export']
    dataset = source / 'training_dataset'
    # Dataset verification belongs to the original-revision audit: its locks
    # intentionally contain tracked file paths in the frozen original checkout.
    output.mkdir(parents=True)
    refs, locks = inference.references(dataset, output)
    locks.update(audit['source_locks'])
    locks.update(source_locks)
    locks.update(snapshot)
    locks.update(subject['hashes'])
    state = {'status': 'running' if execute else 'dry_run', 'git': audit['inference_git'],
        'inference_git': audit['inference_git'], 'training_git': audit['training_git'],
        'source': str(source), 'source_code': audit['source_code'], 'method': method, 'step': step,
        'subject_step': 1000, 'dataset': str(dataset), 'training_images': 9, 'expected_images': len(rows),
        'sampling': prompts.SAMPLING, 'prompt_source': prompts.source_record(),
        'reused_images': len(prefix), 'newly_generated_images': 0, 'generated_images': len(prefix),
        'subject_training': subject, 'source_color_training': color, 'source_snapshot_hashes': snapshot,
        'internal_adapter_slots': {'subject': '<S*>', 'material': '<C*> (historical second-slot name)'},
        'monitoring': False, 'automatic_retry': False}
    (output / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
    completed = list(prefix)
    try:
        with (output / 'generation_status.jsonl').open('x', encoding='utf8') as ledger:
            for row in prefix:
                original, target = source / f'step{step}' / method / row['image_path'], output / row['image_path']
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
                _image(target, row['image_sha256'])
                ledger.write(json.dumps(row) + '\n')
            ledger.flush()
            if execute:
                exported = training.export_step(color, step, output / 'checkpoints' / 'color')
                if (exported.get('source_snapshot_hashes') != snapshot
                        or exported.get('source_hashes') != color['hashes']
                        or exported.get('token') != '<C*>' or exported.get('method') != method
                        or exported.get('steps') != step or exported.get('training_steps') != 1000
                        or exported.get('paired_embedding_finite') is not True):
                    raise ValueError('new Color export differs from the audited same-step pair')
                if old_export and any(exported.get(key) != old_export.get(key) for key in
                        ('source_adapter_sha256', 'source_text_encoder_sha256')):
                    raise ValueError('new and original exports differ in source or paired weight hashes')
                inference.check_locks(exported['hashes'])
                locks.update(exported['hashes'])
                state['color_training'] = exported
                for arm in ('color_only', 'subject_color'):
                    missing = [row for row in rows[len(prefix):] if row['arm'] == arm]
                    if not missing:
                        continue
                    import torch
                    inference.check_locks(locks)
                    selections = [exported] if arm == 'color_only' else [subject, exported]
                    tokens = ['<C*>'] if arm == 'color_only' else ['<S*>', '<C*>']
                    pipe = evaluate.load_pipeline(selections, device)
                    for row in missing:
                        result = pipe(row['prompt'], num_inference_steps=100, guidance_scale=6.0, width=512, height=512,
                            generator=torch.Generator(device=device).manual_seed(row['seed']),
                            cross_attention_kwargs=evaluate.attention_masks(pipe, row['prompt'], tokens, 6.0))
                        if result.nsfw_content_detected is None or len(result.nsfw_content_detected) != 1:
                            raise ValueError('missing safety checker result')
                        image = result.images[0]
                        if image.mode != 'RGB' or image.size != (512, 512):
                            raise ValueError('inference must return512px RGB')
                        target = output / row['image_path']
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with target.open('xb') as handle:
                            image.save(handle, format='PNG')
                        record = {**row, 'status': 'safety_filtered' if result.nsfw_content_detected[0] else 'ok',
                                  'image_sha256': sha256(target)}
                        completed.append(record)
                        ledger.write(json.dumps(record) + '\n')
                        ledger.flush()
                        state['newly_generated_images'] += 1
                    del pipe
                    torch.cuda.empty_cache()
                if len(completed) != 403:
                    raise ValueError('incomplete parallel inference matrix')
                state['status'] = 'succeeded'
        inference.check_locks(locks)
        if git_info() != audit['inference_git']:
            raise ValueError('inference checkout changed')
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        state.update(generated_images=len(completed), source_locks=locks,
                     safety_filtered=sum(row['status'] == 'safety_filtered' for row in completed))
        inference.write(output / 'provenance.json', state)
        inference.write(output / 'reuse_manifest.json', {'source': str(source / f'step{step}' / method),
            'reused_images': len(prefix), 'rows': prefix, 'ignored_tail': inspected['ignored_tail']})
        by_id = {row['id']: row for row in completed}
        inference.review(output, refs, [by_id.get(row['id'], row) for row in rows], state)
    return state
