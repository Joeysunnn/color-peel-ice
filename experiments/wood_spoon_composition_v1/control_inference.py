"""Fixed-seed native, single-token and dual-token wood-spoon Texture500 controls."""

import argparse
import html
import json
from pathlib import Path
import shutil

from PIL import Image

from experiments.wood_spoon_composition_v1 import inference as base

CASES = {
    'mug_without_t': {'prompt': 'a photo of a mug', 'use_subject': False, 'use_texture': False},
    'mug_with_t': {'prompt': 'a photo of a mug with <T*> texture', 'use_subject': False, 'use_texture': True},
    'spoon_without_t': {'prompt': 'a photo of a spoon', 'use_subject': False, 'use_texture': False},
    'spoon_with_t': {'prompt': 'a photo of a spoon with <T*> texture', 'use_subject': False, 'use_texture': True},
    'wood_spoon_without_t': {'prompt': 'a photo of a wood spoon', 'use_subject': False, 'use_texture': False},
    'wood_spoon_with_t': {'prompt': 'a photo of a wood spoon with <T*> texture', 'use_subject': False, 'use_texture': True},
    'subject_without_t': {'prompt': 'a photo of <S*> spoon', 'use_subject': True, 'use_texture': False},
    'subject_with_t': {'prompt': 'a photo of <S*> spoon with <T*> texture', 'use_subject': True, 'use_texture': True},
}
METHOD = 'lora_r64'
SAMPLING = {**base.SAMPLING, 'seeds': [42]}


def prompt_row(case):
    if case not in CASES:
        raise ValueError('select one of the eight fixed wood-spoon controls')
    spec = CASES[case]
    return {'id': case, 'case_id': case, 'method': METHOD, 'seed': 42, **spec,
        'subject_step': 600 if spec['use_subject'] else None,
        'attribute_step': 500 if spec['use_texture'] else None,
        'image_path': 'images/seed42.png', 'subject_adapter_loaded': spec['use_subject'],
        'texture_adapter_loaded': spec['use_texture'], 'attribute_adapter_loaded': spec['use_texture']}


def references(audit, output, row):
    refs, copies = {}, {}
    for name, enabled in (('subject', row['use_subject']), ('texture', row['use_texture'])):
        if not enabled:
            continue
        rows = audit.get('references', {}).get(name, [])
        if len(rows) != base.COUNTS[name]:
            raise ValueError('control review requires every loaded branch training RGB and caption')
        folder = output / 'training_references' / name
        folder.mkdir(parents=True)
        refs[name] = []
        for number, item in enumerate(rows):
            source = Path(item['image'])
            digest = audit['source_locks'].get(str(source))
            if (not source.is_absolute() or not digest or base.sha256(source) != digest
                    or item.get('image_sha256', digest) != digest
                    or not isinstance(item.get('caption'), str) or item['caption'].count(base.TOKENS[name]) != 1):
                raise ValueError('control training reference RGB or caption differs from its source locks')
            target = folder / f'train_{number:03d}{source.suffix}'
            shutil.copyfile(source, target)
            if base.sha256(target) != digest:
                raise ValueError('control training reference copy differs')
            copies[str(target)] = digest
            refs[name].append({**item, 'review_image': target.relative_to(output).as_posix()})
    base.write(output / 'training_references.json', {'references': refs})
    return refs, copies


def review(output, row, state, refs):
    image = (f'<img width="512" src="{row["image_path"]}">' if row.get('status') else '<p>Planned; inference not executed.</p>')
    tokens = [base.TOKENS[name] for name, enabled in (('subject', row['use_subject']), ('texture', row['use_texture'])) if enabled]
    reference_pages = []
    for name, items in refs.items():
        cards = ''.join(f'<figure><img loading="lazy" width="200" src="{html.escape(item["review_image"], quote=True)}">'
                       f'<figcaption>{html.escape(item["caption"])}</figcaption></figure>' for item in items)
        reference_pages.append(f'<details><summary>{name}: {len(items)} actual training RGB images and captions</summary>'
                               f'<section>{cards}</section></details>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        + '<style>body{font:16px sans-serif}section{display:flex;flex-wrap:wrap;gap:10px}figure{margin:0;width:210px}</style>'
        + f'<h1>{html.escape(row["prompt"])}</h1><p>seed42; {state["status"]}; PNDM100 / CFG3.5 / 512px</p>'
        + '<p>Loaded tokens and matching K/V: ' + (html.escape(', '.join(tokens)) if tokens else 'none; native SD1.5') + '</p>'
        + image + ''.join(reference_pages), encoding='utf8')


def run(audit_path, output, case, device='cuda:0', execute=False):
    row = prompt_row(case)
    audit_path, output = Path(audit_path).resolve(), base.resolve_run_dir(Path(output))
    audit, revision = base.read_json(audit_path), base.git_info()
    subject, texture = base.selections(audit, METHOD, 'texture', attribute_step=500,
                                       subject_step=600, subject_images=72)
    selected = ([subject] if row['use_subject'] else []) + ([texture] if row['use_texture'] else [])
    paths = [Path(item[field]).resolve() for group in audit['sources'].values() for item in group.values()
             for field in ('run', 'checkpoint')]
    paths += [Path(item['image']).resolve() for group in audit['references'].values() for item in group]
    if (output.exists() or output.is_relative_to(base.ROOT) or audit_path.is_relative_to(output)
            or any(output.is_relative_to(path) or path.is_relative_to(output) for path in paths)):
        raise ValueError('control inference requires a fresh external output separate from every source')
    locks = {**audit['source_locks'], str(audit_path): base.sha256(audit_path)}
    output.mkdir(parents=True)
    refs, copies = {}, {}
    state = {'status': 'running' if execute else 'dry_run', 'case_id': case, 'method': METHOD,
        'prompt': row['prompt'], 'seed': 42, 'inference_git': revision, 'training_gits': audit.get('training_gits', {}),
        'audit': str(audit_path), 'sampling': SAMPLING, 'expected_images': 1, 'generated_images': 0,
        'subject_step': row['subject_step'], 'attribute_step': row['attribute_step'],
        'subject_adapter_loaded': row['use_subject'], 'texture_adapter_loaded': row['use_texture'],
        'attribute_adapter_loaded': row['use_texture'],
        'subject_training': subject if row['use_subject'] else None,
        'texture_training': texture if row['use_texture'] else None,
        'training_images': {}, 'retraining': False, 'reexporting': False, 'monitoring': False}
    try:
        refs, copies = references(audit, output, row)
        state['training_images'] = {name: len(items) for name, items in refs.items()}
        (output / 'generation_manifest.jsonl').write_text(json.dumps(row) + '\n', encoding='utf8')
        if execute:
            import torch
            base.check_locks(locks)
            pipe = base.evaluate.load_pipeline(selected, device)
            tokens = [item['token'] for item in selected]
            kwargs = ({'cross_attention_kwargs': base.evaluate.attention_masks(pipe, row['prompt'], tokens, 3.5)}
                      if tokens else {})
            result = pipe(row['prompt'], num_inference_steps=100, guidance_scale=3.5,
                width=512, height=512, generator=torch.Generator(device=device).manual_seed(42), **kwargs)
            flags = result.nsfw_content_detected
            if flags is None or len(flags) != 1:
                raise ValueError('missing control safety checker result')
            if len(result.images) != 1 or result.images[0].mode != 'RGB' or result.images[0].size != (512, 512):
                raise ValueError('control inference must return one512px RGB image')
            image = output / row['image_path']
            image.parent.mkdir()
            with image.open('xb') as handle:
                result.images[0].save(handle, format='PNG')
            with Image.open(image) as saved:
                if saved.format != 'PNG' or saved.mode != 'RGB' or saved.size != (512, 512):
                    raise ValueError('saved control image differs from RGB512 PNG')
            row = {**row, 'status': 'safety_filtered' if bool(flags[0]) else 'ok',
                   'image_sha256': base.sha256(image), 'copied': False}
            (output / 'generation_status.jsonl').write_text(json.dumps(row) + '\n', encoding='utf8')
            del pipe
            torch.cuda.empty_cache()
            state.update(status='succeeded', generated_images=1, safety_filtered=int(bool(flags[0])))
        base.check_locks(locks)
        if copies:
            base.check_locks(copies)
        if base.git_info() != revision:
            raise ValueError('control inference checkout changed')
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        state['source_locks'], state['reference_copy_hashes'] = locks, copies
        base.write(output / 'provenance.json', state)
        review(output, row, state, refs)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--case', required=True, choices=tuple(CASES))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--execute-inference', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.audit, args.output, args.case, args.device, args.execute_inference)))
