"""Render frozen wood-spoon S200/400/600 with independent M1000 or T500/750/1000 K/V."""

import argparse
import html
import json
from pathlib import Path
import shutil

from experiments.sd15_token_local_kv_v1 import evaluate
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from scripts.launch.colorpeel_run import git_info, read_config, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('fullrank', 'lora_r64')
BRANCHES = ('material', 'texture')
SEEDS = tuple(range(30, 61))
TOKENS = {'subject': '<S*>', 'material': '<M*>', 'texture': '<T*>'}
COUNTS = {'subject': 72, 'material': 8, 'texture': 16}
PROMPTS = {'material': 'a photo of <S*> spoon made of <M*>',
           'texture': 'a photo of <S*> spoon with <T*> texture'}
SAMPLING = {'scheduler': 'PNDM', 'num_inference_steps': 100, 'guidance_scale': 3.5,
            'width': 512, 'height': 512, 'dtype': 'float16', 'safety_checker': True,
            'generator': 'fresh_device_generator_per_image', 'seeds': list(SEEDS)}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def check_locks(locks):
    if not locks or any(not Path(path).is_absolute() or sha256(path) != digest for path, digest in locks.items()):
        raise ValueError('missing or changed composition source locks')


def _check_scope(method, branch, attribute_step, subject_step):
    if (method not in METHODS or branch not in BRANCHES or type(attribute_step) is not int
            or attribute_step not in ((500, 750, 1000) if branch == 'texture' else (1000,))
            or type(subject_step) is not int or subject_step not in (200, 400, 600)):
        raise ValueError('expected a declared method with Subject200/400/600 and Material1000 or Texture500/750/1000')


def prompt_rows(method, branch, attribute_step=1000, subject_step=600):
    _check_scope(method, branch, attribute_step, subject_step)
    return [{'id': f'{method}__{branch}__seed{seed}', 'method': method, 'branch': branch,
             'condition': 'subject_' + branch, 'subject_step': subject_step, 'attribute_step': attribute_step,
             'seed': seed, 'prompt': PROMPTS[branch], 'image_path': f'images/seed{seed}.png'}
            for seed in SEEDS]


def selections(audit, method, branch, attribute_step=1000, subject_step=600):
    _check_scope(method, branch, attribute_step, subject_step)
    if audit.get('inference_git') != git_info():
        raise ValueError('prepared composition audit differs from the clean inference revision')
    locks = audit.get('source_locks', {})
    check_locks(locks)
    result = []
    for name in ('subject', branch):
        item = audit.get('sources', {}).get(name, {}).get(method, {})
        step = subject_step if name == 'subject' else attribute_step
        training_steps = 600 if name == 'subject' else 1000
        token = TOKENS[name]
        weight = 'pytorch_token_local_kv_weights.bin' if method == 'fullrank' else 'pytorch_lora_kv_weights.bin'
        hashes = item.get('hashes', {})
        if (item.get('method') != method or item.get('branch') != name or item.get('token') != token
                or item.get('status') != 'verified' or item.get('steps') != step
                or item.get('step') != step or item.get('training_steps') != training_steps or item.get('weight') != weight
                or item.get('concept_id') != ('wood_spoon' if name == 'subject' else 'wood_spoon_' + name)
                or item.get('paired_embedding_finite') is not True or not hashes
                or any(locks.get(path) != digest for path, digest in hashes.items())):
            raise ValueError(f'composition requires matching audited Subject{subject_step} and attribute{attribute_step} pairs')
        run, checkpoint = Path(item['run']), Path(item['checkpoint'])
        required = [run / 'config.yaml', checkpoint / weight, checkpoint / (token + '.bin'),
                    checkpoint / 'adaptation_config.json']
        if any(not path.is_absolute() or str(path) not in hashes for path in required):
            raise ValueError('composition adapter, token and configuration must be source-locked')
        config = read_config(run / 'config.yaml')
        initializer = {'subject': 'spoon', 'material': 'material', 'texture': 'texture'}[name]
        if (config['args'].get('modifier_token') != token or config['args'].get('max_train_steps') != training_steps
                or config['args'].get('initializer_token') != initializer):
            raise ValueError('composition source token, steps or initializer differs')
        adaptation = read_json(checkpoint / 'adaptation_config.json')
        mode = 'token_local_kv' if method == 'fullrank' else (
            'lora_material_kv' if name == 'material' else 'lora_subject_kv')
        if (adaptation.get('modifier_tokens') != [token] or adaptation.get('adaptation_mode') != mode
                or (method == 'lora_r64' and any(adaptation.get(key) != value for key, value in
                    {'rank': 64, 'alpha': 64, 'mode': 'token_local_kv'}.items()))):
            raise ValueError('composition adapter metadata differs from its token and method')
        result.append(item)
    return result


def references(audit, branch, output):
    result, copies = {}, {}
    for name in ('subject', branch):
        rows = audit.get('references', {}).get(name, [])
        if len(rows) != COUNTS[name]:
            raise ValueError('review must display every actual selected training RGB and caption')
        folder = output / 'training_references' / name
        folder.mkdir(parents=True)
        result[name] = []
        for number, row in enumerate(rows):
            source = Path(row['image'])
            digest = audit['source_locks'].get(str(source))
            if (not source.is_absolute() or not digest or sha256(source) != digest
                    or row.get('image_sha256', digest) != digest
                    or not isinstance(row.get('caption'), str) or row['caption'].count(TOKENS[name]) != 1):
                raise ValueError('actual training reference RGB or caption differs from the prepared audit')
            target = folder / f'train_{number:03d}{source.suffix}'
            shutil.copyfile(source, target)
            if sha256(target) != digest:
                raise ValueError('training reference copy differs')
            copies[str(target)] = digest
            result[name].append({**row, 'review_image': target.relative_to(output).as_posix()})
    write(output / 'training_references.json', {'references': result})
    return result, copies


def review(output, refs, rows, state):
    style = '<style>body{font:16px sans-serif}section{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;width:240px}img{width:240px}figcaption{overflow-wrap:anywhere}</style>'
    links = []
    for branch, images in refs.items():
        page = f'references_{branch}.html'
        links.append(f'<li><a href="{page}">{branch}: {len(images)} actual training RGB images and captions</a></li>')
        cards = ''.join(f'<figure><img loading="lazy" src="{html.escape(row["review_image"], quote=True)}">'
                        f'<figcaption>{html.escape(row["caption"])}</figcaption></figure>' for row in images)
        (output / page).write_text('<!doctype html><meta charset="utf-8">' + style
            + f'<a href="index.html">Results</a><h1>{branch}: actual training references</h1><section>{cards}</section>', encoding='utf8')
    cards = ''.join('<figure>' + (f'<img loading="lazy" src="{row["image_path"]}">' if row.get('status') else '<p>Planned</p>')
        + f'<figcaption>seed {row["seed"]}: {html.escape(row.get("status", "planned"))}</figcaption></figure>' for row in rows)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">' + style
        + f'<p><a href="../../index.html">Both methods comparison</a></p><h1>{html.escape(state["prompt"])}</h1>'
        + f'<p>{state["method"]}; S{state["subject_step"]} + {"M" if state["branch"] == "material" else "T"}{state["attribute_step"]}'
        + f' ({state["branch"]}); {state["status"]}; PNDM100 / CFG3.5</p>'
        + '<ul>' + ''.join(links) + '</ul><h2>Seeds30–60</h2><section>' + cards + '</section>', encoding='utf8')


def run(audit_path, output, method, branch, device='cuda:0', execute=False, attribute_step=1000, subject_step=600):
    rows = prompt_rows(method, branch, attribute_step, subject_step)
    audit_path, output = Path(audit_path).resolve(), resolve_run_dir(Path(output))
    audit, revision = read_json(audit_path), git_info()
    selected = selections(audit, method, branch, attribute_step, subject_step)
    paths = [Path(item[field]).resolve() for group in audit['sources'].values() for item in group.values()
             for field in ('run', 'checkpoint')]
    paths += [Path(row['image']).resolve() for group in audit['references'].values() for row in group]
    if (output.exists() or output.is_relative_to(ROOT) or audit_path.is_relative_to(output)
            or any(output.is_relative_to(path) or path.is_relative_to(output) for path in paths)):
        raise ValueError('composition requires a fresh external output separate from every source')
    locks = {**audit['source_locks'], str(audit_path): sha256(audit_path)}
    output.mkdir(parents=True)
    refs, generated, copies = {}, [], {}
    state = {'status': 'running' if execute else 'dry_run', 'inference_git': revision,
        'training_gits': audit.get('training_gits', {}), 'audit': str(audit_path),
        'method': method, 'branch': branch, 'prompt': PROMPTS[branch], 'subject_step': subject_step,
        'attribute_step': attribute_step, 'expected_images': len(rows), 'generated_images': 0,
        'sampling': SAMPLING, 'retraining': False, 'reexporting': False, 'monitoring': False,
        'subject_training': selected[0], 'attribute_training': selected[1],
        'internal_adapter_slots': {'subject': '<S*>', 'material': TOKENS[branch]},
        'attribute_semantics': branch, 'historical_second_slot_name': 'material'}
    try:
        refs, copies = references(audit, branch, output)
        state['training_images'] = {name: len(images) for name, images in refs.items()}
        (output / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
        if execute:
            import torch
            check_locks(locks)
            pipe = evaluate.load_pipeline(selected, device)
            with (output / 'generation_status.jsonl').open('x', encoding='utf8') as ledger:
                for row in rows:
                    result = pipe(row['prompt'], num_inference_steps=100, guidance_scale=3.5,
                        width=512, height=512, generator=torch.Generator(device=device).manual_seed(row['seed']),
                        cross_attention_kwargs=evaluate.attention_masks(pipe, row['prompt'], ['<S*>', TOKENS[branch]], 3.5))
                    if result.nsfw_content_detected is None or len(result.nsfw_content_detected) != 1:
                        raise ValueError('missing safety checker result')
                    if result.images[0].mode != 'RGB' or result.images[0].size != (512, 512):
                        raise ValueError('composition inference must return512px RGB')
                    image = output / row['image_path']
                    image.parent.mkdir(parents=True, exist_ok=True)
                    with image.open('xb') as handle:
                        result.images[0].save(handle, format='PNG')
                    generated.append({**row, 'status': 'safety_filtered' if result.nsfw_content_detected[0] else 'ok',
                        'subject_adapter_loaded': True, 'attribute_adapter_loaded': True,
                        'attribute_token': TOKENS[branch], 'image_sha256': sha256(image)})
                    ledger.write(json.dumps(generated[-1]) + '\n')
                    ledger.flush()
            del pipe
            torch.cuda.empty_cache()
            if len(generated) != len(rows) or len({row['id'] for row in generated}) != len(rows):
                raise ValueError('incomplete composition matrix')
            state.update(status='succeeded', generated_images=len(generated),
                         safety_filtered=sum(row['status'] == 'safety_filtered' for row in generated))
        check_locks(locks)
        check_locks(copies)
        if git_info() != revision:
            raise ValueError('inference checkout changed')
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}', generated_images=len(generated))
        raise
    finally:
        state['source_locks'], state['reference_copy_hashes'] = locks, copies
        write(output / 'provenance.json', state)
        review(output, refs, generated or rows, state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--method', required=True, choices=METHODS)
    parser.add_argument('--branch', required=True, choices=BRANCHES)
    parser.add_argument('--attribute-step', type=int, choices=(500, 750, 1000), default=1000)
    parser.add_argument('--subject-step', type=int, choices=(200, 400, 600), default=600)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--execute-inference', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.audit, args.output, args.method, args.branch, args.device,
                         args.execute_inference, args.attribute_step, args.subject_step)))
