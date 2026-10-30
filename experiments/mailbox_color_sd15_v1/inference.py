"""Audit and render the requested Color-only and independently composed S+C probes."""

import argparse
import html
import json
from pathlib import Path
import shutil

from experiments.mailbox_color_sd15_v1 import data, prompts, training
from experiments.sd15_token_local_kv_v1 import evaluate
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.subject_multicolor_multimaterial_b.inference import export_subject_step
from experiments.subject_trellis_dataset_v1 import run_prompt_forms as subject_source
from scripts.launch.colorpeel_run import git_info, resolve_run_dir


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def check_locks(locks):
    if not locks or any(not Path(path).is_absolute() or sha256(path) != digest for path, digest in locks.items()):
        raise ValueError('missing or changed inference source locks')


def prepare_subject(output):
    output = resolve_run_dir(Path(output))
    if (output.exists() or output.is_relative_to(training.ROOT)
            or output.is_relative_to(subject_source.SOURCE) or output.is_relative_to(subject_source.SOURCE_CHECKOUT)):
        raise ValueError('Subject audit requires a new external output')
    output.mkdir(parents=True)
    state = {'status': 'auditing', 'retraining': False, 'subject_step': 1000,
             'git': git_info(), 'training_git': subject_source.TRAINING_GIT}
    try:
        selections, locks = subject_source.source_audit(output)
        runs = [export_subject_step(selection, 1000, output / 'checkpoints' / selection['method'])
                for selection in selections]
        for selection in runs:
            locks.update(selection['hashes'])
        check_locks(locks)
        state.update(status='verified', runs=runs, source_locks=locks,
                     source=str(subject_source.SOURCE), source_checkout=str(subject_source.SOURCE_CHECKOUT))
        write(output / 'subject_index.json', state)
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(output / 'status.json', state)
    return state


def subject_selection(index_path, method):
    index = read_json(index_path)
    if (index.get('status') != 'verified' or index.get('subject_step') != 1000
            or index.get('training_git') != subject_source.TRAINING_GIT
            or len(index.get('runs', [])) != 2
            or {row.get('method') for row in index['runs']} != set(prompts.METHODS)):
        raise ValueError('require the prepared fixed1000 Subject pair')
    # The strict original-revision audit was already performed once in prepare_subject.
    matches = [row for row in index['runs'] if row.get('method') == method]
    selection = matches[0]
    if (selection.get('status') != 'verified' or selection.get('token') != '<S*>'
            or selection.get('branch') != 'subject' or selection.get('steps') != 1000
            or selection.get('training_steps') != 1000 or selection.get('paired_embedding_finite') is not True):
        raise ValueError('Subject adapter and embedding must be the audited1000-step pair')
    if any(index.get('source_locks', {}).get(path) != digest for path, digest in selection['hashes'].items()):
        raise ValueError('prepared Subject selection differs from its recorded source locks')
    check_locks(selection['hashes'])
    return selection, {**selection['hashes'], str(Path(index_path).resolve()): sha256(index_path)}


def color_selection(index_path, dataset, method):
    index = read_json(index_path)
    if index.get('git') != git_info() or Path(index.get('dataset', '')).resolve() != dataset:
        raise ValueError('Color training index revision or dataset differs')
    matches = [row for row in index.get('runs', []) if row.get('method') == method]
    if len(matches) != 1:
        raise ValueError('require exactly one completed Color method in the index')
    selected = matches[0]
    if selected.get('token') != '<C*>' or selected.get('branch') != 'color':
        raise ValueError('Color inference requires an actual Color training selection')
    audited = training.audit_run(method, Path(selected['run']), dataset)
    if selected != audited:
        raise ValueError('Color training selection differs from its actual audit')
    return audited


def references(dataset, output):
    rows = [json.loads(line) for line in (dataset / 'training_assets_manifest.jsonl').read_text(encoding='utf8').splitlines()]
    if len(rows) != 9:
        raise ValueError('review must display all nine actual Color training images')
    locks, result = {}, []
    folder = output / 'training_references'
    folder.mkdir()
    for number, row in enumerate(rows):
        source = Path(row['image']).resolve()
        digest = sha256(source)
        if row.get('image_sha256', digest) != digest or row['caption'].count('<C*>') != 1:
            raise ValueError('actual Color training RGB or caption differs')
        target = folder / f'train_{number:02d}.png'
        shutil.copyfile(source, target)
        if sha256(target) != digest:
            raise ValueError('training reference copy differs')
        locks[str(source)] = digest
        if row.get('mask'):
            mask = Path(row['mask']).resolve()
            locks[str(mask)] = sha256(mask)
        result.append({**row, 'review_image': target.relative_to(output).as_posix()})
    for name in ('concepts.json', 'training_assets_manifest.jsonl', 'source_locks.json', 'dataset_audit.json'):
        path = dataset / name
        if path.is_file():
            locks[str(path)] = sha256(path)
    write(output / 'training_references.json', {'images': result})
    return result, locks


def review(output, refs, rows, state):
    style = '<style>body{font:16px sans-serif}section{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;width:220px}img{width:220px}figcaption{overflow-wrap:anywhere}</style>'
    training_html = '<h2>Nine ACTUAL Color training RGB images and captions</h2><section>' + ''.join(
        f'<figure><img src="{html.escape(row["review_image"], quote=True)}"><figcaption>{html.escape(row["caption"])}</figcaption></figure>'
        for row in refs) + '</section>'
    links = []
    for condition in prompts.conditions():
        filename = condition['condition'] + '.html'
        links.append(f'<li><a href="{filename}">{html.escape(condition["prompt"])}</a></li>')
        group = [row for row in rows if row['condition'] == condition['condition']]
        cards = ''.join('<figure>' + (f'<img loading="lazy" src="{row["image_path"]}">' if row.get('status') else '<p>Planned; inference not executed</p>')
            + f'<figcaption>seed {row["seed"]}: {html.escape(row.get("status", "planned"))}</figcaption></figure>' for row in group)
        (output / filename).write_text('<!doctype html><meta charset="utf-8">' + style
            + f'<p><a href="index.html">Index</a></p><h1>{html.escape(condition["prompt"])}</h1>'
            + f'<p>{state["method"]}; Color step{state["step"]}; Subject1000 for joint probes; PNDM100 / CFG6.0</p>'
            + training_html + '<h2>Seeds30–60</h2><section>' + cards + '</section>', encoding='utf8')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">' + style
        + f'<h1>Mailbox Color transfer: {state["method"]}, step{state["step"]}</h1><p>Status: {state["status"]}</p>'
        + training_html + '<h2>Ten Color-only and three Subject+Color prompts</h2><ul>' + ''.join(links) + '</ul>', encoding='utf8')


def run(dataset, output, method, step, training_index=None, subject_index=None, execute_inference=False, device='cuda:0'):
    dataset, output = Path(dataset).resolve(), resolve_run_dir(Path(output))
    if (output.exists() or output.is_relative_to(training.ROOT) or output.is_relative_to(dataset)
            or dataset.is_relative_to(output) or output.is_relative_to(subject_source.SOURCE)
            or output.is_relative_to(subject_source.SOURCE_CHECKOUT)):
        raise ValueError('inference requires a new external directory separate from data')
    data.verify_dataset(dataset)
    rows, revision = prompts.prompt_rows(method, step), git_info()
    output.mkdir(parents=True)
    refs, locks = references(dataset, output)
    state = {'status': 'running' if execute_inference else 'dry_run', 'git': revision,
             'method': method, 'step': step, 'subject_step': 1000, 'dataset': str(dataset),
             'training_images': 9, 'expected_images': 403, 'generated_images': 0,
             'sampling': prompts.SAMPLING, 'prompt_source': prompts.source_record(),
             'internal_adapter_slots': {'subject': '<S*>', 'material': '<C*> (historical second-slot name)'},
             'training_index': str(training_index) if training_index else None,
             'subject_index': str(subject_index) if subject_index else None}
    (output / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
    generated = []
    try:
        if execute_inference:
            if training_index is None or subject_index is None:
                raise ValueError('executed transfer requires audited Color and Subject indices')
            color = color_selection(training_index, dataset, method)
            color = training.export_step(color, step, output / 'checkpoints' / 'color')
            subject, subject_locks = subject_selection(subject_index, method)
            locks.update(subject_locks)
            locks.update(color['hashes'])
            locks[str(Path(training_index).resolve())] = sha256(training_index)
            state['color_training'], state['subject_training'] = color, subject
            import torch
            with (output / 'generation_status.jsonl').open('x', encoding='utf8') as ledger:
                for arm in ('color_only', 'subject_color'):
                    selections = [color] if arm == 'color_only' else [subject, color]
                    tokens = ['<C*>'] if arm == 'color_only' else ['<S*>', '<C*>']
                    check_locks(locks)
                    pipe = evaluate.load_pipeline(selections, device)
                    for row in (row for row in rows if row['arm'] == arm):
                        result = pipe(row['prompt'], num_inference_steps=100, guidance_scale=6.0,
                            width=512, height=512, generator=torch.Generator(device=device).manual_seed(row['seed']),
                            cross_attention_kwargs=evaluate.attention_masks(pipe, row['prompt'], tokens, 6.0))
                        if result.nsfw_content_detected is None or len(result.nsfw_content_detected) != 1:
                            raise ValueError('missing safety checker result')
                        image = output / row['image_path']
                        image.parent.mkdir(parents=True, exist_ok=True)
                        if result.images[0].mode != 'RGB' or result.images[0].size != (512, 512):
                            raise ValueError('inference must return512px RGB')
                        result.images[0].save(image)
                        generated.append({**row, 'status': 'safety_filtered' if result.nsfw_content_detected[0] else 'ok',
                                          'image_sha256': sha256(image)})
                        ledger.write(json.dumps(generated[-1]) + '\n')
                        ledger.flush()
                    del pipe
                    torch.cuda.empty_cache()
            if len(generated) != 403 or len({row['id'] for row in generated}) != 403:
                raise ValueError('incomplete transfer matrix')
            state.update(status='succeeded', generated_images=len(generated),
                         safety_filtered=sum(row['status'] == 'safety_filtered' for row in generated))
        check_locks(locks)
        data.verify_dataset(dataset)
        if git_info() != revision:
            raise ValueError('inference checkout changed')
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}', generated_images=len(generated))
        raise
    finally:
        state['source_locks'] = locks
        write(output / 'provenance.json', state)
        review(output, refs, generated or rows, state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--method', required=True, choices=prompts.METHODS)
    parser.add_argument('--step', required=True, type=int, choices=prompts.STEPS)
    parser.add_argument('--training-index', type=Path)
    parser.add_argument('--subject-index', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--execute-inference', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.output, args.method, args.step, args.training_index,
                         args.subject_index, args.execute_inference, args.device)))
