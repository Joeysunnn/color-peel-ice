"""Screen eight frozen Texture settings with fixed Mailbox S1000/orange C150."""

import argparse
import html
import json
import os
from pathlib import Path

from experiments.sd21_background_diagnosis_v1 import run as diagnosis

sct = diagnosis.sct
ROOT = Path(__file__).resolve().parents[2]
SEEDS = (31, 38)
CONDITIONS = {
    't400': {'step': 400, 'key_scale': 1.0, 'value_scale': 1.0},
    't600': {'step': 600, 'key_scale': 1.0, 'value_scale': 1.0},
    't800': {'step': 800, 'key_scale': 1.0, 'value_scale': 1.0},
    't1000': {'step': 1000, 'key_scale': 1.0, 'value_scale': 1.0},
    't1000_scale025': {'step': 1000, 'key_scale': 0.25, 'value_scale': 0.25},
    't1000_scale050': {'step': 1000, 'key_scale': 0.5, 'value_scale': 0.5},
    't1000_k0_v1': {'step': 1000, 'key_scale': 0.0, 'value_scale': 1.0},
    't1000_k1_v0': {'step': 1000, 'key_scale': 1.0, 'value_scale': 0.0},
}


def plan():
    return [{'id': f'{method}__{condition}__{scene}__seed{seed}', 'method': method,
             'condition': condition, 'scene': scene, 'seed': seed,
             'prompt': sct.PROMPT + ' ' + suffix,
             'steps': {'subject': 1000, 'color': 150, 'texture': setting['step']},
             'texture_scales': {'key': setting['key_scale'], 'value': setting['value_scale']},
             'image_path': f'images/{method}/{scene}/{condition}/seed{seed}.png'}
            for method in sct.METHODS for condition, setting in CONDITIONS.items()
            for scene, suffix in sct.SCENES.items() for seed in SEEDS]


def prepare(previous_path):
    previous_path = Path(previous_path).resolve()
    audit = diagnosis.prepare(previous_path)
    audit['texture_sources'] = {'1000': audit['sources']['texture']}
    for step in (400, 600, 800):
        audit['texture_sources'][str(step)] = {}
        for method in sct.METHODS:
            item, locks = sct.st.source._pair(sct.st.source.TEXTURE_ROOT, 'texture', method, step,
                sct.st.source.TEXTURE_GIT, audit['sources']['subject'][method]['base_model'],
                recipe_contract=sct.st.shared.contract)
            audit['texture_sources'][str(step)][method] = item
            sct.st.source._merge(audit['source_locks'], locks)
    baselines = {method: {} for method in sct.METHODS}
    for method in sct.METHODS:
        folder = previous_path.parent / 'methods' / method
        ledger = [json.loads(line) for line in (folder / 'generation_status.jsonl').read_text(encoding='utf8').splitlines()]
        for scene, suffix in sct.SCENES.items():
            baselines[method][scene] = {}
            for seed in SEEDS:
                matches = [row for row in ledger if row.get('scene') == scene and row.get('seed') == seed]
                if len(matches) != 1 or matches[0]['prompt'] != sct.PROMPT + ' ' + suffix:
                    raise ValueError('previous scene baseline differs or is ambiguous')
                row = matches[0]
                image = (folder / row['image_path']).resolve()
                if not image.is_relative_to(folder.resolve()) or sct.sha256(image) != row['image_sha256']:
                    raise ValueError('previous scene baseline image path or bytes differ')
                sct.st.source._merge(audit['source_locks'], {str(image): row['image_sha256']})
                baselines[method][scene][str(seed)] = {'image': str(image), 'sha256': row['image_sha256'],
                                                     'status': row['status']}
    audit.update(baselines=baselines, texture_conditions=CONDITIONS,
                 generation_prompts=[sct.PROMPT + ' ' + suffix for suffix in sct.SCENES.values()])
    sct.st.shared.snapshots.check_locks(audit['source_locks'])
    if sct.st.git_info() != audit['inference_git']:
        raise ValueError('checkout changed during Texture source audit')
    return audit


def selections(audit, method, condition):
    selected = {branch: audit['sources'][branch][method] for branch in ('subject', 'color')}
    selected['texture'] = audit['texture_sources'][str(CONDITIONS[condition]['step'])][method]
    locks = {}
    for item in selected.values():
        sct.st.source._merge(locks, item['hashes'])
    return {'sources': selected, 'source_locks': locks, 'training_gits': audit['training_gits'],
            'inference_git': audit['inference_git'], 'texture_setting': CONDITIONS[condition]}


def review(output, rows, generated, state):
    completed = {row['id']: row for row in generated}
    links = []
    style = '<style>body{font:16px sans-serif}td,th{vertical-align:top;padding:8px}img{width:256px}td{min-width:256px}</style>'
    for method in sct.METHODS:
        for scene in sct.SCENES:
            page = output / 'review' / method / f'{scene}.html'
            page.parent.mkdir(parents=True, exist_ok=True)
            selected = [row for row in rows if row['method'] == method and row['scene'] == scene]
            cells = []
            for seed in SEEDS:
                cells.append(f'<tr><th>seed{seed}</th>')
                for condition in CONDITIONS:
                    row = next(row for row in selected if row['seed'] == seed and row['condition'] == condition)
                    record = completed.get(row['id'])
                    label = 'planned' if record is None else record['status']
                    content = '' if record is None else f'<a href="../../{row["image_path"]}"><img src="../../{row["image_path"]}"></a>'
                    if record is not None and 'baseline_sha_equal' in record:
                        label += f'; prior SHA equal: {record["baseline_sha_equal"]}'
                    cells.append(f'<td>{content}<p>{label}</p></td>')
                cells.append('</tr>')
            page.write_text('<!doctype html><meta charset="utf-8">' + style
                + f'<a href="../../index.html">All methods/scenes</a><h1>{method} / {scene}</h1>'
                + f'<p>{html.escape(selected[0]["prompt"])}</p><table><tr><th>Seed</th>'
                + ''.join(f'<th>{name}<br>T{value["step"]}; K={value["key_scale"]}, V={value["value_scale"]}</th>'
                          for name, value in CONDITIONS.items()) + '</tr>' + ''.join(cells) + '</table>', encoding='utf8')
            links.append(f'<li><a href="{page.relative_to(output).as_posix()}">{method} / {scene}: eight conditions, two seeds</a></li>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">' + style
        + '<h1>SD2.1 Texture checkpoint / K/V scale screen</h1><p>S1000 + C150 fixed; PNDM100 / CFG3.5 / 512px.</p>'
        + f'<p>{state["status"]}: {state["generated_images"]}/{state["expected_images"]} images.</p>'
        + '<p>Checkpoint changes pair the Texture adapter and embedding at the same step. Scale cases retain T1000 embedding.</p>'
        + '<ul>' + ''.join(links) + '</ul>', encoding='utf8')


def run(output, previous_path):
    output, previous_path = sct.st.resolve_run_dir(Path(output)), Path(previous_path).resolve()
    roots = (ROOT, sct.st.SUBJECT_PARENT, sct.st.source.TEXTURE_ROOT, sct.COLOR_ROOT, previous_path.parent)
    if output.exists() or any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots):
        raise ValueError('require fresh external output separate from all sources and code')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sct.st.require_idle_gpu0()
    audit = prepare(previous_path)
    sct.st.require_idle_gpu0()
    output.mkdir(parents=True)
    rows, generated, recorded_gates = plan(), [], {}
    selected = {method: {condition: selections(audit, method, condition) for condition in CONDITIONS}
                for method in sct.METHODS}
    state = {'status': 'running', 'expected_images': len(rows), 'generated_images': 0, 'gpu': 0,
             'inference_git': audit['inference_git'], 'previous_inference_git': audit['previous_inference_git'],
             'training_gits': audit['training_gits'], 'retraining': False, 'fixed_steps': {'subject': 1000, 'color': 150},
             'texture_conditions': CONDITIONS, 'sampling': {**sct.st.shared.inference.SAMPLING, 'seeds': list(SEEDS)},
             'baseline_checks': {'equal': 0, 'different': 0}, 'completed_conditions': {m: [] for m in sct.METHODS}}
    sct.write(output / 'source_audit.json', audit)
    sct.write(output / 'source_selections.json', selected)
    sct.write(output / 'status.json', state)
    (output / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
    review(output, rows, generated, state)
    try:
        import torch
        from experiments.sd21_mailbox_sct_v1.attention import install_sct_token_local_kv, attention_masks
        from experiments.sd21_texture_repair_v1.attention import scale_texture_residuals

        with torch.no_grad(), (output / 'generation_status.jsonl').open('x', encoding='utf8') as ledger:
            for method in sct.METHODS:
                recorded_gates[method] = {}
                pipe, loaded_step = None, None
                for condition, setting in CONDITIONS.items():
                    state.update(active_method=method, active_condition=condition)
                    sct.write(output / 'status.json', state)
                    new_step = setting['step'] != loaded_step
                    if new_step:
                        if pipe is not None:
                            del pipe
                            torch.cuda.empty_cache()
                        pipe = sct.st.shared.inference.load_pipeline([], 'cuda:0')
                    sources = selected[method][condition]['sources']
                    weights = {branch: torch.load(Path(item['checkpoint']) / item['weight'], map_location='cpu')
                               for branch, item in sources.items()}
                    install_sct_token_local_kv(pipe.unet, weights, method)
                    del weights
                    scale_texture_residuals(pipe.unet, setting['key_scale'], setting['value_scale'])
                    if new_step:
                        for item in sources.values():
                            pipe.load_textual_inversion(item['checkpoint'], weight_name=item['token'] + '.bin')
                        loaded_step = setting['step']
                    pipe.set_progress_bar_config(disable=True)
                    cases = [row for row in rows if row['method'] == method and row['condition'] == condition]
                    gates = {prompt: attention_masks(pipe, prompt, guidance=3.5) for prompt in dict.fromkeys(row['prompt'] for row in cases)}
                    recorded_gates[method][condition] = {prompt: {branch: mask.cpu().tolist()
                        for branch, mask in gate['modifier_token_mask'].items()} for prompt, gate in gates.items()}
                    sct.write(output / 'token_gates.json', recorded_gates)
                    for row in cases:
                        state.update(active_scene=row['scene'], active_seed=row['seed'])
                        sct.write(output / 'status.json', state)
                        result = pipe(row['prompt'], num_inference_steps=100, guidance_scale=3.5, width=512, height=512,
                            generator=torch.Generator(device='cuda:0').manual_seed(row['seed']), cross_attention_kwargs=gates[row['prompt']])
                        if (result.nsfw_content_detected is None or len(result.nsfw_content_detected) != 1
                                or len(result.images) != 1 or result.images[0].mode != 'RGB' or result.images[0].size != (512, 512)):
                            raise ValueError('require one512px RGB image with safety status')
                        path = output / row['image_path']
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with path.open('xb') as stream:
                            result.images[0].save(stream, format='PNG')
                        record = {**row, 'image_sha256': sct.sha256(path), 'loaded_branches': list(sct.TOKENS),
                                  'status': 'safety_filtered' if result.nsfw_content_detected[0] else 'ok'}
                        if condition == 't1000':
                            baseline = audit['baselines'][method][row['scene']][str(row['seed'])]
                            record.update(previous_baseline=baseline, baseline_sha_equal=record['image_sha256'] == baseline['sha256'])
                            state['baseline_checks']['equal' if record['baseline_sha_equal'] else 'different'] += 1
                        generated.append(record)
                        ledger.write(json.dumps(record) + '\n')
                        ledger.flush()
                        state['generated_images'] = len(generated)
                        sct.write(output / 'status.json', state)
                    state['completed_conditions'][method].append(condition)
                    sct.write(output / 'status.json', state)
                    review(output, rows, generated, state)
                del pipe
                torch.cuda.empty_cache()
        sct.st.shared.snapshots.check_locks(audit['source_locks'])
        if sct.st.git_info() != audit['inference_git'] or len(generated) != len(rows):
            raise ValueError('inference checkout changed or Texture screen incomplete')
        state['status'] = 'succeeded'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        sct.write(output / 'status.json', state)
        review(output, rows, generated, state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-audit', required=True, type=Path)
    args = parser.parse_args()
    run(args.output, args.source_audit)
