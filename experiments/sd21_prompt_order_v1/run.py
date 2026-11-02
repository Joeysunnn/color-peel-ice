"""P0 suffix versus P1 same-word prefix; require six exact T400 screen baselines."""

import argparse
import html
import json
import os
from pathlib import Path

from experiments.sd21_texture_repair_v1 import run as screen

sct = screen.sct
ROOT = Path(__file__).resolve().parents[2]
ORDERS = ('P0_suffix', 'P1_prefix')
METHOD, SEEDS = 'lora_r64', (31, 38)
STEPS = {'subject': 1000, 'color': 150, 'texture': 400}


def plan():
    return [{'id': f'{order}__{scene}__seed{seed}', 'condition': order, 'method': METHOD,
             'scene': scene, 'seed': seed, 'steps': STEPS.copy(),
             'prompt': sct.PROMPT + ' ' + suffix if order == 'P0_suffix' else suffix + ' ' + sct.PROMPT,
             'image_path': f'images/{scene}/{order}/seed{seed}.png'}
            for order in ORDERS for scene, suffix in sct.SCENES.items() for seed in SEEDS]


def screen_baselines(folder, ledger):
    folder = Path(folder).resolve()
    if len(ledger) != 96 or {row['id'] for row in ledger} != {row['id'] for row in screen.plan()}:
        raise ValueError('require the complete original96 screen ledger')
    result, locks = {}, {}
    for expected in (row for row in screen.plan() if row['method'] == METHOD and row['condition'] == 't400'):
        rows = [row for row in ledger if row['id'] == expected['id']]
        if len(rows) != 1 or any(rows[0].get(key) != value for key, value in expected.items()):
            raise ValueError('T400 baseline prompt, scales, checkpoint, seed or path differs')
        row = rows[0]
        path = (folder / row['image_path']).resolve()
        if (row.get('status') not in ('ok', 'safety_filtered') or not path.is_relative_to(folder)
                or sct.sha256(path) != row.get('image_sha256')):
            raise ValueError('T400 baseline image or safety evidence changed')
        result[f'{row["scene"]}:{row["seed"]}'] = {'image': str(path), 'sha256': row['image_sha256'],
                                                  'status': row['status'], 'source_id': row['id']}
        locks[str(path)] = row['image_sha256']
    return result, locks


def prepare(screen_audit_path):
    path = Path(screen_audit_path).resolve()
    previous = sct.st.read_json(path)
    status_path, ledger_path = path.parent / 'status.json', path.parent / 'generation_status.jsonl'
    status = sct.st.read_json(status_path)
    if (status.get('status') != 'succeeded' or status.get('expected_images') != 96
            or status.get('generated_images') != 96 or status.get('inference_git') != previous.get('inference_git')
            or status.get('sampling') != {**sct.st.shared.inference.SAMPLING, 'seeds': list(SEEDS)}):
        raise ValueError('require completed screen96 with unchanged generation settings')
    audit = screen.prepare(Path(previous['previous_source_audit']))
    for key in ('sources', 'texture_sources', 'training_gits', 'references', 'texture_conditions'):
        if previous.get(key) != audit.get(key):
            raise ValueError(f'screen source differs from live verified sources: {key}')
    sct.st.shared.snapshots.check_locks(previous['source_locks'])
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding='utf8').splitlines()]
    baselines, locks = screen_baselines(path.parent, ledger)
    locks.update({str(item): sct.sha256(item) for item in (path, status_path, ledger_path)})
    sct.st.source._merge(audit['source_locks'], locks)
    audit.update(screen_source_audit=str(path), screen_inference_git=previous['inference_git'],
                 prompt_order_baselines=baselines, prompt_order_steps=STEPS)
    return audit


def require_baselines(records):
    rows = [row for row in records if row['condition'] == 'P0_suffix']
    if (len(rows) != 6 or {(row['scene'], row['seed']) for row in rows}
            != {(scene, seed) for scene in sct.SCENES for seed in SEEDS}
            or any(row.get('baseline_sha_equal') is not True or row.get('baseline_status_equal') is not True for row in rows)):
        raise ValueError('P1 blocked: all six P0 baseline SHA and safety states must match screen96')


def review(output, records, state):
    by_id = {row['id']: row for row in records}
    cells = []
    for scene in sct.SCENES:
        for seed in SEEDS:
            cells.append(f'<tr><th>{scene} / seed{seed}</th>')
            for order in ORDERS:
                row = by_id.get(f'{order}__{scene}__seed{seed}')
                cells.append('<td>pending</td>' if row is None else
                    f'<td><img width="320" src="{row["image_path"]}"><p>{html.escape(row["status"])}</p>'
                    f'<p>{html.escape(row["prompt"])}</p></td>')
            cells.append('</tr>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>SD2.1 T400 prompt order</h1>'
        f'<p>{html.escape(state["status"])}; P0 verified {state["baseline_equal"]}/6; no retraining.</p>'
        '<table><tr><th>Case</th><th>P0 suffix</th><th>P1 prefix</th></tr>' + ''.join(cells) + '</table>', encoding='utf8')


def run(output, screen_audit_path):
    output, screen_audit_path = sct.st.resolve_run_dir(Path(output)), Path(screen_audit_path).resolve()
    roots = (ROOT, screen_audit_path.parent, sct.st.SUBJECT_PARENT, sct.st.source.TEXTURE_ROOT, sct.COLOR_ROOT)
    if output.exists() or any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots):
        raise ValueError('require a fresh external output separate from all sources')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sct.st.require_idle_gpu0()
    audit = prepare(screen_audit_path)
    selected = screen.selections(audit, METHOD, 't400')
    sct.st.require_idle_gpu0()
    output.mkdir(parents=True)
    rows, generated = plan(), []
    state = {'status': 'running', 'expected_images': 12, 'generated_images': 0, 'baseline_equal': 0,
             'method': METHOD, 'steps': STEPS, 'gpu': 0, 'retraining': False,
             'sampling': {**sct.st.shared.inference.SAMPLING, 'seeds': list(SEEDS)},
             'inference_git': audit['inference_git'], 'screen_inference_git': audit['screen_inference_git']}
    for name, value in (('source_audit.json', audit), ('source_selection.json', selected), ('status.json', state)):
        sct.write(output / name, value)
    (output / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
    try:
        import torch
        from experiments.sd21_mailbox_sct_v1.attention import attention_masks, install_sct_token_local_kv
        from .conditioning import capture_initial_input, diagnose, tokenize, validate_gates

        pipe = sct.st.shared.inference.load_pipeline([], 'cuda:0')
        sources = selected['sources']
        weights = {branch: torch.load(Path(item['checkpoint']) / item['weight'], map_location='cpu')
                   for branch, item in sources.items()}
        install_sct_token_local_kv(pipe.unet, weights, METHOD)
        del weights
        for item in sources.values():
            pipe.load_textual_inversion(item['checkpoint'], weight_name=item['token'] + '.bin')
        pipe.set_progress_bar_config(disable=True)
        sct.write(output / 'text_conditioning_audit.json', diagnose(pipe, sct.PROMPT, sct.SCENES, sct.TOKENS))
        gates, recorded = {}, {}
        for prompt in dict.fromkeys(row['prompt'] for row in rows):
            encoded, positions = tokenize(pipe, prompt, sct.TOKENS)
            gate = attention_masks(pipe, prompt, guidance=3.5)
            validate_gates(encoded.input_ids, gate, pipe.tokenizer, sct.TOKENS)
            gates[prompt] = gate
            recorded[prompt] = {'input_ids': encoded.input_ids.tolist(), 'modifier_positions': positions,
                                'cfg_masks': {branch: mask.cpu().tolist() for branch, mask in gate['modifier_token_mask'].items()}}
        sct.write(output / 'token_gates.json', recorded)
        initial_inputs = {}
        with torch.no_grad(), (output / 'generation_status.jsonl').open('x', encoding='utf8') as ledger:
            for row in rows:
                if row['condition'] == 'P1_prefix':
                    require_baselines(generated)
                captured = {}
                hook = capture_initial_input(pipe.unet, captured)
                try:
                    result = pipe(row['prompt'], num_inference_steps=100, guidance_scale=3.5, width=512, height=512,
                        generator=torch.Generator(device='cuda:0').manual_seed(row['seed']), cross_attention_kwargs=gates[row['prompt']])
                finally:
                    hook.remove()
                if (not captured or captured['shape'] != [2, 4, 64, 64]
                        or result.nsfw_content_detected is None or len(result.nsfw_content_detected) != 1
                        or len(result.images) != 1 or result.images[0].mode != 'RGB' or result.images[0].size != (512, 512)):
                    raise ValueError('require observed initial latent and one512px RGB image with safety status')
                expected_input = initial_inputs.setdefault(row['seed'], captured)
                if captured != expected_input:
                    raise ValueError('same-seed initial denoiser input changed across prompts')
                path = output / row['image_path']
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    result.images[0].save(stream, format='PNG')
                record = {**row, 'image_sha256': sct.sha256(path), 'initial_denoiser_input': captured,
                          'status': 'safety_filtered' if result.nsfw_content_detected[0] else 'ok',
                          'loaded_branches': list(sct.TOKENS)}
                if row['condition'] == 'P0_suffix':
                    baseline = audit['prompt_order_baselines'][f'{row["scene"]}:{row["seed"]}']
                    record.update(previous_baseline=baseline, baseline_sha_equal=record['image_sha256'] == baseline['sha256'],
                                  baseline_status_equal=record['status'] == baseline['status'])
                    state['baseline_equal'] += int(record['baseline_sha_equal'] and record['baseline_status_equal'])
                generated.append(record)
                ledger.write(json.dumps(record) + '\n')
                ledger.flush()
                state['generated_images'] = len(generated)
                sct.write(output / 'status.json', state)
                review(output, generated, state)
                if row['condition'] == 'P0_suffix' and state['baseline_equal'] != len(generated):
                    raise ValueError('P0 differs from screen96; stop before generating any P1')
        require_baselines(generated)
        sct.st.shared.snapshots.check_locks(audit['source_locks'])
        if len(generated) != 12 or sct.st.git_info() != audit['inference_git']:
            raise ValueError('prompt-order generation incomplete or checkout changed')
        state['status'] = 'succeeded'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        sct.write(output / 'status.json', state)
        review(output, generated, state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--screen-audit', required=True, type=Path)
    args = parser.parse_args()
    run(args.output, args.screen_audit)
