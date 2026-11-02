"""Frozen S/C/T residual ablations for flowers, plus a literal stock control."""

import argparse
import copy
import html
import json
import os
from pathlib import Path

from experiments.sd21_mailbox_sct_v1 import run as sct

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (31, 38)
PROMPT = sct.PROMPT + ' surrounded by flowers'
STOCK_PROMPT = 'a photo of a mailbox surrounded by flowers'
DISABLED = {'baseline': (), 'all_residual_off': tuple(sct.TOKENS),
            'subject_residual_off': ('subject',), 'color_residual_off': ('color',),
            'texture_residual_off': ('texture',)}


def plan():
    rows = [{'method': method, 'condition': condition, 'seed': seed, 'prompt': PROMPT,
             'image_path': f'images/{method}/{condition}/seed{seed}.png'}
            for method in sct.METHODS for condition in DISABLED for seed in SEEDS]
    return rows + [{'method': 'stock', 'condition': 'stock', 'seed': seed, 'prompt': STOCK_PROMPT,
                    'image_path': f'images/stock/seed{seed}.png'} for seed in SEEDS]


def condition_gates(gates, condition):
    if condition not in DISABLED:
        raise ValueError('unknown residual ablation')
    result = copy.deepcopy(gates)
    for branch in DISABLED[condition]:
        result['modifier_token_mask'][branch][...] = False
    return result


def prepare(previous_path):
    previous_path = Path(previous_path).resolve()
    previous_hash = sct.sha256(previous_path)
    previous = sct.st.read_json(previous_path)
    audit = sct.prepare()  # Recheck the selected adapter/token tensors, data and original training Git.
    for key in ('sources', 'source_locks', 'training_gits', 'steps', 'training_images', 'references'):
        if previous.get(key) != audit[key]:
            raise ValueError(f'previous scene experiment has different sources: {key}')
    old_status_path = previous_path.parent / 'status.json'
    old_status = sct.st.read_json(old_status_path)
    if old_status.get('status') != 'succeeded' or old_status.get('scene_transfer') is not True:
        raise ValueError('require the completed scene-transfer experiment')
    locks = {str(previous_path): previous_hash, str(old_status_path): sct.sha256(old_status_path)}
    baselines = {}
    for method in sct.METHODS:
        folder = previous_path.parent / 'methods' / method
        ledger_path = folder / 'generation_status.jsonl'
        ledger = [json.loads(line) for line in ledger_path.read_text(encoding='utf8').splitlines()]
        locks[str(ledger_path)] = sct.sha256(ledger_path)
        baselines[method] = {}
        for seed in SEEDS:
            matches = [row for row in ledger if row.get('scene') == 'flowers' and row.get('seed') == seed]
            if len(matches) != 1 or matches[0]['prompt'] != PROMPT:
                raise ValueError('previous flowers baseline differs or is ambiguous')
            row = matches[0]
            image = (folder / row['image_path']).resolve()
            if not image.is_relative_to(folder.resolve()) or sct.sha256(image) != row['image_sha256']:
                raise ValueError('previous baseline image path or bytes differ')
            locks[str(image)] = row['image_sha256']
            baselines[method][str(seed)] = {'image': str(image), 'sha256': row['image_sha256'],
                                           'status': row['status']}
    sct.st.source._merge(audit['source_locks'], locks)
    sct.st.shared.snapshots.check_locks(audit['source_locks'])
    audit.update(previous_source_audit=str(previous_path), previous_inference_git=previous['inference_git'],
                 diagnosis_prompt=PROMPT, stock_prompt=STOCK_PROMPT, baselines=baselines)
    return audit


def run(output, previous_path):
    output, previous_path = sct.st.resolve_run_dir(Path(output)), Path(previous_path).resolve()
    roots = (ROOT, sct.st.SUBJECT_PARENT, sct.st.source.TEXTURE_ROOT, sct.COLOR_ROOT, previous_path.parent)
    if output.exists() or any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots):
        raise ValueError('require fresh external output separate from code and source experiments')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sct.st.require_idle_gpu0()
    audit = prepare(previous_path)
    sct.st.require_idle_gpu0()
    output.mkdir(parents=True)
    rows, generated, recorded_gates = plan(), [], {}
    state = {'status': 'running', 'expected_images': len(rows), 'generated_images': 0,
             'inference_git': audit['inference_git'], 'previous_inference_git': audit['previous_inference_git'],
             'training_gits': audit['training_gits'], 'steps': audit['steps'], 'gpu': 0, 'retraining': False,
             'sampling': {**sct.st.shared.inference.SAMPLING, 'seeds': list(SEEDS)},
             'conditions': DISABLED, 'embeddings_retained_for_all_ablations': True,
             'stock_control_has_learned_tokens_or_adapters': False}
    sct.write(output / 'source_audit.json', audit)
    sct.write(output / 'status.json', state)
    (output / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
    try:
        import torch
        from experiments.sd21_mailbox_sct_v1.attention import install_sct_token_local_kv, attention_masks

        with torch.no_grad(), (output / 'generation_status.jsonl').open('x', encoding='utf8') as ledger:
            for method in (*sct.METHODS, 'stock'):
                pipe = sct.st.shared.inference.load_pipeline([], 'cuda:0', stock=method == 'stock')
                if method != 'stock':
                    selections = {branch: audit['sources'][branch][method] for branch in sct.TOKENS}
                    weights = {branch: torch.load(Path(item['checkpoint']) / item['weight'], map_location='cpu')
                               for branch, item in selections.items()}
                    install_sct_token_local_kv(pipe.unet, weights, method)
                    del weights
                    for item in selections.values():
                        pipe.load_textual_inversion(item['checkpoint'], weight_name=item['token'] + '.bin')
                    original_gates = attention_masks(pipe, PROMPT, guidance=3.5)
                    gates = {condition: condition_gates(original_gates, condition) for condition in DISABLED}
                    recorded_gates[method] = {condition: {branch: mask.cpu().tolist()
                        for branch, mask in gate['modifier_token_mask'].items()} for condition, gate in gates.items()}
                    sct.write(output / 'token_gates.json', recorded_gates)
                pipe.set_progress_bar_config(disable=True)
                for row in (item for item in rows if item['method'] == method):
                    kwargs = {} if method == 'stock' else {'cross_attention_kwargs': gates[row['condition']]}
                    result = pipe(row['prompt'], num_inference_steps=100, guidance_scale=3.5, width=512, height=512,
                                  generator=torch.Generator(device='cuda:0').manual_seed(row['seed']), **kwargs)
                    if (result.nsfw_content_detected is None or len(result.nsfw_content_detected) != 1
                            or len(result.images) != 1 or result.images[0].mode != 'RGB' or result.images[0].size != (512, 512)):
                        raise ValueError('require one512px RGB image with safety status')
                    path = output / row['image_path']
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('xb') as stream:
                        result.images[0].save(stream, format='PNG')
                    record = {**row, 'image_sha256': sct.sha256(path),
                              'status': 'safety_filtered' if result.nsfw_content_detected[0] else 'ok'}
                    if row['condition'] == 'baseline':
                        previous = audit['baselines'][method][str(row['seed'])]
                        record.update(previous_baseline=previous,
                                      baseline_sha_equal=record['image_sha256'] == previous['sha256'])
                    generated.append(record)
                    ledger.write(json.dumps(record) + '\n')
                    ledger.flush()
                    state['generated_images'] = len(generated)
                    sct.write(output / 'status.json', state)
                del pipe
                torch.cuda.empty_cache()
        sct.st.shared.snapshots.check_locks(audit['source_locks'])
        if sct.st.git_info() != audit['inference_git'] or len(generated) != len(rows):
            raise ValueError('inference checkout changed or diagnosis incomplete')
        state['status'] = 'succeeded'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        sct.write(output / 'status.json', state)
        (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>SD2.1 frozen residual diagnosis</h1>'
            + '<p>S1000/C150/T1000; flowers; seeds31/38; PNDM100/CFG3.5. Ablations retain all three embeddings.</p>'
            + ''.join(f'<figure style="display:inline-block;width:300px"><img width="300" src="{row["image_path"]}">'
                f'<figcaption>{html.escape(row["method"])} / {row["condition"]} / seed{row["seed"]}: {row["status"]}'
                + (f'; baseline SHA equal: {row["baseline_sha_equal"]}' if 'baseline_sha_equal' in row else '')
                + '</figcaption></figure>' for row in generated), encoding='utf8')
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-audit', required=True, type=Path)
    args = parser.parse_args()
    run(args.output, args.source_audit)
