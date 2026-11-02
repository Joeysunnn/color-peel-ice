"""18 frozen LoRA images: original SCT, practical S+C, and Texture-only context routing."""

import argparse
import html
import json
import os
from pathlib import Path

from experiments.sd21_prompt_order_v1 import run as previous

sct, screen = previous.sct, previous.screen
ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ('baseline_sct', 'true_SC', 't_residual_initializer')
SEEDS, METHOD, STEPS = previous.SEEDS, previous.METHOD, previous.STEPS


def plan():
    return [{'id': f'{condition}__{scene}__seed{seed}', 'condition': condition, 'method': METHOD,
             'scene': scene, 'seed': seed, 'steps': STEPS.copy(),
             'prompt': ('a photo of <S*> mailbox in <C*> color' if condition == 'true_SC' else sct.PROMPT) + ' ' + suffix,
             'image_path': f'images/{scene}/{condition}/seed{seed}.png'}
            for condition in CONDITIONS for scene, suffix in sct.SCENES.items() for seed in SEEDS]


def require_baselines(records):
    # Reuse the six-image SHA/safety gate without changing our stored condition labels.
    previous.require_baselines([{**row, 'condition': 'P0_suffix'} for row in records if row['condition'] == 'baseline_sct'])


def review(output, records, state):
    by_id, cells = {row['id']: row for row in records}, []
    for scene in sct.SCENES:
        for seed in SEEDS:
            cells.append(f'<tr><th>{scene} / seed{seed}</th>')
            for condition in CONDITIONS:
                row = by_id.get(f'{condition}__{scene}__seed{seed}')
                cells.append('<td>pending</td>' if row is None else
                    f'<td><a href="{row["image_path"]}"><img width="320" src="{row["image_path"]}"></a>'
                    f'<p>{html.escape(row["status"])}</p><p>{html.escape(row["prompt"])}</p></td>')
            cells.append('</tr>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>SD2.1 T400 context routing</h1>'
        f'<p>{html.escape(state["status"])}; {state["generated_images"]}/18; baseline exact {state["baseline_equal"]}/6.</p>'
        '<p>True SC removes the Texture phrase and disables its residual. Q1 keeps the SCT prompt and routes auxiliary context only to Texture residuals. No retraining.</p>'
        '<table><tr><th>Case</th>' + ''.join(f'<th>{name}</th>' for name in CONDITIONS)
        + '</tr>' + ''.join(cells) + '</table>', encoding='utf8')


def run(output, screen_audit_path):
    output, screen_audit_path = sct.st.resolve_run_dir(Path(output)), Path(screen_audit_path).resolve()
    roots = (ROOT, screen_audit_path.parent, sct.st.SUBJECT_PARENT, sct.st.source.TEXTURE_ROOT, sct.COLOR_ROOT)
    if output.exists() or any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots):
        raise ValueError('require a fresh external output separate from all sources')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sct.st.require_idle_gpu0()
    audit = previous.prepare(screen_audit_path)
    selected = screen.selections(audit, METHOD, 't400')
    sct.st.require_idle_gpu0()
    output.mkdir(parents=True)
    rows, generated = plan(), []
    state = {'status': 'running', 'expected_images': 18, 'generated_images': 0, 'baseline_equal': 0,
        'method': METHOD, 'steps': STEPS, 'gpu': 0, 'retraining': False,
        'sampling': {**sct.st.shared.inference.SAMPLING, 'seeds': list(SEEDS)},
        'inference_git': audit['inference_git'], 'screen_inference_git': audit['screen_inference_git']}
    for name, value in (('source_audit.json', audit), ('source_selection.json', selected), ('status.json', state)):
        sct.write(output / name, value)
    (output / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
    try:
        import torch
        from experiments.sd21_mailbox_sct_v1.attention import install_sct_token_local_kv
        from experiments.sd21_prompt_order_v1.conditioning import capture_initial_input
        from .attention import install_context_routing
        from .conditioning import prepare_conditioning

        pipe = sct.st.shared.inference.load_pipeline([], 'cuda:0')
        sources = selected['sources']
        weights = {branch: torch.load(Path(item['checkpoint']) / item['weight'], map_location='cpu') for branch, item in sources.items()}
        install_sct_token_local_kv(pipe.unet, weights, METHOD)
        del weights
        install_context_routing(pipe.unet)
        for item in sources.values():
            pipe.load_textual_inversion(item['checkpoint'], weight_name=item['token'] + '.bin')
        pipe.set_progress_bar_config(disable=True)
        norms = {name: pipe.unet.get_submodule(name.removesuffix('.processor')).norm_cross
                 for name in pipe.unet.attn_processors if name.endswith('attn2.processor')}
        if any(bool(value) for value in norms.values()):
            raise ValueError('this pinned diagnostic requires no cross-attention context normalization')
        gates, conditioning = {}, {'cross_attention_norms': norms, 'conditions': {}}
        for condition in CONDITIONS:
            conditioning['conditions'][condition] = {}
            for row in (row for row in rows if row['condition'] == condition and row['seed'] == SEEDS[0]):
                gate, record = prepare_conditioning(pipe, row['prompt'], condition)
                gates[condition, row['scene']] = gate
                conditioning['conditions'][condition][row['scene']] = record
        for scene in sct.SCENES:
            baseline = conditioning['conditions']['baseline_sct'][scene]
            routed = conditioning['conditions']['t_residual_initializer'][scene]
            if any(baseline[key] != routed[key] for key in ('main_input_ids', 'negative_input_ids',
                    'modifier_positions', 'cfg_masks', 'main_hidden_sha256')):
                raise ValueError('Q1 changed the original main prompt encoding or gates')
            routed['main_conditioning_equals_baseline'] = True
        sct.write(output / 'text_conditioning_audit.json', conditioning)
        initial_inputs = {}
        with torch.no_grad(), (output / 'generation_status.jsonl').open('x', encoding='utf8') as ledger:
            for row in rows:
                if row['condition'] != 'baseline_sct':
                    require_baselines(generated)
                captured = {}
                hook = capture_initial_input(pipe.unet, captured)
                try:
                    result = pipe(row['prompt'], num_inference_steps=100, guidance_scale=3.5, width=512, height=512,
                        generator=torch.Generator(device='cuda:0').manual_seed(row['seed']),
                        cross_attention_kwargs=gates[row['condition'], row['scene']])
                finally:
                    hook.remove()
                if (not captured or captured['shape'] != [2, 4, 64, 64]
                        or result.nsfw_content_detected is None or len(result.nsfw_content_detected) != 1
                        or len(result.images) != 1 or result.images[0].mode != 'RGB' or result.images[0].size != (512, 512)):
                    raise ValueError('require observed initial latent and one512px RGB image with safety status')
                if captured != initial_inputs.setdefault(row['seed'], captured):
                    raise ValueError('same-seed initial denoiser input changed across conditions')
                path = output / row['image_path']
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    result.images[0].save(stream, format='PNG')
                record = {**row, 'image_sha256': sct.sha256(path), 'initial_denoiser_input': captured,
                    'status': 'safety_filtered' if result.nsfw_content_detected[0] else 'ok',
                    'loaded_branches': list(sct.TOKENS),
                    'active_residual_branches': ['subject', 'color'] if row['condition'] == 'true_SC' else list(sct.TOKENS),
                    'texture_auxiliary_context': row['condition'] == 't_residual_initializer'}
                if row['condition'] == 'baseline_sct':
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
                if row['condition'] == 'baseline_sct' and state['baseline_equal'] != len(generated):
                    raise ValueError('SCT baseline differs from screen96; stop before controls')
        require_baselines(generated)
        sct.st.shared.snapshots.check_locks(audit['source_locks'])
        if len(generated) != 18 or sct.st.git_info() != audit['inference_git']:
            raise ValueError('context-routing generation incomplete or checkout changed')
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
