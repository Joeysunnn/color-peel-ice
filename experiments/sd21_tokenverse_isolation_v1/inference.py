"""Fixed112-image factorial and independent-branch evaluation; no training."""

import argparse
import html
import json
import os
from pathlib import Path

BRANCHES = ('subject', 'color', 'texture')
ARMS = {'c': 'control', 'i': 'isolation'}
COMBINATIONS = ('ccc', 'icc', 'cic', 'cci', 'iic', 'ici', 'cii', 'iii')
SEEDS = (31, 38)
SCENES = {'street': 'in the street', 'snow': 'in the snow',
          'flowers': 'surrounded by flowers', 'beach': 'at the beach'}
GROUPS = (('final', 'sct'), ('mid', 'sct'), ('final', 'subject_only'),
          ('final', 'texture_only'), ('final', 'color_only'))


def plan():
    rows = []

    def add(phase, mode, condition, arms, steps, cases):
        for scene, seed, prompt in cases:
            identity = f'{phase}__{mode}__{condition}__{scene}__seed{seed}'
            rows.append({'id': identity, 'phase': phase, 'mode': mode, 'condition': condition,
                'method': 'lora_r64', 'branch_arms': dict(arms), 'branch_steps': dict(steps),
                'scene': scene, 'seed': seed, 'prompt': prompt, 'negative_prompt': '',
                'image_path': f'images/{phase}/{mode}/{condition}/{scene}/seed{seed}.png'})

    for phase, steps, combinations in (
            ('final', (1000, 150, 1000), COMBINATIONS), ('mid', (500, 100, 500), ('ccc', 'iii'))):
        cases = [(scene, seed, 'a photo of <S*> mailbox in <C*> color with <T*> texture ' + suffix)
                 for scene, suffix in SCENES.items() for seed in SEEDS]
        for combination in combinations:
            add(phase, 'sct', combination, dict(zip(BRANCHES, (ARMS[letter] for letter in combination))),
                dict(zip(BRANCHES, steps)), cases)
    for branch, step in (('subject', 1000), ('texture', 1000), ('color', 150)):
        if branch == 'subject':
            cases = [(scene, seed, 'a photo of <S*> mailbox ' + suffix)
                     for scene, suffix in SCENES.items() for seed in SEEDS]
        elif branch == 'texture':
            anchors = [(obj, seed) for obj in ('mug', 'vase') for seed in SEEDS] + [('chair', 38), ('sofa', 31)]
            cases = [(obj, seed, f'a photo of a {obj} with <T*> texture') for obj, seed in anchors]
        else:
            cases = [(obj, seed, f'a photo of a {obj} in <C*> color') for obj, seed in (('mug', 31), ('vase', 38))]
        for arm in ARMS.values():
            add('final', branch + '_only', arm, {branch: arm}, {branch: step}, cases)
    return rows


def review(output, generated, state):
    planned, records, cards = plan(), {row['id']: row for row in generated}, []
    for phase, mode in GROUPS:
        group = [row for row in planned if (row['phase'], row['mode']) == (phase, mode)]
        conditions = list(dict.fromkeys(row['condition'] for row in group))
        cards.append(f'<h2>{phase} / {mode}</h2><p>Steps: {html.escape(str(group[0]["branch_steps"]))}</p>')
        cards.append('<table><tr><th>scene / seed</th>' + ''.join(f'<th>{name}</th>' for name in conditions) + '</tr>')
        for scene, seed in dict.fromkeys((row['scene'], row['seed']) for row in group):
            cards.append(f'<tr><th>{scene} / {seed}</th>')
            for condition in conditions:
                identity = f'{phase}__{mode}__{condition}__{scene}__seed{seed}'
                row = records.get(identity)
                cards.append('<td>planned</td>' if row is None else
                    f'<td><a href="{row["image_path"]}"><img width="256" src="{row["image_path"]}"></a>'
                    f'<p>{row["status"]}</p><small>{html.escape(row["prompt"])}</small></td>')
            cards.append('</tr>')
        cards.append('</table>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<style>body{font:16px sans-serif}td{vertical-align:top;width:256px}</style>'
        f'<h1>Token isolation: {state["status"]}, {state["generated_images"]}/112</h1>'
        '<p>Combination letters follow Subject / Color / Texture: c=control, i=isolation. '
        'LoRA64; PNDM100 / CFG3.5 / 512px; empty negative prompt; safety checker enabled.</p>'
        '<p>Final factorial, midpoint comparison and single-branch controls are separate. '
        'No automatic quality judgment.</p>' + ''.join(cards), encoding='utf8')


def run(experiment):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import torch
    from experiments.sd21_mailbox_sct_v1 import run as sct
    from experiments.sd21_mailbox_sct_v1.attention import install_sct_token_local_kv, attention_masks
    from experiments.sd15_token_local_kv_v1.evaluate import attention_masks as single_masks
    from experiments.sd21_prompt_order_v1.conditioning import capture_initial_input
    from experiments.sd21_tokenverse_isolation_v1 import contract

    experiment = Path(experiment).resolve()
    audit = sct.st.read_json(experiment / 'source_audit.json')
    exports = sct.st.read_json(experiment / 'exports.json')
    revision = sct.st.git_info()
    if revision != audit['new_training_git']:
        raise ValueError('evaluation checkout differs from the audited training checkout')
    output = experiment / 'inference'
    output.mkdir()
    rows, generated, recorded_gates, loaded_sources, initial_inputs = plan(), [], {}, {}, {}
    state = {'status': 'running', 'expected_images': 112, 'generated_images': 0, 'gpu': 0,
             'inference_git': revision, 'sampling': {**sct.st.shared.inference.SAMPLING, 'negative_prompt': ''}}
    (output / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
    sct.write(output / 'status.json', state)
    review(output, generated, state)
    try:
        sct.st.require_idle_gpu0()
        locks = dict(audit['source_locks'])
        model = sct.st.shared.verify_model(contract.MODEL_DIR)['model']
        # Check every selected checkpoint before creating any CUDA model.
        for branch in BRANCHES:
            for arm in ARMS.values():
                for step in ((100, 150) if branch == 'color' else (500, 1000)):
                    item = exports[branch][arm][str(step)]
                    if (item.get('branch') != branch or item.get('method') != 'lora_r64'
                            or item.get('token') != contract.TOKENS[branch] or item.get('step') != step
                            or item.get('steps') != step or item.get('base_model') != model
                            or item.get('embedding_dim') != 1024 or item.get('status') != 'verified'
                            or item.get('paired_embedding_finite') is not True):
                        raise ValueError('export identity, checkpoint or model differs from the fixed plan')
                    sct.st.source._merge(locks, item['hashes'])
        sct.st.shared.snapshots.check_locks(locks)
        sct.st.require_idle_gpu0()
        with torch.no_grad(), (output / 'generation_status.jsonl').open('x', encoding='utf8') as ledger:
            for key in dict.fromkeys((row['phase'], row['mode'], row['condition']) for row in rows):
                selected = [row for row in rows if (row['phase'], row['mode'], row['condition']) == key]
                first = selected[0]
                sources = {branch: exports[branch][arm][str(first['branch_steps'][branch])]
                           for branch, arm in first['branch_arms'].items()}
                combined = first['mode'] == 'sct'
                pipe = sct.st.shared.inference.load_pipeline([] if combined else list(sources.values()),
                    'cuda:0', model_dir=contract.MODEL_DIR, experiment_contract=contract)
                try:
                    if combined:
                        weights = {branch: torch.load(Path(item['checkpoint']) / item['weight'], map_location='cpu')
                                   for branch, item in sources.items()}
                        install_sct_token_local_kv(pipe.unet, weights, 'lora_r64')
                        del weights
                        for item in sources.values():
                            pipe.load_textual_inversion(item['checkpoint'], weight_name=item['token'] + '.bin')
                    pipe.set_progress_bar_config(disable=True)
                    gates = {row['prompt']: attention_masks(pipe, row['prompt'], guidance=3.5) if combined else
                             single_masks(pipe, row['prompt'], [next(iter(sources.values()))['token']], 3.5)
                             for row in selected}
                    label = '/'.join(key)
                    recorded_gates[label] = {prompt: {branch: mask.cpu().tolist() for branch, mask in gate['modifier_token_mask'].items()}
                        if combined else gate['modifier_token_mask'].cpu().tolist() for prompt, gate in gates.items()}
                    loaded_sources[label] = sources
                    sct.write(output / 'token_gates.json', recorded_gates)
                    sct.write(output / 'loaded_sources.json', loaded_sources)
                    for row in selected:
                        state['active'] = row['id']
                        sct.write(output / 'status.json', state)
                        captured = {}
                        hook = capture_initial_input(pipe.unet, captured)
                        try:
                            result = pipe(row['prompt'], negative_prompt=row['negative_prompt'], num_inference_steps=100,
                                guidance_scale=3.5, width=512, height=512,
                                generator=torch.Generator(device='cuda:0').manual_seed(row['seed']),
                                cross_attention_kwargs=gates[row['prompt']])
                        finally:
                            hook.remove()
                        if not captured or captured != initial_inputs.setdefault(row['seed'], captured):
                            raise ValueError('same-seed initial denoiser input changed across conditions')
                        if (result.nsfw_content_detected is None or len(result.nsfw_content_detected) != 1
                                or len(result.images) != 1 or result.images[0].mode != 'RGB' or result.images[0].size != (512, 512)):
                            raise ValueError('require one512px RGB image with safety status')
                        path = output / row['image_path']
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with path.open('xb') as stream:
                            result.images[0].save(stream, format='PNG')
                        record = {**row, 'loaded_branches': list(sources), 'initial_denoiser_input': captured,
                            'image_sha256': sct.sha256(path), 'status': 'safety_filtered' if result.nsfw_content_detected[0] else 'ok'}
                        generated.append(record)
                        ledger.write(json.dumps(record) + '\n')
                        ledger.flush()
                        state['generated_images'] = len(generated)
                        sct.write(output / 'status.json', state)
                finally:
                    del pipe
                    torch.cuda.empty_cache()
                review(output, generated, state)
        sct.st.shared.snapshots.check_locks(locks)
        if len(generated) != 112 or sct.st.git_info() != revision:
            raise ValueError('evaluation incomplete or checkout changed')
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
    parser.add_argument('--experiment', required=True, type=Path)
    run(parser.parse_args().experiment)
