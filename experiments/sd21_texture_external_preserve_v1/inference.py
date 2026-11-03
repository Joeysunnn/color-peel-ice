"""Exactly56 images: four SCT scenes and retained Texture-only transfer controls."""

import argparse
import html
import json
import os
from pathlib import Path

from experiments.sd21_mailbox_sct_v1 import run as sct
from experiments.sd21_texture_external_preserve_v1 import contract

SCENES = {**sct.SCENES, "beach": "at the beach"}


def plan():
    cases = [('sct', scene, sct.PROMPT + ' ' + suffix, seed) for scene, suffix in SCENES.items() for seed in contract.SEEDS]
    cases += [('texture_only', obj, f'a photo of a {obj} with <T*> texture', seed) for obj in ('mug', 'vase') for seed in contract.SEEDS]
    cases += [('texture_only', obj, f'a photo of a {obj} with <T*> texture', seed) for obj, seed in (('chair', 38), ('sofa', 31))]
    return [{'id': f'{arm}__t{step}__{mode}__{scene}__seed{seed}', 'arm': arm, 'step': step,
             'method': 'lora_r64', 'mode': mode, 'scene': scene, 'seed': seed, 'prompt': prompt,
             'image_path': f'images/{arm}/step{step}/{mode}/{scene}/seed{seed}.png'}
            for arm in contract.ARMS for step in contract.STEPS
            for mode, scene, prompt, seed in cases]


def review(output, generated, state):
    cards = []
    for mode in ('sct', 'texture_only'):
        cards.append(f'<h2>{mode}</h2>')
        for scene in (SCENES if mode == 'sct' else ('mug', 'vase', 'chair', 'sofa')):
            cards.append(f'<h3>{scene}</h3><table><tr><th>seed</th>' + ''.join(
                f'<th>{arm} T{step}</th>' for arm in contract.ARMS for step in contract.STEPS) + '</tr>')
            for seed in contract.SEEDS:
                if mode == 'texture_only' and (scene, seed) in (('chair', 31), ('sofa', 38)):
                    continue
                cards.append(f'<tr><th>{seed}</th>')
                for arm in contract.ARMS:
                    for step in contract.STEPS:
                        row = next((row for row in generated if (row['arm'], row['step'], row['mode'], row['scene'], row['seed'])
                                    == (arm, step, mode, scene, seed)), None)
                        cards.append('<td>planned</td>' if row is None else f'<td><a href="{row["image_path"]}">'
                            f'<img width="256" src="{row["image_path"]}"></a><p>{row["status"]}</p>'
                            f'<small>{html.escape(row["prompt"])}</small></td>')
                cards.append('</tr>')
            cards.append('</table>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px sans-serif}td{vertical-align:top;width:256px}</style>'
        f'<h1>External preservation images: {state["status"]}, {state["generated_images"]}/56</h1>'
        '<p>LoRA only; S1000/C150 fixed for SCT. PNDM100 / CFG3.5 / 512px, safety checker enabled.</p>' + ''.join(cards), encoding='utf8')


def run(experiment):
    import torch
    from experiments.sd21_mailbox_sct_v1.attention import install_sct_token_local_kv, attention_masks
    from experiments.sd15_token_local_kv_v1.evaluate import attention_masks as single_masks
    from experiments.sd21_prompt_order_v1.conditioning import capture_initial_input

    experiment = Path(experiment).resolve()
    audit = sct.st.read_json(experiment / 'source_audit.json')
    exports = sct.st.read_json(experiment / 'exports.json')
    output = experiment / 'inference'
    output.mkdir()
    rows, generated, recorded_gates = plan(), [], {}
    initial_inputs = {}
    state = {'status': 'running', 'expected_images': 56, 'generated_images': 0,
             'inference_git': sct.st.git_info(), 'sampling': sct.st.shared.inference.SAMPLING}
    if state['inference_git'] != audit['new_training_git']:
        raise ValueError('evaluation must use the audited new checkout')
    (output / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
    sct.write(output / 'status.json', state)
    review(output, generated, state)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sct.st.require_idle_gpu0()
    sct.st.shared.snapshots.check_locks(audit['source_locks'])
    sct.st.require_idle_gpu0()
    try:
        with torch.no_grad(), (output / 'generation_status.jsonl').open('x', encoding='utf8') as ledger:
            for arm in contract.ARMS:
                for step in contract.STEPS:
                    texture = exports[arm][str(step)]
                    for mode in ('sct', 'texture_only'):
                        sources = {'subject': audit['sources']['subject']['lora_r64'],
                                   'color': audit['sources']['color']['lora_r64'], 'texture': texture} if mode == 'sct' else {'texture': texture}
                        pipe = sct.st.shared.inference.load_pipeline([] if mode == 'sct' else [texture], 'cuda:0',
                            experiment_contract=contract.recipe(arm, audit['derived_resume']['checkpoint'], audit['preservation_assets']['manifest_path']))
                        if mode == 'sct':
                            states = {branch: torch.load(Path(item['checkpoint']) / item['weight'], map_location='cpu')
                                      for branch, item in sources.items()}
                            install_sct_token_local_kv(pipe.unet, states, 'lora_r64')
                            del states
                            for item in sources.values():
                                pipe.load_textual_inversion(item['checkpoint'], weight_name=item['token'] + '.bin')
                        pipe.set_progress_bar_config(disable=True)
                        selected = [row for row in rows if (row['arm'], row['step'], row['mode']) == (arm, step, mode)]
                        gates = {row['prompt']: attention_masks(pipe, row['prompt'], guidance=3.5) if mode == 'sct'
                                 else single_masks(pipe, row['prompt'], ['<T*>'], 3.5) for row in selected}
                        recorded_gates[f'{arm}/step{step}/{mode}'] = {prompt: {branch: mask.cpu().tolist() for branch, mask in gate['modifier_token_mask'].items()}
                            if mode == 'sct' else gate['modifier_token_mask'].cpu().tolist() for prompt, gate in gates.items()}
                        sct.write(output / 'token_gates.json', recorded_gates)
                        for row in selected:
                            state['active'] = row['id']
                            sct.write(output / 'status.json', state)
                            captured = {}
                            hook = capture_initial_input(pipe.unet, captured)
                            try:
                                result = pipe(row['prompt'], num_inference_steps=100, guidance_scale=3.5, width=512, height=512,
                                    generator=torch.Generator(device='cuda:0').manual_seed(row['seed']), cross_attention_kwargs=gates[row['prompt']])
                            finally:
                                hook.remove()
                            if not captured or captured != initial_inputs.setdefault(row['seed'], captured):
                                raise ValueError('same-seed initial denoiser input changed across paired conditions')
                            if (result.nsfw_content_detected is None or len(result.nsfw_content_detected) != 1
                                    or len(result.images) != 1 or result.images[0].mode != 'RGB' or result.images[0].size != (512, 512)):
                                raise ValueError('require one512px RGB image with safety status')
                            path = output / row['image_path']
                            path.parent.mkdir(parents=True, exist_ok=True)
                            with path.open('xb') as stream:
                                result.images[0].save(stream, format='PNG')
                            record = {**row, 'initial_denoiser_input': captured, 'image_sha256': sct.sha256(path), 'loaded_branches': list(sources),
                                'status': 'safety_filtered' if result.nsfw_content_detected[0] else 'ok'}
                            generated.append(record)
                            ledger.write(json.dumps(record) + '\n')
                            ledger.flush()
                            state['generated_images'] = len(generated)
                            sct.write(output / 'status.json', state)
                        del pipe
                        torch.cuda.empty_cache()
                        review(output, generated, state)
        sct.st.shared.snapshots.check_locks(audit['source_locks'])
        if len(generated) != 56 or sct.st.git_info() != state['inference_git']:
            raise ValueError('evaluation incomplete or checkout changed')
        state['status'] = 'succeeded'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        sct.write(output / 'status.json', state)
        review(output, generated, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, type=Path)
    run(parser.parse_args().experiment)
