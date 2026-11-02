"""SD2.1 Mailbox S1000 + orange C150 + wood T1000, matching methods on GPU0."""

import argparse
import html
import json
import os
from pathlib import Path
import shutil

from experiments.sd21_mailbox_st_v1 import run as st
from experiments.sd21_color_rendered9_150_v1 import contract as color_contract

ROOT = Path(__file__).resolve().parents[2]
COLOR_ROOT = color_contract.PARENT / 'supplemental150/20261101-193942__both150_save100_150__5f3389a__42/experiment'
COLOR_GIT = {'branch': 'codex/sd21-color-rendered9-150-v1', 'commit': '5f3389aad843ae0eb631cb2422941cef30abb9ff'}
TOKENS = {'subject': '<S*>', 'color': '<C*>', 'texture': '<T*>'}
STEPS = {'subject': 1000, 'color': 150, 'texture': 1000}
PROMPT = 'a photo of <S*> mailbox in <C*> color with <T*> texture'
METHODS, SEEDS = st.METHODS, st.SEEDS
SCENES = {'street': 'in the street', 'snow': 'in the snow', 'flowers': 'surrounded by flowers'}
TRANSFER_SEEDS = tuple(range(30, 61))
write, sha256 = st.write, st.sha256


def prompt_rows(method, scene_transfer=False):
    if method not in METHODS:
        raise ValueError('require fullrank or lora_r64, with matching three branches')
    if scene_transfer:
        return [{'id': f'{method}__S1000_C150_T1000__{scene}__seed{seed}', 'scene': scene,
                 'method': method, 'seed': seed, 'prompt': PROMPT + ' ' + suffix, 'steps': STEPS,
                 'image_path': f'images/{scene}/seed{seed}.png'}
                for scene, suffix in SCENES.items() for seed in TRANSFER_SEEDS]
    return [{'id': f'{method}__mailbox_S1000_orange_C150_wood_T1000__seed{seed}',
             'method': method, 'seed': seed, 'prompt': PROMPT, 'steps': STEPS,
             'image_path': f'images/seed{seed}.png'} for seed in SEEDS]


def prepare():
    audit = st.prepare(1000, 1000)
    os.environ[color_contract.DATASET_ENV] = str(COLOR_ROOT / 'dataset')
    data = color_contract.verify_dataset('color')
    if data['rows'] != 9:
        raise ValueError('require original orange rendered9 training data')
    audit['references']['color'] = data['references']
    st.source._merge(audit['source_locks'], data['source_locks'])
    audit['sources']['color'] = {}
    for method in METHODS:
        item, locks = st.source._pair(COLOR_ROOT, 'color', method, 150, COLOR_GIT,
            audit['sources']['subject'][method]['base_model'], recipe_contract=color_contract)
        audit['sources']['color'][method] = item
        st.source._merge(audit['source_locks'], locks)
    audit.update(steps=STEPS, prompt=PROMPT, color_source='mailbox_original_orange_rendered9',
                 training_images={'subject': 11, 'color': 9, 'texture': 16})
    audit['training_gits']['color'] = COLOR_GIT
    if st.git_info() != audit['inference_git']:
        raise ValueError('checkout changed during three-branch audit')
    return audit


def references(audit, output):
    locks, cards = {}, []
    for branch, token in TOKENS.items():
        rows = audit['references'][branch]
        if len(rows) != audit['training_images'][branch]:
            raise ValueError('actual training reference count differs')
        cards.append(f'<h2>{branch}: {len(rows)} training images</h2>')
        for number, row in enumerate(rows):
            path = Path(row['image'])
            digest = audit['source_locks'].get(str(path))
            if not digest or sha256(path) != digest or row['caption'].count(token) != 1:
                raise ValueError('training reference hash or caption differs')
            target = output / 'training_references' / branch / f'{number:03d}.png'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            locks[str(target)] = digest
            cards.append(f'<figure><img width="240" src="{target.relative_to(output).as_posix()}">'
                         f'<figcaption>{html.escape(row["caption"])}</figcaption></figure>')
    (output / 'training_references.html').write_text('<!doctype html><meta charset="utf-8">'
        '<style>figure{display:inline-block;width:240px;vertical-align:top}</style>' + ''.join(cards), encoding='utf8')
    st.shared.snapshots.check_locks(locks)
    return locks


def generate(audit, output, method, scene_transfer=False):
    import torch
    from experiments.sd21_mailbox_sct_v1.attention import install_sct_token_local_kv, attention_masks

    output.mkdir(parents=True)
    copies = references(audit, output)
    rows, generated = prompt_rows(method, scene_transfer), []
    prompts = list(dict.fromkeys(row['prompt'] for row in rows))
    seeds = sorted({row['seed'] for row in rows})
    state = {'status': 'running', 'method': method, 'steps': STEPS, 'prompt': PROMPT,
        'prompts': prompts, 'scene_transfer': scene_transfer,
        'expected_images': len(rows), 'generated_images': 0, 'sampling': {**st.shared.inference.SAMPLING, 'seeds': seeds},
        'training_gits': audit['training_gits'], 'inference_git': audit['inference_git'],
        'adapter_slots': TOKENS, 'adapter_scales': {branch: 1.0 for branch in TOKENS}, 'retraining': False}
    write(output / 'status.json', state)
    (output / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
    try:
        pipe = st.shared.inference.load_pipeline([], 'cuda:0')
        selections = {branch: audit['sources'][branch][method] for branch in TOKENS}
        states = {branch: torch.load(Path(item['checkpoint']) / item['weight'], map_location='cpu')
                  for branch, item in selections.items()}
        install_sct_token_local_kv(pipe.unet, states, method)
        del states
        for item in selections.values():
            pipe.load_textual_inversion(item['checkpoint'], weight_name=item['token'] + '.bin')
        pipe.set_progress_bar_config(disable=True)
        gates = {prompt: attention_masks(pipe, prompt, guidance=3.5) for prompt in prompts}
        recorded_gates = {prompt: {branch: mask.cpu().tolist()
                          for branch, mask in gate['modifier_token_mask'].items()} for prompt, gate in gates.items()}
        write(output / 'token_gates.json', recorded_gates if scene_transfer else recorded_gates[PROMPT])
        with (output / 'generation_status.jsonl').open('x', encoding='utf8') as ledger, torch.no_grad():
            for row in rows:
                result = pipe(row['prompt'], num_inference_steps=100, guidance_scale=3.5, width=512, height=512,
                    generator=torch.Generator(device='cuda:0').manual_seed(row['seed']), cross_attention_kwargs=gates[row['prompt']])
                if (result.nsfw_content_detected is None or len(result.nsfw_content_detected) != 1
                        or len(result.images) != 1 or result.images[0].size != (512, 512) or result.images[0].mode != 'RGB'):
                    raise ValueError('require one512px RGB image with safety status')
                path = output / row['image_path']
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    result.images[0].save(stream, format='PNG')
                generated.append({**row, 'status': 'safety_filtered' if result.nsfw_content_detected[0] else 'ok',
                    'image_sha256': sha256(path), 'loaded_branches': list(TOKENS)})
                ledger.write(json.dumps(generated[-1]) + '\n')
                ledger.flush()
                state['generated_images'] = len(generated)
                write(output / 'status.json', state)
        del pipe
        torch.cuda.empty_cache()
        st.shared.snapshots.check_locks(copies)
        state['status'] = 'succeeded'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        state['reference_copy_hashes'] = copies
        write(output / 'status.json', state)
        (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
            + f'<h1>{method}: S1000 + C150 + T1000</h1>'
            + ''.join(f'<p>{html.escape(prompt)}</p>' for prompt in prompts)
            + '<a href="training_references.html">All actual training images/captions</a>'
            + ''.join(f'<figure style="display:inline-block"><img width="256" src="{row["image_path"]}">'
                f'<figcaption>{row.get("scene", "base")} seed{row["seed"]}: {row["status"]}</figcaption></figure>' for row in generated), encoding='utf8')
    return state


def run(output, scene_transfer=False):
    output = st.resolve_run_dir(Path(output))
    if output.exists() or any(output.is_relative_to(root) for root in (ROOT, st.SUBJECT_PARENT, st.source.TEXTURE_ROOT, COLOR_ROOT)):
        raise ValueError('use fresh external output separate from all source experiments')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    st.require_idle_gpu0()
    audit = prepare()
    planned = prompt_rows(METHODS[0], scene_transfer)
    audit['generation_prompts'] = list(dict.fromkeys(row['prompt'] for row in planned))
    st.require_idle_gpu0()
    output.mkdir(parents=True)
    write(output / 'source_audit.json', audit)
    state = {'status': 'running', 'git': audit['inference_git'], 'steps': STEPS,
             'expected_images': len(planned) * len(METHODS), 'generated_images': 0, 'gpu': 0,
             'scene_transfer': scene_transfer, 'methods': {}, 'retraining': False}
    write(output / 'status.json', state)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>SD2.1 Mailbox S1000 + orange C150 + wood T1000</h1>'
        + f'<p>{html.escape(PROMPT)}; PNDM100/CFG3.5/512px; seeds30–{60 if scene_transfer else 40}</p>'
        + ('<p>Scenes: in the street / in the snow / surrounded by flowers</p>' if scene_transfer else '')
        + ''.join(f'<p><a href="methods/{m}/index.html">{m}: results and training references</a></p>' for m in METHODS)
        + '<table><tr><th>Scene / seed</th><th>Fullrank</th><th>LoRA rank64</th></tr>'
        + ''.join(f'<tr><th>{row.get("scene", "base")} / {row["seed"]}</th>'
          + ''.join(f'<td><img loading="lazy" width="384" src="methods/{m}/{row["image_path"]}"></td>'
          for m in METHODS) + '</tr>' for row in planned) + '</table>', encoding='utf8')
    try:
        for method in METHODS:
            state['methods'][method] = generate(audit, output / 'methods' / method, method, scene_transfer)
            state['generated_images'] += state['methods'][method]['generated_images']
            write(output / 'status.json', state)
        st.shared.snapshots.check_locks(audit['source_locks'])
        if st.git_info() != audit['inference_git']:
            raise ValueError('inference checkout changed')
        state['status'] = 'succeeded'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(output / 'status.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scene-transfer', action='store_true')
    args = parser.parse_args()
    run(args.output, args.scene_transfer)
