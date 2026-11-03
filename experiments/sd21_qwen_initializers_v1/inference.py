"""Fixed112-image native/literal/learned and four-branch SD2.1 evaluation."""

import argparse
import html
import json
import os
from pathlib import Path

BRANCHES = ('subject', 'color', 'material', 'texture')
TOKENS = dict(zip(BRANCHES, ('<S*>', '<C*>', '<M*>', '<T*>')))
SEEDS = (31, 38)
SCENES = {'plain': '', 'street': 'in the street', 'snow': 'in the snow', 'flowers': 'surrounded by flowers'}
OBJECTS = ('chair', 'mug', 'vase', 'sofa')
STEPS = {'final': {'subject': 1000, 'color': 150, 'material': 1000, 'texture': 1000},
         'mid': {'subject': 500, 'color': 100, 'material': 500, 'texture': 500}}
SAMPLING = {'scheduler': 'PNDM', 'num_inference_steps': 100, 'width': 512, 'height': 512,
            'dtype': 'float16', 'safety_checker': True, 'negative_prompt': '',
            'generator': 'fresh_device_generator_per_image', 'guidance_scale_by_mode': {'color_only': 6.0, 'other': 3.5}}


def _article(phrase):
    return ('an ' if phrase[0].lower() in 'aeiou' else 'a ') + phrase


def plan(manifest):
    entries = manifest['branches']
    category = manifest['subject_category']
    if (set(entries) != set(BRANCHES) or not isinstance(category, str) or not category.strip()
            or category != category.strip() or any(token in category for token in TOKENS.values())):
        raise ValueError('require the validated four-branch manifest and ordinary subject category')
    for branch, item in entries.items():
        word = item['initializer']
        if (item['token'] != TOKENS[branch] or not isinstance(word, str) or not word
                or any(character.isspace() for character in word) or '<' in word or '>' in word):
            raise ValueError('literal initializers must be validated ordinary single words')
    rows = []

    def add(phase, mode, condition, case, seed, prompt, loaded):
        identity = f'{phase}__{mode}__{condition}__{case}__seed{seed}'
        rows.append({'id': identity, 'phase': phase, 'mode': mode, 'condition': condition, 'case': case,
            'method': 'lora_r64' if loaded else 'stock', 'seed': seed, 'prompt': prompt,
            'control_role': ('four_branch_composition' if mode == 'scmt' else 'learned_single_branch' if loaded
                             else 'generic_object_base' if mode == 'subject_only' and condition == 'native'
                             else 'subject_category_base' if mode == 'subject_only'
                             else 'ordinary_attribute_base' if condition == 'literal' else 'ordinary_object_base'),
            'negative_prompt': '', 'guidance_scale': 6.0 if mode == 'color_only' else 3.5,
            'branch_steps': {branch: STEPS[phase][branch] for branch in loaded},
            'target_initializers': {branch: entries[branch]['initializer'] for branch in
                                    (BRANCHES if mode == 'scmt' else (mode.removesuffix('_only'),))},
            'image_path': f'images/{phase}/{mode}/{condition}/{case}/seed{seed}.png'})

    for branch in BRANCHES:
        for condition in ('native', 'literal', 'learned'):
            for case, suffix in (SCENES.items() if branch == 'subject' else ((obj, '') for obj in OBJECTS)):
                if branch == 'subject':
                    subject = ('an object' if condition == 'native' else _article(category)
                               if condition == 'literal' else '<S*> ' + category)
                    prompt = ('a photo of ' + subject + ' ' + suffix).strip()
                else:
                    prompt = 'a photo of ' + _article(case)
                    if condition != 'native':
                        word = TOKENS[branch] if condition == 'learned' else entries[branch]['initializer']
                        prompt += {'color': f' in {word} color',
                                   'material': f' with {"a " + word if condition == "learned" else _article(word)} surface',
                                   'texture': f' with {word} texture'}[branch]
                for seed in SEEDS:
                    add('final', branch + '_only', condition, case, seed, prompt,
                        (branch,) if condition == 'learned' else ())
    for phase in ('final', 'mid'):
        body = f'a photo of <S*> {category} in <C*> color with a <M*> surface with <T*> texture'
        for scene, suffix in SCENES.items():
            for seed in SEEDS:
                add(phase, 'scmt', 'learned', scene, seed, (body + ' ' + suffix).strip(), BRANCHES)
    return rows


def generation_kwargs(row, device='cuda:0'):
    import torch
    return {'negative_prompt': '', 'num_inference_steps': 100, 'guidance_scale': row['guidance_scale'],
            'width': 512, 'height': 512, 'generator': torch.Generator(device=device).manual_seed(row['seed'])}


def register_initial_input(initial_inputs, seed, captured):
    if not captured or captured != initial_inputs.setdefault(seed, dict(captured)):
        raise ValueError('same-seed initial denoiser input changed across conditions')


def prompt_conditioning(pipe, row):
    from experiments.sd15_token_local_kv_v1.evaluate import attention_masks as single_masks
    from experiments.sd21_qwen_initializers_v1.attention import attention_masks
    tokenizer, prompt = pipe.tokenizer, row['prompt']
    if (tokenizer.model_max_length != 77 or pipe.maybe_convert_prompt(prompt, tokenizer) != prompt
            or len(tokenizer.encode(prompt, add_special_tokens=True)) > 77):
        raise ValueError('inference refuses expanded or truncated CLIP77 prompts')
    loaded = row['branch_steps']
    if any(prompt.count(token) != int(branch in loaded) for branch, token in TOKENS.items()):
        raise ValueError('prompt learned tokens differ from loaded branches')
    ids = tokenizer(prompt, padding='max_length', max_length=77, truncation=False, return_tensors='pt').input_ids
    if not loaded:
        gate, recorded = {}, {}
    elif row['mode'] == 'scmt':
        gate = attention_masks(pipe, prompt)
        recorded = {branch: value.cpu().tolist() for branch, value in gate['modifier_token_mask'].items()}
    else:
        branch = next(iter(loaded))
        gate = single_masks(pipe, prompt, [TOKENS[branch]], row['guidance_scale'])
        mask = gate['modifier_token_mask']
        if tuple(mask.shape) != (2, 77) or bool(mask[0].any()) or int(mask[1].sum()) != 1:
            raise ValueError('single-branch CFG mask differs from the prompt')
        recorded = {branch: mask.cpu().tolist()}
    return gate, {'input_ids': ids.tolist(), 'gates': recorded, 'prompt': prompt}


def review(output, rows, generated, state):
    records, tables = {row['id']: row for row in generated}, []
    for mode in (*(branch + '_only' for branch in BRANCHES), 'scmt'):
        group = [row for row in rows if row['mode'] == mode]
        columns = list(dict.fromkeys((row['phase'], row['condition']) for row in group))
        tables.append(f'<h2>{mode}</h2><table><tr><th>case / seed</th>' + ''.join(
            f'<th>{phase} / {condition}</th>' for phase, condition in columns) + '</tr>')
        for case, seed in dict.fromkeys((row['case'], row['seed']) for row in group):
            tables.append(f'<tr><th>{case} / {seed}</th>')
            for phase, condition in columns:
                row = records.get(f'{phase}__{mode}__{condition}__{case}__seed{seed}')
                tables.append('<td>planned</td>' if row is None else
                    f'<td><a href="{row["image_path"]}"><img width="256" src="{row["image_path"]}"></a>'
                    f'<p>{row["status"]}; CFG{row["guidance_scale"]}</p><small>{html.escape(row["prompt"])}</small></td>')
            tables.append('</tr>')
        tables.append('</table>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<style>body{font:16px sans-serif}td{vertical-align:top;width:256px}</style>'
        f'<h1>Qwen initializers: {state["status"]}, {state["generated_images"]}/112</h1>'
        '<p>Native/literal controls use stock base weights and no learned embeddings. Subject native is a generic '
        'object; Subject literal is the validated category. Other literal controls use the selected ordinary '
        'initializer. Learned controls load only the named branch; SCMT loads all four paired branches.</p>'
        '<p>PNDM100 / FP16 / 512px / empty negative / checker enabled. Color-only CFG6; all others CFG3.5. '
        'Material is visible surface finish; Texture is spatial pattern. No automatic quality judgment.</p>'
        + ''.join(tables), encoding='utf8')


def run(experiment):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import torch
    from transformers import CLIPTokenizer
    from experiments.sd21_mailbox_sct_v1 import run as sct
    from experiments.sd21_wood_branches_v1 import inference as single, snapshots
    from experiments.sd21_qwen_initializers_v1 import contract
    from experiments.sd21_qwen_initializers_v1.attention import install_four_branch_lora
    from experiments.sd21_prompt_order_v1.conditioning import capture_initial_input, tensor_sha256
    from src.train.initializer_token_utils import load_initializer_manifest

    experiment = Path(experiment).resolve()
    audit_path, exports_path = experiment / 'source_audit.json', experiment / 'exports.json'
    audit, exports = sct.st.read_json(audit_path), sct.st.read_json(exports_path)
    revision = sct.st.git_info()
    if revision != audit['new_training_git']:
        raise ValueError('evaluation checkout differs from the audited training checkout')
    output = experiment / 'inference'
    output.mkdir()
    rows, generated, recorded_gates, loaded_sources, initial_inputs = [], [], {}, {}, {}
    state = {'status': 'running', 'expected_images': 112, 'generated_images': 0, 'gpu': 0,
             'inference_git': revision, 'sampling': SAMPLING, 'quality_verified': False}
    sct.write(output / 'status.json', state)
    try:
        sct.st.require_idle_gpu0()
        locks = dict(audit['source_locks'])
        sct.st.source._merge(locks, {str(path): sct.sha256(path) for path in (audit_path, exports_path)})
        model_receipt = sct.st.shared.verify_model(contract.MODEL_DIR)
        model = model_receipt['model']
        sct.st.source._merge(locks, snapshots.model_locks(contract.MODEL_DIR, model_receipt))
        tokenizer = CLIPTokenizer.from_pretrained(str(contract.MODEL_DIR), subfolder='tokenizer', local_files_only=True)
        manifest_path = Path(audit['initializer_manifest'])
        if not manifest_path.is_absolute():
            raise ValueError('require an absolute initializer manifest path')
        manifest = load_initializer_manifest(manifest_path, tokenizer, contract.MODEL_DIR)
        if manifest != audit['initializers']:
            raise ValueError('frozen initializer manifest differs from source audit')
        sct.st.source._merge(locks, {**manifest['source_locks'], str(manifest_path): sct.sha256(manifest_path)})
        rows = plan(manifest)
        (output / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
        review(output, rows, generated, state)
        tokenizer.add_tokens(list(TOKENS.values()))
        if any(len(tokenizer.encode(row['prompt'], add_special_tokens=True)) > 77 for row in rows):
            raise ValueError('planned prompt exceeds77 CLIP positions')
        shapes, empty = snapshots.adapter_schema('lora_r64')
        if set(exports) != set(BRANCHES):
            raise ValueError('require exactly four exported branches')
        for branch in BRANCHES:
            if set(exports[branch]) != {str(STEPS[phase][branch]) for phase in STEPS}:
                raise ValueError('export checkpoints differ from final/mid plan')
            for phase in STEPS:
                step, token = STEPS[phase][branch], TOKENS[branch]
                item = exports[branch][str(step)]
                if (item.get('branch') != branch or item.get('method') != 'lora_r64' or item.get('token') != token
                        or item.get('step') != step or item.get('steps') != step or item.get('base_model') != model
                        or item.get('embedding_dim') != 1024 or item.get('status') != 'verified'
                        or item.get('weight') != 'pytorch_lora_kv_weights.bin'
                        or item.get('paired_embedding_finite') is not True):
                    raise ValueError('export branch, checkpoint, base or pairing differs')
                provenance = item['initializer_provenance']
                expected = {'initializer_manifest': str(manifest_path),
                    'initializer_manifest_sha256': sct.sha256(manifest_path),
                    'initializer': manifest['branches'][branch]['initializer'],
                    'initializer_token_id': manifest['branches'][branch]['initializer_token_id']}
                if any(provenance.get(key) != value for key, value in expected.items()):
                    raise ValueError('export initializer provenance differs from frozen manifest')
                audit_file = provenance['initializer_audit']
                if (not Path(audit_file).is_absolute()
                        or audit['source_locks'].get(audit_file) != provenance['initializer_audit_sha256']):
                    raise ValueError('initializer copy audit must be source-locked')
                sct.st.source._merge(locks, item['hashes'])
                directory = Path(item['checkpoint'])
                weights = torch.load(directory / item['weight'], map_location='cpu')
                snapshots.shared._check_state(torch, weights, shapes, empty, f'{branch}/{step}')
                if any(value.dtype != torch.float32 or not bool(torch.isfinite(value.half()).all())
                       for value in weights.values()):
                    raise ValueError('require FP32 adapter exports finite after inference conversion')
                embedding = torch.load(directory / (token + '.bin'), map_location='cpu')
                if (not isinstance(embedding, dict) or set(embedding) != {token}
                        or not isinstance(embedding[token], torch.Tensor) or embedding[token].shape != (1024,)
                        or embedding[token].dtype != torch.float32 or not bool(torch.isfinite(embedding[token]).all())
                        or not bool(torch.isfinite(embedding[token].half()).all())):
                    raise ValueError('paired learned embedding is not a finite1024-vector')
                del weights, embedding
        snapshots.check_locks(locks)
        sct.st.require_idle_gpu0()
        with torch.no_grad(), (output / 'generation_status.jsonl').open('x', encoding='utf8') as ledger:
            for key in dict.fromkeys((row['phase'], row['mode'], row['condition']) for row in rows):
                selected = [row for row in rows if (row['phase'], row['mode'], row['condition']) == key]
                first = selected[0]
                sources = {branch: exports[branch][str(step)] for branch, step in first['branch_steps'].items()}
                combined = first['mode'] == 'scmt'
                pipe = single.load_pipeline([] if combined else list(sources.values()), 'cuda:0',
                    model_dir=contract.MODEL_DIR, stock=not sources, experiment_contract=contract)
                try:
                    if combined:
                        weights = {branch: torch.load(Path(item['checkpoint']) / item['weight'], map_location='cpu')
                                   for branch, item in sources.items()}
                        install_four_branch_lora(pipe.unet, weights)
                        del weights
                        for item in sources.values():
                            pipe.load_textual_inversion(item['checkpoint'], weight_name=item['token'] + '.bin')
                    loaded_embeddings = {}
                    if {branch for branch, token in TOKENS.items() if token in pipe.tokenizer.get_vocab()} != set(sources):
                        raise ValueError('pipeline token inventory differs from exactly the loaded branches')
                    for branch, item in sources.items():
                        vector = torch.load(Path(item['checkpoint']) / (item['token'] + '.bin'), map_location='cpu')[item['token']]
                        actual = pipe.text_encoder.get_input_embeddings().weight[pipe.tokenizer.convert_tokens_to_ids(item['token'])]
                        if not torch.equal(actual, vector.to(device=actual.device, dtype=actual.dtype)):
                            raise ValueError('loaded learned vector differs from selected paired export')
                        loaded_embeddings[branch] = {'token_id': pipe.tokenizer.convert_tokens_to_ids(item['token']),
                            'loaded_fp16_sha256': tensor_sha256(actual), 'source_fp32_sha256': tensor_sha256(vector)}
                        del actual, vector
                    pipe.set_progress_bar_config(disable=True)
                    label, gates = '/'.join(key), {}
                    recorded_gates[label] = {}
                    for row in selected:
                        gates[row['prompt']], recorded_gates[label][row['prompt']] = prompt_conditioning(pipe, row)
                    loaded_sources[label] = {'sources': sources, 'loaded_embeddings': loaded_embeddings,
                                             'stock_base_only': not sources}
                    sct.write(output / 'token_gates.json', recorded_gates)
                    sct.write(output / 'loaded_sources.json', loaded_sources)
                    for row in selected:
                        state['active'] = row['id']
                        sct.write(output / 'status.json', state)
                        captured = {}
                        hook = capture_initial_input(pipe.unet, captured)
                        try:
                            kwargs = {'cross_attention_kwargs': gates[row['prompt']]} if sources else {}
                            result = pipe(row['prompt'], **generation_kwargs(row), **kwargs)
                        finally:
                            hook.remove()
                        register_initial_input(initial_inputs, row['seed'], captured)
                        if (result.nsfw_content_detected is None or len(result.nsfw_content_detected) != 1
                                or len(result.images) != 1 or result.images[0].mode != 'RGB' or result.images[0].size != (512, 512)):
                            raise ValueError('require one512px RGB image with safety status')
                        path = output / row['image_path']
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with path.open('xb') as stream:
                            result.images[0].save(stream, format='PNG')
                        record = {**row, 'loaded_branches': list(sources), 'loaded_source_group': label,
                            'initial_denoiser_input': captured, 'image_sha256': sct.sha256(path),
                            'status': 'safety_filtered' if result.nsfw_content_detected[0] else 'ok'}
                        generated.append(record)
                        ledger.write(json.dumps(record) + '\n')
                        ledger.flush()
                        state['generated_images'] = len(generated)
                        sct.write(output / 'status.json', state)
                finally:
                    del pipe
                    torch.cuda.empty_cache()
                review(output, rows, generated, state)
        snapshots.check_locks(locks)
        if len(generated) != 112 or sct.st.git_info() != revision:
            raise ValueError('evaluation incomplete or checkout changed')
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
    parser.add_argument('--experiment', required=True, type=Path)
    run(parser.parse_args().experiment)
