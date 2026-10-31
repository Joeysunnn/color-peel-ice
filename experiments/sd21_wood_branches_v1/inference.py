"""SD2.1-bound independent wood Subject/Material/Texture probes and stock spoon controls."""

import argparse
import html
import json
from pathlib import Path
import shutil
import sys
import textwrap

from PIL import Image, ImageDraw

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd15_token_local_kv_v1.evaluate import attention_masks
from experiments.sd_native_comparison_v1.generate import write_json
from experiments.sd21_wood_branches_v1 import contract, snapshots
from experiments.subject_trellis_spoons_v1.inference import blue_green_prompt_rows
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = contract.MODEL_DIR
TRANSFER_PROTOCOL = ROOT / 'experiments/natural_texture_material_v1/transfer_protocol.json'
SEEDS = tuple(range(30, 61))
SAMPLING = {'scheduler': 'PNDM', 'num_inference_steps': 100, 'guidance_scale': 3.5,
            'width': 512, 'height': 512, 'dtype': 'float16', 'safety_checker': True,
            'generator': 'fresh_device_generator_per_image'}


def _seeds(seeds):
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError('select unique nonnegative integer generation seeds')
    return seeds


def prompt_rows(branch, method, step, seeds=SEEDS, smoke=False):
    seeds = _seeds(seeds)
    if (branch not in contract.BRANCHES or method not in contract.METHODS or type(smoke) is not bool
            or type(step) is not int or step not in ((2,) if smoke else contract.STEPS)):
        raise ValueError('select a declared SD2.1 branch, method, saved step and unique nonnegative seeds')
    seeds = (42,) if smoke else seeds
    rows = []
    if branch == 'subject':
        for template in blue_green_prompt_rows(500):
            if template['method'] != method or template['seed'] != 42:
                continue
            for seed in seeds:
                label = template.get('material', template.get('condition'))
                identity = template['id'].rsplit('__seed', 1)[0] + f'__seed{seed}'
                rows.append({'id': identity, 'branch': branch, 'method': method, 'step': step, 'seed': seed,
                    'condition': 'learned_subject', 'label': label, 'prompt': template['prompt'],
                    'image_path': f'images/{identity}.png'})
    else:
        protocol = read_json(TRANSFER_PROTOCOL)
        for obj in protocol['objects']:
            for seed in seeds:
                for condition in ('base', 'literal', 'learned_' + branch):
                    if condition == 'base':
                        prompt = protocol['base_template'].format(object=obj)
                    else:
                        attribute = contract.TOKENS[branch] if condition.startswith('learned_') else protocol['literal_' + branch]['wood_spoon']
                        prompt = protocol[branch + '_template'].format(object=obj, attribute=attribute)
                    identity = f'{method}__wood_spoon_{branch}__{obj}__{condition}__seed{seed}'
                    rows.append({'id': identity, 'branch': branch, 'method': method, 'step': step, 'seed': seed,
                        'condition': condition, 'label': obj, 'prompt': prompt, 'image_path': f'images/{identity}.png'})
    return [next(row for row in rows if row['condition'].startswith('learned_'))] if smoke else rows


def literal_rows(seeds=SEEDS, smoke=False):
    seeds = _seeds(seeds)
    return [{'id': f'cfg{str(cfg).replace(".", "p")}__seed{seed}', 'cfg': cfg, 'seed': seed,
        'prompt': 'a photo of a spoon', 'condition': 'stock', 'label': f'CFG{cfg:g}',
        'image_path': f'images/cfg{str(cfg).replace(".", "p")}__seed{seed}.png'}
        for cfg in ((3.5,) if smoke else (3.5, 5.0, 7.5)) for seed in ((42,) if smoke else seeds)]


def load_pipeline(selections, device, *, model_dir=MODEL_DIR, stock=False):
    import torch
    from diffusers import PNDMScheduler, StableDiffusionPipeline
    from experiments.sd21_wood_branches_v1.prepare_model import verify_model

    model_dir = Path(model_dir).resolve()
    receipt = verify_model(model_dir)
    if len(selections) > 1 or (stock and selections):
        raise ValueError('SD2.1 first-stage probes load one independent branch or a stock base')
    for selection in selections:
        if (selection.get('base_model') != receipt['model'] or selection.get('embedding_dim') != snapshots.WIDTH
                or selection.get('status') != 'verified' or selection.get('paired_embedding_finite') is not True
                or selection.get('method') not in contract.METHODS
                or selection.get('token') != contract.TOKENS.get(selection.get('branch'))):
            raise ValueError('adapter/token selection belongs to another base or branch')
        snapshots.check_locks(selection.get('hashes', {}))
    scheduler = PNDMScheduler.from_pretrained(str(model_dir), subfolder='scheduler', local_files_only=True)
    kwargs = {'scheduler': scheduler, 'torch_dtype': torch.float16,
              'local_files_only': True, 'low_cpu_mem_usage': False}
    if not stock:
        sys.path.insert(0, str(ROOT / 'src/train'))
        from custom_attention.unet_2d_condition_custom import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(str(model_dir), subfolder='unet',
                                                  local_files_only=True, torch_dtype=torch.float16)
        if selections:
            item = selections[0]
            if item['method'] == 'fullrank':
                unet.load_attn_procs(item['checkpoint'], weight_name=item['weight'], adaptation_mode='token_local_kv')
            else:
                from experiments.lora_kv_subject_v1.attention import install_subject_lora_kv
                state = torch.load(Path(item['checkpoint']) / item['weight'], map_location='cpu')
                install_subject_lora_kv(unet, state, 'token_local_kv', 64, 64)
        kwargs['unet'] = unet
    pipe = StableDiffusionPipeline.from_pretrained(str(model_dir), **kwargs).to(device)
    if (pipe.unet.config.cross_attention_dim != snapshots.WIDTH or not pipe.unet.config.use_linear_projection
            or pipe.text_encoder.get_input_embeddings().weight.shape[1] != snapshots.WIDTH
            or pipe.safety_checker is None or not isinstance(pipe.scheduler, PNDMScheduler)
            or any(token in pipe.tokenizer.get_vocab() for token in ('<S*>', '<C*>', '<M*>', '<T*>'))):
        raise ValueError('expected the unmodified1024-dimensional SD2.1-base512 model, PNDM and checker')
    if stock and (not type(pipe).__module__.startswith('diffusers.')
            or not type(pipe.unet).__module__.startswith('diffusers.')
            or any(not type(proc).__module__.startswith('diffusers.') or 'lora' in type(proc).__name__.lower()
                   for proc in pipe.unet.attn_processors.values())):
        raise ValueError('literal control requires stock Diffusers pipeline/UNet/attention')
    for item in selections:
        pipe.load_textual_inversion(item['checkpoint'], weight_name=item['token'] + '.bin')
    return pipe


def reference_review(output, references):
    from experiments.natural_texture_material_v1.probe_review import _font
    font = _font()
    copied, hashes = [], {}
    if references:
        folder = output / 'training_references'
        folder.mkdir()
    for number, row in enumerate(references):
        source = Path(row['image'])
        if sha256(source) != row['image_sha256']:
            raise ValueError('actual SD2.1 training reference changed')
        target = folder / f'train_{number:03d}{source.suffix}'
        shutil.copyfile(source, target)
        hashes[str(target)] = sha256(target)
        if hashes[str(target)] != row['image_sha256']:
            raise ValueError('SD2.1 training reference copy differs')
        copied.append({**row, 'review_image': target.relative_to(output).as_posix()})
    write_json(output / 'training_references.json', {'images': copied})
    cards = ''.join(f'<figure><img width="180" src="{html.escape(row["review_image"], quote=True)}">'
                    f'<figcaption>{html.escape(row["caption"])}</figcaption></figure>' for row in copied)
    (output / 'training_references.html').write_text('<!doctype html><meta charset="utf-8">'
        + f'<h1>{len(copied)} actual training RGB images and full captions</h1><section style="display:flex;flex-wrap:wrap">'
        + cards + '</section>', encoding='utf8')
    if copied:
        lines = [textwrap.wrap(row['caption'], 27) for row in copied]
        height = 175 + 17 * max(map(len, lines))
        sheet = Image.new('RGB', (1000, ((len(copied) + 4) // 5) * height + 30), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 5), 'ACTUAL training RGB and captions; full-resolution copies linked in training_references.html', font=font, fill='black')
        for number, row in enumerate(copied):
            x, y = number % 5 * 200, number // 5 * height + 30
            with Image.open(output / row['review_image']) as image:
                image = image.convert('RGB')
                image.thumbnail((184, 160))
                sheet.paste(image, (x + 5, y))
            for line, text in enumerate(lines[number]):
                draw.text((x + 5, y + 162 + line * 17), text, font=font, fill='black')
        sheet.save(output / 'training_references_sheet.png')
    return hashes


def review(output, rows, *, training=False):
    from experiments.natural_texture_material_v1.probe_review import _font
    font = _font()
    columns = min(5, max(1, len(rows)))
    height = 270
    sheet = Image.new('RGB', (columns * 230, ((len(rows) + columns - 1) // columns) * height + 35), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), 'SD2.1-base512 | PNDM100 | fp16 | checker | per-image seeded generator', font=font, fill='black')
    cards = []
    for number, row in enumerate(rows):
        x, y = number % columns * 230 + 4, number // columns * height + 35
        if row.get('status'):
            with Image.open(output / row['image_path']) as image:
                sheet.paste(image.resize((220, 220)), (x, y))
        draw.text((x, y + 222), f'{row["label"]}; seed{row["seed"]}', font=font, fill='black')
        draw.text((x, y + 240), row.get('status', 'planned'), font=font, fill='black')
        cards.append('<figure>' + (f'<img width="256" src="{row["image_path"]}">' if row.get('status') else '<p>Planned</p>')
                     + f'<figcaption>{html.escape(row["prompt"])}; seed{row["seed"]}</figcaption></figure>')
    sheet.save(output / 'contact_sheet.png')
    reference = ('<p><a href="../training_references.html">Actual training RGB and full captions</a></p>'
        '<a href="../training_references_sheet.png"><img width="800" src="../training_references_sheet.png"></a>' if training else '')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>SD2.1 independent branch probes</h1>'
        + reference + '<a href="contact_sheet.png">Result contact sheet</a><section style="display:flex;flex-wrap:wrap">'
        + ''.join(cards) + '</section>', encoding='utf8')


def _generate(pipe, rows, output, device, token=None):
    import torch
    generated = []
    with (output / 'generation_status.jsonl').open('a', encoding='utf8') as ledger:
        for row in rows:
            kwargs = {'cross_attention_kwargs': attention_masks(pipe, row['prompt'], [token], row.get('cfg', 3.5))} if token else {}
            result = pipe(row['prompt'], num_inference_steps=100, guidance_scale=row.get('cfg', 3.5), width=512, height=512,
                          generator=torch.Generator(device=device).manual_seed(row['seed']), **kwargs)
            if (result.nsfw_content_detected is None or len(result.nsfw_content_detected) != 1
                    or len(result.images) != 1 or result.images[0].mode != 'RGB' or result.images[0].size != (512, 512)):
                raise ValueError('SD2.1 inference requires one512px RGB image and a safety result')
            path = output / row['image_path']
            path.parent.mkdir(exist_ok=True)
            with path.open('xb') as handle:
                result.images[0].save(handle, format='PNG')
            record = {**row, 'status': 'safety_filtered' if bool(result.nsfw_content_detected[0]) else 'ok',
                'image_sha256': sha256(path), 'subject_adapter_loaded': token == '<S*>',
                'material_adapter_loaded': token == '<M*>', 'texture_adapter_loaded': token == '<T*>'}
            generated.append(record)
            ledger.write(json.dumps(record) + '\n')
            ledger.flush()
    return generated


def run(branch, method, train_run, output, seeds=SEEDS, smoke=False, device='cuda:0', execute=False):
    from experiments.sd21_wood_branches_v1.prepare_model import verify_model
    literal = branch == 'literal'
    seeds = _seeds(seeds)
    if type(smoke) is not bool or (not literal and (branch not in contract.BRANCHES or method not in contract.METHODS)):
        raise ValueError('select literal stock controls or a declared independent branch/method')
    output = resolve_run_dir(Path(output))
    if (output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(Path(MODEL_DIR).resolve())
            or (not literal and (train_run is None or output.is_relative_to(Path(train_run).resolve())
                                 or Path(train_run).resolve().is_relative_to(output)))):
        raise ValueError('SD2.1 inference requires a fresh external directory separate from training and model')
    revision = git_info()
    receipt = verify_model(MODEL_DIR)
    locks = snapshots.model_locks(MODEL_DIR, receipt)
    output.mkdir(parents=True)
    state = {'status': 'running' if execute else 'dry_run', 'git': revision, 'branch': branch, 'method': method,
        'base_model': receipt['model'], 'sampling': {**SAMPLING, 'seeds': [42] if smoke else list(seeds)},
        'smoke': smoke, 'training': False, 'monitoring': False, 'generated_images': 0, 'completed_steps': []}
    reference_hashes = {}
    try:
        exported = None if literal else snapshots.audit_export(train_run, branch, method, output / 'ckpts', smoke)
        if exported:
            locks.update(exported['source_locks'])
        references = [] if literal else exported['references']
        reference_hashes = reference_review(output, references)
        state['training_images'] = len(references)
        groups = {'stock': literal_rows(seeds, smoke)} if literal else {
            step: prompt_rows(branch, method, step, seeds, smoke) for step in exported['steps']}
        state['expected_images'] = sum(map(len, groups.values()))
        if literal:
            state['sampling'].update(cfgs=[3.5] if smoke else [3.5, 5.0, 7.5], seeds=[42] if smoke else list(seeds))
        for step, rows in groups.items():
            directory = output / ('stock' if literal else f'step{step}')
            directory.mkdir()
            (directory / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
            generated = []
            if execute:
                import torch
                snapshots.check_locks(locks)
                if literal:
                    pipe = load_pipeline([], device, model_dir=MODEL_DIR, stock=True)
                    generated.extend(_generate(pipe, rows, directory, device))
                    del pipe
                else:
                    learned = [row for row in rows if row['condition'].startswith('learned_')]
                    pipe = load_pipeline([exported['steps'][step]], device, model_dir=MODEL_DIR)
                    generated.extend(_generate(pipe, learned, directory, device, contract.TOKENS[branch]))
                    del pipe
                    native = [row for row in rows if not row['condition'].startswith('learned_')]
                    if native:
                        pipe = load_pipeline([], device, model_dir=MODEL_DIR)
                        generated.extend(_generate(pipe, native, directory, device))
                        del pipe
                torch.cuda.empty_cache()
                if len(generated) != len(rows) or len({row['id'] for row in generated}) != len(rows):
                    raise ValueError('incomplete SD2.1 probe matrix')
                state['generated_images'] += len(generated)
            review(directory, generated or rows, training=not literal)
            state['completed_steps'].append(step)
            write_json(output / 'status.json', state)
        snapshots.check_locks(locks)
        if reference_hashes:
            snapshots.check_locks(reference_hashes)
        if git_info() != revision:
            raise ValueError('SD2.1 checkout changed during inference')
        state['status'] = 'succeeded' if execute else 'dry_run_verified'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        state.update(source_locks=locks, reference_copy_hashes=reference_hashes)
        write_json(output / 'status.json', state)
        write_json(output / 'provenance.json', state)
        links = ''.join(f'<p><a href="{("stock" if step == "stock" else "step" + str(step))}/index.html">{step}</a></p>'
                        for step in state['completed_steps'])
        (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>SD2.1-base512 wood branch probes</h1>'
            + f'<p>{state["status"]};{state["generated_images"]} generated images</p>'
            + '<p><a href="training_references.html">Actual training RGB and captions</a></p>' + links, encoding='utf8')
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch', required=True, choices=(*contract.BRANCHES, 'literal'))
    parser.add_argument('--method', choices=contract.METHODS)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    run(args.branch, args.method, args.run, args.output, args.seeds, args.smoke, args.device, args.execute)
