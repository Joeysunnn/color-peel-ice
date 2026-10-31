"""Stock SD1.5 spoon prompt: three CFG values by five independent seeds."""

import argparse
import html
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

from experiments.sd15_token_local_kv_v1.contract import PROTOCOL, read_json, sha256
from experiments.sd_native_comparison_v1.generate import package_versions, write_json
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
PROMPT = 'a photo of a spoon'
CFGS = (3.5, 5.0, 7.5)
SEEDS = (42, 43, 44, 45, 46)
TOKENS = ('<S*>', '<C*>', '<M*>', '<T*>')
PREVIOUS = Path('/var/tmp/colorpeel-r12user5/runs/wood_spoon_composition_v1/'
    '20261031-200526__T500_rank64_seed42_controls8__a504a40__42/experiment/cases/spoon_without_t')


def rows():
    return [{'id': f'cfg{str(cfg).replace(".", "p")}__seed{seed}', 'cfg': cfg, 'seed': seed,
        'prompt': PROMPT, 'image_path': f'images/cfg{str(cfg).replace(".", "p")}__seed{seed}.png'}
        for cfg in CFGS for seed in SEEDS]


def review(output, state):
    existing = {(r['cfg'], r['seed']): r for r in state['records']}
    cells = []
    for seed in SEEDS:
        cells.append(f'<tr><th>seed{seed}</th>')
        for cfg in CFGS:
            row = existing.get((cfg, seed))
            cells.append('<td>' + (f'<a href="{row["image_path"]}"><img width="384" src="{row["image_path"]}"></a>'
                + f'<p>{row["status"]}</p>' if row else '<p>Pending</p>') + '</td>')
        cells.append('</tr>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Stock SD1.5 spoon CFG</title>'
        + '<style>body{font:16px sans-serif}td,th{padding:8px;vertical-align:top}img{max-width:100%}</style>'
        + f'<h1>{html.escape(PROMPT)}</h1><p>{state["status"]};{len(state["records"])}/15 images;'
        + ' stock StableDiffusionPipeline/UNet/attention;PNDM100;512px;fp16;checker;no learned tokens or adapters.</p>'
        + '<table><tr><th>Seed</th>' + ''.join(f'<th>CFG{cfg:g}</th>' for cfg in CFGS) + '</tr>'
        + ''.join(cells) + '</table>', encoding='utf8')


def contact_sheet(output, records):
    canvas = Image.new('RGB', (3 * 384, 80 + 5 * 420), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 8), 'Stock SD1.5 | a photo of a spoon | PNDM100 |512px|fp16|checker', font=font, fill='black')
    by_key = {(r['cfg'], r['seed']): r for r in records}
    for col, cfg in enumerate(CFGS):
        draw.text((col * 384 + 8, 40), f'CFG{cfg:g}', font=font, fill='black')
        for line, seed in enumerate(SEEDS):
            row = by_key[cfg, seed]
            with Image.open(output / row['image_path']) as image:
                canvas.paste(image.resize((376, 376)), (col * 384 + 4, 80 + line * 420))
            draw.text((col * 384 + 8, 460 + line * 420), f'seed{seed} | {row["status"]}', font=font, fill='black')
    canvas.save(output / 'contact_sheet.png')


def compare_previous(output, row):
    if row['cfg'] != 3.5 or row['seed'] != 42 or not (PREVIOUS / 'generation_status.jsonl').is_file():
        return
    old = json.loads((PREVIOUS / 'generation_status.jsonl').read_text().splitlines()[0])
    if (old['prompt'] != PROMPT or old['seed'] != 42 or old['subject_adapter_loaded']
            or old['texture_adapter_loaded']):
        raise ValueError('previous unadapted spoon baseline differs')
    image = PREVIOUS / old['image_path']
    if sha256(image) != old['image_sha256']:
        raise ValueError('previous baseline image changed')
    with Image.open(image) as before, Image.open(output / row['image_path']) as after:
        pixels_equal = before.convert('RGB').tobytes() == after.convert('RGB').tobytes()
    write_json(output / 'stock_vs_custom_seed42.json', {'previous_image': str(image),
        'previous_image_sha256': old['image_sha256'], 'stock_image_sha256': row['image_sha256'],
        'pixels_identical': pixels_equal, 'cfg': 3.5, 'seed': 42, 'prompt': PROMPT,
        'note': 'Read-only comparison with the existing no-adapter common-loader baseline.'})


def report(output, state):
    write_json(output / 'provenance.json', state)
    folder = output / 'repro_outputs'
    folder.mkdir(exist_ok=True)
    write_json(folder / 'status.json', state)
    (folder / 'SUMMARY.md').write_text(f"{state['status']};{len(state['records'])}/15 images. Stock SD1.5 spoon CFG sweep;no training.\n", encoding='utf8')
    (folder / 'COMMANDS.md').write_text(state['command'] + '\n', encoding='utf8')
    (folder / 'COMPARABILITY_REPORT.md').write_text(
        'Only CFG varies at each fixed seed. Prompt=a photo of a spoon;PNDM100,512px,fp16,shared checker. '
        'All15 are freshly generated through stock Diffusers pipeline/UNet/attention. No custom UNet, learned embedding, '
        'adapter, negative prompt, reference image or training. Reset CUDA generator for every image. '
        'The prior custom-entry seed42/CFG3.5 PNG is compared read-only when available.\n', encoding='utf8')
    review(output, state)


def run(output, device='cuda:0', execute=False):
    output = resolve_run_dir(Path(output))
    revision = git_info()
    spec = read_json(PROTOCOL)['model']
    model = Path(spec['root'])
    if output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(model):
        raise ValueError('use a fresh external output, preserving old results and model files')
    receipt_path = model / 'model_source.json'
    receipt = read_json(receipt_path)
    if receipt['model'] != spec:
        raise ValueError('frozen SD1.5 model receipt differs')
    locks = {str(model / name): value for name, value in receipt['files'].items()}
    locks.update({str(receipt_path): sha256(receipt_path), str(PROTOCOL): sha256(PROTOCOL), str(Path(__file__)): sha256(__file__)})
    if any(sha256(path) != digest for path, digest in locks.items()):
        raise ValueError('frozen model or source bytes changed')
    output.mkdir(parents=True)
    state = {'status': 'running' if execute else 'dry_run_verified', 'git': revision, 'model': spec,
        'source_locks': locks, 'prompt': PROMPT, 'sampling': {'scheduler': 'PNDM', 'num_inference_steps': 100,
        'cfgs': list(CFGS), 'seeds': list(SEEDS), 'width': 512, 'height': 512, 'dtype': 'float16',
        'safety_checker': True, 'generator': 'fresh_device_generator_per_image', 'negative_prompt': None},
        'expected_images': 15, 'records': [], 'training': False, 'monitoring': False, 'learned_tokens': [],
        'adapters': False, 'environment': package_versions(), 'safety_checker_source': receipt['safety_checker_source'],
        'command': f'{sys.executable} -u -B -m experiments.sd15_native_spoon_cfg_v1.run --output {output} --device {device}'
            + (' --execute' if execute else '')}
    (output / 'generation_manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows()), encoding='utf8')
    report(output, state)
    if not execute:
        return state
    try:
        import torch
        from diffusers import PNDMScheduler, StableDiffusionPipeline
        scheduler = PNDMScheduler.from_pretrained(str(model), subfolder='scheduler', local_files_only=True)
        pipe = StableDiffusionPipeline.from_pretrained(str(model), scheduler=scheduler,
            torch_dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=False).to(device)
        processors = sorted({type(p).__module__ + '.' + type(p).__name__ for p in pipe.unet.attn_processors.values()})
        if (not type(pipe).__module__.startswith('diffusers.') or not type(pipe.unet).__module__.startswith('diffusers.')
                or any(not name.startswith('diffusers.') or 'lora' in name.lower() for name in processors)
                or any(token in pipe.tokenizer.get_vocab() for token in TOKENS)
                or pipe.safety_checker is None or not isinstance(pipe.scheduler, PNDMScheduler)):
            raise ValueError('require stock pipeline/UNet/attention, unmodified tokens, no LoRA, PNDM and checker')
        state['runtime'] = {'pipeline_class': type(pipe).__module__ + '.' + type(pipe).__name__,
            'unet_class': type(pipe.unet).__module__ + '.' + type(pipe.unet).__name__, 'attention_processors': processors,
            'scheduler_config': dict(pipe.scheduler.config), 'gpu': torch.cuda.get_device_name(device),
            'dtype': str(pipe.unet.dtype), 'safety_checker': True}
        pipe.set_progress_bar_config(disable=True)
        for row in rows():
            result = pipe(row['prompt'], num_inference_steps=100, guidance_scale=row['cfg'], width=512, height=512,
                generator=torch.Generator(device=device).manual_seed(row['seed']))
            if (result.nsfw_content_detected is None or len(result.nsfw_content_detected) != 1
                    or len(result.images) != 1 or result.images[0].mode != 'RGB' or result.images[0].size != (512, 512)):
                raise ValueError('require one512px RGB image and an explicit safety result')
            path = output / row['image_path']
            path.parent.mkdir(exist_ok=True)
            with path.open('xb') as handle:
                result.images[0].save(handle, format='PNG')
            record = {**row, 'status': 'safety_filtered' if bool(result.nsfw_content_detected[0]) else 'ok',
                      'image_sha256': sha256(path), 'subject_adapter_loaded': False, 'texture_adapter_loaded': False}
            state['records'].append(record)
            with (output / 'generation_status.jsonl').open('a', encoding='utf8') as ledger:
                ledger.write(json.dumps(record) + '\n')
            compare_previous(output, record)
            report(output, state)
            print('STOCK_SPOON_CFG', row['id'], record['status'], flush=True)
        if git_info() != revision or any(sha256(path) != digest for path, digest in locks.items()):
            raise ValueError('execution revision or frozen sources changed')
        contact_sheet(output, state['records'])
        state.update(status='succeeded', generated_images=15,
                     safety_filtered=sum(r['status'] == 'safety_filtered' for r in state['records']))
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report(output, state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.device, args.execute)))
