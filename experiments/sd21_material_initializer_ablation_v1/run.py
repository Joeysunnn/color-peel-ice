"""Three fresh Material8 runs differing only in initializer; automatic150/200 transfer."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.sd15_token_local_kv_v1.contract import read_json
from experiments.sd21_qwen_initializers_v1 import contract
from experiments.sd21_qwen_initializers_v1.inference import register_initial_input
from experiments.sd21_qwen_initializers_v1.run import ENVIRONMENT, require_idle
from experiments.sd21_qwen_material250_v1 import run as material
from experiments.subject_multicolor_multimaterial_b.training import _write
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
STUDY = 'sd21_material_initializer_ablation_v1'
WORDS = ('painted', 'glossy', 'material')
STEPS = (150, 200)
GPUS = dict(zip(WORDS, (0, 1, 2)))


def comparable_config(config):
    """Remove only experiment identity, GPU assignment and the requested intervention."""
    value = copy.deepcopy(config)
    for key in ('study', 'variant'):
        value['run'].pop(key)
    value['environment'].pop('CUDA_VISIBLE_DEVICES')
    for key in ('initializer_token', 'initializer_manifest'):
        value['args'].pop(key)
    value['data_lineage'].pop('initializer_manifest')
    return value


def verify_comparability(configs):
    if set(configs) != set(WORDS):
        raise ValueError('require all three requested initializer configs')
    reference = comparable_config(configs['painted'])
    if any(comparable_config(item) != reference for item in configs.values()):
        raise ValueError('training changed beyond initializer, experiment identity or GPU')
    for word, config in configs.items():
        if (config['args']['initializer_token'] != word or config['args']['max_train_steps'] != 200
                or config['args']['checkpoint_steps'] != [150, 200] or config['run']['seed'] != 42):
            raise ValueError('initializer, saved steps or training seed differs')


def comparison_pages(output, manifest):
    links = []
    for step in STEPS:
        body = [f'<h1>Material{step}: initializer comparison</h1>',
            '<p>Same Material8, training seed42, LoRA64. Material only; PNDM100 /512px /FP16 /CFG3.5 /empty negative. '
            'Images appear as runs finish. Filter status and execution status are in each variant gallery.</p>',
            '<p><a href="index.html">All runs</a></p><table><tr><th>case / seed</th>'
            + ''.join(f'<th>{word}</th>' for word in WORDS) + '</tr>']
        for row in material.plan(manifest, step):
            body.append(f'<tr><th>{row["case"]} / {row["seed"]}</th>')
            for word in WORDS:
                path = f'{word}/inference/step{step}/{row["image_path"]}'
                body.append(f'<td><a href="{path}"><img loading="lazy" width="256" src="{path}" '
                            f'alt="{word}: pending image"></a></td>')
            body.append('</tr>')
        name = f'compare_step{step}.html'
        (output / name).write_text('<!doctype html><meta charset="utf-8">'
            '<style>body{font:16px sans-serif}td{vertical-align:top}</style>' + ''.join(body) + '</table>', encoding='utf8')
        links.append(f'<p><a href="{name}">Step{step}: painted / glossy / material</a></p>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Material initializer ablation</h1>'
        '<p>3 initializers x2 checkpoints x4 objects x11 seeds =264 images. See status.json for execution state.</p>'
        + ''.join(links) + ''.join(f'<p><a href="{word}/inference/index.html">{word}: individual galleries</a></p>'
                                   for word in WORDS), encoding='utf8')


def run(output):
    from transformers import CLIPTokenizer
    from src.train.initializer_token_utils import build_material_ablation_manifest, load_initializer_manifest

    os.environ.update(ENVIRONMENT)
    require_idle(GPUS.values())
    output = resolve_run_dir(Path(output))
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output)
            for p in (ROOT, material.SOURCE.resolve(), contract.MODEL_DIR.resolve())):
        raise ValueError('require fresh output outside protected code, source and model directories')
    revision = git_info()
    source = read_json(material.SOURCE / 'source_audit.json')
    tokenizer = CLIPTokenizer.from_pretrained(str(contract.MODEL_DIR), subfolder='tokenizer', local_files_only=True)
    parent = load_initializer_manifest(source['initializer_manifest'], tokenizer, contract.MODEL_DIR)
    # Validate every initializer before any GPU training starts; keep original Qwen bytes intact.
    manifests = {word: build_material_ablation_manifest(source['initializer_manifest'], tokenizer,
                 contract.MODEL_DIR, word) for word in WORDS}
    output.mkdir(parents=True)
    (output / 'preflight').mkdir()
    configs = {}
    for word, manifest in manifests.items():
        path = output / 'preflight' / (word + '.json')
        _write(path, manifest)
        configs[word] = material.build_config(material.SOURCE, GPUS[word], STEPS, initializer_manifest=path)
    verify_comparability(configs)
    _write(output / 'config_comparability.json', {'verified': True, 'configs': configs,
        'allowed_differences': ['initializer_token', 'initializer_manifest/provenance', 'study/variant', 'GPU'],
        'no_bitwise_training_equivalence_claim': True})
    comparison_pages(output, parent)
    state = {'status': 'running', 'expected_images': 264, 'generated_images': 0,
        'initializers': list(WORDS), 'checkpoint_steps': list(STEPS), 'gpu_mapping': GPUS,
        'training_seed': 42, 'new_training_git': revision, 'fresh_training': True, 'quality_verified': False}
    _write(output / 'status.json', state)

    def launch(word):
        command = [sys.executable, '-u', '-B', '-m', 'experiments.sd21_qwen_material250_v1.run',
                   '--output', str(output / word), '--gpu', str(GPUS[word]),
                   '--checkpoint-steps', '150', '200', '--initializer', word]
        _write(output / f'{word}_command.json', {'command': command, 'gpu': GPUS[word]})
        with (output / f'{word}.log').open('x', encoding='utf8') as stream:
            subprocess.run(command, cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(GPUS[word])},
                           stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, check=True)

    try:
        failures = {}
        require_idle(GPUS.values())
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(launch, word): word for word in WORDS}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    failures[futures[future]] = str(error)
        if failures:
            raise RuntimeError(f'variant failure; all outputs preserved: {failures}')
        verify_comparability({word: read_json(output / word / 'config.json') for word in WORDS})
        generated, initial_inputs = [], {}
        for word in WORDS:
            result = read_json(output / word / 'status.json')
            if result['status'] != 'succeeded' or result['generated_images'] != 88:
                raise ValueError('each variant must complete88 images')
            manifest = read_json(output / word / 'source_audit.json')['initializers']
            for step in STEPS:
                folder = output / word / 'inference' / f'step{step}'
                for row in material.collect(folder, material.plan(manifest, step)):
                    register_initial_input(initial_inputs, row['seed'], row['initial_denoiser_input'])
                    generated.append({**row, 'id': word + '__' + row['id'], 'initializer_variant': word,
                        'image_path': f'{word}/inference/step{step}/{row["image_path"]}'})
        if len(generated) != 264 or git_info() != revision:
            raise ValueError('image count or checkout changed')
        (output / 'generation_ledger.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in generated), encoding='utf8')
        _write(output / 'initial_input_audit.json', {'same_seed_equal_across_initializers_and_checkpoints': True,
            'generated_images': len(generated), 'by_seed': initial_inputs})
        state.update(status='succeeded', generated_images=len(generated),
                     safety_filtered_images=sum(row['nsfw_content_detected'] for row in generated))
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        _write(output / 'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
